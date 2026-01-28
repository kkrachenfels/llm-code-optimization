import random
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging
from pathlib import Path

from .compiler import CppCompiler
from .reward import AdaptiveRewardFunction
from .datasets import CodeDataset

logger = logging.getLogger(__name__)


class SimpleCodeOptimizationTrainer:
    """Simplified RL trainer for code optimization using REINFORCE."""

    def __init__(
        self,
        model_name: str,
        dataset: CodeDataset,
        sampling_strategy: str = "random",
        output_dir: str = "checkpoints",
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        max_length: int = 6144,
        use_8bit: bool = False,
        train_programs: Optional[int] = None,
        test_programs: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model_name: HuggingFace model name
            dataset: CodeDataset containing programs to optimize
            sampling_strategy: How to sample programs ('random' or 'sequential')
            output_dir: Directory to save checkpoints
            learning_rate: Learning rate
            batch_size: Batch size for training
            max_length: Maximum sequence length
            use_8bit: Whether to use 8-bit quantization
            train_programs: Number of programs per epoch for training (for epoch mode)
            test_programs: Number of programs to hold out for testing (for epoch mode)
            seed: Random seed for reproducible train/test splits
        """
        self.dataset = dataset
        self.train_programs = train_programs
        self.test_programs = test_programs

        # Split dataset if using epoch mode
        if train_programs is not None and test_programs is not None:
            total_needed = train_programs + test_programs
            if len(dataset) < total_needed:
                raise ValueError(
                    f"Dataset has {len(dataset)} programs, but need {total_needed} "
                    f"({train_programs} train + {test_programs} test)"
                )
            # Create train/test indices with random split
            all_indices = list(range(len(dataset)))
            if seed is not None:
                random.seed(seed)
            random.shuffle(all_indices)
            self.test_indices = all_indices[:test_programs]
            self.train_indices = all_indices[test_programs:]
            logger.info(f"Train/test split: {len(self.train_indices)} train, {len(self.test_indices)} test")
        else:
            self.train_indices = None
            self.test_indices = None
        self.sampling_strategy = sampling_strategy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.max_length = max_length

        # Current program state (will be updated each step in dataset mode)
        self.current_program = None
        self.original_code = None
        self.baseline_runtime = None
        self.reward_function = None
        self.compiler = None  # Will be created per program with appropriate config
        self.kernel_info = None  # Set when kernel_markers are configured
        self.correctness_compiler = None  # Set when correctness_config is configured
        self.reference_output = None  # Reference output for correctness checking

        # Track programs that failed to compile to avoid retrying them
        self._failed_programs: set = set()

        # Load and benchmark the first program (retry if compilation fails)
        max_retries = len(dataset)
        for attempt in range(max_retries):
            try:
                self._load_next_program()
                break
            except RuntimeError as e:
                logger.warning(f"Initial program load failed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise RuntimeError("All programs failed to compile during initialization")

        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "device_map": "auto",  # Automatically distribute across available GPUs
            "torch_dtype": torch.bfloat16,  # Use bf16 for better numerical stability than fp16
        }
        if use_8bit:
            model_kwargs["load_in_8bit"] = True
            del model_kwargs["torch_dtype"]  # 8-bit handles its own dtype

        logger.info(f"Loading model with device_map='auto' to use all available GPUs...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # With device_map="auto", model is already on device(s)
        self.device = self.model.device
        logger.info(f"Model loaded. Device: {self.device}, Memory footprint: {self.model.get_memory_footprint() / 1e9:.2f} GB")

        # Get the model's actual max position embeddings and use the smaller of user-specified and model limit
        model_max_length = getattr(self.model.config, 'max_position_embeddings', None) or \
                           getattr(self.model.config, 'n_positions', None) or \
                           self.max_length
        if self.max_length > model_max_length:
            logger.warning(f"Requested max_length ({self.max_length}) exceeds model's max position embeddings ({model_max_length}). Using {model_max_length}.")
            self.max_length = model_max_length

        self.model.train()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _extract_kernel_region(self, code: str, markers: tuple) -> Optional[dict]:
        """
        Extract the kernel region delimited by markers (e.g. #pragma scop / #pragma endscop).

        Returns a dict with:
            prefix: everything up to and including the start marker line
            kernel_body: the code between the markers
            suffix: everything from the end marker line onward
            kernel_context: the kernel function signature + variable declarations for prompt context

        Returns None if markers are not found in the code.
        """
        start_marker, end_marker = markers
        lines = code.split('\n')

        start_idx = None
        end_idx = None
        for i, line in enumerate(lines):
            if start_marker in line and start_idx is None:
                start_idx = i
            elif end_marker in line and start_idx is not None:
                end_idx = i
                break

        if start_idx is None or end_idx is None:
            return None

        prefix = '\n'.join(lines[:start_idx + 1]) + '\n'
        kernel_body = '\n'.join(lines[start_idx + 1:end_idx])
        suffix = '\n'.join(lines[end_idx:])

        # Extract kernel function context: walk backward from start_marker
        # to find the function signature and variable declarations
        context_lines = []
        for i in range(start_idx - 1, -1, -1):
            line = lines[i]
            context_lines.insert(0, line)
            # Stop when we hit the function opening brace or 'static' keyword
            if line.strip() == '{' or line.strip().startswith('static'):
                break

        kernel_context = '\n'.join(context_lines)

        return {
            'prefix': prefix,
            'kernel_body': kernel_body,
            'suffix': suffix,
            'kernel_context': kernel_context,
        }

    def _outputs_match(self, reference: str, candidate: str, rtol: float = 1e-4) -> bool:
        """
        Check if two program outputs match approximately.

        Parses all floating-point numbers from both outputs and compares them
        with relative tolerance to allow for FP reordering effects.
        """
        ref_floats = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', reference)
        cand_floats = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', candidate)

        if len(ref_floats) != len(cand_floats):
            logger.debug(f"Output mismatch: {len(ref_floats)} vs {len(cand_floats)} values")
            return False

        for i, (r, c) in enumerate(zip(ref_floats, cand_floats)):
            try:
                rv, cv = float(r), float(c)
            except ValueError:
                continue
            if rv == 0.0 and cv == 0.0:
                continue
            if rv == 0.0:
                if abs(cv) > rtol:
                    logger.debug(f"Output mismatch at index {i}: ref=0.0, cand={cv}")
                    return False
            elif abs(rv - cv) / max(abs(rv), 1e-15) > rtol:
                logger.debug(f"Output mismatch at index {i}: ref={rv}, cand={cv}")
                return False

        return True

    def _load_next_program(self):
        """Load the next program from the dataset and compute its baseline."""
        # Get next program based on sampling strategy
        if self.sampling_strategy == "random":
            self.current_program = self.dataset.get_random_program()
        else:  # sequential
            self.current_program = self.dataset.get_next_program()

        self.original_code = self.current_program['code']
        program_name = self.current_program['name']

        logger.info(f"Loading program: {program_name}")

        # Create compiler with program-specific configuration
        compiler_config = self.current_program.get('compiler_config', {})
        compiler_kwargs = {k: v for k, v in compiler_config.items() if k not in ('kernel_markers', 'correctness_config')}
        self.compiler = CppCompiler(**compiler_kwargs)

        # Compute baseline for this program
        logger.info("Computing baseline runtime...")
        success, self.baseline_runtime, error = self.compiler.compile_and_run(
            self.original_code, num_runs=5
        )
        if not success:
            raise RuntimeError(
                f"Failed to benchmark baseline for {program_name}: {error}"
            )
        logger.info(f"Baseline runtime: {self.baseline_runtime:.2f} microseconds")

        # Create reward function for this program
        self.reward_function = AdaptiveRewardFunction(self.baseline_runtime)

        # Extract kernel region if markers are configured
        kernel_markers = compiler_config.get('kernel_markers')
        if kernel_markers:
            kernel_info = self._extract_kernel_region(self.original_code, kernel_markers)
            if kernel_info:
                self.kernel_info = kernel_info
                logger.info(f"Kernel-only mode: extracted {len(kernel_info['kernel_body'])} chars of kernel code")
            else:
                self.kernel_info = None
                logger.warning(f"Kernel markers configured but not found in {program_name}, using full-file mode")
        else:
            self.kernel_info = None

        # Setup correctness checking if configured
        correctness_config = compiler_config.get('correctness_config')
        if correctness_config:
            self.correctness_compiler = CppCompiler(**correctness_config)
            success, stdout, stderr, error = self.correctness_compiler.compile_and_get_output(self.original_code)
            if success:
                self.reference_output = stderr
                logger.info(f"Correctness check: captured {len(self.reference_output)} chars of reference output")
            else:
                self.correctness_compiler = None
                self.reference_output = None
                logger.warning(f"Correctness check setup failed for {program_name}: {error}")
        else:
            self.correctness_compiler = None
            self.reference_output = None

    def create_prompt(self, code: str) -> str:
        """Create optimization prompt for the model."""
        # Kernel-only mode: only ask for the optimized kernel body
        if self.kernel_info is not None:
            kernel_context = self.kernel_info['kernel_context']
            kernel_body = self.kernel_info['kernel_body']
            prompt = f"""<|im_start|>system
You are a C code optimizer specializing in high-performance loop optimization. You ONLY output valid C loop code. No explanations, no markdown, no function signatures. Just the optimized loop body.
<|im_end|>
<|im_start|>user
Optimize the following loop body for maximum runtime performance.

Function context (do NOT output this, just use it to understand variable types):
{kernel_context}

Loop body to optimize:
{kernel_body}

Focus on: loop tiling, loop reordering for cache efficiency, reducing redundant computations, enabling vectorization.
Output ONLY the optimized loop code. No #pragma lines, no function wrapper, no explanations.
<|im_end|>
<|im_start|>assistant
"""
            logger.info(f"Created kernel-only prompt ({len(kernel_body)} chars of kernel code)")
            return prompt

        # Detect if this is C or C++ code
        is_cpp = 'iostream' in code or 'std::' in code or 'class ' in code or 'namespace' in code
        lang = "C++" if is_cpp else "C"

        # Detect if this is polybench code
        is_polybench = 'polybench.h' in code or 'POLYBENCH_' in code

        if is_polybench:
            prompt = f"""<|im_start|>system
You are a C code optimizer specializing in high-performance computing. You ONLY output valid C code. No explanations, no markdown. Just the raw optimized C code.
<|im_end|>
<|im_start|>user
Optimize this PolyBench/C code for maximum runtime performance.

Context about PolyBench macros (do not modify these, just use them):
- DATA_TYPE is typically double
- POLYBENCH_2D(arr,N,M,n,m) declares a 2D array
- POLYBENCH_1D(arr,N,n) declares a 1D array
- Array indices use standard C syntax: arr[i][j]

Focus on optimizing:
- Loop ordering for cache efficiency
- Loop tiling/blocking
- Reducing redundant computations
- Enabling vectorization

Keep all #include statements, function signatures, and macro usage exactly the same. Only optimize the loop bodies and computations.

{code}
<|im_end|>
<|im_start|>assistant
"""
        else:
            prompt = f"""<|im_start|>system
You are a {lang} code optimizer. You ONLY output valid {lang} code. No explanations, no markdown, no comments about changes. Just the raw optimized {lang} code. Preserve all #include statements and function signatures.
<|im_end|>
<|im_start|>user
Optimize this {lang} code for maximum runtime performance. Keep the same function signatures and #include statements:

{code}
<|im_end|>
<|im_start|>assistant
"""
        logger.info(f"Created prompt for {lang} code with {is_polybench and 'PolyBench' or 'standard'} context")
        # logger.debug(f"Prompt: {prompt}")
        return prompt

    def generate_optimizations(self, num_samples: int) -> tuple:
        """
        Generate optimized code samples.

        Returns:
            Tuple of (prompts, responses, log_probs)
        """
        prompt = self.create_prompt(self.original_code)
        prompts = [prompt] * num_samples

        # Tokenize prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Move inputs to the same device as the model's first layer
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        # Log input length to help debug truncation issues
        input_len = inputs['input_ids'].shape[1]
        logger.info(f"Prompt tokenized to {input_len} tokens")

        # Calculate max_new_tokens to stay within model's max_length
        max_new_tokens = self.max_length - input_len
        if max_new_tokens < 100:
            logger.warning(f"Very little room for generation ({max_new_tokens} tokens). Consider shorter prompts.")

        # Generate with log probabilities
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        generated_sequences = outputs.sequences

        # Decode responses
        responses = []
        for seq in generated_sequences:
            # Remove prompt tokens
            response_tokens = seq[inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            responses.append(response)

        # Skip log_probs computation during generation - we'll compute with gradients in train_step
        # This avoids potential indexing issues and saves memory
        log_probs_list = [0.0] * len(responses)

        return prompts, responses, log_probs_list

    def evaluate_code(self, responses: List[str]) -> tuple:
        """Evaluate generated code and compute rewards.

        Returns:
            Tuple of (rewards, runtimes) where runtimes contains float values
            for successful runs and None for failures.
        """
        rewards = []
        runtimes = []

        for idx, response in enumerate(responses):
            logger.debug(f"--- Evaluating response {idx + 1}/{len(responses)} ---")

            if self.kernel_info is not None:
                # Kernel-only mode: response is the kernel code directly
                # Just clean up special tokens and splice back into the full file
                code = re.sub(r'<\|im_end\|>.*', '', response, flags=re.DOTALL).strip()
                code = re.sub(r'<\|endoftext\|>.*', '', code, flags=re.DOTALL).strip()
                # Strip markdown code blocks if the model wrapped output in them
                code = re.sub(r'^```(?:c|cpp|c\+\+)?\s*\n', '', code)
                code = re.sub(r'\n```\s*$', '', code)

                if not code.strip():
                    logger.warning(f"Response {idx + 1}: Empty kernel output")
                    reward = self.reward_function.compute_reward(
                        False, None, "Empty kernel output"
                    )
                    rewards.append(reward)
                    runtimes.append(None)
                    continue

                logger.debug(f"Response {idx + 1}: Kernel code {len(code)} chars")
                code = self.kernel_info['prefix'] + code + '\n' + self.kernel_info['suffix']
            else:
                # Full-file mode: extract code from response
                code = self.compiler.extract_code_from_llm_output(response)

                if code is None:
                    logger.warning(f"Response {idx + 1}: Failed to extract code from LLM output")
                    reward = self.reward_function.compute_reward(
                        False, None, "Failed to extract code"
                    )
                    rewards.append(reward)
                    runtimes.append(None)
                    continue

                logger.debug(f"Response {idx + 1}: Extracted {len(code)} chars of code")

            # Compile and run
            success, runtime, error = self.compiler.compile_and_run(code, num_runs=3)

            # Check correctness for successful runs
            if success and self.correctness_compiler is not None and self.reference_output is not None:
                ok, _, stderr, cerr = self.correctness_compiler.compile_and_get_output(code)
                if not ok:
                    logger.info(f"Correctness check: compilation failed ({cerr}), treating as incorrect")
                    success = False
                    error = "Incorrect output (correctness compile failed)"
                elif not self._outputs_match(self.reference_output, stderr):
                    logger.info(f"Correctness check: output mismatch, treating as incorrect")
                    success = False
                    error = "Incorrect output"

            # Compute reward
            reward = self.reward_function.compute_reward(success, runtime, error)
            rewards.append(reward)
            runtimes.append(runtime if success else None)

            if success:
                logger.info(
                    f"Success! Runtime: {runtime:.2f}μs "
                    f"(baseline: {self.baseline_runtime:.2f}μs), "
                    f"Reward: {reward:.3f}"
                )
            else:
                logger.info(f"Failed: {error}, Reward: {reward:.3f}")

        return rewards, runtimes

    def train_step(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """Perform one training step using REINFORCE."""
        prefix = f"[Epoch {epoch}] " if epoch else ""

        # Generate samples
        logger.debug(f"{prefix}Generating {self.batch_size} optimization candidates...")
        prompts, responses, log_probs_old = self.generate_optimizations(self.batch_size)

        # Evaluate and get rewards
        logger.debug(f"{prefix}Evaluating generated code...")
        rewards, runtimes = self.evaluate_code(responses)

        # Compute speedup from best successful runtime in this batch
        # Filter out runtimes that would produce unreasonable speedups (> max_speedup)
        # These are likely measurement errors or invalid optimizations
        max_speedup = self.reward_function.max_speedup
        min_valid_runtime = self.baseline_runtime / max_speedup
        valid_runtimes = [r for r in runtimes if r is not None and r >= min_valid_runtime]
        if valid_runtimes:
            best_batch_runtime = min(valid_runtimes)
            speedup = self.baseline_runtime / best_batch_runtime
        else:
            speedup = 1.0  # No valid speedups, report as 1.0x

        # Compute normalized rewards (baseline subtraction)
        rewards_tensor = torch.tensor(rewards, device=self.device)
        normalized_rewards = rewards_tensor - rewards_tensor.mean()

        # Compute loss using REINFORCE
        # We need to recompute log probs with gradients
        logger.debug(f"{prefix}Computing gradients...")
        self.optimizer.zero_grad()

        total_loss = 0.0
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # Tokenize prompt separately to get its length
            prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
            prompt_length = prompt_tokens['input_ids'].shape[1]

            # Tokenize full text (prompt + response)
            full_text = prompt + response
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Create labels with prompt tokens masked out (-100 is ignored by CrossEntropyLoss)
            labels = inputs['input_ids'].clone()
            labels[:, :prompt_length] = -100

            # Forward pass - loss is only computed on response tokens now
            outputs = self.model(**inputs, labels=labels)

            # Count response tokens for scaling mean -> sum
            num_response_tokens = (labels != -100).sum().item()
            if num_response_tokens == 0:
                logger.warning(f"Sample {i}: No response tokens after truncation, skipping")
                continue

            # REINFORCE loss: -log_prob * (reward - baseline)
            # Multiply by num_response_tokens to convert mean to sum of log probs
            loss = -outputs.loss * num_response_tokens * normalized_rewards[i]
            total_loss += loss.item()

            # Backward pass (accumulate gradients)
            loss.backward()

        # Optimizer step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Compute metrics
        metrics = {
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "loss": total_loss / len(rewards),
            "speedup": speedup,
        }

        return metrics

    def train(self, num_steps: int = 100, save_every: int = 10):
        """Run training loop."""
        logger.info(f"Starting training for {num_steps} steps...")
        logger.info(f"Dataset size: {len(self.dataset)} programs")

        for step in range(num_steps):
            logger.info(f"\n--- Step {step + 1}/{num_steps} ---")

            # Load new program if dataset has multiple programs
            if len(self.dataset) > 1:
                # Try to load next program, retrying on compilation failures
                loaded = False
                for _ in range(len(self.dataset)):
                    try:
                        self._load_next_program()
                        loaded = True
                        break
                    except RuntimeError as e:
                        logger.warning(f"Program failed to compile, trying next: {e}")
                if not loaded:
                    logger.error("All remaining programs failed to compile, stopping training")
                    break

            metrics = self.train_step()

            logger.info(
                f"Step {step + 1} ({self.current_program['name']}): "
                f"mean_reward={metrics['mean_reward']:.3f}, "
                f"loss={metrics['loss']:.3f}, "
                f"best_speedup={metrics['speedup']:.2f}x"
            )

            # Save checkpoint if reward > 1 (indicating meaningful speedup)
            if metrics['max_reward'] > 1:
                checkpoint_path = self.output_dir / f"checkpoint-step{step + 1}-reward{metrics['max_reward']:.2f}"
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path} (max_reward={metrics['max_reward']:.3f}, speedup={metrics['speedup']:.2f}x)")

        logger.info("Training complete!")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _load_program_by_index(self, index: int, epoch: Optional[int] = None):
        """Load a specific program by dataset index and compute its baseline.

        Raises:
            RuntimeError: If the program fails to compile (also adds to _failed_programs)
        """
        self.current_program = self.dataset.get_program(index)
        self.original_code = self.current_program['code']
        program_name = self.current_program['name']

        prefix = f"[Epoch {epoch}] " if epoch else ""
        logger.info(f"{prefix}Loading program: {program_name}")

        # Create compiler with program-specific configuration
        compiler_config = self.current_program.get('compiler_config', {})
        compiler_kwargs = {k: v for k, v in compiler_config.items() if k not in ('kernel_markers', 'correctness_config')}
        self.compiler = CppCompiler(**compiler_kwargs)

        # Compute baseline for this program
        logger.debug(f"{prefix}Computing baseline runtime...")
        success, self.baseline_runtime, error = self.compiler.compile_and_run(
            self.original_code, num_runs=5
        )
        if not success:
            self._failed_programs.add(index)
            raise RuntimeError(
                f"Failed to benchmark baseline for {program_name}: {error}"
            )
        logger.debug(f"{prefix}Baseline runtime: {self.baseline_runtime:.2f} microseconds")

        # Create reward function for this program
        self.reward_function = AdaptiveRewardFunction(self.baseline_runtime)

        # Extract kernel region if markers are configured
        kernel_markers = compiler_config.get('kernel_markers')
        if kernel_markers:
            kernel_info = self._extract_kernel_region(self.original_code, kernel_markers)
            if kernel_info:
                self.kernel_info = kernel_info
                logger.info(f"{prefix}Kernel-only mode: extracted {len(kernel_info['kernel_body'])} chars of kernel code")
            else:
                self.kernel_info = None
                logger.warning(f"{prefix}Kernel markers configured but not found in {program_name}, using full-file mode")
        else:
            self.kernel_info = None

        # Setup correctness checking if configured
        correctness_config = compiler_config.get('correctness_config')
        if correctness_config:
            self.correctness_compiler = CppCompiler(**correctness_config)
            success, stdout, stderr, error = self.correctness_compiler.compile_and_get_output(self.original_code)
            if success:
                self.reference_output = stderr
                logger.info(f"{prefix}Correctness check: captured {len(self.reference_output)} chars of reference output")
            else:
                self.correctness_compiler = None
                self.reference_output = None
                logger.warning(f"{prefix}Correctness check setup failed for {program_name}: {error}")
        else:
            self.correctness_compiler = None
            self.reference_output = None

    def evaluate_on_program(self, program_index: int, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model on a single program without training.

        Returns:
            Dict with mean_reward, max_reward, speedup for this program
        """
        self._load_program_by_index(program_index, epoch=epoch)

        # Generate samples (same as training)
        prompts, responses, _ = self.generate_optimizations(self.batch_size)

        # Evaluate and get rewards
        rewards, runtimes = self.evaluate_code(responses)

        # Compute speedup from best successful runtime in this batch
        # Filter out runtimes that would produce unreasonable speedups (> max_speedup)
        max_speedup = self.reward_function.max_speedup
        min_valid_runtime = self.baseline_runtime / max_speedup
        valid_runtimes = [r for r in runtimes if r is not None and r >= min_valid_runtime]
        if valid_runtimes:
            best_batch_runtime = min(valid_runtimes)
            speedup = self.baseline_runtime / best_batch_runtime
        else:
            speedup = 1.0  # No valid speedups, report as 1.0x

        return {
            "program": self.current_program['name'],
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "speedup": speedup,
        }

    def evaluate_test_set(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate model on the held-out test set.

        Returns:
            Dict with aggregated test metrics
        """
        if self.test_indices is None:
            raise ValueError("No test set configured. Use train_programs and test_programs.")

        prefix = f"[Epoch {epoch}] " if epoch else ""
        logger.info(f"\n{'='*40}")
        logger.info(f"{prefix}Evaluating on {len(self.test_indices)} test programs...")
        logger.info(f"{'='*40}")

        all_rewards = []
        all_speedups = []
        program_results = []

        for idx in self.test_indices:
            if idx in self._failed_programs:
                program = self.dataset.get_program(idx)
                logger.warning(f"{prefix}Skipping previously failed test program: {program['name']}")
                continue
            try:
                result = self.evaluate_on_program(idx, epoch=epoch)
                all_rewards.append(result['mean_reward'])
                all_speedups.append(result['speedup'])
                program_results.append(result)
                logger.info(
                    f"{prefix}Test [{result['program']}]: "
                    f"mean_reward={result['mean_reward']:.3f}, "
                    f"speedup={result['speedup']:.2f}x"
                )
            except RuntimeError as e:
                logger.warning(f"{prefix}Test program failed to compile, skipping: {e}")

        return {
            "test_mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
            "test_max_reward": max(all_rewards) if all_rewards else 0.0,
            "test_mean_speedup": sum(all_speedups) / len(all_speedups) if all_speedups else 1.0,
            "test_max_speedup": max(all_speedups) if all_speedups else 1.0,
            "program_results": program_results,
        }

    def train_epochs(self, num_epochs: int = 10, save_every: int = 1):
        """
        Run epoch-based training with train/test evaluation.

        Each epoch:
        1. Train on train_programs programs from the training set
        2. Evaluate on the held-out test set
        """
        if self.train_indices is None or self.test_indices is None:
            raise ValueError("Epoch training requires train_programs and test_programs to be set")

        logger.info(f"Starting epoch training for {num_epochs} epochs...")
        logger.info(f"Train set: {len(self.train_indices)} programs, Test set: {len(self.test_indices)} programs")
        logger.info(f"Programs per epoch: {self.train_programs}")

        # Epoch 0: Evaluate on test set before any training
        logger.info(f"\n{'#'*60}")
        logger.info(f"EPOCH 0 (Pre-training baseline)")
        logger.info(f"{'#'*60}")
        self.model.eval()
        with torch.no_grad():
            test_metrics = self.evaluate_test_set(epoch=0)
        self.model.train()
        logger.info(f"\n{'='*60}")
        logger.info(f"EPOCH 0 SUMMARY (Pre-training baseline):")
        logger.info(f"  Test:  mean_reward={test_metrics['test_mean_reward']:.3f}, "
                   f"mean_speedup={test_metrics['test_mean_speedup']:.2f}x")
        logger.info(f"{'='*60}")

        for epoch in range(num_epochs):
            logger.info(f"\n{'#'*60}")
            logger.info(f"EPOCH {epoch + 1}/{num_epochs}")
            logger.info(f"{'#'*60}")

            # Sample train_programs indices from training set
            if self.sampling_strategy == "random":
                epoch_indices = random.sample(
                    self.train_indices,
                    min(self.train_programs, len(self.train_indices))
                )
            else:  # sequential
                start = (epoch * self.train_programs) % len(self.train_indices)
                epoch_indices = []
                for i in range(self.train_programs):
                    idx = self.train_indices[(start + i) % len(self.train_indices)]
                    epoch_indices.append(idx)

            # Training phase
            logger.info(f"\n--- [Epoch {epoch + 1}] Training on {len(epoch_indices)} programs ---")
            epoch_train_rewards = []
            epoch_train_speedups = []

            # Track which indices we've already used/tried this epoch
            used_indices = set(epoch_indices)

            for step, prog_idx in enumerate(epoch_indices):
                # Try to load the program, finding a replacement if it fails to compile
                current_idx = prog_idx
                while True:
                    if current_idx in self._failed_programs:
                        # Already know this one fails, skip it
                        logger.warning(
                            f"[Epoch {epoch + 1}] Skipping previously failed program index {current_idx}"
                        )
                    else:
                        try:
                            self._load_program_by_index(current_idx, epoch=epoch + 1)
                            break  # Success, continue with training
                        except RuntimeError as e:
                            logger.warning(
                                f"[Epoch {epoch + 1}] Program failed to compile, will try replacement: {e}"
                            )

                    # Find a replacement program from the training set
                    available = [
                        idx for idx in self.train_indices
                        if idx not in used_indices and idx not in self._failed_programs
                    ]
                    if not available:
                        logger.warning(
                            f"[Epoch {epoch + 1}] No more replacement programs available, skipping this step"
                        )
                        current_idx = None
                        break
                    current_idx = available[0]
                    used_indices.add(current_idx)
                    logger.info(f"[Epoch {epoch + 1}] Trying replacement program index {current_idx}")

                if current_idx is None:
                    continue  # Skip this training step

                metrics = self.train_step(epoch=epoch + 1)

                epoch_train_rewards.append(metrics['mean_reward'])
                epoch_train_speedups.append(metrics['speedup'])

                logger.info(
                    f"[Epoch {epoch + 1}] Train step {step + 1}/{len(epoch_indices)} ({self.current_program['name']}): "
                    f"mean_reward={metrics['mean_reward']:.3f}, "
                    f"loss={metrics['loss']:.3f}, "
                    f"speedup={metrics['speedup']:.2f}x"
                )

            # Compute epoch training metrics
            if epoch_train_rewards:
                train_metrics = {
                    "train_mean_reward": sum(epoch_train_rewards) / len(epoch_train_rewards),
                    "train_max_reward": max(epoch_train_rewards),
                    "train_mean_speedup": sum(epoch_train_speedups) / len(epoch_train_speedups),
                    "train_max_speedup": max(epoch_train_speedups),
                }
            else:
                logger.warning(f"[Epoch {epoch + 1}] No programs successfully trained this epoch!")
                train_metrics = {
                    "train_mean_reward": 0.0,
                    "train_max_reward": 0.0,
                    "train_mean_speedup": 1.0,
                    "train_max_speedup": 1.0,
                }

            # Test evaluation phase
            self.model.eval()
            with torch.no_grad():
                test_metrics = self.evaluate_test_set(epoch=epoch + 1)
            self.model.train()

            # Log epoch summary
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch + 1} SUMMARY:")
            logger.info(f"  Train: mean_reward={train_metrics['train_mean_reward']:.3f}, "
                       f"mean_speedup={train_metrics['train_mean_speedup']:.2f}x")
            logger.info(f"  Test:  mean_reward={test_metrics['test_mean_reward']:.3f}, "
                       f"mean_speedup={test_metrics['test_mean_speedup']:.2f}x")
            logger.info(f"{'='*60}")

            # Save checkpoint periodically
            if (epoch + 1) % save_every == 0:
                checkpoint_path = self.output_dir / f"checkpoint-epoch{epoch + 1}"
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        logger.info("\nEpoch training complete!")
