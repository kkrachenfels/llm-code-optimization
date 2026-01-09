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
        max_length: int = 1024,
        use_8bit: bool = False,
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
        """
        self.dataset = dataset
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

        # Load and benchmark the first program
        self._load_next_program()

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

        self.model.train()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

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
        self.compiler = CppCompiler(**compiler_config)

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

    def create_prompt(self, code: str) -> str:
        """Create optimization prompt for the model."""
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

        # Tokenize - use larger max_length to avoid truncating source code
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096  # Increased from 512 to handle full source files
        )
        # Move inputs to the same device as the model's first layer
        first_device = next(self.model.parameters()).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}

        # Log input length to help debug truncation issues
        input_len = inputs['input_ids'].shape[1]
        logger.info(f"Prompt tokenized to {input_len} tokens")

        # Generate with log probabilities
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,  # Increased from 512 to allow full code generation
                do_sample=True,
                top_p=0.9,
                top_k=50,
                temperature=0.8,
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

        # Compute log probabilities for generated tokens
        log_probs_list = []
        for i, seq in enumerate(generated_sequences):
            response_tokens = seq[inputs['input_ids'].shape[1]:]
            log_prob = 0.0

            # Simple approximation: compute probability of the sequence
            with torch.no_grad():
                full_output = self.model(seq.unsqueeze(0))
                logits = full_output.logits[0, inputs['input_ids'].shape[1]-1:-1, :]
                log_probs = torch.log_softmax(logits, dim=-1)

                for j, token in enumerate(response_tokens):
                    if j < log_probs.shape[0]:
                        log_prob += log_probs[j, token].item()

            log_probs_list.append(log_prob)

        return prompts, responses, log_probs_list

    def evaluate_code(self, responses: List[str]) -> List[float]:
        """Evaluate generated code and compute rewards."""
        rewards = []

        for idx, response in enumerate(responses):
            logger.debug(f"--- Evaluating response {idx + 1}/{len(responses)} ---")

            # Extract code from response
            code = self.compiler.extract_code_from_llm_output(response)

            if code is None:
                logger.warning(f"Response {idx + 1}: Failed to extract code from LLM output")
                reward = self.reward_function.compute_reward(
                    False, None, "Failed to extract code"
                )
                rewards.append(reward)
                continue

            logger.debug(f"Response {idx + 1}: Extracted {len(code)} chars of code")

            # Compile and run
            success, runtime, error = self.compiler.compile_and_run(code, num_runs=3)

            # Compute reward
            reward = self.reward_function.compute_reward(success, runtime, error)
            rewards.append(reward)

            if success:
                logger.info(
                    f"Success! Runtime: {runtime:.2f}μs "
                    f"(baseline: {self.baseline_runtime:.2f}μs), "
                    f"Reward: {reward:.3f}"
                )
            else:
                logger.info(f"Failed: {error}, Reward: {reward:.3f}")

        return rewards

    def train_step(self) -> Dict[str, float]:
        """Perform one training step using REINFORCE."""
        # Generate samples
        logger.info(f"Generating {self.batch_size} optimization candidates...")
        prompts, responses, log_probs_old = self.generate_optimizations(self.batch_size)

        # Evaluate and get rewards
        logger.info("Evaluating generated code...")
        rewards = self.evaluate_code(responses)

        # Compute normalized rewards (baseline subtraction)
        rewards_tensor = torch.tensor(rewards, device=self.device)
        normalized_rewards = rewards_tensor - rewards_tensor.mean()

        # Compute loss using REINFORCE
        # We need to recompute log probs with gradients
        logger.info("Computing gradients...")
        self.optimizer.zero_grad()

        total_loss = 0.0
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # Tokenize prompt + response
            full_text = prompt + response
            inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Forward pass
            outputs = self.model(**inputs, labels=inputs['input_ids'])

            # REINFORCE loss: -log_prob * (reward - baseline)
            loss = -outputs.loss * normalized_rewards[i]
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
            "best_runtime": self.reward_function.best_runtime,
            "speedup": self.baseline_runtime / self.reward_function.best_runtime,
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
                self._load_next_program()

            # Track best runtime before step to detect improvement
            best_runtime_before = self.reward_function.best_runtime

            metrics = self.train_step()

            logger.info(
                f"Step {step + 1} ({self.current_program['name']}): "
                f"mean_reward={metrics['mean_reward']:.3f}, "
                f"loss={metrics['loss']:.3f}, "
                f"best_speedup={metrics['speedup']:.2f}x"
            )

            # Save checkpoint if reward > 1 and we improved best runtime
            best_runtime_improved = metrics['best_runtime'] < best_runtime_before
            if metrics['max_reward'] > 1 and best_runtime_improved:
                checkpoint_path = self.output_dir / f"checkpoint-step{step + 1}-reward{metrics['max_reward']:.2f}"
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path} (max_reward={metrics['max_reward']:.3f}, new best runtime={metrics['best_runtime']:.2f}μs)")

        logger.info("Training complete!")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
