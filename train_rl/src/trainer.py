import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from typing import List, Dict, Optional
import logging
from pathlib import Path

from .compiler import CppCompiler
from .reward import RewardFunction, AdaptiveRewardFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeOptimizationTrainer:
    """RL trainer for code optimization using PPO."""

    def __init__(
        self,
        model_name: str,
        program_path: str,
        output_dir: str = "checkpoints",
        learning_rate: float = 1.41e-5,
        batch_size: int = 4,
        mini_batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        ppo_epochs: int = 4,
        max_length: int = 1024,
        use_8bit: bool = False,
    ):
        """
        Initialize the trainer.

        Args:
            model_name: HuggingFace model name (e.g., "Salesforce/codegen-350M-mono")
            program_path: Path to the original C++ program
            output_dir: Directory to save checkpoints
            learning_rate: Learning rate for PPO
            batch_size: Batch size for training
            mini_batch_size: Mini-batch size for PPO
            gradient_accumulation_steps: Gradient accumulation steps
            ppo_epochs: Number of PPO epochs
            max_length: Maximum sequence length
            use_8bit: Whether to use 8-bit quantization
        """
        self.program_path = Path(program_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Load original program
        with open(self.program_path, 'r') as f:
            self.original_code = f.read()

        # Initialize compiler and get baseline
        self.compiler = CppCompiler()
        logger.info("Computing baseline runtime...")
        success, self.baseline_runtime, error = self.compiler.compile_and_run(
            self.original_code, num_runs=5
        )
        if not success:
            raise RuntimeError(f"Failed to benchmark baseline: {error}")
        logger.info(f"Baseline runtime: {self.baseline_runtime:.2f} microseconds")

        # Initialize reward function
        self.reward_function = AdaptiveRewardFunction(self.baseline_runtime)

        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if use_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            **model_kwargs
        )
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            **model_kwargs
        )

        # PPO config
        self.ppo_config = PPOConfig(
            learning_rate=learning_rate,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_ppo_epochs=ppo_epochs,
            output_dir=output_dir,
        )

        # Initialize PPO trainer
        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

        self.max_length = max_length

    def create_prompt(self, code: str) -> str:
        """
        Create optimization prompt for the model.

        Args:
            code: Original C++ code

        Returns:
            Formatted prompt
        """
        prompt = f"""Optimize the following C++ code for runtime performance. Provide only the complete optimized C++ code without explanations.

Original code:
```cpp
{code}
```

Optimized code:
```cpp
"""
        return prompt

    def generate_optimizations(self, prompts: List[str]) -> List[str]:
        """
        Generate optimized code using the model.

        Args:
            prompts: List of prompts

        Returns:
            List of generated texts
        """
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move to device
        device = self.model.pretrained_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        # Remove prompts from outputs
        responses = []
        for prompt, generated in zip(prompts, generated_texts):
            response = generated[len(prompt):]
            responses.append(response)

        return responses

    def evaluate_code(self, responses: List[str]) -> tuple:
        """
        Evaluate generated code and compute rewards.

        Args:
            responses: List of generated code responses

        Returns:
            Tuple of (rewards, runtimes) where runtimes contains float values
            for successful runs and None for failures.
        """
        rewards = []
        runtimes = []

        for response in responses:
            # Extract code from response
            code = self.compiler.extract_code_from_llm_output(response)

            if code is None:
                # Failed to extract code
                reward = self.reward_function.compute_reward(
                    False, None, "Failed to extract code"
                )
                rewards.append(reward)
                runtimes.append(None)
                continue

            # Compile and run
            success, runtime, error = self.compiler.compile_and_run(code, num_runs=3)

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

    def train_step(self, batch_size: int = 4) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            batch_size: Number of samples to generate

        Returns:
            Dictionary of training metrics
        """
        # Create prompts
        prompt = self.create_prompt(self.original_code)
        prompts = [prompt] * batch_size

        # Tokenize queries
        query_tensors = []
        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt, return_tensors="pt")
            query_tensors.append(tokens.squeeze())

        # Generate responses
        logger.info("Generating optimizations...")
        response_tensors = []
        for query in query_tensors:
            query = query.to(self.model.pretrained_model.device)
            response = self.ppo_trainer.generate(
                query.unsqueeze(0),
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
            )
            response_tensors.append(response.squeeze())

        # Decode responses
        responses = [
            self.tokenizer.decode(r, skip_special_tokens=True)
            for r in response_tensors
        ]

        # Evaluate and get rewards
        logger.info("Evaluating generated code...")
        rewards, runtimes = self.evaluate_code(responses)
        reward_tensors = [torch.tensor(r) for r in rewards]

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

        # Run PPO step
        logger.info("Running PPO update...")
        stats = self.ppo_trainer.step(query_tensors, response_tensors, reward_tensors)

        # Compute metrics
        metrics = {
            "mean_reward": sum(rewards) / len(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "speedup": speedup,
        }

        return metrics

    def train(self, num_steps: int = 100, save_every: int = 10):
        """
        Run training loop.

        Args:
            num_steps: Number of training steps
            save_every: Save checkpoint every N steps
        """
        logger.info(f"Starting training for {num_steps} steps...")

        for step in range(num_steps):
            logger.info(f"\n--- Step {step + 1}/{num_steps} ---")

            metrics = self.train_step(batch_size=self.ppo_config.batch_size)

            logger.info(
                f"Step {step + 1}: "
                f"mean_reward={metrics['mean_reward']:.3f}, "
                f"best_speedup={metrics['speedup']:.2f}x"
            )

            # Save checkpoint
            if (step + 1) % save_every == 0:
                checkpoint_path = self.output_dir / f"checkpoint-{step + 1}"
                self.save_checkpoint(checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        logger.info("Training complete!")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(exist_ok=True, parents=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
