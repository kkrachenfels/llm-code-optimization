import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import logging
from pathlib import Path

from .compiler import CppCompiler
from .reward import AdaptiveRewardFunction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleCodeOptimizationTrainer:
    """Simplified RL trainer for code optimization using REINFORCE."""

    def __init__(
        self,
        model_name: str,
        program_path: str,
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
            program_path: Path to the original C++ program
            output_dir: Directory to save checkpoints
            learning_rate: Learning rate
            batch_size: Batch size for training
            max_length: Maximum sequence length
            use_8bit: Whether to use 8-bit quantization
        """
        self.program_path = Path(program_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.batch_size = batch_size
        self.max_length = max_length

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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )

        # Setup device
        if not use_8bit:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
        else:
            self.device = self.model.device

        self.model.train()

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def create_prompt(self, code: str) -> str:
        """Create optimization prompt for the model."""
        prompt = f"""Optimize the following C++ code for runtime performance. Provide only the complete optimized C++ code without explanations.

Original code:
```cpp
{code}
```

Optimized code:
```cpp
"""
        return prompt

    def generate_optimizations(self, num_samples: int) -> tuple:
        """
        Generate optimized code samples.

        Returns:
            Tuple of (prompts, responses, log_probs)
        """
        prompt = self.create_prompt(self.original_code)
        prompts = [prompt] * num_samples

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate with log probabilities
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
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

        for response in responses:
            # Extract code from response
            code = self.compiler.extract_code_from_llm_output(response)

            if code is None:
                reward = self.reward_function.compute_reward(
                    False, None, "Failed to extract code"
                )
                rewards.append(reward)
                continue

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

        for step in range(num_steps):
            logger.info(f"\n--- Step {step + 1}/{num_steps} ---")

            metrics = self.train_step()

            logger.info(
                f"Step {step + 1}: "
                f"mean_reward={metrics['mean_reward']:.3f}, "
                f"loss={metrics['loss']:.3f}, "
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
