"""
GRPO (Group Relative Policy Optimization) trainer for CUDA kernel optimization.

Based on DeepSeek's GRPO algorithm as used in Kevin-32B.
"""

import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .task_sampler import TaskSampler, KernelTask
from .prompt_builder import PromptBuilder, TurnFeedback
from .code_extractor import CUDACodeExtractor
from .reward_calculator import (
    RewardCalculator,
    RewardMode,
    EvaluationResult,
    compute_discounted_rewards,
)


@dataclass
class TrajectoryStep:
    """A single step in a trajectory."""

    turn: int
    prompt: str
    response: str
    cuda_code: Optional[str]
    cuda_file: Optional[str]
    eval_result: Optional[EvaluationResult]
    reward: float


@dataclass
class Trajectory:
    """A complete trajectory for a task."""

    task: KernelTask
    steps: List[TrajectoryStep] = field(default_factory=list)
    total_reward: float = 0.0

    def add_step(self, step: TrajectoryStep):
        self.steps.append(step)

    def compute_total_reward(self, gamma: float = 0.4):
        """Compute discounted total reward."""
        rewards = [step.reward for step in self.steps]
        self.total_reward = compute_discounted_rewards(rewards, gamma)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9

    # GRPO
    group_size: int = 8
    kl_coef: float = 0.05
    clip_ratio: float = 0.2
    gamma: float = 0.4

    # Training
    tasks_per_batch: int = 4
    max_turns: int = 3
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 5000
    save_interval: int = 500
    eval_interval: int = 100
    log_interval: int = 10
    seed: int = 42

    # Reward
    reward_mode: str = "speed_correct"
    correctness_bonus: float = 0.3
    max_speedup_reward: float = 5.0

    # Paths
    output_dir: str = "outputs"
    temp_dir: Optional[str] = None


class GRPOTrainer:
    """
    GRPO trainer for CUDA kernel optimization.

    Implements Group Relative Policy Optimization where rewards are normalized
    within groups of responses to the same prompt, eliminating the need for
    a separate value network.
    """

    def __init__(
        self,
        config: GRPOConfig,
        task_sampler: TaskSampler,
        reward_calculator: RewardCalculator,
        device: str = "cuda",
    ):
        """
        Initialize the GRPO trainer.

        Args:
            config: Training configuration
            task_sampler: Task sampler for loading tasks
            reward_calculator: Reward calculator using robust_kbench
            device: Device for model
        """
        self.config = config
        self.task_sampler = task_sampler
        self.reward_calculator = reward_calculator
        self.device = device

        # Initialize components
        self.prompt_builder = PromptBuilder()
        self.code_extractor = CUDACodeExtractor(temp_dir=config.temp_dir)

        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model - use device_map for automatic placement
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)

        # Enable gradient checkpointing to reduce activation memory
        self.model.gradient_checkpointing_enable()

        # Keep a reference model for KL computation
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
        )

        # Training state
        self.global_step = 0
        self.best_reward = float("-inf")

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sanity check: test model generation
        self._test_model_generation()

    def _test_model_generation(self):
        """Quick sanity check that model generation works."""
        print("Testing model generation...")
        test_messages = [{"role": "user", "content": "Write a simple hello world in Python."}]
        prompt = self.tokenizer.apply_chat_template(
            test_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Test generation result: {response[:200]}")

        if len(response.strip()) < 10 or not any(c.isalpha() for c in response):
            print("WARNING: Model generation appears broken - output is garbage!")
        else:
            print("Model generation test passed.")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        num_samples: int = 1,
    ) -> List[str]:
        """
        Generate responses for a prompt.

        Args:
            messages: Chat messages
            num_samples: Number of samples to generate

        Returns:
            List of generated responses
        """
        # Format prompt
        prompt = self.prompt_builder.format_for_generation(messages, self.tokenizer)

        # Debug: print prompt (first 500 chars)
        print(f"[DEBUG] Prompt (first 500 chars):\n{prompt[:500]}")

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.device)

        print(f"[DEBUG] Input token count: {inputs['input_ids'].shape[1]}")

        # Switch to eval mode and disable gradient checkpointing for generation
        was_training = self.model.training
        self.model.eval()

        # Ensure attention mask is set
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        responses = []
        with torch.no_grad():
            for _ in range(num_samples):
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                # Decode response (only the generated part)
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                responses.append(response)

        # Restore training mode if needed
        if was_training:
            self.model.train()

        return responses

    def run_trajectory(
        self,
        task: KernelTask,
        trajectory_id: int,
    ) -> Trajectory:
        """
        Run a single trajectory for a task.

        Args:
            task: The kernel task
            trajectory_id: ID for this trajectory

        Returns:
            Completed trajectory
        """
        trajectory = Trajectory(task=task)
        messages = self.prompt_builder.build_initial_prompt(task)

        for turn in range(self.config.max_turns):
            # Generate response
            responses = self.generate_response(messages, num_samples=1)
            response = responses[0]

            # Debug: print the generated response
            print(f"\n{'='*60}")
            print(f"[DEBUG] Task: {task.name}, Trajectory: {trajectory_id}, Turn: {turn}")
            print(f"{'='*60}")
            print(f"[DEBUG] Generated response (first 2000 chars):")
            print(response[:2000])
            if len(response) > 2000:
                print(f"... (truncated, total length: {len(response)})")
            print(f"{'='*60}")

            # Extract CUDA code
            cuda_code, extract_error = self.code_extractor.extract(response)

            # Debug: print extraction result
            if cuda_code is not None:
                print(f"[DEBUG] Successfully extracted CUDA code ({len(cuda_code)} chars)")
                print(f"[DEBUG] CUDA code preview (first 500 chars):")
                print(cuda_code[:500])
            else:
                print(f"[DEBUG] Failed to extract CUDA code!")
                print(f"[DEBUG] Extract error: {extract_error}")

            step = TrajectoryStep(
                turn=turn,
                prompt=messages[-1]["content"],
                response=response,
                cuda_code=cuda_code,
                cuda_file=None,
                eval_result=None,
                reward=self.config.correctness_bonus
                * -1,  # Default penalty for no code
            )

            if cuda_code is None:
                # Failed to extract code
                step.reward = self.reward_calculator.compile_fail_penalty
                trajectory.add_step(step)

                # Build feedback for next turn
                feedback = TurnFeedback(
                    compiled=False,
                    compile_error=extract_error or "Failed to extract CUDA code",
                    correct=False,
                    max_diff=None,
                    speedup=None,
                    torch_time_ms=None,
                    cuda_time_ms=None,
                    profile_info=None,
                )
            else:
                # Save CUDA code and evaluate
                cuda_file = self.code_extractor.save_to_file(
                    cuda_code, task.name, turn, trajectory_id, forward=task.forward
                )
                step.cuda_file = cuda_file

                # Evaluate in isolated process (won't crash main training if compilation fails badly)
                eval_result = self.reward_calculator.evaluate_kernel_isolated(
                    cuda_file, task.task_dir, forward=task.forward
                )
                step.eval_result = eval_result
                step.reward = eval_result.reward

                # Debug: print evaluation result
                print(f"[DEBUG] Eval result:")
                print(f"  compiled: {eval_result.compiled}")
                print(f"  compile_error: {eval_result.compile_error}")
                print(f"  correct: {eval_result.correct}")
                print(f"  max_diff: {eval_result.max_diff}")
                print(f"  speedup: {eval_result.speedup}")
                print(f"  reward: {eval_result.reward}")

                trajectory.add_step(step)

                # Build feedback
                feedback = TurnFeedback(
                    compiled=eval_result.compiled,
                    compile_error=eval_result.compile_error,
                    correct=eval_result.correct,
                    max_diff=eval_result.max_diff,
                    speedup=eval_result.speedup,
                    torch_time_ms=eval_result.torch_time_ms,
                    cuda_time_ms=eval_result.cuda_time_ms,
                    profile_info=eval_result.profile_info,
                )

                # If correct and fast enough, no need for more turns
                if eval_result.correct and eval_result.speedup and eval_result.speedup > 1.5:
                    break

            # Build refinement prompt for next turn
            if turn < self.config.max_turns - 1:
                messages = self.prompt_builder.build_refinement_prompt(
                    task, messages, response, feedback
                )

        trajectory.compute_total_reward(self.config.gamma)
        return trajectory

    def sample_trajectories(
        self,
        tasks: List[KernelTask],
    ) -> List[List[Trajectory]]:
        """
        Sample trajectories for a batch of tasks.

        Args:
            tasks: List of tasks

        Returns:
            List of trajectory lists (one list per task)
        """
        all_trajectories = []

        for task in tqdm(tasks, desc="Sampling trajectories"):
            task_trajectories = []
            for traj_id in range(self.config.group_size):
                trajectory = self.run_trajectory(task, traj_id)
                task_trajectories.append(trajectory)
            all_trajectories.append(task_trajectories)

        return all_trajectories

    def compute_grpo_loss(
        self,
        trajectories: List[List[Trajectory]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss for a batch of trajectories.

        GRPO normalizes rewards within each group (trajectories for same task)
        and uses these normalized advantages for policy gradient.

        Args:
            trajectories: List of trajectory lists (grouped by task)

        Returns:
            Tuple of (loss tensor, metrics dict)
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_pg_loss = torch.tensor(0.0, device=self.device)
        total_kl_loss = torch.tensor(0.0, device=self.device)
        num_samples = 0
        all_rewards = []

        for task_trajectories in trajectories:
            # Get rewards for this group
            rewards = torch.tensor(
                [t.total_reward for t in task_trajectories],
                device=self.device,
            )
            all_rewards.extend(rewards.tolist())

            # Normalize rewards within group (GRPO)
            if rewards.std() > 1e-8:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            else:
                advantages = torch.zeros_like(rewards)

            # Compute loss for each trajectory
            for traj, advantage in zip(task_trajectories, advantages):
                for step in traj.steps:
                    if step.cuda_code is None:
                        continue

                    # Get log probabilities
                    messages = self.prompt_builder.build_initial_prompt(traj.task)
                    prompt = self.prompt_builder.format_for_generation(
                        messages, self.tokenizer
                    )
                    full_text = prompt + step.response

                    inputs = self.tokenizer(
                        full_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=8192,
                    ).to(self.device)

                    prompt_inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=8192,
                    ).to(self.device)

                    prompt_len = prompt_inputs["input_ids"].shape[1]

                    # Forward pass
                    outputs = self.model(**inputs)
                    ref_outputs = self.ref_model(**inputs)

                    # Get log probs for response tokens
                    logits = outputs.logits[:, prompt_len - 1 : -1, :]
                    ref_logits = ref_outputs.logits[:, prompt_len - 1 : -1, :]
                    labels = inputs["input_ids"][:, prompt_len:]

                    log_probs = F.log_softmax(logits, dim=-1)
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)

                    # Gather log probs for actual tokens
                    token_log_probs = log_probs.gather(
                        -1, labels.unsqueeze(-1)
                    ).squeeze(-1)
                    ref_token_log_probs = ref_log_probs.gather(
                        -1, labels.unsqueeze(-1)
                    ).squeeze(-1)

                    # Policy gradient loss with clipping
                    ratio = torch.exp(token_log_probs - ref_token_log_probs.detach())
                    clipped_ratio = torch.clamp(
                        ratio,
                        1 - self.config.clip_ratio,
                        1 + self.config.clip_ratio,
                    )
                    pg_loss = -torch.min(
                        ratio * advantage, clipped_ratio * advantage
                    ).mean()

                    # KL divergence loss
                    kl_loss = (
                        torch.exp(ref_token_log_probs)
                        * (ref_token_log_probs - token_log_probs)
                    ).mean()

                    # Combined loss
                    step_loss = pg_loss + self.config.kl_coef * kl_loss

                    total_loss = total_loss + step_loss
                    total_pg_loss = total_pg_loss + pg_loss.detach()
                    total_kl_loss = total_kl_loss + kl_loss.detach()
                    num_samples += 1

        # Average losses
        if num_samples == 0:
            print("WARNING: No valid samples found (all cuda_code was None). Using zero loss.")
        if num_samples > 0:
            total_loss = total_loss / num_samples
            total_pg_loss = total_pg_loss / num_samples
            total_kl_loss = total_kl_loss / num_samples

        metrics = {
            "loss": total_loss.item(),
            "pg_loss": total_pg_loss.item(),
            "kl_loss": total_kl_loss.item(),
            "mean_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0,
            "max_reward": max(all_rewards) if all_rewards else 0,
            "min_reward": min(all_rewards) if all_rewards else 0,
            "num_samples": num_samples,
        }

        return total_loss, metrics

    def train_step(self) -> Dict[str, float]:
        """
        Execute a single training step.

        Returns:
            Dictionary of metrics
        """
        self.model.train()

        # Sample tasks
        tasks = self.task_sampler.sample_batch(self.config.tasks_per_batch)

        # Sample trajectories
        trajectories = self.sample_trajectories(tasks)

        # Compute loss
        loss, metrics = self.compute_grpo_loss(trajectories)

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

        self.global_step += 1

        return metrics

    def save_checkpoint(self, path: Optional[str] = None):
        """Save a checkpoint."""
        if path is None:
            path = self.output_dir / f"checkpoint-{self.global_step}"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        # Save training state
        state = {
            "global_step": self.global_step,
            "best_reward": self.best_reward,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(state, path / "training_state.pt")

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load a checkpoint."""
        path = Path(path)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device,
        )

        # Load training state
        state = torch.load(path / "training_state.pt", weights_only=True)
        self.global_step = state["global_step"]
        self.best_reward = state["best_reward"]
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])

        print(f"Loaded checkpoint from {path} (step {self.global_step})")

    def train(self):
        """Run the full training loop."""
        print(f"Starting training from step {self.global_step}")
        print(f"Config: {self.config}")

        progress_bar = tqdm(
            total=self.config.max_steps,
            initial=self.global_step,
            desc="Training",
        )

        while self.global_step < self.config.max_steps:
            metrics = self.train_step()

            # Log metrics
            if self.global_step % self.config.log_interval == 0:
                tqdm.write(
                    f"Step {self.global_step}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"reward={metrics['mean_reward']:.4f}, "
                    f"max_reward={metrics['max_reward']:.4f}"
                )

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint()

            # Update best reward
            if metrics["mean_reward"] > self.best_reward:
                self.best_reward = metrics["mean_reward"]
                self.save_checkpoint(self.output_dir / "best")

            progress_bar.update(1)

        progress_bar.close()

        # Final save
        self.save_checkpoint(self.output_dir / "final")
        print("Training complete!")
