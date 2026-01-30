"""
GRPO (Group Relative Policy Optimization) trainer for CUDA kernel optimization.
Based on the DeepSeek approach, adapted for multi-turn training.
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler
)
from accelerate import Accelerator
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from .config import Config
from .dataset import KernelTask
from .trajectory import Trajectory, RefinementStep
from .reward import TrajectoryRewards, RewardComputer, normalize_rewards_grpo


@dataclass
class GRPOBatch:
    """A batch of training data for GRPO."""
    prompts: List[str]
    responses: List[str]
    advantages: torch.Tensor
    old_log_probs: Optional[torch.Tensor] = None

    # Metadata for logging
    task_ids: Optional[List[str]] = None
    step_indices: Optional[List[int]] = None


class GRPODataset(Dataset):
    """Dataset for GRPO training samples."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 4096
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Tokenize prompt
        prompt_tokens = self.tokenizer(
            sample['prompt'],
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors='pt'
        )

        # Tokenize response
        response_tokens = self.tokenizer(
            sample['response'],
            truncation=True,
            max_length=self.max_length // 2,
            return_tensors='pt'
        )

        return {
            'prompt_input_ids': prompt_tokens['input_ids'].squeeze(0),
            'prompt_attention_mask': prompt_tokens['attention_mask'].squeeze(0),
            'response_input_ids': response_tokens['input_ids'].squeeze(0),
            'response_attention_mask': response_tokens['attention_mask'].squeeze(0),
            'advantage': torch.tensor(sample['advantage'], dtype=torch.float32),
            'metadata': sample['metadata']
        }


def collate_grpo_batch(
    batch: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 4096
) -> Dict[str, torch.Tensor]:
    """Collate function for GRPO batches."""

    # Combine prompt and response for each sample
    combined_texts = []
    prompt_lengths = []

    for sample in batch:
        prompt_ids = sample['prompt_input_ids']
        response_ids = sample['response_input_ids']

        prompt_lengths.append(len(prompt_ids))
        combined_texts.append(torch.cat([prompt_ids, response_ids]))

    # Pad to same length
    max_len = min(max(len(t) for t in combined_texts), max_length)

    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

    for i, (text, prompt_len) in enumerate(zip(combined_texts, prompt_lengths)):
        seq_len = min(len(text), max_len)
        input_ids[i, :seq_len] = text[:seq_len]
        attention_mask[i, :seq_len] = 1

        # Only compute loss on response tokens
        response_start = min(prompt_len, seq_len)
        labels[i, response_start:seq_len] = text[response_start:seq_len]

    advantages = torch.stack([sample['advantage'] for sample in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'advantages': advantages,
        'prompt_lengths': torch.tensor(prompt_lengths)
    }


class GRPOTrainer:
    """
    GRPO trainer for multi-turn CUDA kernel optimization.

    Key features:
    - Group-based reward normalization (no value network needed)
    - Per-step training from trajectory decomposition
    - Aggressive gradient clipping (0.05)
    - Zero KL coefficient (allow policy deviation)
    """

    def __init__(
        self,
        config: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        ref_model: Optional[PreTrainedModel] = None
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.ref_model = ref_model  # For KL penalty if needed

        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision='bf16' if config.training.bf16 else ('fp16' if config.training.fp16 else 'no')
        )

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Reward computer
        self.reward_computer = RewardComputer(config.reward)

        # Logging
        self.global_step = 0
        self.epoch = 0

    def compute_log_probs(
        self,
        model: PreTrainedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for the response tokens.
        """
        with torch.no_grad() if model != self.model else torch.enable_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

        # Shift for autoregressive
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probs for actual tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)

        # Mask out padding and prompt tokens (labels=-100)
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # Sum log probs per sequence
        sequence_log_probs = token_log_probs.sum(dim=-1)

        return sequence_log_probs

    def compute_grpo_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss for a batch.

        GRPO loss = -E[advantage * log_prob(response)]

        Since advantages are already normalized within the group,
        this directly optimizes the policy without a value baseline.
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        advantages = batch['advantages']

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        # Compute per-token log probs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1).clamp(min=0)
        ).squeeze(-1)

        # Mask padding
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # Mean log prob per sequence (normalized by response length)
        response_lengths = mask.sum(dim=-1).clamp(min=1)
        mean_log_probs = token_log_probs.sum(dim=-1) / response_lengths

        # GRPO loss: negative advantage-weighted log prob
        # advantages should be normalized within batch
        loss = -(advantages * mean_log_probs).mean()

        # Compute metrics
        metrics = {
            'loss': loss.item(),
            'mean_log_prob': mean_log_probs.mean().item(),
            'mean_advantage': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }

        return loss, metrics

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute a single training step."""
        self.model.train()

        with self.accelerator.accumulate(self.model):
            loss, metrics = self.compute_grpo_loss(batch)

            self.accelerator.backward(loss)

            # Aggressive gradient clipping (as in Kevin-32B)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

        self.global_step += 1
        return metrics

    def train_epoch(
        self,
        dataloader: DataLoader,
        desc: str = "Training"
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'mean_log_prob': 0.0,
            'mean_advantage': 0.0
        }
        num_batches = 0

        progress_bar = tqdm(
            dataloader,
            desc=desc,
            disable=not self.accelerator.is_local_main_process
        )

        for batch in progress_bar:
            # Move to device
            batch = {
                k: v.to(self.accelerator.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            metrics = self.train_step(batch)

            # Accumulate metrics
            for key in epoch_metrics:
                if key in metrics:
                    epoch_metrics[key] += metrics[key]
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'step': self.global_step
            })

            # Log to wandb
            if self.global_step % self.config.training.logging_steps == 0:
                if self.accelerator.is_main_process and HAS_WANDB and wandb.run:
                    wandb.log({
                        'train/loss': metrics['loss'],
                        'train/mean_log_prob': metrics['mean_log_prob'],
                        'train/mean_advantage': metrics['mean_advantage'],
                        'train/step': self.global_step
                    })

        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= max(num_batches, 1)

        self.epoch += 1
        return epoch_metrics

    def train_on_trajectories(
        self,
        all_trajectories: List[List[Trajectory]],
        all_rewards: List[List[TrajectoryRewards]]
    ) -> Dict[str, float]:
        """
        Train on a batch of trajectories.

        This implements the per-step training from the MDP formulation:
        each refinement step becomes its own training sample with
        discounted cumulative rewards as the advantage.

        Args:
            all_trajectories: [task_idx][traj_idx] -> Trajectory
            all_rewards: [task_idx][traj_idx] -> TrajectoryRewards

        Returns:
            Training metrics
        """
        # Flatten trajectories into training samples
        samples = []
        max_steps = self.config.training.max_refinement_steps

        for task_idx, (task_trajectories, task_rewards) in enumerate(
            zip(all_trajectories, all_rewards)
        ):
            for traj_idx, (trajectory, rewards) in enumerate(
                zip(task_trajectories, task_rewards)
            ):
                for step_idx, step in enumerate(trajectory.steps):
                    if step_idx >= len(rewards.step_rewards):
                        continue

                    step_reward = rewards.step_rewards[step_idx]

                    sample = {
                        'prompt': step.prompt,
                        'response': step.response,
                        'advantage': step_reward.discounted_cumulative_reward,
                        'metadata': {
                            'task_id': trajectory.task.task_id,
                            'trajectory_id': trajectory.trajectory_id,
                            'step_idx': step_idx
                        }
                    }
                    samples.append(sample)

        if not samples:
            return {'loss': 0.0}

        # Normalize advantages within batch (GRPO)
        advantages = [s['advantage'] for s in samples]
        normalized_advantages = normalize_rewards_grpo(advantages)
        for sample, norm_adv in zip(samples, normalized_advantages):
            sample['advantage'] = norm_adv

        # Create dataset and dataloader
        dataset = GRPODataset(
            samples,
            self.tokenizer,
            self.config.model.max_seq_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.per_device_train_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_grpo_batch(
                b, self.tokenizer, self.config.model.max_seq_length
            )
        )

        # Prepare with accelerator
        dataloader = self.accelerator.prepare(dataloader)

        # Train for GRPO steps
        all_metrics = []
        for grpo_step in range(self.config.training.grpo_steps_per_batch):
            metrics = self.train_epoch(
                dataloader,
                desc=f"GRPO Step {grpo_step + 1}/{self.config.training.grpo_steps_per_batch}"
            )
            all_metrics.append(metrics)

        # Average metrics across GRPO steps
        avg_metrics = {}
        for key in all_metrics[0]:
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        return avg_metrics

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(
                path,
                save_function=self.accelerator.save
            )
            self.tokenizer.save_pretrained(path)

            # Save training state
            state = {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'optimizer_state': self.optimizer.state_dict()
            }
            torch.save(state, os.path.join(path, 'trainer_state.pt'))

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        state_path = os.path.join(path, 'trainer_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu')
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.optimizer.load_state_dict(state['optimizer_state'])


class OnlineGRPOTrainer(GRPOTrainer):
    """
    Online GRPO trainer that alternates between:
    1. Generating trajectories
    2. Evaluating and computing rewards
    3. Training on the trajectories
    """

    def __init__(
        self,
        config: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        trajectory_generator: Any,
        kernel_evaluator: Any,
        ref_model: Optional[PreTrainedModel] = None
    ):
        super().__init__(config, model, tokenizer, ref_model)
        self.trajectory_generator = trajectory_generator
        self.kernel_evaluator = kernel_evaluator

    def training_iteration(
        self,
        tasks: List[KernelTask]
    ) -> Dict[str, float]:
        """
        Execute one full training iteration:
        1. Generate trajectories
        2. Compute rewards
        3. Train on trajectories
        """
        # Generate trajectories
        self.model.eval()
        all_trajectories = self.trajectory_generator.generate_batch(
            tasks=tasks,
            evaluator=self.kernel_evaluator,
            trajectories_per_task=self.config.training.trajectories_per_task,
            max_steps=self.config.training.max_refinement_steps
        )

        # Compute rewards
        all_rewards = []
        for task_idx, task_trajectories in enumerate(all_trajectories):
            task_rewards = []
            for traj_idx, trajectory in enumerate(task_trajectories):
                eval_results = [
                    step.eval_result for step in trajectory.steps
                    if step.eval_result is not None
                ]

                rewards = self.reward_computer.compute_trajectory(
                    eval_results,
                    trajectory_id=trajectory.trajectory_id,
                    task_id=trajectory.task.task_id
                )
                task_rewards.append(rewards)
            all_rewards.append(task_rewards)

        # Train on trajectories
        train_metrics = self.train_on_trajectories(all_trajectories, all_rewards)

        # Compute additional metrics
        total_trajectories = sum(len(t) for t in all_trajectories)
        total_correct = sum(
            1 for task_rewards in all_rewards
            for rewards in task_rewards
            if rewards.final_correct
        )
        total_speedups = [
            rewards.final_speedup
            for task_rewards in all_rewards
            for rewards in task_rewards
            if rewards.final_speedup > 0
        ]

        metrics = {
            **train_metrics,
            'trajectories/total': total_trajectories,
            'trajectories/correct_rate': total_correct / max(total_trajectories, 1),
            'trajectories/mean_speedup': sum(total_speedups) / max(len(total_speedups), 1),
            'trajectories/max_speedup': max(total_speedups) if total_speedups else 0
        }

        return metrics

    def train(
        self,
        train_tasks: List[KernelTask],
        eval_tasks: Optional[List[KernelTask]] = None,
        num_iterations: Optional[int] = None
    ):
        """
        Main training loop.
        """
        if num_iterations is None:
            num_iterations = self.config.training.max_steps

        tasks_per_batch = self.config.training.tasks_per_batch

        for iteration in range(num_iterations):
            # Sample tasks for this iteration
            import random
            batch_tasks = random.sample(
                train_tasks,
                min(tasks_per_batch, len(train_tasks))
            )

            # Run training iteration
            metrics = self.training_iteration(batch_tasks)

            # Log
            if self.accelerator.is_main_process:
                print(f"Iteration {iteration + 1}/{num_iterations}")
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  Correct rate: {metrics['trajectories/correct_rate']:.2%}")
                print(f"  Mean speedup: {metrics['trajectories/mean_speedup']:.2f}x")

                if HAS_WANDB and wandb.run:
                    wandb.log({
                        f'iteration/{k}': v for k, v in metrics.items()
                    }, step=iteration)

            # Save checkpoint
            if (iteration + 1) % self.config.training.save_steps == 0:
                checkpoint_path = os.path.join(
                    self.config.training.checkpoint_dir,
                    f'checkpoint-{iteration + 1}'
                )
                self.save_checkpoint(checkpoint_path)

            # Evaluation
            if eval_tasks and (iteration + 1) % self.config.training.eval_steps == 0:
                eval_metrics = self.evaluate(eval_tasks)
                if self.accelerator.is_main_process:
                    print(f"  Eval correct rate: {eval_metrics['correct_rate']:.2%}")
                    print(f"  Eval mean speedup: {eval_metrics['mean_speedup']:.2f}x")

                    if HAS_WANDB and wandb.run:
                        wandb.log({
                            f'eval/{k}': v for k, v in eval_metrics.items()
                        }, step=iteration)

    @torch.no_grad()
    def evaluate(
        self,
        tasks: List[KernelTask],
        num_samples: int = 1
    ) -> Dict[str, float]:
        """Evaluate on a set of tasks."""
        self.model.eval()

        all_trajectories = self.trajectory_generator.generate_batch(
            tasks=tasks,
            evaluator=self.kernel_evaluator,
            trajectories_per_task=num_samples,
            max_steps=self.config.training.max_refinement_steps,
            temperature=0.3  # Lower temperature for evaluation
        )

        # Compute metrics
        total = 0
        correct = 0
        speedups = []

        for task_trajectories in all_trajectories:
            for trajectory in task_trajectories:
                total += 1
                if trajectory.final_result and trajectory.final_result.is_correct:
                    correct += 1
                    speedups.append(trajectory.final_result.speedup)

        return {
            'correct_rate': correct / max(total, 1),
            'mean_speedup': sum(speedups) / max(len(speedups), 1),
            'max_speedup': max(speedups) if speedups else 0.0,
            'total_evaluated': total
        }
