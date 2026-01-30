#!/usr/bin/env python3
"""
Main training script for CUDA kernel RL optimization.
Implements multi-turn GRPO training as described in Kevin-32B.
"""

import os
import sys
import argparse
import random
from pathlib import Path

import torch
import numpy as np

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config, get_default_config, get_7b_config
from src.dataset import KernelBenchDataset
from src.model_utils import load_model_for_training, load_model_for_inference
from src.kernel_evaluator import KernelEvaluator
from src.trajectory import TrajectoryGenerator, BatchTrajectoryGenerator
from src.reward import RewardComputer
from src.grpo_trainer import OnlineGRPOTrainer


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CUDA kernel optimization model with GRPO"
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Model name or path"
    )
    parser.add_argument(
        "--use_7b",
        action="store_true",
        help="Use 7B model configuration"
    )

    # Training
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--tasks_per_batch",
        type=int,
        default=4,
        help="Number of tasks per batch"
    )
    parser.add_argument(
        "--trajectories_per_task",
        type=int,
        default=8,
        help="Number of parallel trajectories per task"
    )
    parser.add_argument(
        "--max_refinement_steps",
        type=int,
        default=6,
        help="Maximum refinement steps per trajectory"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-6,
        help="Learning rate"
    )

    # Memory optimization
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        default=True,
        help="Use 4-bit quantization"
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA"
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="LoRA rank"
    )

    # Paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory"
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default="./data/cache",
        help="Data cache directory"
    )

    # Logging
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cuda-kernel-rl",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="cuda_kernel_grpo",
        help="Experiment name"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable W&B logging"
    )

    # Hardware
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU device(s) to use, e.g. '0' or '0,1,2,3'"
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set GPU devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    print(f"Using GPU(s): {args.gpus}")

    # Set seed
    set_seed(args.seed)

    # Create config
    if args.use_7b:
        config = get_7b_config()
    else:
        config = get_default_config()

    # Override config with args
    config.model.model_name = args.model_name
    config.model.use_lora = not args.no_lora
    config.model.lora_r = args.lora_r
    config.training.tasks_per_batch = args.tasks_per_batch
    config.training.trajectories_per_task = args.trajectories_per_task
    config.training.max_refinement_steps = args.max_refinement_steps
    config.training.learning_rate = args.learning_rate
    config.training.output_dir = args.output_dir
    config.training.checkpoint_dir = args.checkpoint_dir
    config.dataset.cache_dir = args.data_cache_dir
    config.experiment_name = args.experiment_name
    config.wandb_project = args.wandb_project
    config.wandb_entity = args.wandb_entity
    config.seed = args.seed

    # Create directories
    os.makedirs(config.training.output_dir, exist_ok=True)
    os.makedirs(config.training.checkpoint_dir, exist_ok=True)
    os.makedirs(config.dataset.cache_dir, exist_ok=True)

    # Save config
    config.save(os.path.join(config.training.output_dir, "config.yaml"))

    # Initialize W&B
    if not args.no_wandb and HAS_WANDB:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.experiment_name,
            config={
                "model": config.model.model_name,
                "learning_rate": config.training.learning_rate,
                "tasks_per_batch": config.training.tasks_per_batch,
                "trajectories_per_task": config.training.trajectories_per_task,
                "max_refinement_steps": config.training.max_refinement_steps,
                "discount_factor": config.reward.discount_factor,
                "use_lora": config.model.use_lora,
                "lora_r": config.model.lora_r
            }
        )
    elif not args.no_wandb and not HAS_WANDB:
        print("Warning: wandb not installed, logging disabled")

    print("=" * 60)
    print("CUDA Kernel RL Training")
    print("=" * 60)
    print(f"Model: {config.model.model_name}")
    print(f"Tasks per batch: {config.training.tasks_per_batch}")
    print(f"Trajectories per task: {config.training.trajectories_per_task}")
    print(f"Max refinement steps: {config.training.max_refinement_steps}")
    print(f"Learning rate: {config.training.learning_rate}")
    print("=" * 60)

    # Load dataset
    print("\nLoading KernelBench dataset...")
    dataset = KernelBenchDataset(config.dataset)
    dataset.load()

    train_tasks, eval_tasks = dataset.get_train_test_split(seed=config.seed)
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Eval tasks: {len(eval_tasks)}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = load_model_for_training(
        config.model,
        use_4bit=args.use_4bit
    )

    # Initialize components
    print("\nInitializing training components...")
    kernel_evaluator = KernelEvaluator(config.evaluation)
    trajectory_generator = BatchTrajectoryGenerator(config, model, tokenizer)

    # Create trainer
    trainer = OnlineGRPOTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        trajectory_generator=trajectory_generator,
        kernel_evaluator=kernel_evaluator
    )

    # Resume if specified
    if args.resume_from:
        print(f"\nResuming from {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)

    # Train
    print("\nStarting training...")
    try:
        trainer.train(
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            num_iterations=args.num_iterations
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint(
            os.path.join(config.training.checkpoint_dir, "interrupted")
        )

    # Save final model
    print("\nSaving final model...")
    trainer.save_checkpoint(
        os.path.join(config.training.checkpoint_dir, "final")
    )

    # Finish W&B
    if HAS_WANDB and wandb.run:
        wandb.finish()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
