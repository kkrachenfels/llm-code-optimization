#!/usr/bin/env python3
"""
Main training script for RL-based CUDA kernel optimization.

Usage:
    python -m rl_kernel_opt.scripts.train --config configs/default.yaml
    python -m rl_kernel_opt.scripts.train --config configs/default.yaml --reward_mode speed_correct_profile
    python -m rl_kernel_opt.scripts.train --resume outputs/checkpoint-1000
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from rl_kernel_opt.src.task_sampler import TaskSampler
from rl_kernel_opt.src.reward_calculator import RewardCalculator, RewardMode
from rl_kernel_opt.src.grpo_trainer import GRPOTrainer, GRPOConfig
from rl_kernel_opt.src.utils import load_config, merge_configs, set_seed, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train CUDA kernel optimization model")

    parser.add_argument(
        "--config",
        type=str,
        default="rl_kernel_opt/configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        choices=["speed", "speed_correct", "speed_correct_profile"],
        default=None,
        help="Override reward mode from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Override max training steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with reduced settings",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Override GPU IDs. Single int (e.g., '3') or comma-separated (e.g., '0,1,2'). "
             "First GPU is used for model, all are used for evaluation.",
    )

    return parser.parse_args()


def parse_gpu_ids(gpu_str):
    """Parse GPU ID string/list into list of integers."""
    if gpu_str is None:
        return None
    if isinstance(gpu_str, list):
        return [int(x) for x in gpu_str]
    if isinstance(gpu_str, int):
        return [gpu_str]
    return [int(x.strip()) for x in str(gpu_str).split(",")]


def build_config(args) -> GRPOConfig:
    """Build training config from file and command line args."""
    # Load base config
    config_dict = load_config(args.config)

    # Apply command line overrides
    if args.reward_mode:
        config_dict.setdefault("reward", {})["mode"] = args.reward_mode
    if args.output_dir:
        config_dict.setdefault("training", {})["output_dir"] = args.output_dir
    if args.max_steps:
        config_dict.setdefault("training", {})["max_steps"] = args.max_steps
    if args.seed:
        config_dict.setdefault("training", {})["seed"] = args.seed

    # Debug mode overrides
    if args.debug:
        config_dict.setdefault("training", {}).update({
            "tasks_per_batch": 1,
            "max_turns": 1,
            "max_steps": 10,
            "log_interval": 1,
            "save_interval": 5,
        })
        config_dict.setdefault("grpo", {})["group_size"] = 2
        config_dict.setdefault("evaluation", {}).update({
            "warmup_time": 5,
            "repetition_time": 100,
            "timeout": 60,
        })

    # Build GRPOConfig
    model_cfg = config_dict.get("model", {})
    grpo_cfg = config_dict.get("grpo", {})
    training_cfg = config_dict.get("training", {})
    reward_cfg = config_dict.get("reward", {})

    grpo_config = GRPOConfig(
        # Model
        model_name=model_cfg.get("name", "Qwen/Qwen2.5-Coder-3B-Instruct"),
        max_new_tokens=model_cfg.get("max_new_tokens", 4096),
        temperature=model_cfg.get("temperature", 0.7),
        top_p=model_cfg.get("top_p", 0.9),
        # GRPO
        group_size=grpo_cfg.get("group_size", 8),
        kl_coef=grpo_cfg.get("kl_coef", 0.05),
        clip_ratio=grpo_cfg.get("clip_ratio", 0.2),
        gamma=grpo_cfg.get("gamma", 0.4),
        # Training
        tasks_per_batch=training_cfg.get("tasks_per_batch", 4),
        max_turns=training_cfg.get("max_turns", 3),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 2),
        learning_rate=training_cfg.get("learning_rate", 1e-6),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        warmup_steps=training_cfg.get("warmup_steps", 100),
        max_steps=training_cfg.get("max_steps", 5000),
        save_interval=training_cfg.get("save_interval", 500),
        eval_interval=training_cfg.get("eval_interval", 100),
        log_interval=training_cfg.get("log_interval", 10),
        seed=training_cfg.get("seed", 42),
        # Reward
        reward_mode=reward_cfg.get("mode", "speed_correct"),
        correctness_bonus=reward_cfg.get("correctness_bonus", 0.3),
        max_speedup_reward=reward_cfg.get("max_speedup_reward", 5.0),
        # Output
        output_dir=training_cfg.get("output_dir", "outputs"),
    )

    return grpo_config, config_dict


def main():
    args = parse_args()

    # Build config
    grpo_config, full_config = build_config(args)

    # Set seed
    set_seed(grpo_config.seed)

    # Setup logging
    log_dir = setup_logging(
        full_config.get("logging", {}).get("log_dir", "outputs/logs"),
        use_wandb=full_config.get("logging", {}).get("use_wandb", False),
        project=full_config.get("logging", {}).get("wandb_project", "rl-kernel-opt"),
    )

    print("=" * 60)
    print("RL-Based CUDA Kernel Optimization")
    print("=" * 60)
    print(f"Model: {grpo_config.model_name}")
    print(f"Reward mode: {grpo_config.reward_mode}")
    print(f"Group size: {grpo_config.group_size}")
    print(f"Max turns: {grpo_config.max_turns}")
    print(f"Max steps: {grpo_config.max_steps}")
    print(f"Output dir: {grpo_config.output_dir}")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This training requires GPU.")
        sys.exit(1)

    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(args.gpu) or parse_gpu_ids(
        full_config.get("hardware", {}).get("gpu_ids", [0])
    )
    model_gpu = gpu_ids[0]
    eval_gpu_ids = gpu_ids

    print(f"CUDA devices available: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        marker = " <--" if i in gpu_ids else ""
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}{marker}")

    print(f"\nUsing GPU {model_gpu} for model, GPUs {eval_gpu_ids} for evaluation")

    # Initialize task sampler
    task_dirs = full_config.get("tasks", {}).get("task_dirs", [])
    if not task_dirs:
        print("ERROR: No task directories specified in config")
        sys.exit(1)

    print(f"\nLoading tasks from: {task_dirs}")
    task_sampler = TaskSampler(
        task_dirs=task_dirs,
        forward=full_config.get("tasks", {}).get("forward", True),
        seed=grpo_config.seed,
    )

    if len(task_sampler) == 0:
        print("ERROR: No valid tasks found")
        sys.exit(1)

    print(f"Loaded {len(task_sampler)} tasks")

    # Initialize reward calculator
    eval_cfg = full_config.get("evaluation", {})
    reward_mode = RewardMode(grpo_config.reward_mode)

    reward_calculator = RewardCalculator(
        mode=reward_mode,
        correctness_bonus=grpo_config.correctness_bonus,
        max_speedup_reward=grpo_config.max_speedup_reward,
        warmup_time=eval_cfg.get("warmup_time", 25),
        repetition_time=eval_cfg.get("repetition_time", 10000),
        timeout=eval_cfg.get("timeout", 300),
        multi_init_settings=eval_cfg.get("multi_init_settings", True),
        multi_input_settings=eval_cfg.get("multi_input_settings", True),
        op_atol=eval_cfg.get("op_atol", 1e-5),
        op_rtol=eval_cfg.get("op_rtol", 1e-5),
        eval_type=eval_cfg.get("eval_type", "kernelbench"),
        torch_prof=reward_mode == RewardMode.SPEED_CORRECT_PROFILE,
        gpu_ids=eval_gpu_ids,
    )

    print(f"\nReward calculator initialized with mode: {reward_mode.value}")

    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = GRPOTrainer(
        config=grpo_config,
        task_sampler=task_sampler,
        reward_calculator=reward_calculator,
        device=f"cuda:{model_gpu}",
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(Path(grpo_config.output_dir) / "interrupted")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        trainer.save_checkpoint(Path(grpo_config.output_dir) / "error")
        raise

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
