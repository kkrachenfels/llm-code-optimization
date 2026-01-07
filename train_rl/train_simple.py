#!/usr/bin/env python3
"""
Simple RL training script using REINFORCE algorithm.

Usage:
    python train_simple.py --model MODEL_NAME --program PROGRAM_PATH [OPTIONS]

Example:
    python train_simple.py --model Salesforce/codegen-350M-mono --program programs/bubble_sort.cpp --steps 10
"""

import argparse
import logging
from pathlib import Path

from src.simple_trainer import SimpleCodeOptimizationTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an LLM to optimize C++ code using REINFORCE"
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., 'Salesforce/codegen-350M-mono')"
    )
    parser.add_argument(
        "--program",
        type=str,
        default="programs/bubble_sort.cpp",
        help="Path to the C++ program to optimize (default: programs/bubble_sort.cpp)"
    )

    # Training arguments
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of training steps (default: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )

    # Model arguments
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length (default: 1024)"
    )
    parser.add_argument(
        "--use-8bit",
        action="store_true",
        help="Use 8-bit quantization to reduce memory usage"
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints (default: checkpoints)"
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N steps (default: 10)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Simple RL-Based Code Optimization Training (REINFORCE)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Program: {args.program}")
    logger.info(f"Training steps: {args.steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Verify program exists
    program_path = Path(args.program)
    if not program_path.exists():
        logger.error(f"Program not found: {args.program}")
        return 1

    # Initialize trainer
    try:
        trainer = SimpleCodeOptimizationTrainer(
            model_name=args.model,
            program_path=str(program_path),
            output_dir=args.output_dir,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_8bit=args.use_8bit,
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Train
    try:
        trainer.train(num_steps=args.steps, save_every=args.save_every)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # Save final checkpoint
        final_path = Path(args.output_dir) / "checkpoint-interrupted"
        trainer.save_checkpoint(final_path)
        logger.info(f"Saved interrupted checkpoint to {final_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    logger.info("Training completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
