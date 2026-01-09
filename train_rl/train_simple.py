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
import os
import sys
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Configure logging for all modules."""
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True  # Override any existing configuration
    )

    # Explicitly set level for our modules
    for module_name in ['src.compiler', 'src.simple_trainer', 'src.reward', 'src.datasets', '__main__']:
        logging.getLogger(module_name).setLevel(log_level)

    # Reduce noise from other libraries
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# Note: src.* imports are done inside main() after logging is configured
# This ensures --verbose flag affects all debug logging


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

    # Dataset or single program mode
    parser.add_argument(
        "--program",
        type=str,
        default=None,
        help="Path to a single C++ program to optimize (use this OR --dataset)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=['polybench', 'directory'],
        help="Dataset type to use: 'polybench' or 'directory' (use this OR --program)"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset (required if --dataset is specified)"
    )
    parser.add_argument(
        "--dataset-sampling",
        type=str,
        default="random",
        choices=['random', 'sequential'],
        help="How to sample programs from dataset: 'random' or 'sequential' (default: random)"
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
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU IDs to use (e.g., '2,3' for GPUs 2 and 3)"
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging to see generated code and compilation details"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging based on verbosity - must be done before any logging calls
    setup_logging(verbose=args.verbose)

    # Now import src modules (after logging is configured)
    from src.simple_trainer import SimpleCodeOptimizationTrainer
    from src.datasets import create_dataset, SingleProgramDataset

    # Validate arguments
    if args.program is None and args.dataset is None:
        logger.error("Must specify either --program or --dataset")
        return 1

    if args.program is not None and args.dataset is not None:
        logger.error("Cannot specify both --program and --dataset")
        return 1

    if args.dataset is not None and args.dataset_path is None:
        logger.error("Must specify --dataset-path when using --dataset")
        return 1

    # Set GPU devices before any CUDA initialization
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        logger.info(f"Using GPUs: {args.gpus}")

    logger.info("=" * 80)
    logger.info("Simple RL-Based Code Optimization Training (REINFORCE)")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")

    # Initialize dataset
    dataset = None
    if args.dataset is not None:
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Dataset path: {args.dataset_path}")
        logger.info(f"Sampling strategy: {args.dataset_sampling}")
        try:
            if args.dataset == 'polybench':
                dataset = create_dataset('polybench', polybench_dir=args.dataset_path)
            elif args.dataset == 'directory':
                dataset = create_dataset('directory', directory=args.dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return 1
    else:
        logger.info(f"Program: {args.program}")
        # Single program mode - verify it exists
        program_path = Path(args.program)
        if not program_path.exists():
            logger.error(f"Program not found: {args.program}")
            return 1
        # Wrap single program as a dataset for consistency
        dataset = SingleProgramDataset(args.program)

    logger.info(f"Training steps: {args.steps}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Initialize trainer
    try:
        trainer = SimpleCodeOptimizationTrainer(
            model_name=args.model,
            dataset=dataset,
            sampling_strategy=args.dataset_sampling,
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
