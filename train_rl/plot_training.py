#!/usr/bin/env python3
"""
Parse training logs and create plots of training progress.

Supports both step-based and epoch-based training logs.

Usage:
    python plot_training.py <log_file> [--output OUTPUT_DIR]

Example:
    python plot_training.py training.log --output plots/
"""

import argparse
import re
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def parse_step_log(log_path: str) -> dict:
    """Parse step-based training log and extract metrics."""
    metrics = {
        'steps': [],
        'programs': [],
        'mean_reward': [],
        'loss': [],
        'best_speedup': [],
        'successes': [],
        'failures': [],
    }

    step_pattern = re.compile(
        r'Step (\d+) \(([^)]+)\): mean_reward=([-\d.]+), loss=([-\d.]+), best_speedup=([\d.]+)x'
    )
    success_pattern = re.compile(r'Success! Runtime: ([\d.]+).*Reward: ([-\d.]+)')
    failure_pattern = re.compile(r'Failed:.*Reward: ([-\d.]+)')

    current_step_successes = 0
    current_step_failures = 0

    with open(log_path, 'r') as f:
        for line in f:
            # Check for step summary
            match = step_pattern.search(line)
            if match:
                step = int(match.group(1))
                program = match.group(2)
                mean_reward = float(match.group(3))
                loss = float(match.group(4))
                speedup = float(match.group(5))

                metrics['steps'].append(step)
                metrics['programs'].append(program)
                metrics['mean_reward'].append(mean_reward)
                metrics['loss'].append(loss)
                metrics['best_speedup'].append(speedup)
                metrics['successes'].append(current_step_successes)
                metrics['failures'].append(current_step_failures)

                # Reset counters for next step
                current_step_successes = 0
                current_step_failures = 0
                continue

            # Count successes and failures
            if success_pattern.search(line):
                current_step_successes += 1
            elif failure_pattern.search(line):
                current_step_failures += 1

    return metrics


def parse_epoch_log(log_path: str) -> dict:
    """Parse epoch-based training log and extract metrics."""
    metrics = {
        'epochs': [],
        'train_mean_reward': [],
        'train_mean_speedup': [],
        'test_mean_reward': [],
        'test_mean_speedup': [],
        # Per-step data within epochs
        'all_steps': [],
        'step_programs': [],
        'step_mean_reward': [],
        'step_loss': [],
        'step_speedup': [],
        'step_successes': [],
        'step_failures': [],
    }

    # Epoch summary patterns
    epoch_pattern = re.compile(r'EPOCH (\d+) SUMMARY:')
    train_summary_pattern = re.compile(
        r'Train: mean_reward=([-\d.]+), mean_speedup=([\d.]+)x'
    )
    test_summary_pattern = re.compile(
        r'Test:\s+mean_reward=([-\d.]+), mean_speedup=([\d.]+)x'
    )

    # Per-step patterns within epochs
    train_step_pattern = re.compile(
        r'Train step (\d+)/(\d+) \(([^)]+)\): mean_reward=([-\d.]+), loss=([-\d.]+), speedup=([\d.]+)x'
    )
    success_pattern = re.compile(r'Success! Runtime: ([\d.]+).*Reward: ([-\d.]+)')
    failure_pattern = re.compile(r'Failed:.*Reward: ([-\d.]+)')

    current_epoch = None
    current_train_reward = None
    current_train_speedup = None
    current_step_successes = 0
    current_step_failures = 0
    global_step = 0

    with open(log_path, 'r') as f:
        for line in f:
            # Check for epoch summary start
            match = epoch_pattern.search(line)
            if match:
                current_epoch = int(match.group(1))
                continue

            # Check for train summary
            match = train_summary_pattern.search(line)
            if match and current_epoch is not None:
                current_train_reward = float(match.group(1))
                current_train_speedup = float(match.group(2))
                continue

            # Check for test summary (completes the epoch)
            match = test_summary_pattern.search(line)
            if match and current_epoch is not None and current_train_reward is not None:
                metrics['epochs'].append(current_epoch)
                metrics['train_mean_reward'].append(current_train_reward)
                metrics['train_mean_speedup'].append(current_train_speedup)
                metrics['test_mean_reward'].append(float(match.group(1)))
                metrics['test_mean_speedup'].append(float(match.group(2)))

                # Reset for next epoch
                current_epoch = None
                current_train_reward = None
                current_train_speedup = None
                continue

            # Check for train step
            match = train_step_pattern.search(line)
            if match:
                global_step += 1
                metrics['all_steps'].append(global_step)
                metrics['step_programs'].append(match.group(3))
                metrics['step_mean_reward'].append(float(match.group(4)))
                metrics['step_loss'].append(float(match.group(5)))
                metrics['step_speedup'].append(float(match.group(6)))
                metrics['step_successes'].append(current_step_successes)
                metrics['step_failures'].append(current_step_failures)

                current_step_successes = 0
                current_step_failures = 0
                continue

            # Count successes and failures
            if success_pattern.search(line):
                current_step_successes += 1
            elif failure_pattern.search(line):
                current_step_failures += 1

    return metrics


def detect_log_format(log_path: str) -> str:
    """Detect whether log is step-based or epoch-based."""
    with open(log_path, 'r') as f:
        content = f.read()
        if 'EPOCH' in content and 'Train:' in content and 'Test:' in content:
            return 'epoch'
        return 'step'


def plot_step_metrics(metrics: dict, output_dir: str = None, log_file: str = None):
    """Create plots from step-based metrics."""
    if not metrics['steps']:
        print("No training data found in log file!")
        return

    output_path = Path(output_dir) if output_dir else Path('.')
    output_path.mkdir(exist_ok=True, parents=True)

    # Use log file prefix for output filename
    if log_file:
        prefix = Path(log_file).stem
    else:
        prefix = 'training_progress'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Progress (Step-based)', fontsize=14)

    steps = metrics['steps']

    # Plot 1: Mean Reward
    ax1 = axes[0, 0]
    ax1.plot(steps, metrics['mean_reward'], 'b-o', markersize=4)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Mean Reward per Step')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Best Speedup
    ax2 = axes[0, 1]
    ax2.plot(steps, metrics['best_speedup'], 'g-o', markersize=4)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Best Speedup per Step')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Success/Failure Rate
    ax3 = axes[1, 0]
    width = 0.35
    ax3.bar([s - width/2 for s in steps], metrics['successes'], width, label='Successes', color='green', alpha=0.7)
    ax3.bar([s + width/2 for s in steps], metrics['failures'], width, label='Failures', color='red', alpha=0.7)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Count')
    ax3.set_title('Compilation Success/Failure per Step')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Loss
    ax4 = axes[1, 1]
    ax4.plot(steps, metrics['loss'], 'r-o', markersize=4)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss')
    ax4.set_title('Training Loss per Step')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_path / f'{prefix}.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")

    # Print summary
    print_step_summary(metrics)
    plt.show()


def plot_epoch_metrics(metrics: dict, output_dir: str = None, log_file: str = None):
    """Create plots from epoch-based metrics."""
    if not metrics['epochs']:
        print("No epoch data found in log file!")
        return

    output_path = Path(output_dir) if output_dir else Path('.')
    output_path.mkdir(exist_ok=True, parents=True)

    # Use log file prefix for output filename
    if log_file:
        prefix = Path(log_file).stem
    else:
        prefix = 'training_progress'

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Progress (Epoch-based)', fontsize=14)

    epochs = metrics['epochs']

    # Plot 1: Train vs Test Mean Reward
    ax1 = axes[0, 0]
    ax1.plot(epochs, metrics['train_mean_reward'], 'b-o', markersize=6, label='Train')
    ax1.plot(epochs, metrics['test_mean_reward'], 'r-s', markersize=6, label='Test')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Reward')
    ax1.set_title('Mean Reward: Train vs Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Train vs Test Speedup
    ax2 = axes[0, 1]
    ax2.plot(epochs, metrics['train_mean_speedup'], 'b-o', markersize=6, label='Train')
    ax2.plot(epochs, metrics['test_mean_speedup'], 'r-s', markersize=6, label='Test')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Speedup (x)')
    ax2.set_title('Mean Speedup: Train vs Test')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Per-step reward within epochs
    ax3 = axes[1, 0]
    if metrics['all_steps']:
        ax3.plot(metrics['all_steps'], metrics['step_mean_reward'], 'b-', alpha=0.7, linewidth=1)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        # Add vertical lines for epoch boundaries
        steps_per_epoch = len(metrics['all_steps']) // len(epochs) if epochs else 0
        if steps_per_epoch > 0:
            for i, e in enumerate(epochs[:-1]):
                ax3.axvline(x=(i + 1) * steps_per_epoch, color='red', linestyle=':', alpha=0.3)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Mean Reward')
    ax3.set_title('Per-Step Mean Reward')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Per-step loss
    ax4 = axes[1, 1]
    if metrics['all_steps']:
        ax4.plot(metrics['all_steps'], metrics['step_loss'], 'r-', alpha=0.7, linewidth=1)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Loss')
    ax4.set_title('Per-Step Training Loss')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = output_path / f'{prefix}.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")

    # Print summary
    print_epoch_summary(metrics)
    plt.show()


def print_step_summary(metrics: dict):
    """Print summary for step-based training."""
    print("\n" + "="*70)
    print("Training Summary (Step-based)")
    print("="*70)
    print(f"{'Step':<6} {'Program':<20} {'Mean Reward':<12} {'Speedup':<10} {'Success/Fail'}")
    print("-"*70)
    for i, step in enumerate(metrics['steps']):
        print(f"{step:<6} {metrics['programs'][i]:<20} {metrics['mean_reward'][i]:<12.3f} {metrics['best_speedup'][i]:<10.2f}x {metrics['successes'][i]}/{metrics['failures'][i]}")
    print("="*70)

    # Overall stats
    avg_reward = sum(metrics['mean_reward']) / len(metrics['mean_reward'])
    max_speedup = max(metrics['best_speedup'])
    total_success = sum(metrics['successes'])
    total_fail = sum(metrics['failures'])
    success_rate = total_success / (total_success + total_fail) * 100 if (total_success + total_fail) > 0 else 0

    print(f"\nOverall Statistics:")
    print(f"  Average Mean Reward: {avg_reward:.3f}")
    print(f"  Max Speedup: {max_speedup:.2f}x")
    print(f"  Total Successes: {total_success}")
    print(f"  Total Failures: {total_fail}")
    print(f"  Success Rate: {success_rate:.1f}%")


def print_epoch_summary(metrics: dict):
    """Print summary for epoch-based training."""
    print("\n" + "="*80)
    print("Training Summary (Epoch-based)")
    print("="*80)
    print(f"{'Epoch':<8} {'Train Reward':<14} {'Train Speedup':<14} {'Test Reward':<14} {'Test Speedup':<14}")
    print("-"*80)
    for i, epoch in enumerate(metrics['epochs']):
        print(f"{epoch:<8} {metrics['train_mean_reward'][i]:<14.3f} {metrics['train_mean_speedup'][i]:<14.2f}x {metrics['test_mean_reward'][i]:<14.3f} {metrics['test_mean_speedup'][i]:<14.2f}x")
    print("="*80)

    # Overall stats
    print(f"\nOverall Statistics:")
    print(f"  Final Train Mean Reward: {metrics['train_mean_reward'][-1]:.3f}")
    print(f"  Final Test Mean Reward:  {metrics['test_mean_reward'][-1]:.3f}")
    print(f"  Final Train Speedup:     {metrics['train_mean_speedup'][-1]:.2f}x")
    print(f"  Final Test Speedup:      {metrics['test_mean_speedup'][-1]:.2f}x")

    # Check for overfitting
    train_improvement = metrics['train_mean_reward'][-1] - metrics['train_mean_reward'][0]
    test_improvement = metrics['test_mean_reward'][-1] - metrics['test_mean_reward'][0]
    print(f"\n  Train Reward Change: {train_improvement:+.3f}")
    print(f"  Test Reward Change:  {test_improvement:+.3f}")

    if train_improvement > 0.1 and test_improvement < 0:
        print("\n  WARNING: Possible overfitting detected (train improving, test declining)")
    elif test_improvement > 0:
        print("\n  Model appears to be generalizing well!")


def main():
    parser = argparse.ArgumentParser(description='Plot training progress from log files')
    parser.add_argument('log_file', help='Path to training log file')
    parser.add_argument('--output', '-o', default='plots', help='Output directory for plots')

    args = parser.parse_args()

    print(f"Parsing log file: {args.log_file}")

    # Detect format and parse accordingly
    log_format = detect_log_format(args.log_file)
    print(f"Detected log format: {log_format}")

    if log_format == 'epoch':
        metrics = parse_epoch_log(args.log_file)
        print(f"Found {len(metrics['epochs'])} epochs, {len(metrics['all_steps'])} training steps")
        plot_epoch_metrics(metrics, args.output, args.log_file)
    else:
        metrics = parse_step_log(args.log_file)
        print(f"Found {len(metrics['steps'])} training steps")
        plot_step_metrics(metrics, args.output, args.log_file)


if __name__ == '__main__':
    main()
