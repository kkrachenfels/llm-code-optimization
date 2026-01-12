#!/usr/bin/env python3
"""
Parse training logs and create plots of training progress.

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


def parse_log_file(log_path: str) -> dict:
    """Parse training log and extract metrics."""
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


def plot_metrics(metrics: dict, output_dir: str = None):
    """Create plots from parsed metrics."""
    if not metrics['steps']:
        print("No training data found in log file!")
        return

    output_path = Path(output_dir) if output_dir else Path('.')
    output_path.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Training Progress', fontsize=14)

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
    plot_path = output_path / 'training_progress.png'
    plt.savefig(plot_path, dpi=150)
    print(f"Saved plot to: {plot_path}")

    # Also create a summary table
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    print(f"{'Step':<6} {'Program':<20} {'Mean Reward':<12} {'Speedup':<10} {'Success/Fail'}")
    print("-"*70)
    for i, step in enumerate(steps):
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

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot training progress from log files')
    parser.add_argument('log_file', help='Path to training log file')
    parser.add_argument('--output', '-o', default='plots', help='Output directory for plots')

    args = parser.parse_args()

    print(f"Parsing log file: {args.log_file}")
    metrics = parse_log_file(args.log_file)

    print(f"Found {len(metrics['steps'])} training steps")
    plot_metrics(metrics, args.output)


if __name__ == '__main__':
    main()
