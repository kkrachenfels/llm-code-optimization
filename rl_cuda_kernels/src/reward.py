"""
Reward computation for CUDA kernel RL training.
Implements MDP formulation with discounted cumulative rewards.
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from .config import RewardConfig
from .kernel_evaluator import EvaluationResult


@dataclass
class StepReward:
    """Reward for a single refinement step."""
    step_idx: int
    correctness_reward: float
    performance_reward: float
    total_raw_reward: float
    discounted_cumulative_reward: float


@dataclass
class TrajectoryRewards:
    """Rewards for an entire trajectory."""
    trajectory_id: str
    task_id: str
    step_rewards: List[StepReward]
    final_reward: float
    final_speedup: float
    final_correct: bool


def compute_step_reward(
    eval_result: EvaluationResult,
    config: RewardConfig
) -> Tuple[float, float, float]:
    """
    Compute raw reward for a single step.

    Returns:
        Tuple of (correctness_reward, performance_reward, total_reward)
    """
    # Base rewards
    correctness_reward = 0.0
    performance_reward = 0.0

    # Handle failure cases
    if not eval_result.compilation_success:
        return config.compilation_failure_penalty, 0.0, config.compilation_failure_penalty

    if not eval_result.runtime_success:
        return config.runtime_error_penalty, 0.0, config.runtime_error_penalty

    if eval_result.error_type == 'Timeout':
        return config.timeout_penalty, 0.0, config.timeout_penalty

    # Correctness reward
    if eval_result.is_correct:
        correctness_reward = config.correctness_reward

        # Performance reward (speedup)
        speedup = eval_result.speedup
        speedup = max(config.min_speedup, min(speedup, config.max_speedup))
        performance_reward = speedup

        # Bonus for significant improvement
        if speedup >= config.improvement_threshold:
            performance_reward += config.improvement_bonus

    total_reward = correctness_reward + performance_reward

    return correctness_reward, performance_reward, total_reward


def compute_trajectory_rewards(
    eval_results: List[EvaluationResult],
    config: RewardConfig,
    trajectory_id: str = "",
    task_id: str = ""
) -> TrajectoryRewards:
    """
    Compute discounted cumulative rewards for a trajectory.

    Following the Kevin-32B MDP formulation:
    - Each refinement step becomes its own training sample
    - Reward per step = discounted sum of current and all subsequent kernel scores
    - Uses discount factor (gamma) of 0.4

    Args:
        eval_results: List of evaluation results for each step
        config: Reward configuration
        trajectory_id: Identifier for this trajectory
        task_id: Identifier for the task

    Returns:
        TrajectoryRewards with per-step discounted rewards
    """
    n_steps = len(eval_results)
    gamma = config.discount_factor

    # Compute raw rewards for each step
    raw_rewards = []
    for result in eval_results:
        _, _, total = compute_step_reward(result, config)
        raw_rewards.append(total)

    # Compute discounted cumulative rewards (backwards)
    # R_t = r_t + gamma * R_{t+1}
    discounted_rewards = [0.0] * n_steps

    # Start from the last step
    discounted_rewards[-1] = raw_rewards[-1]

    # Work backwards
    for t in range(n_steps - 2, -1, -1):
        discounted_rewards[t] = raw_rewards[t] + gamma * discounted_rewards[t + 1]

    # Build step rewards
    step_rewards = []
    for i, (result, raw, discounted) in enumerate(zip(
        eval_results, raw_rewards, discounted_rewards
    )):
        correctness, performance, _ = compute_step_reward(result, config)
        step_rewards.append(StepReward(
            step_idx=i,
            correctness_reward=correctness,
            performance_reward=performance,
            total_raw_reward=raw,
            discounted_cumulative_reward=discounted
        ))

    # Get final metrics
    final_result = eval_results[-1]
    final_speedup = final_result.speedup if final_result.success else 0.0
    final_correct = final_result.is_correct if final_result.success else False

    return TrajectoryRewards(
        trajectory_id=trajectory_id,
        task_id=task_id,
        step_rewards=step_rewards,
        final_reward=discounted_rewards[0],  # Total trajectory reward
        final_speedup=final_speedup,
        final_correct=final_correct
    )


def normalize_rewards_grpo(
    rewards: List[float],
    eps: float = 1e-8
) -> List[float]:
    """
    Normalize rewards within a group for GRPO.

    GRPO normalizes rewards within the group of responses sampled from the same prompt,
    eliminating the need for a value network baseline.

    Args:
        rewards: List of rewards for responses from the same prompt
        eps: Small constant for numerical stability

    Returns:
        Normalized rewards (mean=0, std=1 within group)
    """
    if len(rewards) <= 1:
        return [0.0] * len(rewards)

    rewards_array = np.array(rewards)
    mean = rewards_array.mean()
    std = rewards_array.std()

    if std < eps:
        return [0.0] * len(rewards)

    normalized = (rewards_array - mean) / (std + eps)
    return normalized.tolist()


def compute_advantages_grpo(
    trajectory_rewards: List[TrajectoryRewards],
    step_idx: int
) -> List[float]:
    """
    Compute GRPO advantages for a specific step across trajectories.

    For GRPO, advantages are computed by normalizing rewards within the group
    of responses for the same prompt.

    Args:
        trajectory_rewards: List of TrajectoryRewards for parallel trajectories on same task
        step_idx: The refinement step index to compute advantages for

    Returns:
        List of advantages for each trajectory at this step
    """
    # Extract rewards at this step
    step_rewards = []
    for tr in trajectory_rewards:
        if step_idx < len(tr.step_rewards):
            step_rewards.append(tr.step_rewards[step_idx].discounted_cumulative_reward)
        else:
            # Trajectory ended early
            step_rewards.append(0.0)

    return normalize_rewards_grpo(step_rewards)


class RewardComputer:
    """
    Main class for computing rewards in the training loop.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def compute_single_step(
        self,
        eval_result: EvaluationResult
    ) -> Tuple[float, float, float]:
        """Compute reward for a single evaluation."""
        return compute_step_reward(eval_result, self.config)

    def compute_trajectory(
        self,
        eval_results: List[EvaluationResult],
        trajectory_id: str = "",
        task_id: str = ""
    ) -> TrajectoryRewards:
        """Compute rewards for an entire trajectory."""
        return compute_trajectory_rewards(
            eval_results, self.config, trajectory_id, task_id
        )

    def compute_batch_advantages(
        self,
        all_trajectory_rewards: List[List[TrajectoryRewards]],
        max_steps: int
    ) -> List[List[List[float]]]:
        """
        Compute advantages for a batch of tasks.

        Args:
            all_trajectory_rewards: List of lists of TrajectoryRewards
                                   [task_idx][trajectory_idx]
            max_steps: Maximum number of refinement steps

        Returns:
            Advantages [task_idx][trajectory_idx][step_idx]
        """
        all_advantages = []

        for task_trajectories in all_trajectory_rewards:
            task_advantages = [[] for _ in task_trajectories]

            for step_idx in range(max_steps):
                step_advantages = compute_advantages_grpo(
                    task_trajectories, step_idx
                )

                for traj_idx, adv in enumerate(step_advantages):
                    task_advantages[traj_idx].append(adv)

            all_advantages.append(task_advantages)

        return all_advantages

    def get_feedback_string(
        self,
        eval_result: EvaluationResult,
        include_suggestions: bool = True
    ) -> str:
        """
        Generate a feedback string for the model.

        Args:
            eval_result: The evaluation result
            include_suggestions: Whether to include improvement suggestions

        Returns:
            Human-readable feedback string
        """
        feedback_parts = [eval_result.feedback]

        if include_suggestions:
            if not eval_result.compilation_success:
                feedback_parts.append(
                    "Suggestion: Check for syntax errors and missing imports."
                )
            elif not eval_result.runtime_success:
                feedback_parts.append(
                    "Suggestion: Verify tensor shapes and device placement."
                )
            elif not eval_result.is_correct:
                feedback_parts.append(
                    f"Suggestion: Your kernel produces incorrect results. "
                    f"Max error: {eval_result.max_abs_error:.2e}. "
                    "Check numerical precision and algorithm correctness."
                )
            elif eval_result.speedup < 1.0:
                feedback_parts.append(
                    f"Suggestion: Your kernel is {1/eval_result.speedup:.2f}x slower "
                    "than reference. Consider memory coalescing, shared memory, "
                    "or reducing thread divergence."
                )
            elif eval_result.speedup < 1.5:
                feedback_parts.append(
                    "Suggestion: Good start! To improve further, consider "
                    "loop unrolling, prefetching, or using tensor cores."
                )

        return " ".join(feedback_parts)


def aggregate_training_stats(
    all_trajectory_rewards: List[TrajectoryRewards]
) -> dict:
    """
    Aggregate statistics across all trajectories for logging.
    """
    if not all_trajectory_rewards:
        return {}

    final_rewards = [tr.final_reward for tr in all_trajectory_rewards]
    final_speedups = [tr.final_speedup for tr in all_trajectory_rewards if tr.final_speedup > 0]
    correct_count = sum(1 for tr in all_trajectory_rewards if tr.final_correct)

    stats = {
        'mean_final_reward': np.mean(final_rewards),
        'std_final_reward': np.std(final_rewards),
        'max_final_reward': np.max(final_rewards),
        'min_final_reward': np.min(final_rewards),
        'mean_speedup': np.mean(final_speedups) if final_speedups else 0.0,
        'max_speedup': np.max(final_speedups) if final_speedups else 0.0,
        'correctness_rate': correct_count / len(all_trajectory_rewards),
        'total_trajectories': len(all_trajectory_rewards)
    }

    # Per-step statistics
    max_steps = max(len(tr.step_rewards) for tr in all_trajectory_rewards)
    for step in range(max_steps):
        step_rewards = [
            tr.step_rewards[step].discounted_cumulative_reward
            for tr in all_trajectory_rewards
            if step < len(tr.step_rewards)
        ]
        if step_rewards:
            stats[f'step_{step}_mean_reward'] = np.mean(step_rewards)
            stats[f'step_{step}_std_reward'] = np.std(step_rewards)

    return stats


if __name__ == "__main__":
    # Test reward computation
    from .kernel_evaluator import EvaluationResult

    config = RewardConfig()
    computer = RewardComputer(config)

    # Simulate a trajectory with improving results
    eval_results = [
        EvaluationResult(
            success=False,
            compilation_success=True,
            runtime_success=True,
            is_correct=False,
            feedback="Incorrect results"
        ),
        EvaluationResult(
            success=True,
            compilation_success=True,
            runtime_success=True,
            is_correct=True,
            speedup=0.8,
            feedback="Correct but slower"
        ),
        EvaluationResult(
            success=True,
            compilation_success=True,
            runtime_success=True,
            is_correct=True,
            speedup=1.5,
            feedback="1.5x speedup achieved"
        ),
    ]

    trajectory_rewards = computer.compute_trajectory(
        eval_results,
        trajectory_id="test_0",
        task_id="matmul"
    )

    print("Trajectory Rewards:")
    for step in trajectory_rewards.step_rewards:
        print(f"  Step {step.step_idx}: raw={step.total_raw_reward:.3f}, "
              f"discounted={step.discounted_cumulative_reward:.3f}")
    print(f"Final reward: {trajectory_rewards.final_reward:.3f}")
    print(f"Final speedup: {trajectory_rewards.final_speedup:.2f}x")
