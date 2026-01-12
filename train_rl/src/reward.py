import numpy as np
from typing import Optional


class RewardFunction:
    """Compute rewards for code optimization based on runtime improvements."""

    def __init__(
        self,
        baseline_runtime: float,
        speedup_weight: float = 1.0,
        compilation_penalty: float = -1.0,
        timeout_penalty: float = -2.0,
        correctness_bonus: float = 0.5
    ):
        """
        Initialize reward function.

        Args:
            baseline_runtime: Baseline runtime in microseconds
            speedup_weight: Weight for speedup reward
            compilation_penalty: Penalty for compilation failures
            timeout_penalty: Penalty for timeouts
            correctness_bonus: Bonus for producing correct, compilable code
        """
        self.baseline_runtime = baseline_runtime
        self.speedup_weight = speedup_weight
        self.compilation_penalty = compilation_penalty
        self.timeout_penalty = timeout_penalty
        self.correctness_bonus = correctness_bonus

    def compute_reward(
        self,
        success: bool,
        runtime: Optional[float],
        error_message: str
    ) -> float:
        """
        Compute reward based on execution results.

        Args:
            success: Whether compilation and execution succeeded
            runtime: Runtime in microseconds (None if failed)
            error_message: Error message if failed

        Returns:
            Reward value
        """
        if not success:
            # Penalize failures
            if "timeout" in error_message.lower():
                return self.timeout_penalty
            else:
                return self.compilation_penalty

        # Compute speedup
        speedup = self.baseline_runtime / runtime

        # Reward formula: log-based for smooth gradient
        # log(speedup) is positive for speedup > 1, negative for slowdown
        # This provides a continuous reward signal without discontinuities
        reward = self.speedup_weight * np.log(speedup) + self.correctness_bonus

        return reward

    def compute_batch_rewards(
        self,
        successes: list,
        runtimes: list,
        error_messages: list
    ) -> list:
        """
        Compute rewards for a batch of results.

        Args:
            successes: List of success flags
            runtimes: List of runtimes (None for failures)
            error_messages: List of error messages

        Returns:
            List of reward values
        """
        rewards = []
        for success, runtime, error in zip(successes, runtimes, error_messages):
            reward = self.compute_reward(success, runtime, error)
            rewards.append(reward)
        return rewards


class AdaptiveRewardFunction(RewardFunction):
    """Adaptive reward function that adjusts based on training progress."""

    def __init__(
        self,
        baseline_runtime: float,
        initial_speedup_weight: float = 1.0,
        **kwargs
    ):
        super().__init__(baseline_runtime, speedup_weight=initial_speedup_weight, **kwargs)
        self.initial_speedup_weight = initial_speedup_weight
        self.best_runtime = baseline_runtime

    def update_best_runtime(self, runtime: float):
        """Update the best runtime seen so far."""
        if runtime < self.best_runtime:
            self.best_runtime = runtime
            # Optionally increase weight as we find better solutions
            improvement_ratio = self.baseline_runtime / self.best_runtime
            if improvement_ratio > 2.0:
                self.speedup_weight = self.initial_speedup_weight * 1.5

    def compute_reward(
        self,
        success: bool,
        runtime: Optional[float],
        error_message: str
    ) -> float:
        """
        Compute reward and update best runtime if improved.

        Args:
            success: Whether compilation and execution succeeded
            runtime: Runtime in microseconds (None if failed)
            error_message: Error message if failed

        Returns:
            Reward value
        """
        reward = super().compute_reward(success, runtime, error_message)

        if success and runtime is not None:
            self.update_best_runtime(runtime)

        return reward
