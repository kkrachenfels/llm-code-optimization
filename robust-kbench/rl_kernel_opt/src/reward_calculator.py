"""
Reward calculator using robust_kbench for CUDA kernel evaluation.
"""

import os
import multiprocessing as mp
from multiprocessing import Process, Queue
import traceback

# Use spawn method to avoid CUDA context issues with fork
# This must be set before any CUDA operations
try:
    mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass  # Already set
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from robust_kbench.parallel import ParallelKernelExecutor


def _evaluate_kernel_worker(
    cuda_file: str,
    task_dir: str,
    eval_config: Dict[str, Any],
    result_queue: Queue,
):
    """
    Worker function that runs in a separate process to evaluate a CUDA kernel.

    This isolates compilation/testing from the main training process.
    """
    try:
        # Create executor in this process
        executor = ParallelKernelExecutor(
            task_dir=task_dir,
            op_atol=eval_config["op_atol"],
            op_rtol=eval_config["op_rtol"],
            warmup_time=eval_config["warmup_time"],
            repetition_time=eval_config["repetition_time"],
            eval_type=eval_config["eval_type"],
            multi_init_settings=eval_config["multi_init_settings"],
            multi_input_settings=eval_config["multi_input_settings"],
            timeout=eval_config["timeout"],
            torch_prof=eval_config["torch_prof"],
            ncu_prof=eval_config["ncu_prof"],
            forward=eval_config.get("forward", True),
        )

        raw_results = {
            "compile": None,
            "test": None,
            "eval": None,
            "profile": None,
            "torch": None,
        }

        # Step 1: Compile
        try:
            compile_results = executor.compile([cuda_file])
            raw_results["compile"] = compile_results[0] if compile_results else None
        except Exception as e:
            result_queue.put({
                "success": False,
                "error": f"Compilation exception: {str(e)}",
                "compiled": False,
                "correct": False,
                "raw_results": raw_results,
            })
            return

        if raw_results["compile"] is None or raw_results["compile"].get("error", True):
            error_msg = (
                raw_results["compile"].get("error_msg", "Unknown error")
                if raw_results["compile"]
                else "Compilation returned no results"
            )
            result_queue.put({
                "success": False,
                "error": error_msg,
                "compiled": False,
                "correct": False,
                "raw_results": raw_results,
            })
            return

        # Step 2: Test correctness
        try:
            test_results = executor.test([cuda_file])
            raw_results["test"] = test_results[0] if test_results else None
        except Exception as e:
            result_queue.put({
                "success": False,
                "error": f"Test exception: {str(e)}",
                "compiled": True,
                "correct": False,
                "raw_results": raw_results,
            })
            return

        if raw_results["test"] is None:
            result_queue.put({
                "success": False,
                "error": "Test returned no results",
                "compiled": True,
                "correct": False,
                "raw_results": raw_results,
            })
            return

        test_summary = raw_results["test"].get("summary", {})
        is_correct = test_summary.get("correct", False)
        max_diff = test_summary.get("max_diff")

        if not is_correct:
            result_queue.put({
                "success": True,
                "compiled": True,
                "correct": False,
                "max_diff": max_diff,
                "torch_time_ms": None,
                "cuda_time_ms": None,
                "speedup": None,
                "raw_results": raw_results,
            })
            return

        # Step 3: Evaluate runtime
        try:
            eval_results = executor.evaluate([cuda_file])
            raw_results["eval"] = eval_results[0] if eval_results else None
        except Exception as e:
            # Correct but couldn't measure runtime
            result_queue.put({
                "success": True,
                "compiled": True,
                "correct": True,
                "max_diff": max_diff,
                "torch_time_ms": None,
                "cuda_time_ms": None,
                "speedup": None,
                "raw_results": raw_results,
            })
            return

        # Get torch baseline
        try:
            torch_results = executor.torch_eval(gpu_id=eval_config.get("gpu_id", 0))
            raw_results["torch"] = torch_results
        except Exception:
            torch_results = None

        # Calculate speedup
        torch_time = None
        cuda_time = None
        speedup = None

        if raw_results["eval"] and not raw_results["eval"].get("error", False):
            eval_summary = raw_results["eval"].get("summary", {})
            cuda_time = eval_summary.get("avg_mean_time")

        if torch_results and not torch_results.get("error", False):
            native_results = torch_results.get("native", torch_results)
            if isinstance(native_results, dict):
                torch_summary = native_results.get("summary", {})
                torch_time = torch_summary.get("avg_mean_time")

        if torch_time is not None and cuda_time is not None and cuda_time > 0:
            speedup = torch_time / cuda_time

        result_queue.put({
            "success": True,
            "compiled": True,
            "correct": True,
            "max_diff": max_diff,
            "torch_time_ms": torch_time,
            "cuda_time_ms": cuda_time,
            "speedup": speedup,
            "raw_results": raw_results,
        })

    except Exception as e:
        result_queue.put({
            "success": False,
            "error": f"Worker exception: {str(e)}\n{traceback.format_exc()}",
            "compiled": False,
            "correct": False,
            "raw_results": {},
        })


class RewardMode(Enum):
    """Reward computation modes."""

    SPEED = "speed"  # Only runtime speed
    SPEED_CORRECT = "speed_correct"  # Speed + correctness
    SPEED_CORRECT_PROFILE = "speed_correct_profile"  # Speed + correctness + profiling


@dataclass
class EvaluationResult:
    """Result from evaluating a CUDA kernel."""

    compiled: bool
    compile_error: Optional[str]
    correct: bool
    max_diff: Optional[float]
    torch_time_ms: Optional[float]
    cuda_time_ms: Optional[float]
    speedup: Optional[float]
    profile_info: Optional[Dict[str, Any]]
    reward: float
    raw_results: Dict[str, Any]


class RewardCalculator:
    """
    Calculates rewards for CUDA kernels using robust_kbench.

    Supports three reward modes:
    1. SPEED: reward = speedup_ratio (torch_time / cuda_time)
    2. SPEED_CORRECT: reward = correctness_bonus + speedup_ratio
    3. SPEED_CORRECT_PROFILE: reward = correctness_bonus + speedup_ratio + profile_bonus
    """

    def __init__(
        self,
        mode: RewardMode = RewardMode.SPEED_CORRECT,
        correctness_bonus: float = 0.3,
        max_speedup_reward: float = 5.0,
        compile_fail_penalty: float = -1.0,
        incorrect_penalty: float = 0.0,
        profile_weights: Optional[Dict[str, float]] = None,
        # Evaluation settings
        warmup_time: int = 25,
        repetition_time: int = 10000,
        timeout: int = 300,
        multi_init_settings: bool = True,
        multi_input_settings: bool = True,
        op_atol: float = 1e-5,
        op_rtol: float = 1e-5,
        eval_type: str = "kernelbench",
        torch_prof: bool = False,
        ncu_prof: bool = False,
        # GPU settings
        gpu_id: Optional[int] = None,
        gpu_ids: Optional[List[int]] = None,
    ):
        """
        Initialize the reward calculator.

        Args:
            mode: Reward computation mode
            correctness_bonus: Bonus for correct implementations
            max_speedup_reward: Maximum speedup reward (cap)
            compile_fail_penalty: Penalty for failed compilation
            incorrect_penalty: Penalty for incorrect results
            profile_weights: Weights for profile-based rewards
            warmup_time: Warmup time for benchmarking (ms)
            repetition_time: Total repetition time for benchmarking (ms)
            timeout: Timeout for evaluation (seconds)
            multi_init_settings: Test multiple initialization settings
            multi_input_settings: Test multiple input configurations
            op_atol: Absolute tolerance for correctness
            op_rtol: Relative tolerance for correctness
            eval_type: Evaluation type ("kernelbench", "torch_bench", "triton")
            torch_prof: Enable torch profiling
            ncu_prof: Enable NCU profiling
            gpu_id: Single GPU ID to use (for single-GPU mode)
            gpu_ids: List of GPU IDs for multi-GPU parallel evaluation
        """
        self.mode = mode
        self.correctness_bonus = correctness_bonus
        self.max_speedup_reward = max_speedup_reward
        self.compile_fail_penalty = compile_fail_penalty
        self.incorrect_penalty = incorrect_penalty

        # Default profile weights
        self.profile_weights = profile_weights or {
            "sm_utilization": 0.4,
            "memory_throughput": 0.3,
            "occupancy": 0.3,
            "total_weight": 0.1,
        }

        # Evaluation settings
        self.warmup_time = warmup_time
        self.repetition_time = repetition_time
        self.timeout = timeout
        self.multi_init_settings = multi_init_settings
        self.multi_input_settings = multi_input_settings
        self.op_atol = op_atol
        self.op_rtol = op_rtol
        self.eval_type = eval_type
        self.torch_prof = torch_prof or (mode == RewardMode.SPEED_CORRECT_PROFILE)
        self.ncu_prof = ncu_prof

        # GPU settings
        # If gpu_ids is provided, use those; otherwise use gpu_id or default to [0]
        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        elif gpu_id is not None:
            self.gpu_ids = [gpu_id]
        else:
            self.gpu_ids = [0]

        self._gpu_idx = 0  # Round-robin counter for multi-GPU

        # Cache for torch baseline results
        self._torch_cache: Dict[str, Dict[str, Any]] = {}

    def _get_next_gpu(self) -> int:
        """Get next GPU ID in round-robin fashion."""
        gpu_id = self.gpu_ids[self._gpu_idx % len(self.gpu_ids)]
        self._gpu_idx += 1
        return gpu_id

    def create_executor(self, task_dir: str, gpu_id: Optional[int] = None, forward: bool = True) -> ParallelKernelExecutor:
        """Create a ParallelKernelExecutor for a task.

        Args:
            task_dir: Task directory path
            gpu_id: Specific GPU to use. If None, uses round-robin from gpu_ids.
            forward: Whether this is a forward pass (True) or backward pass (False).
        """
        if gpu_id is None:
            gpu_id = self._get_next_gpu()

        # Set CUDA_VISIBLE_DEVICES for this executor
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        return ParallelKernelExecutor(
            task_dir=task_dir,
            op_atol=self.op_atol,
            op_rtol=self.op_rtol,
            warmup_time=self.warmup_time,
            repetition_time=self.repetition_time,
            eval_type=self.eval_type,
            multi_init_settings=self.multi_init_settings,
            multi_input_settings=self.multi_input_settings,
            timeout=self.timeout,
            torch_prof=self.torch_prof,
            ncu_prof=self.ncu_prof,
            forward=forward,
        )

    def _get_eval_config(self, forward: bool = True) -> Dict[str, Any]:
        """Get evaluation configuration for worker process.

        Args:
            forward: Whether this is a forward pass (True) or backward pass (False).
        """
        return {
            "op_atol": self.op_atol,
            "op_rtol": self.op_rtol,
            "warmup_time": self.warmup_time,
            "repetition_time": self.repetition_time,
            "eval_type": self.eval_type,
            "multi_init_settings": self.multi_init_settings,
            "multi_input_settings": self.multi_input_settings,
            "timeout": self.timeout,
            "torch_prof": self.torch_prof,
            "ncu_prof": self.ncu_prof,
            "gpu_id": self._get_next_gpu(),
            "forward": forward,
        }

    def evaluate_kernel_isolated(
        self,
        cuda_file: str,
        task_dir: str,
        process_timeout: Optional[int] = None,
        forward: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a CUDA kernel in a separate process.

        This isolates the compilation/testing from the main training process,
        preventing crashes or hangs from disrupting training.

        Args:
            cuda_file: Path to the CUDA source file
            task_dir: Task directory
            process_timeout: Timeout in seconds for the entire evaluation process.
                           If None, uses self.timeout + 60 seconds buffer.
            forward: Whether this is a forward pass (True) or backward pass (False).

        Returns:
            EvaluationResult with all evaluation info and reward
        """
        if process_timeout is None:
            process_timeout = self.timeout + 60  # Add buffer for compilation

        # Create queue for results
        result_queue = mp.Queue()

        # Get eval config with forward/backward flag
        eval_config = self._get_eval_config(forward=forward)

        # Start worker process
        process = Process(
            target=_evaluate_kernel_worker,
            args=(cuda_file, task_dir, eval_config, result_queue),
        )
        process.start()

        # Wait for result with timeout
        result = None
        eval_result = None
        try:
            process.join(timeout=process_timeout)

            if process.is_alive():
                # Process timed out
                print(f"[WARNING] Evaluation process timed out after {process_timeout}s, terminating...")
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()

                eval_result = EvaluationResult(
                    compiled=False,
                    compile_error=f"Evaluation timed out after {process_timeout} seconds",
                    correct=False,
                    max_diff=None,
                    torch_time_ms=None,
                    cuda_time_ms=None,
                    speedup=None,
                    profile_info=None,
                    reward=self.compile_fail_penalty,
                    raw_results={},
                )
                return eval_result

            # Get result from queue
            if not result_queue.empty():
                result = result_queue.get_nowait()
            else:
                eval_result = EvaluationResult(
                    compiled=False,
                    compile_error="Worker process completed but returned no result",
                    correct=False,
                    max_diff=None,
                    torch_time_ms=None,
                    cuda_time_ms=None,
                    speedup=None,
                    profile_info=None,
                    reward=self.compile_fail_penalty,
                    raw_results={},
                )
                return eval_result

        except Exception as e:
            # Handle any exceptions during process management
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()

            eval_result = EvaluationResult(
                compiled=False,
                compile_error=f"Process management error: {str(e)}",
                correct=False,
                max_diff=None,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.compile_fail_penalty,
                raw_results={},
            )
            return eval_result
        finally:
            # Clean up the queue to avoid semaphore leaks
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception:
                pass

        # Convert worker result to EvaluationResult
        if not result.get("success", False) and not result.get("compiled", False):
            return EvaluationResult(
                compiled=False,
                compile_error=result.get("error", "Unknown error"),
                correct=False,
                max_diff=None,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.compile_fail_penalty,
                raw_results=result.get("raw_results", {}),
            )

        if not result.get("correct", False):
            return EvaluationResult(
                compiled=result.get("compiled", False),
                compile_error=result.get("error"),
                correct=False,
                max_diff=result.get("max_diff"),
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.incorrect_penalty,
                raw_results=result.get("raw_results", {}),
            )

        # Calculate reward for correct result
        reward = self._calculate_reward(
            is_correct=True,
            speedup=result.get("speedup"),
            profile_info=None,
        )

        return EvaluationResult(
            compiled=True,
            compile_error=None,
            correct=True,
            max_diff=result.get("max_diff"),
            torch_time_ms=result.get("torch_time_ms"),
            cuda_time_ms=result.get("cuda_time_ms"),
            speedup=result.get("speedup"),
            profile_info=None,
            reward=reward,
            raw_results=result.get("raw_results", {}),
        )

    def get_torch_baseline(
        self,
        task_dir: str,
        executor: Optional[ParallelKernelExecutor] = None,
        gpu_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get PyTorch baseline timing for a task.

        Results are cached to avoid redundant evaluation.
        """
        if task_dir in self._torch_cache:
            return self._torch_cache[task_dir]

        if executor is None:
            executor = self.create_executor(task_dir, gpu_id=gpu_id)

        try:
            torch_results = executor.torch_eval(gpu_id=gpu_id or self.gpu_ids[0])
            self._torch_cache[task_dir] = torch_results
            return torch_results
        except Exception as e:
            print(f"Warning: Failed to get torch baseline for {task_dir}: {e}")
            return {"error": True, "error_msg": str(e)}

    def evaluate_kernel(
        self,
        cuda_file: str,
        task_dir: str,
        executor: Optional[ParallelKernelExecutor] = None,
        forward: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a single CUDA kernel and compute its reward.

        Args:
            cuda_file: Path to the CUDA source file
            task_dir: Task directory
            executor: Optional pre-created executor
            forward: Whether this is a forward pass (True) or backward pass (False).

        Returns:
            EvaluationResult with all evaluation info and reward
        """
        if executor is None:
            executor = self.create_executor(task_dir, forward=forward)

        raw_results = {
            "compile": None,
            "test": None,
            "eval": None,
            "profile": None,
            "torch": None,
        }

        # Step 1: Compile
        try:
            compile_results = executor.compile([cuda_file])
            raw_results["compile"] = compile_results[0] if compile_results else None
        except Exception as e:
            return EvaluationResult(
                compiled=False,
                compile_error=str(e),
                correct=False,
                max_diff=None,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.compile_fail_penalty,
                raw_results=raw_results,
            )

        if (
            raw_results["compile"] is None
            or raw_results["compile"].get("error", True)
        ):
            error_msg = (
                raw_results["compile"].get("error_msg", "Unknown error")
                if raw_results["compile"]
                else "Compilation returned no results"
            )
            return EvaluationResult(
                compiled=False,
                compile_error=error_msg,
                correct=False,
                max_diff=None,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.compile_fail_penalty,
                raw_results=raw_results,
            )

        # Step 2: Test correctness
        try:
            test_results = executor.test([cuda_file])
            raw_results["test"] = test_results[0] if test_results else None
        except Exception as e:
            return EvaluationResult(
                compiled=True,
                compile_error=None,
                correct=False,
                max_diff=None,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.incorrect_penalty,
                raw_results=raw_results,
            )

        if raw_results["test"] is None:
            return EvaluationResult(
                compiled=True,
                compile_error=None,
                correct=False,
                max_diff=None,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.incorrect_penalty,
                raw_results=raw_results,
            )

        test_summary = raw_results["test"].get("summary", {})
        is_correct = test_summary.get("correct", False)
        max_diff = test_summary.get("max_diff")

        if not is_correct:
            return EvaluationResult(
                compiled=True,
                compile_error=None,
                correct=False,
                max_diff=max_diff,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.incorrect_penalty,
                raw_results=raw_results,
            )

        # Step 3: Evaluate runtime
        try:
            eval_results = executor.evaluate([cuda_file])
            raw_results["eval"] = eval_results[0] if eval_results else None
        except Exception as e:
            # Correct but couldn't measure runtime - give partial reward
            return EvaluationResult(
                compiled=True,
                compile_error=None,
                correct=True,
                max_diff=max_diff,
                torch_time_ms=None,
                cuda_time_ms=None,
                speedup=None,
                profile_info=None,
                reward=self.correctness_bonus,
                raw_results=raw_results,
            )

        # Get torch baseline
        torch_results = self.get_torch_baseline(task_dir, executor)
        raw_results["torch"] = torch_results

        # Calculate speedup
        torch_time = None
        cuda_time = None
        speedup = None

        if raw_results["eval"] and not raw_results["eval"].get("error", False):
            eval_summary = raw_results["eval"].get("summary", {})
            cuda_time = eval_summary.get("avg_mean_time")

        if torch_results and not torch_results.get("error", False):
            # Use native torch results (not torch.compile)
            native_results = torch_results.get("native", torch_results)
            if isinstance(native_results, dict):
                torch_summary = native_results.get("summary", {})
                torch_time = torch_summary.get("avg_mean_time")

        if torch_time is not None and cuda_time is not None and cuda_time > 0:
            speedup = torch_time / cuda_time

        # Step 4: Profile (if enabled)
        profile_info = None
        if self.mode == RewardMode.SPEED_CORRECT_PROFILE:
            try:
                profile_results = executor.profile([cuda_file])
                raw_results["profile"] = (
                    profile_results[0] if profile_results else None
                )
                profile_info = self._extract_profile_info(raw_results["profile"])
            except Exception as e:
                print(f"Warning: Profiling failed: {e}")

        # Calculate reward
        reward = self._calculate_reward(
            is_correct=True,
            speedup=speedup,
            profile_info=profile_info,
        )

        return EvaluationResult(
            compiled=True,
            compile_error=None,
            correct=True,
            max_diff=max_diff,
            torch_time_ms=torch_time,
            cuda_time_ms=cuda_time,
            speedup=speedup,
            profile_info=profile_info,
            reward=reward,
            raw_results=raw_results,
        )

    def evaluate_batch(
        self,
        cuda_files: List[str],
        task_dir: str,
        forward: bool = True,
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of CUDA kernels for the same task.

        More efficient than evaluating one by one as it reuses the executor.
        """
        executor = self.create_executor(task_dir, forward=forward)

        # Pre-fetch torch baseline
        self.get_torch_baseline(task_dir, executor)

        results = []
        for cuda_file in cuda_files:
            result = self.evaluate_kernel(cuda_file, task_dir, executor)
            results.append(result)

        # Cleanup
        try:
            executor.cleanup_extension_dirs()
        except Exception:
            pass

        return results

    def _calculate_reward(
        self,
        is_correct: bool,
        speedup: Optional[float],
        profile_info: Optional[Dict[str, Any]],
    ) -> float:
        """Calculate reward based on mode and results."""
        if not is_correct:
            return self.incorrect_penalty

        reward = 0.0

        # Add correctness bonus for modes that include it
        if self.mode in [RewardMode.SPEED_CORRECT, RewardMode.SPEED_CORRECT_PROFILE]:
            reward += self.correctness_bonus

        # Add speedup reward
        if speedup is not None:
            # Cap the speedup to avoid reward hacking
            capped_speedup = min(speedup, self.max_speedup_reward)
            reward += capped_speedup

        # Add profile bonus for profile mode
        if self.mode == RewardMode.SPEED_CORRECT_PROFILE and profile_info:
            profile_reward = self._calculate_profile_reward(profile_info)
            reward += profile_reward * self.profile_weights["total_weight"]

        return reward

    def _extract_profile_info(
        self, profile_results: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Extract relevant profiling information."""
        if profile_results is None:
            return None

        info = {}

        # Extract from torch profiler
        torch_prof = profile_results.get("torch")
        if torch_prof and isinstance(torch_prof, dict):
            # Find the main CUDA kernel event
            for event_name, event_data in torch_prof.items():
                if "cuda" in event_name.lower() or "kernel" in event_name.lower():
                    if isinstance(event_data, dict):
                        info["device_time_us"] = event_data.get("device_time_total", 0)
                        info["device_memory_mb"] = (
                            event_data.get("device_memory_usage", 0) / 1e6
                        )
                    break

        # Extract from NCU profiler
        ncu_prof = profile_results.get("ncu")
        if ncu_prof and isinstance(ncu_prof, dict):
            metrics = ncu_prof.get("metrics", {})

            # SM utilization
            sm_metric = metrics.get(
                "sm__throughput.avg.pct_of_peak_sustained_elapsed", {}
            )
            if sm_metric:
                info["sm_utilization"] = sm_metric.get("avg_value", 0)

            # Memory throughput
            mem_metric = metrics.get(
                "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed", {}
            )
            if mem_metric:
                info["memory_throughput"] = mem_metric.get("avg_value", 0)

            # Occupancy
            occ_metric = metrics.get("launch__occupancy_achieved", {})
            if occ_metric:
                info["occupancy"] = occ_metric.get("avg_value", 0)

        return info if info else None

    def _calculate_profile_reward(self, profile_info: Dict[str, Any]) -> float:
        """Calculate reward from profiling information."""
        reward = 0.0

        # SM utilization (0-100%)
        sm_util = profile_info.get("sm_utilization", 0)
        reward += (sm_util / 100.0) * self.profile_weights["sm_utilization"]

        # Memory throughput (0-100%)
        mem_tp = profile_info.get("memory_throughput", 0)
        reward += (mem_tp / 100.0) * self.profile_weights["memory_throughput"]

        # Occupancy (0-1)
        occupancy = profile_info.get("occupancy", 0)
        reward += occupancy * self.profile_weights["occupancy"]

        return reward


def compute_discounted_rewards(
    turn_rewards: List[float], gamma: float = 0.4
) -> float:
    """
    Compute discounted sum of rewards across turns.

    Following Kevin-32B: total_reward = r₁ + γ*r₂ + γ²*r₃ + ...
    """
    total = 0.0
    discount = 1.0
    for reward in turn_rewards:
        total += discount * reward
        discount *= gamma
    return total
