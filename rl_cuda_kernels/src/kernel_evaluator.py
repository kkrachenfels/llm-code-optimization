"""
Sandboxed CUDA kernel evaluation.
Implements compilation, correctness checking, and performance profiling.
"""

import os
import sys
import time
import tempfile
import subprocess
import traceback
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import importlib.util
import signal
import resource

import torch
import torch.nn as nn
import numpy as np

from .config import EvaluationConfig


@dataclass
class EvaluationResult:
    """Result of kernel evaluation."""
    # Status
    success: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Correctness
    is_correct: bool = False
    max_abs_error: float = float('inf')
    max_rel_error: float = float('inf')

    # Performance
    reference_time_ms: float = 0.0
    kernel_time_ms: float = 0.0
    speedup: float = 0.0

    # Detailed feedback
    compilation_success: bool = False
    runtime_success: bool = False
    feedback: str = ""


def set_memory_limit(max_bytes: int):
    """Set memory limit for the process."""
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, hard))


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Kernel execution timed out")


class KernelSandbox:
    """
    Sandboxed environment for kernel execution.
    Prevents crashes from illegal memory access from affecting the main process.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.temp_dir = Path(config.temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _create_kernel_module(
        self,
        kernel_code: str,
        module_name: str
    ) -> Path:
        """Write kernel code to a temporary module file."""
        module_path = self.temp_dir / f"{module_name}.py"

        # Add necessary imports
        full_code = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

# Try to import triton if available
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Try to import CUDA extension utilities
try:
    from torch.utils.cpp_extension import load_inline
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False

{kernel_code}
'''
        with open(module_path, 'w') as f:
            f.write(full_code)

        return module_path

    def _load_module(self, module_path: Path):
        """Dynamically load a Python module from file."""
        spec = importlib.util.spec_from_file_location(
            module_path.stem,
            module_path
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_path.stem] = module
        spec.loader.exec_module(module)
        return module

    def _run_in_subprocess(
        self,
        func,
        args: tuple,
        timeout: float
    ) -> Any:
        """Run a function in a subprocess with timeout and memory limits."""
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()

        def worker(queue, fn, arguments):
            try:
                # Set memory limit
                max_mem = self.config.max_memory_mb * 1024 * 1024
                set_memory_limit(max_mem)

                result = fn(*arguments)
                queue.put(('success', result))
            except Exception as e:
                queue.put(('error', (type(e).__name__, str(e), traceback.format_exc())))

        process = ctx.Process(target=worker, args=(result_queue, func, args))
        process.start()
        process.join(timeout=timeout)

        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
            raise TimeoutError(f"Process timed out after {timeout}s")

        if result_queue.empty():
            raise RuntimeError("Process terminated without result (likely segfault)")

        status, result = result_queue.get()
        if status == 'error':
            error_type, error_msg, tb = result
            raise RuntimeError(f"{error_type}: {error_msg}\n{tb}")

        return result


def evaluate_kernel_worker(
    kernel_code: str,
    reference_code: str,
    config_dict: Dict,
    device: str = "cuda:0"
) -> Dict:
    """
    Worker function to evaluate a kernel in isolation.
    This runs in a subprocess for safety.
    """
    import torch
    import torch.nn as nn
    import time
    import tempfile
    import importlib.util
    import sys
    from pathlib import Path

    result = {
        'success': False,
        'compilation_success': False,
        'runtime_success': False,
        'is_correct': False,
        'error_type': None,
        'error_message': None,
        'reference_time_ms': 0.0,
        'kernel_time_ms': 0.0,
        'speedup': 0.0,
        'max_abs_error': float('inf'),
        'max_rel_error': float('inf'),
        'feedback': ''
    }

    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Load reference module
        ref_path = temp_dir / "reference.py"
        ref_code_full = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
{reference_code}
'''
        with open(ref_path, 'w') as f:
            f.write(ref_code_full)

        spec = importlib.util.spec_from_file_location("reference", ref_path)
        ref_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_module)

        # Load kernel module
        kernel_path = temp_dir / "kernel.py"
        kernel_code_full = f'''
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import triton
    import triton.language as tl
except ImportError:
    pass
{kernel_code}
'''
        with open(kernel_path, 'w') as f:
            f.write(kernel_code_full)

        spec = importlib.util.spec_from_file_location("kernel", kernel_path)
        kernel_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(kernel_module)

        result['compilation_success'] = True

        # Get inputs
        inputs = ref_module.get_inputs()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]

        init_inputs = ref_module.get_init_inputs()

        # Create reference model
        ref_model = ref_module.Model(*init_inputs).to(device).eval()

        # Create kernel model
        kernel_model = kernel_module.Model(*init_inputs).to(device).eval()

        result['runtime_success'] = True

        # Warmup and correctness check
        with torch.no_grad():
            ref_output = ref_model(*inputs)
            kernel_output = kernel_model(*inputs)

        # Check correctness
        if isinstance(ref_output, tuple):
            ref_output = ref_output[0]
        if isinstance(kernel_output, tuple):
            kernel_output = kernel_output[0]

        abs_error = torch.abs(ref_output - kernel_output)
        rel_error = abs_error / (torch.abs(ref_output) + 1e-8)

        result['max_abs_error'] = abs_error.max().item()
        result['max_rel_error'] = rel_error.max().item()

        rtol = config_dict.get('rtol', 1e-3)
        atol = config_dict.get('atol', 1e-5)

        is_correct = torch.allclose(ref_output, kernel_output, rtol=rtol, atol=atol)
        result['is_correct'] = is_correct

        if not is_correct:
            result['feedback'] = (
                f"Correctness check failed. "
                f"Max absolute error: {result['max_abs_error']:.6e}, "
                f"Max relative error: {result['max_rel_error']:.6e}. "
                f"Required: rtol={rtol}, atol={atol}"
            )
            result['success'] = False
            return result

        # Performance benchmarking
        warmup_iters = config_dict.get('warmup_iterations', 5)
        bench_iters = config_dict.get('benchmark_iterations', 20)

        # Warmup reference
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = ref_model(*inputs)
        torch.cuda.synchronize()

        # Benchmark reference
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                _ = ref_model(*inputs)
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / bench_iters * 1000  # ms

        # Warmup kernel
        with torch.no_grad():
            for _ in range(warmup_iters):
                _ = kernel_model(*inputs)
        torch.cuda.synchronize()

        # Benchmark kernel
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(bench_iters):
                _ = kernel_model(*inputs)
        torch.cuda.synchronize()
        kernel_time = (time.perf_counter() - start) / bench_iters * 1000  # ms

        result['reference_time_ms'] = ref_time
        result['kernel_time_ms'] = kernel_time
        result['speedup'] = ref_time / max(kernel_time, 1e-6)

        result['feedback'] = (
            f"Kernel is correct! "
            f"Reference time: {ref_time:.3f}ms, "
            f"Kernel time: {kernel_time:.3f}ms, "
            f"Speedup: {result['speedup']:.2f}x"
        )
        result['success'] = True

    except SyntaxError as e:
        result['error_type'] = 'SyntaxError'
        result['error_message'] = str(e)
        result['feedback'] = f"Compilation failed with syntax error: {e}"

    except ImportError as e:
        result['error_type'] = 'ImportError'
        result['error_message'] = str(e)
        result['feedback'] = f"Import error (missing dependency?): {e}"

    except AttributeError as e:
        result['error_type'] = 'AttributeError'
        result['error_message'] = str(e)
        result['feedback'] = f"Missing required attribute (Model class or functions?): {e}"

    except RuntimeError as e:
        if 'CUDA' in str(e):
            result['error_type'] = 'CUDAError'
            result['feedback'] = f"CUDA runtime error: {e}"
        else:
            result['error_type'] = 'RuntimeError'
            result['feedback'] = f"Runtime error: {e}"
        result['error_message'] = str(e)

    except Exception as e:
        result['error_type'] = type(e).__name__
        result['error_message'] = str(e)
        result['feedback'] = f"Unexpected error ({type(e).__name__}): {e}"

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return result


class KernelEvaluator:
    """
    Main evaluator class for CUDA kernels.
    Uses subprocess-based sandboxing for safety.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config

        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible_devices

    def evaluate(
        self,
        kernel_code: str,
        reference_code: str,
        use_subprocess: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a kernel against the reference implementation.

        Args:
            kernel_code: The generated CUDA kernel code
            reference_code: The reference PyTorch implementation
            use_subprocess: Whether to use subprocess isolation (safer but slower)

        Returns:
            EvaluationResult with correctness and performance metrics
        """
        config_dict = {
            'rtol': self.config.rtol,
            'atol': self.config.atol,
            'warmup_iterations': self.config.warmup_iterations,
            'benchmark_iterations': self.config.benchmark_iterations
        }

        if use_subprocess:
            try:
                ctx = mp.get_context('spawn')
                result_queue = ctx.Queue()

                def worker(queue):
                    try:
                        result = evaluate_kernel_worker(
                            kernel_code,
                            reference_code,
                            config_dict,
                            device="cuda:0"
                        )
                        queue.put(('success', result))
                    except Exception as e:
                        queue.put(('error', {
                            'error_type': type(e).__name__,
                            'error_message': str(e),
                            'feedback': f"Subprocess error: {e}"
                        }))

                process = ctx.Process(target=worker, args=(result_queue,))
                process.start()
                process.join(timeout=self.config.timeout_seconds)

                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()

                    return EvaluationResult(
                        success=False,
                        error_type='Timeout',
                        error_message=f'Execution timed out after {self.config.timeout_seconds}s',
                        feedback=f'Kernel execution timed out after {self.config.timeout_seconds} seconds. '
                                 'Consider optimizing or simplifying your implementation.'
                    )

                if result_queue.empty():
                    return EvaluationResult(
                        success=False,
                        error_type='Crash',
                        error_message='Process crashed (likely segfault or illegal memory access)',
                        feedback='The kernel caused a crash, possibly due to illegal memory access. '
                                 'Check array bounds and memory allocation.'
                    )

                status, result_dict = result_queue.get()

                if status == 'error':
                    return EvaluationResult(
                        success=False,
                        error_type=result_dict.get('error_type', 'Unknown'),
                        error_message=result_dict.get('error_message', ''),
                        feedback=result_dict.get('feedback', 'Unknown error occurred')
                    )

                return EvaluationResult(**result_dict)

            except Exception as e:
                return EvaluationResult(
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    feedback=f"Evaluation failed: {e}"
                )

        else:
            # Direct evaluation (unsafe, for debugging only)
            try:
                result_dict = evaluate_kernel_worker(
                    kernel_code,
                    reference_code,
                    config_dict,
                    device="cuda:0"
                )
                return EvaluationResult(**result_dict)
            except Exception as e:
                return EvaluationResult(
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                    feedback=f"Evaluation failed: {e}"
                )

    def evaluate_batch(
        self,
        kernel_codes: List[str],
        reference_code: str,
        num_workers: int = 4
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple kernels in parallel.
        """
        results = []

        # Use process pool for parallel evaluation
        with mp.Pool(processes=num_workers) as pool:
            config_dict = {
                'rtol': self.config.rtol,
                'atol': self.config.atol,
                'warmup_iterations': self.config.warmup_iterations,
                'benchmark_iterations': self.config.benchmark_iterations
            }

            async_results = []
            for kernel_code in kernel_codes:
                ar = pool.apply_async(
                    evaluate_kernel_worker,
                    (kernel_code, reference_code, config_dict, "cuda:0")
                )
                async_results.append(ar)

            for ar in async_results:
                try:
                    result_dict = ar.get(timeout=self.config.timeout_seconds)
                    results.append(EvaluationResult(**result_dict))
                except mp.TimeoutError:
                    results.append(EvaluationResult(
                        success=False,
                        error_type='Timeout',
                        feedback='Evaluation timed out'
                    ))
                except Exception as e:
                    results.append(EvaluationResult(
                        success=False,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        feedback=f"Evaluation failed: {e}"
                    ))

        return results


def extract_kernel_code(response: str) -> Optional[str]:
    """
    Extract kernel code from a model response.
    Looks for code blocks and extracts the implementation.
    """
    import re

    # Try to find Python code blocks
    code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        # Return the longest code block (likely the main implementation)
        return max(code_blocks, key=len)

    # Try generic code blocks
    code_blocks = re.findall(r'```\n(.*?)```', response, re.DOTALL)
    if code_blocks:
        return max(code_blocks, key=len)

    # If no code blocks, try to extract based on class definition
    if 'class Model' in response:
        # Find the start of the Model class and extract until the end
        start = response.find('class Model')
        if start != -1:
            return response[start:]

    return None


if __name__ == "__main__":
    # Test evaluation
    config = EvaluationConfig()
    evaluator = KernelEvaluator(config)

    reference_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
'''

    kernel_code = '''
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))

def get_inputs():
    return [torch.randn(1024, 1024, device='cuda')]

def get_init_inputs():
    return []
'''

    result = evaluator.evaluate(kernel_code, reference_code)
    print(f"Success: {result.success}")
    print(f"Correct: {result.is_correct}")
    print(f"Speedup: {result.speedup:.2f}x")
    print(f"Feedback: {result.feedback}")
