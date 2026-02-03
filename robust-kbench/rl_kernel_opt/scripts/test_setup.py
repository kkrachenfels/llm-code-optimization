#!/usr/bin/env python3
"""
Test script to verify the RL kernel optimization setup.

Usage:
    python -m rl_kernel_opt.scripts.test_setup
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from rl_kernel_opt.src.task_sampler import TaskSampler
        print("  ✓ TaskSampler")
    except ImportError as e:
        print(f"  ✗ TaskSampler: {e}")
        return False

    try:
        from rl_kernel_opt.src.prompt_builder import PromptBuilder
        print("  ✓ PromptBuilder")
    except ImportError as e:
        print(f"  ✗ PromptBuilder: {e}")
        return False

    try:
        from rl_kernel_opt.src.code_extractor import CUDACodeExtractor
        print("  ✓ CUDACodeExtractor")
    except ImportError as e:
        print(f"  ✗ CUDACodeExtractor: {e}")
        return False

    try:
        from rl_kernel_opt.src.reward_calculator import RewardCalculator, RewardMode
        print("  ✓ RewardCalculator")
    except ImportError as e:
        print(f"  ✗ RewardCalculator: {e}")
        return False

    try:
        from rl_kernel_opt.src.grpo_trainer import GRPOTrainer, GRPOConfig
        print("  ✓ GRPOTrainer")
    except ImportError as e:
        print(f"  ✗ GRPOTrainer: {e}")
        return False

    return True


def test_robust_kbench():
    """Test that robust_kbench is available."""
    print("\nTesting robust_kbench...")

    try:
        from robust_kbench.parallel import ParallelKernelExecutor
        print("  ✓ ParallelKernelExecutor")
    except ImportError as e:
        print(f"  ✗ ParallelKernelExecutor: {e}")
        print("    Make sure robust_kbench is installed: pip install -e .")
        return False

    return True


def test_torch():
    """Test PyTorch and CUDA availability."""
    print("\nTesting PyTorch...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch: {e}")
        return False

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.device_count()} device(s)")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("  ⚠ CUDA not available (training requires GPU)")

    return True


def test_transformers():
    """Test transformers library."""
    print("\nTesting transformers...")

    try:
        import transformers
        print(f"  ✓ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers: {e}")
        return False

    return True


def test_task_loading():
    """Test loading a sample task."""
    print("\nTesting task loading...")

    from rl_kernel_opt.src.task_sampler import TaskSampler

    # Try to load tasks
    task_dirs = [
        "tasks/layernorm",
        "tasks/mnist_cross_entropy",
        "tasks/mnist_linear",
    ]

    available_dirs = [d for d in task_dirs if Path(d).exists()]

    if not available_dirs:
        print(f"  ⚠ No task directories found: {task_dirs}")
        return True  # Not a failure, just no tasks

    sampler = TaskSampler(available_dirs, forward=True)

    if len(sampler) > 0:
        print(f"  ✓ Loaded {len(sampler)} tasks")
        for task in sampler:
            print(f"    - {task.name}")
    else:
        print("  ⚠ No tasks could be loaded")

    return True


def test_code_extraction():
    """Test CUDA code extraction."""
    print("\nTesting code extraction...")

    from rl_kernel_opt.src.code_extractor import CUDACodeExtractor

    extractor = CUDACodeExtractor()

    # Test case with valid CUDA code
    test_output = '''
Here's a CUDA kernel for the operation:

```cuda
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void my_kernel(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = input[idx] * 2.0f;
    }
}

torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    my_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward");
}
```

This kernel multiplies each element by 2.
'''

    code, error = extractor.extract(test_output)

    if code is not None:
        print("  ✓ Code extraction works")
        print(f"    Extracted {len(code)} characters")
    else:
        print(f"  ✗ Code extraction failed: {error}")
        return False

    extractor.cleanup()
    return True


def test_prompt_building():
    """Test prompt building."""
    print("\nTesting prompt building...")

    from rl_kernel_opt.src.task_sampler import TaskSampler, KernelTask
    from rl_kernel_opt.src.prompt_builder import PromptBuilder

    # Create a mock task
    task = KernelTask(
        name="test_task",
        task_dir="tasks/test",
        pytorch_code="def forward_fn(x): return x * 2",
        input_specs={"input_names": ["x"]},
        output_spec="Returns x * 2",
        docstring="Multiplies input by 2",
        forward=True,
        config={},
    )

    builder = PromptBuilder()
    messages = builder.build_initial_prompt(task)

    if len(messages) >= 2:
        print("  ✓ Prompt building works")
        print(f"    Generated {len(messages)} messages")
    else:
        print("  ✗ Prompt building failed")
        return False

    return True


def main():
    print("=" * 60)
    print("RL Kernel Optimization Setup Test")
    print("=" * 60)

    all_passed = True

    all_passed &= test_imports()
    all_passed &= test_robust_kbench()
    all_passed &= test_torch()
    all_passed &= test_transformers()
    all_passed &= test_task_loading()
    all_passed &= test_code_extraction()
    all_passed &= test_prompt_building()

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Setup is ready.")
    else:
        print("Some tests failed. Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
