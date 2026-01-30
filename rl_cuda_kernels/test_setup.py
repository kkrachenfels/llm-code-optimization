#!/usr/bin/env python3
"""
Test script to verify the environment setup and download the dataset.
"""

import os
import sys
import argparse
from pathlib import Path

# Parse args early to set CUDA_VISIBLE_DEVICES before torch import
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--gpus", type=str, default=None, help="GPU device(s) to use")
_parser.add_argument("--test-model", action="store_true", help="Also test model loading")
_args, _ = _parser.parse_known_args()

if _args.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = _args.gpus


def check_imports():
    """Test that all required packages can be imported."""
    print("Checking imports...")
    errors = []

    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("peft", "PEFT (LoRA)"),
        ("trl", "TRL"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
    ]

    for module, name in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            errors.append((name, str(e)))

    # Optional packages
    optional = [
        ("wandb", "Weights & Biases"),
        ("vllm", "vLLM"),
        ("triton", "Triton"),
    ]

    print("\nOptional packages:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} (not installed)")

    return errors


def check_cuda():
    """Test CUDA availability."""
    print("\nChecking CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available")
            print(f"    Device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"    GPU {i}: {props.name} ({props.total_memory // 1024**3} GB)")
            return True
        else:
            print("  ✗ CUDA not available")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def download_dataset():
    """Download the KernelBench dataset."""
    print("\nDownloading KernelBench dataset...")
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            "ScalingIntelligence/KernelBench",
            cache_dir="./data/cache"
        )

        print(f"  ✓ Dataset downloaded successfully")
        print(f"    Splits: {list(dataset.keys())}")
        for split_name, split_data in dataset.items():
            print(f"    {split_name}: {len(split_data)} examples")

        # Show a sample
        print("\n  Sample task (level_1, first example):")
        sample = dataset['level_1'][0]
        print(f"    Name: {sample['name']}")
        print(f"    Problem ID: {sample['problem_id']}")
        print(f"    Code preview: {sample['code'][:200]}...")

        return True
    except Exception as e:
        print(f"  ✗ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test loading a small model (optional)."""
    print("\nTesting model loading (this may take a while)...")
    try:
        from transformers import AutoTokenizer

        # Just test tokenizer loading (faster)
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B-Instruct",
            trust_remote_code=True
        )

        print(f"  ✓ Tokenizer loaded")
        print(f"    Vocab size: {tokenizer.vocab_size}")

        # Test tokenization
        test_text = "def hello_world():\n    print('Hello, World!')"
        tokens = tokenizer(test_text)
        print(f"    Test tokenization: {len(tokens['input_ids'])} tokens")

        return True
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        return False


def test_src_imports():
    """Test that our source modules can be imported."""
    print("\nTesting source module imports...")

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent))

    modules = [
        "src.config",
        "src.dataset",
        "src.kernel_evaluator",
        "src.reward",
        "src.trajectory",
        "src.grpo_trainer",
        "src.model_utils",
    ]

    errors = []
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except Exception as e:
            print(f"  ✗ {module}: {e}")
            errors.append((module, str(e)))

    return errors


def main():
    print("=" * 60)
    print("CUDA Kernel RL Training - Setup Test")
    print("=" * 60)
    if _args.gpus:
        print(f"Using GPU(s): {_args.gpus}")

    all_passed = True

    # Check imports
    import_errors = check_imports()
    if import_errors:
        print(f"\n⚠ {len(import_errors)} required packages failed to import")
        all_passed = False

    # Check CUDA
    cuda_ok = check_cuda()
    if not cuda_ok:
        print("\n⚠ CUDA not available - training will be slow or fail")

    # Test source imports
    src_errors = test_src_imports()
    if src_errors:
        print(f"\n⚠ {len(src_errors)} source modules failed to import")
        all_passed = False

    # Download dataset
    dataset_ok = download_dataset()
    if not dataset_ok:
        print("\n⚠ Dataset download failed")
        all_passed = False

    # Test model loading (optional, slow)
    print("\nSkipping full model loading test (use --test-model to enable)")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Environment is ready for training.")
    else:
        print("⚠ Some tests failed. Please fix the issues above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    if _args.test_model:
        # Will add model loading test
        pass

    sys.exit(main())
