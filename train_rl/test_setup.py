#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly.
"""

import sys
from pathlib import Path

print("=" * 80)
print("Testing RL Code Optimization Setup")
print("=" * 80)

# Test 1: Check Python modules
print("\n[1/4] Checking Python dependencies...")
try:
    import torch
    import transformers
    import trl
    import numpy as np
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ Transformers: {transformers.__version__}")
    print(f"✓ TRL: {trl.__version__}")
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Test 2: Check C++ compiler
print("\n[2/4] Checking C++ compiler...")
import subprocess
try:
    result = subprocess.run(
        ["g++", "--version"],
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        version_line = result.stdout.split('\n')[0]
        print(f"✓ {version_line}")
    else:
        print("✗ g++ not working properly")
        sys.exit(1)
except FileNotFoundError:
    print("✗ g++ not found. Please install a C++ compiler.")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error checking g++: {e}")
    sys.exit(1)

# Test 3: Test compilation and execution
print("\n[3/4] Testing C++ compilation and execution...")
try:
    from src.compiler import CppCompiler, get_baseline_runtime

    program_path = "programs/bubble_sort.cpp"
    if not Path(program_path).exists():
        print(f"✗ Program not found: {program_path}")
        sys.exit(1)

    compiler = CppCompiler()
    baseline_runtime = get_baseline_runtime(program_path, compiler, num_runs=3)
    print(f"✓ Baseline runtime: {baseline_runtime:.2f} microseconds")

except Exception as e:
    print(f"✗ Compilation/execution test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test reward function
print("\n[4/4] Testing reward function...")
try:
    from src.reward import RewardFunction, AdaptiveRewardFunction

    reward_fn = RewardFunction(baseline_runtime=baseline_runtime)

    # Test successful optimization (2x speedup)
    reward_speedup = reward_fn.compute_reward(True, baseline_runtime / 2.0, "")
    print(f"✓ Reward for 2x speedup: {reward_speedup:.3f}")

    # Test similar performance
    reward_similar = reward_fn.compute_reward(True, baseline_runtime, "")
    print(f"✓ Reward for similar performance: {reward_similar:.3f}")

    # Test slowdown
    reward_slowdown = reward_fn.compute_reward(True, baseline_runtime * 1.5, "")
    print(f"✓ Reward for 1.5x slowdown: {reward_slowdown:.3f}")

    # Test compilation failure
    reward_fail = reward_fn.compute_reward(False, None, "compilation error")
    print(f"✓ Reward for compilation failure: {reward_fail:.3f}")

except Exception as e:
    print(f"✗ Reward function test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# All tests passed
print("\n" + "=" * 80)
print("✓ All tests passed! Setup is ready.")
print("=" * 80)
print("\nNext steps:")
print("  1. Choose a model (e.g., Salesforce/codegen-350M-mono)")
print("  2. Run training:")
print("     python train.py --model Salesforce/codegen-350M-mono --steps 50")
print("\nFor more options, see: python train.py --help")
print("=" * 80)
