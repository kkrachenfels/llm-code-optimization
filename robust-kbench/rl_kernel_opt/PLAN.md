# RL-Based CUDA Kernel Optimization with robust_kbench Rewards

## Overview

This project implements reinforcement learning for CUDA kernel optimization using a Qwen2.5-Coder-3B-Instruct model with rewards derived from robust_kbench evaluation tools. Inspired by Cognition's Kevin-32B approach, we use GRPO (Group Relative Policy Optimization) for training.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Loop                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐ │
│  │   Task       │     │  Qwen Model  │     │  robust_kbench       │ │
│  │   Sampler    │────▶│  (Generator) │────▶│  Evaluator           │ │
│  └──────────────┘     └──────────────┘     └──────────────────────┘ │
│         │                    │                       │               │
│         │                    │                       ▼               │
│         │                    │              ┌──────────────────────┐ │
│         │                    │              │  Reward Calculator   │ │
│         │                    │              │  - Compile           │ │
│         │                    │              │  - Correctness       │ │
│         │                    │              │  - Runtime Speedup   │ │
│         │                    │              │  - Profiling Metrics │ │
│         │                    │              └──────────────────────┘ │
│         │                    │                       │               │
│         │                    ▼                       ▼               │
│         │            ┌──────────────────────────────────┐           │
│         │            │       GRPO Trainer               │           │
│         │            │  (Group Relative Policy Opt)     │           │
│         │            └──────────────────────────────────┘           │
│         │                           │                                │
│         └───────────────────────────┘                                │
│                    (next iteration)                                  │
└─────────────────────────────────────────────────────────────────────┘
```

## Reward Modes

### Mode 1: Runtime Speed Only
```python
reward = speedup_ratio  # torch_time / cuda_time
# If doesn't compile: reward = -1.0
# If compiles but crashes: reward = 0.0
```

### Mode 2: Runtime Speed + Correctness
```python
if not compiles:
    reward = -1.0
elif not correct:
    reward = 0.0
else:
    reward = correctness_bonus (0.3) + speedup_ratio
```

### Mode 3: Runtime Speed + Correctness + Profiling
```python
if not compiles:
    reward = -1.0
elif not correct:
    reward = 0.0
else:
    reward = (
        correctness_bonus (0.3) +
        speedup_ratio +
        memory_efficiency_bonus +
        occupancy_bonus
    )
```

## Key Components

### 1. Task Sampler (`task_sampler.py`)
- Loads tasks from robust_kbench task directories
- Extracts PyTorch reference implementations
- Provides task descriptions and constraints

### 2. Prompt Builder (`prompt_builder.py`)
- Constructs prompts with:
  - Task description
  - PyTorch reference code
  - Input/output specifications
  - Previous attempt feedback (for multi-turn)

### 3. CUDA Code Extractor (`code_extractor.py`)
- Parses model outputs to extract CUDA code
- Validates basic syntax
- Handles code block formatting

### 4. Reward Calculator (`reward_calculator.py`)
- Uses `ParallelKernelExecutor` for evaluation
- Computes rewards based on selected mode
- Handles error cases gracefully

### 5. GRPO Trainer (`trainer.py`)
- Implements Group Relative Policy Optimization
- Manages batched sampling and gradient updates
- Handles multi-turn refinement

## Directory Structure

```
rl_kernel_opt/
├── PLAN.md                    # This file
├── configs/
│   └── default.yaml           # Training configuration
├── src/
│   ├── __init__.py
│   ├── task_sampler.py        # Task loading and sampling
│   ├── prompt_builder.py      # Prompt construction
│   ├── code_extractor.py      # Extract CUDA from outputs
│   ├── reward_calculator.py   # Compute rewards via kbench
│   ├── grpo_trainer.py        # GRPO implementation
│   └── utils.py               # Shared utilities
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   └── generate.py            # Single-sample generation
└── outputs/                   # Training outputs
```

## Training Configuration

```yaml
# Model settings
model:
  name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  max_new_tokens: 4096
  temperature: 0.7

# GRPO settings
grpo:
  group_size: 8          # Responses per prompt
  kl_coef: 0.05          # KL divergence coefficient
  clip_ratio: 0.2        # PPO-style clipping
  gamma: 0.4             # Discount factor for multi-turn

# Reward settings
reward:
  mode: "speed_correct_profile"  # Options: speed, speed_correct, speed_correct_profile
  correctness_bonus: 0.3
  max_speedup_reward: 5.0        # Cap speedup to avoid outliers
  profile_weight: 0.1

# Training settings
training:
  tasks_per_batch: 4
  trajectories_per_task: 8
  max_turns: 3           # Multi-turn refinement
  gradient_accumulation: 2
  learning_rate: 1e-6
  warmup_steps: 100
  max_steps: 5000
  save_interval: 500
  eval_interval: 100

# Evaluation settings
evaluation:
  warmup_time: 25
  repetition_time: 10000
  timeout: 300
  multi_init_settings: true
  multi_input_settings: true
```

## Multi-Turn Refinement

Following Kevin-32B, we use multi-turn refinement where the model can iterate on its outputs:

```
Turn 1: Initial generation
        ↓ Feedback: compile error / correctness / speedup
Turn 2: Refined generation based on feedback
        ↓ Feedback: ...
Turn 3: Final refinement
```

Each turn contributes to the total reward with discount factor γ=0.4:
```
total_reward = r₁ + γ*r₂ + γ²*r₃
```

## Profiling Metrics for Rewards

When using Mode 3 (speed_correct_profile), we extract these metrics:

### From Torch Profiler:
- `self_device_time_total`: Time spent in CUDA kernels
- `device_memory_usage`: Peak GPU memory

### From NCU (if available):
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`: SM utilization
- `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed`: Memory throughput
- `launch__occupancy_achieved`: Achieved occupancy

Profile reward is computed as:
```python
profile_reward = (
    0.4 * (sm_utilization / 100) +
    0.3 * (memory_throughput / 100) +
    0.3 * (occupancy)
)
```

## Implementation Steps

1. **Phase 1**: Core infrastructure
   - [ ] Task sampler with robust_kbench integration
   - [ ] Prompt builder for CUDA generation
   - [ ] Code extractor with validation
   - [ ] Reward calculator with all three modes

2. **Phase 2**: Training loop
   - [ ] GRPO trainer with HuggingFace transformers
   - [ ] Multi-turn refinement
   - [ ] Logging and checkpointing

3. **Phase 3**: Evaluation and analysis
   - [ ] Evaluation script for trained models
   - [ ] Comparison against baseline
   - [ ] Ablation studies for reward modes

## Dependencies

```
torch>=2.0
transformers>=4.35
accelerate
peft  # For efficient fine-tuning
wandb  # For logging
pyyaml
tqdm
robust_kbench  # This package
```

## Usage

### Training
```bash
# Activate environment
conda activate robust_kbench

# Run training with default config
python -m rl_kernel_opt.scripts.train --config configs/default.yaml

# Run with specific reward mode
python -m rl_kernel_opt.scripts.train --config configs/default.yaml --reward_mode speed_correct

# Resume training
python -m rl_kernel_opt.scripts.train --config configs/default.yaml --resume outputs/checkpoint-1000
```

### Evaluation
```bash
# Evaluate on held-out tasks
python -m rl_kernel_opt.scripts.evaluate --model outputs/final --tasks tasks/
```

### Single Generation
```bash
# Generate CUDA kernel for a single task
python -m rl_kernel_opt.scripts.generate --model Qwen/Qwen2.5-Coder-3B-Instruct --task tasks/layernorm
```
