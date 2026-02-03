# RL-Based CUDA Kernel Optimization

Reinforcement learning for CUDA kernel optimization using Qwen2.5-Coder-3B-Instruct with rewards from robust_kbench evaluation. Inspired by Cognition's Kevin-32B approach, this implementation uses GRPO (Group Relative Policy Optimization) for training.

## Quick Start

```bash
# Activate the conda environment
conda activate robust_kbench

# Install additional dependencies
pip install -r rl_kernel_opt/requirements.txt

# Test the setup
python -m rl_kernel_opt.scripts.test_setup

# Generate a kernel for a single task (no training)
python -m rl_kernel_opt.scripts.generate --task tasks/layernorm --evaluate

# Start training
python -m rl_kernel_opt.scripts.train --config rl_kernel_opt/configs/default.yaml
```

## Reward Modes

Three reward computation modes are supported:

### 1. Speed Only (`speed`)
```
reward = speedup_ratio  # torch_time / cuda_time
# Compile failure: -1.0
# Runtime crash: 0.0
```

### 2. Speed + Correctness (`speed_correct`) [Default]
```
if not compiles: reward = -1.0
elif not correct: reward = 0.0
else: reward = correctness_bonus (0.3) + speedup_ratio
```

### 3. Speed + Correctness + Profiling (`speed_correct_profile`)
```
if not compiles: reward = -1.0
elif not correct: reward = 0.0
else: reward = correctness_bonus + speedup_ratio + profile_bonus
```

Profile bonus incorporates:
- SM utilization
- Memory throughput
- Achieved occupancy

## Multi-Turn Refinement

Following Kevin-32B, the model iterates on its outputs with feedback:

```
Turn 1: Initial generation → Feedback (compile/correct/speed)
Turn 2: Refined generation → Feedback
Turn 3: Final refinement
```

Total reward uses discounted sum: `R = r₁ + γ*r₂ + γ²*r₃` (γ=0.4)

## Project Structure

```
rl_kernel_opt/
├── configs/
│   └── default.yaml           # Training configuration
├── src/
│   ├── task_sampler.py        # Load tasks from robust_kbench
│   ├── prompt_builder.py      # Build prompts for generation
│   ├── code_extractor.py      # Extract CUDA code from outputs
│   ├── reward_calculator.py   # Compute rewards via robust_kbench
│   ├── grpo_trainer.py        # GRPO training loop
│   └── utils.py               # Utilities
├── scripts/
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   ├── generate.py            # Single task generation
│   └── test_setup.py          # Setup verification
└── outputs/                   # Checkpoints and logs
```

## Training

```bash
# Default training
python -m rl_kernel_opt.scripts.train --config rl_kernel_opt/configs/default.yaml

# With specific reward mode
python -m rl_kernel_opt.scripts.train --config rl_kernel_opt/configs/default.yaml --reward_mode speed_correct_profile

# Resume from checkpoint
python -m rl_kernel_opt.scripts.train --resume outputs/checkpoint-1000

# Debug mode (reduced settings for testing)
python -m rl_kernel_opt.scripts.train --config rl_kernel_opt/configs/default.yaml --debug
```

## Evaluation

```bash
# Evaluate on specific tasks
python -m rl_kernel_opt.scripts.evaluate --model outputs/final --tasks tasks/layernorm tasks/mnist_cross_entropy

# Evaluate base model
python -m rl_kernel_opt.scripts.evaluate --model Qwen/Qwen2.5-Coder-3B-Instruct --tasks tasks/
```

## Configuration

See `configs/default.yaml` for all options. Key settings:

```yaml
model:
  name: "Qwen/Qwen2.5-Coder-3B-Instruct"
  max_new_tokens: 4096
  temperature: 0.7

grpo:
  group_size: 8        # Samples per prompt
  gamma: 0.4           # Discount factor

training:
  tasks_per_batch: 4
  max_turns: 3
  max_steps: 5000

reward:
  mode: "speed_correct"
  correctness_bonus: 0.3
  max_speedup_reward: 5.0
```

## Hardware Requirements

- GPU with at least 16GB VRAM for Qwen2.5-Coder-3B-Instruct
- Additional GPU(s) recommended for kernel evaluation
- Training with default settings uses ~24GB peak GPU memory
