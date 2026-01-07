# RL-Based Code Optimization

This project uses Reinforcement Learning (specifically PPO - Proximal Policy Optimization) to train a small language model to optimize C++ programs for runtime performance.

## Overview

The system works as follows:
1. **Prompt**: The LLM is given a C++ program and asked to optimize it for runtime
2. **Generate**: The LLM generates an optimized version of the code
3. **Evaluate**: The code is compiled and executed, measuring runtime
4. **Reward**: A reward is computed based on speedup compared to the baseline
5. **Update**: The LLM is updated using PPO to maximize the reward

## Project Structure

```
train_rl/
├── programs/           # C++ programs to optimize
│   └── bubble_sort.cpp # Example inefficient sorting program
├── benchmarks/         # Compiled binaries (generated)
├── checkpoints/        # Model checkpoints (generated)
├── logs/              # Training logs (generated)
├── src/               # Python source code
│   ├── compiler.py    # C++ compilation and execution
│   ├── reward.py      # Reward function implementations
│   └── trainer.py     # RL training loop (PPO)
├── train.py           # Main training script
└── requirements.txt   # Python dependencies
```

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify C++ compiler

Make sure you have g++ installed:
```bash
g++ --version
```

## Usage

### Basic Training

Train with a small model (recommended for initial experiments):

```bash
python train.py \
  --model Salesforce/codegen-350M-mono \
  --program programs/bubble_sort.cpp \
  --steps 50 \
  --batch-size 4
```

### Available Models

Small models suitable for this task:
- `Salesforce/codegen-350M-mono` (350M parameters, code-focused)
- `bigcode/tiny_starcoder_py` (~160M parameters, very lightweight)
- `Salesforce/codegen-2B-mono` (2B parameters, more capable but slower)

For GPU memory constraints, use the `--use-8bit` flag:

```bash
python train.py \
  --model Salesforce/codegen-2B-mono \
  --program programs/bubble_sort.cpp \
  --use-8bit \
  --batch-size 2
```

### Command-Line Arguments

**Required:**
- `--model`: HuggingFace model name

**Optional:**
- `--program`: Path to C++ program (default: `programs/bubble_sort.cpp`)
- `--steps`: Number of training steps (default: 100)
- `--batch-size`: Batch size (default: 4)
- `--learning-rate`: Learning rate (default: 1.41e-5)
- `--output-dir`: Checkpoint directory (default: `checkpoints`)
- `--save-every`: Save checkpoint every N steps (default: 10)
- `--use-8bit`: Use 8-bit quantization to reduce memory

## Adding More Programs

To train on different C++ programs:

1. Create a new C++ file in `programs/`
2. Ensure it outputs runtime in microseconds to stdout
3. Train: `python train.py --model MODEL_NAME --program programs/your_program.cpp`

Example C++ program template:

```cpp
#include <iostream>
#include <chrono>

// Your code here
void algorithmToOptimize() {
    // ...
}

int main() {
    auto start = std::chrono::high_resolution_clock::now();

    algorithmToOptimize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << duration.count() << std::endl;
    return 0;
}
```

## Reward Function

The reward function is based on runtime speedup:

- **Success + Speedup**: Logarithmic reward based on speedup ratio
- **Success + Similar Performance**: Small positive reward (within 5%)
- **Success + Slowdown**: Negative reward proportional to slowdown
- **Compilation Failure**: Fixed penalty (-1.0)
- **Timeout**: Higher penalty (-2.0)

The reward function is adaptive and tracks the best runtime seen during training.

## Architecture

### PPO (Proximal Policy Optimization)

We use TRL's PPOTrainer which implements:
- Value function for advantage estimation
- Clipped objective to prevent large policy updates
- Multiple epochs over collected data
- KL divergence constraint to reference policy

### Alternative: GRPO

To use GRPO (Group Relative Policy Optimization) instead, you can modify `src/trainer.py` to use TRL's GRPO implementation, which is simpler and doesn't require a value head.

## Monitoring Training

During training, you'll see logs like:

```
Step 1: mean_reward=0.523, best_speedup=1.34x
Success! Runtime: 150234.23μs (baseline: 201752.00μs), Reward: 0.745
```

Key metrics:
- `mean_reward`: Average reward across the batch
- `best_speedup`: Best speedup achieved so far
- Individual results show runtime comparisons

## Extending the Project

### Multiple Programs Dataset

To train on multiple programs:

1. Add more C++ files to `programs/`
2. Modify `trainer.py` to sample from multiple programs
3. Create a dataset configuration file

### Different Optimization Targets

You can modify the reward function to optimize for:
- Binary size (measure output binary size)
- Memory usage (measure peak memory)
- Energy consumption (use power measurement tools)
- Multi-objective (combine multiple metrics)

### Advanced RL Algorithms

The codebase can be extended to support:
- **GRPO**: Simpler than PPO, no value head needed
- **DPO**: Direct Preference Optimization using pairwise comparisons
- **RLOO**: REINFORCE Leave-One-Out for stability

## Troubleshooting

### CUDA Out of Memory

- Reduce `--batch-size` to 1 or 2
- Use `--use-8bit` flag
- Try a smaller model
- Reduce `--max-length`

### Compilation Failures

- Check that generated code is valid C++
- Adjust the prompt in `trainer.py` to be more specific
- Increase penalty for compilation failures

### Poor Performance

- Increase number of training steps
- Adjust learning rate
- Try different models
- Modify reward function weights

## License

This is a research project. Use at your own discretion.
