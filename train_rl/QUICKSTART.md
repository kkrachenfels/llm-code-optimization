# Quick Start Guide

Get started with RL-based code optimization in 5 minutes.

## Prerequisites

- Python 3.8+
- g++ compiler
- CUDA-capable GPU (recommended, but CPU works too)

## Setup

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Test the setup
python test_setup.py
```

## Run Your First Training

```bash
# Train with a small model on the bubble sort example
python train.py \
  --model Salesforce/codegen-350M-mono \
  --program programs/bubble_sort.cpp \
  --steps 20 \
  --batch-size 4
```

This will:
1. Load the CodeGen-350M model
2. Compile and benchmark the original bubble sort
3. Train the model to generate optimized versions
4. Save checkpoints every 10 steps

## What to Expect

The model will attempt to optimize the bubble sort algorithm. You should see output like:

```
Baseline runtime: 201752.00 microseconds
Step 1: mean_reward=0.523, best_speedup=1.34x
Success! Runtime: 150234.23μs (baseline: 201752.00μs), Reward: 0.745
```

## Next Steps

### Try Different Programs

```bash
# Matrix multiplication (more room for optimization)
python train.py \
  --model Salesforce/codegen-350M-mono \
  --program programs/matrix_multiply.cpp \
  --steps 50
```

### Use a Larger Model

```bash
# CodeGen-2B with 8-bit quantization
python train.py \
  --model Salesforce/codegen-2B-mono \
  --program programs/bubble_sort.cpp \
  --use-8bit \
  --batch-size 2
```

### Add Your Own Program

1. Create a C++ file in `programs/` that outputs runtime to stdout
2. Train: `python train.py --model MODEL --program programs/your_program.cpp`

## Common Issues

**CUDA Out of Memory:**
- Use smaller batch size: `--batch-size 1`
- Enable 8-bit: `--use-8bit`
- Use smaller model: `bigcode/tiny_starcoder_py`

**No GPU:**
The code will automatically use CPU (slower but works).

**Model Download Fails:**
Check internet connection. Models are downloaded from HuggingFace.

## Understanding Results

- **Speedup > 1.0**: Code is faster ✓
- **Speedup ≈ 1.0**: Similar performance
- **Speedup < 1.0**: Code is slower
- **Compilation failures**: Model generated invalid code

The model learns to maximize speedup over time.

## Advanced Usage

See [README.md](README.md) for:
- Complete documentation
- Reward function details
- Multi-program training
- Custom optimization targets
