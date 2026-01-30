#!/usr/bin/env python3
"""
Demo script to test model loading and a simple kernel generation.
"""

import os
import sys
import argparse
from pathlib import Path

# Parse args first to set CUDA_VISIBLE_DEVICES before torch import
parser = argparse.ArgumentParser(description="Demo script for CUDA kernel RL")
parser.add_argument("--gpus", type=str, default="0", help="GPU device(s) to use")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.config import get_default_config
from src.dataset import KernelBenchDataset


def main():
    print("=" * 60)
    print("CUDA Kernel RL - Demo")
    print("=" * 60)
    print(f"Using GPU(s): {args.gpus}")

    config = get_default_config()

    # Load dataset
    print("\n1. Loading dataset...")
    dataset = KernelBenchDataset(config.dataset)
    dataset.load()

    # Get a sample task
    train_tasks, _ = dataset.get_train_test_split()
    sample_task = train_tasks[0]
    print(f"   Sample task: {sample_task.name}")

    # Load model with 4-bit quantization
    print("\n2. Loading model (4-bit quantized)...")
    print(f"   Model: {config.model.model_name}")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Model loaded on: {next(model.parameters()).device}")
    print(f"   Memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Generate a kernel
    print("\n3. Generating a CUDA C++ kernel...")

    prompt = f"""You are an expert CUDA C++ kernel developer. Write a custom CUDA C++ kernel to optimize this PyTorch model.

## Reference PyTorch Implementation:
```python
{sample_task.pytorch_code[:1500]}
```

## Requirements:
1. Write a CUDA C++ kernel with __global__ functions
2. Load it using torch.utils.cpp_extension.load_inline
3. Do NOT use PyTorch operations - write raw CUDA C++ code
4. The kernel must be numerically correct
5. Optimize for NVIDIA A6000 GPU

## Example format:
```python
from torch.utils.cpp_extension import load_inline

cuda_source = \"\"\"
__global__ void my_kernel(const float* input, float* output, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {{
        output[idx] = input[idx];  // your logic here
    }}
}}

torch::Tensor forward(torch::Tensor input) {{
    auto output = torch::empty_like(input);
    int n = input.numel();
    my_kernel<<<(n+255)/256, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}}
\"\"\"
```

Provide a brief thought about your optimization strategy, then the full implementation.
"""

    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    print("   Generating (this may take a moment)...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    print("\n4. Generated response:")
    print("-" * 60)
    print(response[:2000])
    if len(response) > 2000:
        print(f"\n... [truncated, total {len(response)} chars]")
    print("-" * 60)

    print("\n✓ Demo completed successfully!")
    print(f"   Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
