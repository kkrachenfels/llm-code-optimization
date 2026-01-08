#!/usr/bin/env python3
"""Debug script to see what the model generates."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

# Load the model
model_name = "Salesforce/codegen2-1B"
print(f"Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# Load the bubble sort code
program_path = Path("programs/bubble_sort.cpp")
with open(program_path, 'r') as f:
    original_code = f.read()

print("\n" + "="*80)
print("ORIGINAL CODE:")
print("="*80)
print(original_code)

# Create the prompt
prompt = f"""Optimize the following C++ code for runtime performance. Provide only the complete optimized C++ code without explanations.

Original code:
```cpp
{original_code}
```

Optimized code:
```cpp
"""

print("\n" + "="*80)
print("PROMPT:")
print("="*80)
print(prompt)

# Generate
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
inputs = {k: v.to(device) for k, v in inputs.items()}

print("\n" + "="*80)
print("GENERATING 3 SAMPLES...")
print("="*80)

for i in range(3):
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode
    response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(response_tokens, skip_special_tokens=True)

    print(f"\n--- SAMPLE {i+1} ---")
    print(response)
    print("\n" + "-"*80)
