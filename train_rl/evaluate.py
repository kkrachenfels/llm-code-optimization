#!/usr/bin/env python3
"""
Evaluate a trained model on a C++ program without training.
Generate optimized code and compare runtime.

Usage:
    python evaluate.py --checkpoint PATH --program PROGRAM [OPTIONS]

Example:
    python evaluate.py --checkpoint checkpoints/checkpoint-50 --program programs/bubble_sort.cpp --num-samples 5
"""

import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.compiler import CppCompiler, get_baseline_runtime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_prompt(code: str) -> str:
    """Create optimization prompt."""
    prompt = f"""Optimize the following C++ code for runtime performance. Provide only the complete optimized C++ code without explanations.

Original code:
```cpp
{code}
```

Optimized code:
```cpp
"""
    return prompt


def generate_optimizations(model, tokenizer, prompt: str, num_samples: int = 5):
    """Generate multiple optimization attempts."""
    logger.info(f"Generating {num_samples} optimization candidates...")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    results = []
    for i in range(num_samples):
        logger.info(f"  Generating sample {i+1}/{num_samples}...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):]
        results.append(response)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on C++ optimization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--program", type=str, required=True, help="Path to C++ program")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of optimization attempts")
    parser.add_argument("--num-runs", type=int, default=5, help="Number of runs for benchmarking")
    parser.add_argument("--output", type=str, help="Save best optimized code to file")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Model Evaluation for Code Optimization")
    logger.info("=" * 80)

    # Load program
    program_path = Path(args.program)
    if not program_path.exists():
        logger.error(f"Program not found: {args.program}")
        return 1

    with open(program_path, 'r') as f:
        original_code = f.read()

    # Get baseline
    logger.info("Computing baseline runtime...")
    compiler = CppCompiler()
    baseline_runtime = get_baseline_runtime(str(program_path), compiler, num_runs=args.num_runs)
    logger.info(f"Baseline: {baseline_runtime:.2f} μs")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded on {device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return 1

    # Generate optimizations
    prompt = create_prompt(original_code)
    responses = generate_optimizations(model, tokenizer, prompt, args.num_samples)

    # Evaluate each
    logger.info("\nEvaluating generated optimizations...")
    logger.info("=" * 80)

    best_runtime = baseline_runtime
    best_code = None
    best_idx = -1
    successful_count = 0

    for i, response in enumerate(responses):
        logger.info(f"\nCandidate {i+1}/{len(responses)}:")

        # Extract code
        code = compiler.extract_code_from_llm_output(response)
        if code is None:
            logger.info("  ✗ Failed to extract code")
            continue

        # Compile and run
        success, runtime, error = compiler.compile_and_run(code, num_runs=args.num_runs)

        if not success:
            logger.info(f"  ✗ {error}")
            continue

        successful_count += 1
        speedup = baseline_runtime / runtime

        logger.info(f"  ✓ Runtime: {runtime:.2f} μs")
        logger.info(f"  ✓ Speedup: {speedup:.2f}x")

        if runtime < best_runtime:
            best_runtime = runtime
            best_code = code
            best_idx = i
            logger.info("  ★ New best!")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Successful compilations: {successful_count}/{len(responses)}")
    logger.info(f"Baseline runtime: {baseline_runtime:.2f} μs")

    if best_code is not None:
        best_speedup = baseline_runtime / best_runtime
        logger.info(f"Best runtime: {best_runtime:.2f} μs")
        logger.info(f"Best speedup: {best_speedup:.2f}x ({(best_speedup-1)*100:.1f}% improvement)")
        logger.info(f"Best candidate: #{best_idx + 1}")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                f.write(best_code)
            logger.info(f"Best code saved to: {args.output}")

    else:
        logger.info("No successful optimizations generated.")

    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
