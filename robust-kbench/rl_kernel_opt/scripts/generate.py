#!/usr/bin/env python3
"""
Single kernel generation script for testing and debugging.

Usage:
    python -m rl_kernel_opt.scripts.generate --task tasks/layernorm
    python -m rl_kernel_opt.scripts.generate --task tasks/layernorm --backward
    python -m rl_kernel_opt.scripts.generate --model outputs/final --task tasks/mnist_cross_entropy
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_kernel_opt.src.task_sampler import TaskSampler
from rl_kernel_opt.src.prompt_builder import PromptBuilder, TurnFeedback
from rl_kernel_opt.src.code_extractor import CUDACodeExtractor
from rl_kernel_opt.src.reward_calculator import RewardCalculator, RewardMode
from rl_kernel_opt.src.utils import format_speedup, format_time_ms


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CUDA kernel for a single task")

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="Model name or checkpoint path",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for generated CUDA code",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="Maximum refinement turns",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the generated kernel",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output including prompts and responses",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Generate a backward (gradient) kernel instead of forward",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID(s) to use. Single int (e.g., '3') or comma-separated (e.g., '0,1,2'). "
             "First GPU is used for model, all are used for evaluation.",
    )

    return parser.parse_args()


def parse_gpu_ids(gpu_str: str) -> list:
    """Parse GPU ID string into list of integers."""
    return [int(x.strip()) for x in gpu_str.split(",")]


def main():
    args = parse_args()

    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(args.gpu)
    model_gpu = gpu_ids[0]  # First GPU for model
    eval_gpu_ids = gpu_ids  # All GPUs available for evaluation

    if torch.cuda.is_available():
        device = f"cuda:{model_gpu}"
        if len(gpu_ids) == 1:
            print(f"Using GPU {model_gpu} for model and evaluation")
        else:
            print(f"Using GPU {model_gpu} for model, GPUs {gpu_ids} for evaluation")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()

    # Load task
    is_forward = not args.backward
    task_sampler = TaskSampler([args.task], forward=is_forward)
    if len(task_sampler) == 0:
        print(f"ERROR: No valid task found in {args.task}")
        sys.exit(1)

    task = task_sampler.tasks[0]
    print(f"\nTask: {task.name}")
    print(f"Description: {task.docstring[:200]}..." if len(task.docstring) > 200 else f"Description: {task.docstring}")

    # Initialize components
    prompt_builder = PromptBuilder()
    code_extractor = CUDACodeExtractor()

    if args.evaluate:
        reward_calculator = RewardCalculator(
            mode=RewardMode.SPEED_CORRECT,
            warmup_time=25,
            repetition_time=10000,
            timeout=300,
            gpu_ids=eval_gpu_ids,
        )
    else:
        reward_calculator = None

    # Generate
    messages = prompt_builder.build_initial_prompt(task)
    best_code = None
    best_speedup = None

    for turn in range(args.max_turns):
        print(f"\n{'='*60}")
        print(f"Turn {turn + 1}/{args.max_turns}")
        print("="*60)

        # Format prompt
        prompt = prompt_builder.format_for_generation(messages, tokenizer)

        if args.verbose:
            print("\n--- PROMPT ---")
            print(prompt[-2000:] if len(prompt) > 2000 else prompt)
            print("--- END PROMPT ---\n")

        # Generate
        print("Generating...")
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
                temperature=args.temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        if args.verbose:
            print("\n--- RESPONSE ---")
            print(response)
            print("--- END RESPONSE ---\n")

        # Extract code
        cuda_code, extract_error = code_extractor.extract(response)

        if cuda_code is None:
            print(f"Failed to extract CUDA code: {extract_error}")

            feedback = TurnFeedback(
                compiled=False,
                compile_error=extract_error,
                correct=False,
                max_diff=None,
                speedup=None,
                torch_time_ms=None,
                cuda_time_ms=None,
                profile_info=None,
            )
        else:
            print(f"Extracted CUDA code ({len(cuda_code)} chars)")
            best_code = cuda_code

            if args.evaluate:
                # Save and evaluate
                cuda_file = code_extractor.save_to_file(cuda_code, task.name, turn, 0, forward=is_forward)
                print(f"Saved to: {cuda_file}")

                print("Evaluating...")
                eval_result = reward_calculator.evaluate_kernel(cuda_file, task.task_dir, forward=is_forward)

                print(f"\nResults:")
                print(f"  Compiled: {eval_result.compiled}")
                if eval_result.compile_error:
                    print(f"  Compile error: {eval_result.compile_error[:200]}...")
                print(f"  Correct: {eval_result.correct}")
                if eval_result.max_diff:
                    print(f"  Max diff: {eval_result.max_diff:.2e}")
                if eval_result.torch_time_ms:
                    print(f"  PyTorch time: {format_time_ms(eval_result.torch_time_ms)}")
                if eval_result.cuda_time_ms:
                    print(f"  CUDA time: {format_time_ms(eval_result.cuda_time_ms)}")
                if eval_result.speedup:
                    print(f"  Speedup: {format_speedup(eval_result.speedup)}")
                print(f"  Reward: {eval_result.reward:.4f}")

                if eval_result.speedup and (best_speedup is None or eval_result.speedup > best_speedup):
                    best_speedup = eval_result.speedup

                feedback = TurnFeedback(
                    compiled=eval_result.compiled,
                    compile_error=eval_result.compile_error,
                    correct=eval_result.correct,
                    max_diff=eval_result.max_diff,
                    speedup=eval_result.speedup,
                    torch_time_ms=eval_result.torch_time_ms,
                    cuda_time_ms=eval_result.cuda_time_ms,
                    profile_info=eval_result.profile_info,
                )

                # Early exit if good
                if eval_result.correct and eval_result.speedup and eval_result.speedup > 1.5:
                    print("\nGood result achieved, stopping early.")
                    break
            else:
                feedback = TurnFeedback(
                    compiled=True,
                    compile_error=None,
                    correct=True,
                    max_diff=None,
                    speedup=None,
                    torch_time_ms=None,
                    cuda_time_ms=None,
                    profile_info=None,
                )

        # Build refinement prompt
        if turn < args.max_turns - 1:
            messages = prompt_builder.build_refinement_prompt(
                task, messages, response, feedback
            )

    # Save final code
    if best_code:
        if args.output:
            output_path = args.output
        else:
            output_path = f"generated_{task.name}.cu"

        with open(output_path, "w") as f:
            f.write(best_code)
        print(f"\nFinal CUDA code saved to: {output_path}")

        if best_speedup:
            print(f"Best speedup achieved: {format_speedup(best_speedup)}")
    else:
        print("\nNo valid CUDA code was generated.")

    # Cleanup
    code_extractor.cleanup()


if __name__ == "__main__":
    main()
