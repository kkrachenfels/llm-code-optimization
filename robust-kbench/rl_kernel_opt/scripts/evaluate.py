#!/usr/bin/env python3
"""
Evaluation script for trained CUDA kernel optimization models.

Usage:
    python -m rl_kernel_opt.scripts.evaluate --model outputs/final --tasks tasks/
    python -m rl_kernel_opt.scripts.evaluate --model Qwen/Qwen2.5-Coder-3B-Instruct --tasks tasks/layernorm
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl_kernel_opt.src.task_sampler import TaskSampler
from rl_kernel_opt.src.prompt_builder import PromptBuilder, TurnFeedback
from rl_kernel_opt.src.code_extractor import CUDACodeExtractor
from rl_kernel_opt.src.reward_calculator import RewardCalculator, RewardMode
from rl_kernel_opt.src.utils import set_seed, format_speedup, format_time_ms


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CUDA kernel optimization model")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint or HuggingFace model name",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="Task directories to evaluate on",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples per task",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU ID(s) to use. Single int (e.g., '3') or comma-separated (e.g., '0,1,2').",
    )

    return parser.parse_args()


def parse_gpu_ids(gpu_str: str) -> list:
    """Parse GPU ID string into list of integers."""
    return [int(x.strip()) for x in gpu_str.split(",")]


def evaluate_task(
    task,
    model,
    tokenizer,
    prompt_builder,
    code_extractor,
    reward_calculator,
    num_samples: int,
    max_turns: int,
    temperature: float,
    device: str,
):
    """Evaluate a single task with multiple samples."""
    results = {
        "task_name": task.name,
        "task_dir": task.task_dir,
        "samples": [],
    }

    for sample_idx in range(num_samples):
        sample_result = {
            "sample_idx": sample_idx,
            "turns": [],
            "final_correct": False,
            "final_speedup": None,
            "best_speedup": None,
        }

        messages = prompt_builder.build_initial_prompt(task)
        best_speedup = None

        for turn in range(max_turns):
            # Generate
            prompt = prompt_builder.format_for_generation(messages, tokenizer)
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
                    temperature=temperature,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Extract code
            cuda_code, extract_error = code_extractor.extract(response)

            turn_result = {
                "turn": turn,
                "extracted_code": cuda_code is not None,
            }

            if cuda_code is None:
                turn_result["error"] = extract_error
                turn_result["compiled"] = False
                turn_result["correct"] = False

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
                # Save and evaluate
                cuda_file = code_extractor.save_to_file(
                    cuda_code, task.name, turn, sample_idx
                )
                eval_result = reward_calculator.evaluate_kernel(
                    cuda_file, task.task_dir
                )

                turn_result.update({
                    "compiled": eval_result.compiled,
                    "compile_error": eval_result.compile_error,
                    "correct": eval_result.correct,
                    "max_diff": eval_result.max_diff,
                    "speedup": eval_result.speedup,
                    "torch_time_ms": eval_result.torch_time_ms,
                    "cuda_time_ms": eval_result.cuda_time_ms,
                    "reward": eval_result.reward,
                })

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

                # Early exit if good enough
                if eval_result.correct and eval_result.speedup and eval_result.speedup > 1.5:
                    sample_result["turns"].append(turn_result)
                    break

            sample_result["turns"].append(turn_result)

            # Build refinement prompt
            if turn < max_turns - 1:
                messages = prompt_builder.build_refinement_prompt(
                    task, messages, response, feedback
                )

        # Final results
        last_turn = sample_result["turns"][-1]
        sample_result["final_correct"] = last_turn.get("correct", False)
        sample_result["final_speedup"] = last_turn.get("speedup")
        sample_result["best_speedup"] = best_speedup

        results["samples"].append(sample_result)

    # Aggregate statistics
    correct_count = sum(1 for s in results["samples"] if s["final_correct"])
    speedups = [s["best_speedup"] for s in results["samples"] if s["best_speedup"]]

    results["summary"] = {
        "accuracy": correct_count / num_samples,
        "mean_speedup": sum(speedups) / len(speedups) if speedups else None,
        "max_speedup": max(speedups) if speedups else None,
        "min_speedup": min(speedups) if speedups else None,
    }

    return results


def main():
    args = parse_args()
    set_seed(args.seed)

    # Parse GPU IDs
    gpu_ids = parse_gpu_ids(args.gpu)
    model_gpu = gpu_ids[0]
    device = f"cuda:{model_gpu}"

    print(f"Using GPU {model_gpu} for model, GPUs {gpu_ids} for evaluation")

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    # Initialize components
    prompt_builder = PromptBuilder()
    code_extractor = CUDACodeExtractor()
    reward_calculator = RewardCalculator(
        mode=RewardMode.SPEED_CORRECT,
        warmup_time=25,
        repetition_time=10000,
        timeout=300,
        gpu_ids=gpu_ids,
    )

    # Load tasks
    task_sampler = TaskSampler(args.tasks, forward=True, seed=args.seed)
    print(f"Loaded {len(task_sampler)} tasks")

    # Evaluate
    all_results = {
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_samples": args.num_samples,
            "max_turns": args.max_turns,
            "temperature": args.temperature,
        },
        "tasks": [],
    }

    for task in tqdm(task_sampler, desc="Evaluating tasks"):
        print(f"\nEvaluating task: {task.name}")
        task_results = evaluate_task(
            task=task,
            model=model,
            tokenizer=tokenizer,
            prompt_builder=prompt_builder,
            code_extractor=code_extractor,
            reward_calculator=reward_calculator,
            num_samples=args.num_samples,
            max_turns=args.max_turns,
            temperature=args.temperature,
            device=device,
        )

        summary = task_results["summary"]
        print(f"  Accuracy: {summary['accuracy']:.1%}")
        if summary["mean_speedup"]:
            print(f"  Mean speedup: {format_speedup(summary['mean_speedup'])}")
            print(f"  Max speedup: {format_speedup(summary['max_speedup'])}")

        all_results["tasks"].append(task_results)

    # Compute overall statistics
    all_correct = sum(t["summary"]["accuracy"] * args.num_samples for t in all_results["tasks"])
    total_samples = len(all_results["tasks"]) * args.num_samples
    all_speedups = [
        t["summary"]["mean_speedup"]
        for t in all_results["tasks"]
        if t["summary"]["mean_speedup"]
    ]

    all_results["overall"] = {
        "accuracy": all_correct / total_samples,
        "mean_speedup": sum(all_speedups) / len(all_speedups) if all_speedups else None,
        "tasks_evaluated": len(all_results["tasks"]),
        "total_samples": total_samples,
    }

    # Save results
    results_file = output_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print("Evaluation Complete")
    print("=" * 60)
    print(f"Overall accuracy: {all_results['overall']['accuracy']:.1%}")
    if all_results["overall"]["mean_speedup"]:
        print(f"Overall mean speedup: {format_speedup(all_results['overall']['mean_speedup'])}")
    print(f"Results saved to: {results_file}")

    # Cleanup
    code_extractor.cleanup()


if __name__ == "__main__":
    main()
