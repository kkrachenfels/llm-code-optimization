"""
Multi-turn trajectory generation for CUDA kernel optimization.
Handles refinement steps and thought summaries.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import copy

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config import Config, PromptConfig
from .dataset import KernelTask
from .kernel_evaluator import EvaluationResult, extract_kernel_code


@dataclass
class RefinementStep:
    """A single refinement step in a trajectory."""
    step_idx: int
    prompt: str
    response: str
    kernel_code: Optional[str]
    thought_summary: str
    eval_result: Optional[EvaluationResult]


@dataclass
class Trajectory:
    """A complete trajectory of refinement steps."""
    trajectory_id: str
    task: KernelTask
    steps: List[RefinementStep] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def final_kernel(self) -> Optional[str]:
        if self.steps:
            return self.steps[-1].kernel_code
        return None

    @property
    def final_result(self) -> Optional[EvaluationResult]:
        if self.steps:
            return self.steps[-1].eval_result
        return None

    def get_context_for_step(self, step_idx: int, max_context_length: int = 4096) -> str:
        """
        Build context for a given refinement step.
        Uses thought summaries instead of full chain-of-thought to manage context length.
        """
        if step_idx == 0:
            return ""

        context_parts = []
        prev_step = self.steps[step_idx - 1]

        # Include previous kernel (without full reasoning)
        if prev_step.kernel_code:
            context_parts.append(f"Previous kernel code:\n```python\n{prev_step.kernel_code}\n```")

        # Include thought summary (compressed reasoning)
        if prev_step.thought_summary:
            context_parts.append(f"Previous approach: {prev_step.thought_summary}")

        # Include evaluation feedback
        if prev_step.eval_result:
            context_parts.append(f"Evaluation: {prev_step.eval_result.feedback}")

        context = "\n\n".join(context_parts)

        # Truncate if too long
        if len(context) > max_context_length:
            context = context[:max_context_length] + "\n... [truncated]"

        return context


def extract_thought_summary(response: str) -> str:
    """
    Extract a brief thought summary from a model response.
    Looks for explicit summaries or extracts from reasoning.
    """
    # Look for explicit thought summary markers
    summary_patterns = [
        r'(?:thought summary|summary|approach|strategy):\s*(.+?)(?:\n\n|```)',
        r'(?:^|\n)(?:my approach|i will|plan):\s*(.+?)(?:\n\n|```)',
    ]

    for pattern in summary_patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            summary = match.group(1).strip()
            # Limit length
            if len(summary) > 500:
                summary = summary[:500] + "..."
            return summary

    # Fallback: extract first paragraph or reasoning
    lines = response.strip().split('\n')
    non_code_lines = []

    in_code_block = False
    for line in lines:
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            continue
        if not in_code_block and line.strip():
            non_code_lines.append(line.strip())
        if len(non_code_lines) >= 3:
            break

    if non_code_lines:
        summary = ' '.join(non_code_lines)
        if len(summary) > 500:
            summary = summary[:500] + "..."
        return summary

    return "No explicit reasoning provided."


class TrajectoryGenerator:
    """
    Generates multi-turn trajectories for kernel optimization.
    """

    def __init__(
        self,
        config: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

    def build_initial_prompt(self, task: KernelTask) -> str:
        """Build the initial prompt for a task."""
        prompt = self.config.prompt.task_prompt_template.format(
            pytorch_code=task.pytorch_code,
            gpu_type="H100"  # Can be made configurable
        )
        return prompt

    def build_refinement_prompt(
        self,
        task: KernelTask,
        trajectory: Trajectory,
        step_idx: int
    ) -> str:
        """Build a refinement prompt based on previous steps."""
        prev_step = trajectory.steps[-1]

        prompt = self.config.prompt.refinement_prompt_template.format(
            step_num=step_idx + 1,
            previous_code=prev_step.kernel_code or "No valid code extracted",
            feedback=prev_step.eval_result.feedback if prev_step.eval_result else "No feedback",
            thought_summary=prev_step.thought_summary
        )

        return prompt

    def build_messages(
        self,
        task: KernelTask,
        trajectory: Optional[Trajectory] = None,
        step_idx: int = 0
    ) -> List[Dict[str, str]]:
        """Build message list for model input."""
        messages = [
            {"role": "system", "content": self.config.prompt.system_prompt}
        ]

        if step_idx == 0 or trajectory is None:
            # Initial prompt
            messages.append({
                "role": "user",
                "content": self.build_initial_prompt(task)
            })
        else:
            # Build conversation history (compressed)
            messages.append({
                "role": "user",
                "content": self.build_initial_prompt(task)
            })

            # Add previous steps (compressed)
            for i, step in enumerate(trajectory.steps):
                # Add assistant response (compressed to kernel + summary)
                assistant_content = f"Thought: {step.thought_summary}\n\n"
                if step.kernel_code:
                    assistant_content += f"```python\n{step.kernel_code}\n```"
                messages.append({
                    "role": "assistant",
                    "content": assistant_content
                })

                # Add user feedback
                if step.eval_result:
                    messages.append({
                        "role": "user",
                        "content": self.build_refinement_prompt(task, trajectory, i + 1)
                    })

        return messages

    @torch.no_grad()
    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_new_tokens: Optional[int] = None
    ) -> str:
        """Generate a model response."""
        if max_new_tokens is None:
            max_new_tokens = self.config.model.max_response_length

        # Format messages for the model
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.model.max_seq_length - max_new_tokens
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        return response

    def generate_trajectory(
        self,
        task: KernelTask,
        trajectory_id: str,
        evaluator: Any,  # KernelEvaluator
        max_steps: Optional[int] = None,
        temperature: float = 0.7
    ) -> Trajectory:
        """
        Generate a complete trajectory for a task.

        Args:
            task: The kernel optimization task
            trajectory_id: Unique identifier for this trajectory
            evaluator: KernelEvaluator instance
            max_steps: Maximum refinement steps (default from config)
            temperature: Sampling temperature

        Returns:
            Complete Trajectory with all refinement steps
        """
        if max_steps is None:
            max_steps = self.config.training.max_refinement_steps

        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            task=task
        )

        for step_idx in range(max_steps):
            # Build messages
            messages = self.build_messages(task, trajectory, step_idx)

            # Generate response
            response = self.generate_response(messages, temperature)

            # Extract kernel code
            kernel_code = extract_kernel_code(response)

            # Extract thought summary
            thought_summary = extract_thought_summary(response)

            # Evaluate kernel
            eval_result = None
            if kernel_code:
                eval_result = evaluator.evaluate(
                    kernel_code,
                    task.pytorch_code
                )
            else:
                eval_result = EvaluationResult(
                    success=False,
                    error_type="ExtractionError",
                    feedback="Could not extract valid kernel code from response."
                )

            # Create step
            step = RefinementStep(
                step_idx=step_idx,
                prompt=messages[-1]['content'] if messages else "",
                response=response,
                kernel_code=kernel_code,
                thought_summary=thought_summary,
                eval_result=eval_result
            )
            trajectory.steps.append(step)

            # Early stopping if kernel is correct and fast
            if eval_result and eval_result.success and eval_result.speedup >= 2.0:
                break

        return trajectory


class BatchTrajectoryGenerator:
    """
    Generates multiple trajectories in parallel for a batch of tasks.
    """

    def __init__(
        self,
        config: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.single_generator = TrajectoryGenerator(config, model, tokenizer)

    def generate_batch(
        self,
        tasks: List[KernelTask],
        evaluator: Any,
        trajectories_per_task: Optional[int] = None,
        max_steps: Optional[int] = None,
        temperature: float = 0.7
    ) -> List[List[Trajectory]]:
        """
        Generate multiple trajectories for each task.

        Args:
            tasks: List of kernel optimization tasks
            evaluator: KernelEvaluator instance
            trajectories_per_task: Number of parallel trajectories per task
            max_steps: Maximum refinement steps
            temperature: Sampling temperature

        Returns:
            List of lists: [task_idx][trajectory_idx] -> Trajectory
        """
        if trajectories_per_task is None:
            trajectories_per_task = self.config.training.trajectories_per_task
        if max_steps is None:
            max_steps = self.config.training.max_refinement_steps

        all_trajectories = []

        for task_idx, task in enumerate(tasks):
            task_trajectories = []

            for traj_idx in range(trajectories_per_task):
                trajectory_id = f"task{task_idx}_traj{traj_idx}"

                trajectory = self.single_generator.generate_trajectory(
                    task=task,
                    trajectory_id=trajectory_id,
                    evaluator=evaluator,
                    max_steps=max_steps,
                    temperature=temperature
                )
                task_trajectories.append(trajectory)

            all_trajectories.append(task_trajectories)

        return all_trajectories


class VLLMTrajectoryGenerator:
    """
    High-throughput trajectory generation using vLLM.
    Generates all responses for a step in parallel.
    """

    def __init__(
        self,
        config: Config,
        vllm_engine: Any  # vllm.LLM
    ):
        self.config = config
        self.engine = vllm_engine

    def generate_batch_step(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Generate responses for a batch of prompts using vLLM.
        """
        from vllm import SamplingParams

        if max_tokens is None:
            max_tokens = self.config.model.max_response_length

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_tokens
        )

        outputs = self.engine.generate(prompts, sampling_params)
        responses = [output.outputs[0].text for output in outputs]

        return responses

    def generate_trajectories_parallel(
        self,
        tasks: List[KernelTask],
        evaluator: Any,
        trajectories_per_task: Optional[int] = None,
        max_steps: Optional[int] = None,
        temperature: float = 0.7
    ) -> List[List[Trajectory]]:
        """
        Generate trajectories with maximum parallelism using vLLM.

        All initial responses are generated in one batch,
        then all step-1 responses, etc.
        """
        if trajectories_per_task is None:
            trajectories_per_task = self.config.training.trajectories_per_task
        if max_steps is None:
            max_steps = self.config.training.max_refinement_steps

        # Initialize trajectories
        all_trajectories = [
            [
                Trajectory(
                    trajectory_id=f"task{task_idx}_traj{traj_idx}",
                    task=task
                )
                for traj_idx in range(trajectories_per_task)
            ]
            for task_idx, task in enumerate(tasks)
        ]

        prompt_config = self.config.prompt

        for step_idx in range(max_steps):
            # Build all prompts for this step
            prompts = []
            prompt_map = []  # (task_idx, traj_idx) for each prompt

            for task_idx, task in enumerate(tasks):
                for traj_idx in range(trajectories_per_task):
                    trajectory = all_trajectories[task_idx][traj_idx]

                    # Skip if trajectory already succeeded well
                    if (trajectory.final_result and
                        trajectory.final_result.success and
                        trajectory.final_result.speedup >= 2.0):
                        continue

                    # Build prompt
                    if step_idx == 0:
                        prompt = prompt_config.task_prompt_template.format(
                            pytorch_code=task.pytorch_code,
                            gpu_type="H100"
                        )
                        full_prompt = f"{prompt_config.system_prompt}\n\n{prompt}"
                    else:
                        prev_step = trajectory.steps[-1]
                        refinement = prompt_config.refinement_prompt_template.format(
                            step_num=step_idx + 1,
                            previous_code=prev_step.kernel_code or "No code",
                            feedback=prev_step.eval_result.feedback if prev_step.eval_result else "No feedback",
                            thought_summary=prev_step.thought_summary
                        )
                        full_prompt = f"{prompt_config.system_prompt}\n\n{refinement}"

                    prompts.append(full_prompt)
                    prompt_map.append((task_idx, traj_idx))

            if not prompts:
                break  # All trajectories completed

            # Generate all responses in parallel
            responses = self.generate_batch_step(prompts, temperature)

            # Process responses and evaluate
            for (task_idx, traj_idx), prompt, response in zip(
                prompt_map, prompts, responses
            ):
                trajectory = all_trajectories[task_idx][traj_idx]
                task = tasks[task_idx]

                kernel_code = extract_kernel_code(response)
                thought_summary = extract_thought_summary(response)

                if kernel_code:
                    eval_result = evaluator.evaluate(kernel_code, task.pytorch_code)
                else:
                    eval_result = EvaluationResult(
                        success=False,
                        error_type="ExtractionError",
                        feedback="Could not extract kernel code."
                    )

                step = RefinementStep(
                    step_idx=step_idx,
                    prompt=prompt,
                    response=response,
                    kernel_code=kernel_code,
                    thought_summary=thought_summary,
                    eval_result=eval_result
                )
                trajectory.steps.append(step)

        return all_trajectories


def flatten_trajectories_for_training(
    all_trajectories: List[List[Trajectory]],
    advantages: List[List[List[float]]]
) -> List[Dict[str, Any]]:
    """
    Flatten trajectories into individual training samples.
    Each refinement step becomes its own training sample.

    Returns list of dicts with:
    - input_ids: tokenized prompt
    - response: model response
    - advantage: GRPO advantage for this step
    - metadata: task_id, trajectory_id, step_idx
    """
    samples = []

    for task_idx, task_trajectories in enumerate(all_trajectories):
        for traj_idx, trajectory in enumerate(task_trajectories):
            for step_idx, step in enumerate(trajectory.steps):
                advantage = advantages[task_idx][traj_idx][step_idx]

                sample = {
                    'prompt': step.prompt,
                    'response': step.response,
                    'advantage': advantage,
                    'metadata': {
                        'task_id': trajectory.task.task_id,
                        'trajectory_id': trajectory.trajectory_id,
                        'step_idx': step_idx,
                        'is_correct': step.eval_result.is_correct if step.eval_result else False,
                        'speedup': step.eval_result.speedup if step.eval_result else 0.0
                    }
                }
                samples.append(sample)

    return samples
