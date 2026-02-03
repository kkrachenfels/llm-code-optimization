"""
Task sampler for loading and sampling kernel optimization tasks from robust_kbench.
"""

import os
import json
import random
import importlib.util
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path


@dataclass
class KernelTask:
    """Represents a kernel optimization task."""

    name: str
    task_dir: str
    pytorch_code: str
    input_specs: Dict[str, Any]
    output_spec: str
    docstring: str
    forward: bool
    config: Dict[str, Any]

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for prompt building."""
        return {
            "name": self.name,
            "pytorch_code": self.pytorch_code,
            "input_specs": self.input_specs,
            "output_spec": self.output_spec,
            "docstring": self.docstring,
            "forward": self.forward,
        }


class TaskSampler:
    """Loads and samples kernel optimization tasks from robust_kbench."""

    def __init__(
        self,
        task_dirs: List[str],
        forward: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize the task sampler.

        Args:
            task_dirs: List of task directory paths
            forward: Whether to use forward pass tasks
            seed: Random seed for reproducibility
        """
        self.task_dirs = [Path(d) for d in task_dirs]
        self.forward = forward
        self.tasks: List[KernelTask] = []

        if seed is not None:
            random.seed(seed)

        self._load_tasks()

    def _load_tasks(self):
        """Load all tasks from the specified directories."""
        for task_dir in self.task_dirs:
            if not task_dir.exists():
                print(f"Warning: Task directory {task_dir} does not exist, skipping.")
                continue

            task = self._load_single_task(task_dir)
            if task is not None:
                self.tasks.append(task)

        print(f"Loaded {len(self.tasks)} tasks")

    def _load_single_task(self, task_dir: Path) -> Optional[KernelTask]:
        """Load a single task from a directory."""
        func_file = "func_forward.py" if self.forward else "func_backward.py"
        config_file = "config_forward.json" if self.forward else "config_backward.json"

        func_path = task_dir / func_file
        config_path = task_dir / config_file

        if not func_path.exists():
            print(f"Warning: {func_path} not found, skipping task.")
            return None

        # Load the function module
        try:
            spec = importlib.util.spec_from_file_location("func_module", func_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            print(f"Warning: Failed to load {func_path}: {e}")
            return None

        import inspect

        if self.forward:
            # Forward tasks: extract forward_fn
            if not hasattr(module, "forward_fn"):
                print(f"Warning: forward_fn not found in {func_path}")
                return None

            fn = module.forward_fn

            try:
                pytorch_code = inspect.getsource(fn)
            except Exception:
                pytorch_code = "# Could not extract source for forward_fn"

            docstring = fn.__doc__ or "No description available."
        else:
            # Backward tasks: extract AutogradFunction class (contains backward logic)
            # The func_backward.py files expose AutogradFunction and forward_fn,
            # not a standalone backward_fn. The backward kernel replaces
            # AutogradFunction.backward_fn which is called from AutogradFunction.backward().
            if not hasattr(module, "AutogradFunction"):
                print(f"Warning: AutogradFunction not found in {func_path}")
                return None

            autograd_cls = module.AutogradFunction
            fn = getattr(module, "forward_fn", None)

            # Show the full AutogradFunction class so the LLM sees both forward
            # context and the backward method it needs to implement as a CUDA kernel.
            # Note: inspect.getsource() fails on torch.autograd.Function subclasses
            # because FunctionMeta makes them appear as built-in classes. We fall back
            # to extracting the class source directly from the file.
            try:
                pytorch_code = inspect.getsource(autograd_cls)
            except (TypeError, OSError):
                pytorch_code = self._extract_class_source(func_path, "AutogradFunction")

            # Use the forward_fn docstring for operation description (more informative)
            docstring = (fn.__doc__ if fn and fn.__doc__ else
                         autograd_cls.__doc__ or "No description available.")

        # Get input names if available
        input_names = getattr(module, "input_names", [])

        # Load config
        config = {}
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)

        # Build input specs from config
        input_specs = self._build_input_specs(config, input_names)

        # Infer output spec from docstring or function
        output_spec = self._infer_output_spec(docstring, fn)

        return KernelTask(
            name=task_dir.name,
            task_dir=str(task_dir),
            pytorch_code=pytorch_code,
            input_specs=input_specs,
            output_spec=output_spec,
            docstring=docstring,
            forward=self.forward,
            config=config,
        )

    @staticmethod
    def _extract_class_source(file_path: Path, class_name: str) -> str:
        """Extract a class definition from a Python source file using AST.

        Falls back to reading raw source when inspect.getsource() fails
        (e.g. for torch.autograd.Function subclasses).
        """
        import ast

        try:
            source = file_path.read_text()
            tree = ast.parse(source)
        except Exception:
            return f"# Could not extract source for {class_name}"

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                lines = source.splitlines()
                # AST line numbers are 1-indexed
                start = node.lineno - 1
                end = node.end_lineno  # end_lineno is inclusive in ast
                return "\n".join(lines[start:end])

        return f"# Could not find class {class_name} in {file_path}"

    def _build_input_specs(
        self, config: Dict[str, Any], input_names: List[str]
    ) -> Dict[str, Any]:
        """Build input specifications from config."""
        specs = {"input_names": input_names}

        # Get example input config
        if "single_input_configs" in config and config["single_input_configs"]:
            specs["example_config"] = config["single_input_configs"][0]
        elif "multi_input_configs" in config and config["multi_input_configs"]:
            specs["example_config"] = config["multi_input_configs"][0]

        # Get shared config
        if "single_shared_configs" in config and config["single_shared_configs"]:
            specs["shared_config"] = config["single_shared_configs"][0]
        elif "multi_shared_configs" in config and config["multi_shared_configs"]:
            specs["shared_config"] = config["multi_shared_configs"][0]

        return specs

    def _infer_output_spec(self, docstring: str, fn) -> str:
        """Infer output specification from docstring or function signature."""
        # Try to extract return type from docstring
        lines = docstring.split("\n")
        for line in lines:
            line_lower = line.lower().strip()
            if "return" in line_lower or "output" in line_lower:
                return line.strip()

        # Default output spec
        return "Returns a torch.Tensor with the computed result."

    def sample(self, n: int = 1) -> List[KernelTask]:
        """Sample n tasks randomly."""
        if n > len(self.tasks):
            print(f"Warning: Requested {n} tasks but only {len(self.tasks)} available.")
            n = len(self.tasks)

        return random.sample(self.tasks, n)

    def sample_batch(self, batch_size: int) -> List[KernelTask]:
        """Sample a batch of tasks, with replacement if necessary."""
        if len(self.tasks) >= batch_size:
            return random.sample(self.tasks, batch_size)
        else:
            # Sample with replacement
            return random.choices(self.tasks, k=batch_size)

    def get_task_by_name(self, name: str) -> Optional[KernelTask]:
        """Get a specific task by name."""
        for task in self.tasks:
            if task.name == name:
                return task
        return None

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)
