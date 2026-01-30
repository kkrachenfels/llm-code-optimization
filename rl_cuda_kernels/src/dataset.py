"""
KernelBench dataset loading and preprocessing.
"""

import re
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from datasets import load_dataset, Dataset, DatasetDict
import torch

from .config import DatasetConfig


@dataclass
class KernelTask:
    """Represents a single kernel optimization task."""
    task_id: str
    name: str
    level: int
    pytorch_code: str
    get_inputs_code: str
    get_init_inputs_code: str
    model_class_code: str

    # Extracted metadata
    input_shapes: Optional[List[Tuple[int, ...]]] = None
    batch_size: Optional[int] = None


def parse_pytorch_code(code: str) -> Dict[str, str]:
    """
    Parse the PyTorch code to extract components.

    Returns dict with keys:
    - model_class: The Model class definition
    - get_inputs: The get_inputs function
    - get_init_inputs: The get_init_inputs function
    - setup_code: Any setup code (batch_size, dimensions, etc.)
    """
    components = {
        'model_class': '',
        'get_inputs': '',
        'get_init_inputs': '',
        'setup_code': ''
    }

    # Extract Model class
    model_pattern = r'(class Model\(.*?\):.*?)(?=\n(?:def |[A-Za-z_][A-Za-z0-9_]*\s*=|\Z))'
    model_match = re.search(model_pattern, code, re.DOTALL)
    if model_match:
        components['model_class'] = model_match.group(1).strip()

    # Extract get_inputs function
    get_inputs_pattern = r'(def get_inputs\(\).*?)(?=\ndef |\Z)'
    get_inputs_match = re.search(get_inputs_pattern, code, re.DOTALL)
    if get_inputs_match:
        components['get_inputs'] = get_inputs_match.group(1).strip()

    # Extract get_init_inputs function
    get_init_pattern = r'(def get_init_inputs\(\).*?)(?=\ndef |\Z)'
    get_init_match = re.search(get_init_pattern, code, re.DOTALL)
    if get_init_match:
        components['get_init_inputs'] = get_init_match.group(1).strip()

    # Extract setup code (variable assignments before functions)
    setup_lines = []
    in_class = False
    in_function = False

    for line in code.split('\n'):
        stripped = line.strip()

        if stripped.startswith('class '):
            in_class = True
        elif stripped.startswith('def '):
            in_function = True
        elif not stripped.startswith(' ') and not stripped.startswith('\t'):
            if stripped and not stripped.startswith('#') and not stripped.startswith('import'):
                if '=' in stripped and not in_class and not in_function:
                    setup_lines.append(line)
            in_class = False
            in_function = False

    components['setup_code'] = '\n'.join(setup_lines)

    return components


def enlarge_tensor_dimensions(code: str, min_elements: int = 1_000_000) -> str:
    """
    Enlarge tensor dimensions to avoid kernel-launch overhead measurement bias.
    This addresses a key correction mentioned in the Kevin-32B paper.
    """
    # Find dimension variables and scale them up
    dim_patterns = [
        (r'\bN\s*=\s*(\d+)', 'N'),
        (r'\bM\s*=\s*(\d+)', 'M'),
        (r'\bK\s*=\s*(\d+)', 'K'),
        (r'\bbatch_size\s*=\s*(\d+)', 'batch_size'),
        (r'\bin_features\s*=\s*(\d+)', 'in_features'),
        (r'\bout_features\s*=\s*(\d+)', 'out_features'),
        (r'\bseq_length\s*=\s*(\d+)', 'seq_length'),
        (r'\bhidden_size\s*=\s*(\d+)', 'hidden_size'),
        (r'\bwidth\s*=\s*(\d+)', 'width'),
        (r'\bheight\s*=\s*(\d+)', 'height'),
    ]

    modified_code = code

    for pattern, var_name in dim_patterns:
        match = re.search(pattern, modified_code)
        if match:
            old_val = int(match.group(1))
            # Scale up small dimensions
            if old_val < 512:
                new_val = max(old_val * 4, 512)
                modified_code = re.sub(
                    pattern,
                    f'{var_name} = {new_val}',
                    modified_code,
                    count=1
                )

    return modified_code


class KernelBenchDataset:
    """
    Wrapper for the KernelBench dataset with preprocessing.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self._raw_dataset: Optional[DatasetDict] = None
        self._tasks: Dict[str, KernelTask] = {}

    def load(self) -> None:
        """Load the dataset from HuggingFace."""
        print(f"Loading KernelBench dataset from {self.config.dataset_name}...")
        self._raw_dataset = load_dataset(
            self.config.dataset_name,
            cache_dir=self.config.cache_dir
        )
        self._process_tasks()
        print(f"Loaded {len(self._tasks)} tasks from levels {self.config.train_levels}")

    def _process_tasks(self) -> None:
        """Process raw dataset into KernelTask objects."""
        for level in self.config.train_levels:
            split_name = f"level_{level}"
            if split_name not in self._raw_dataset:
                print(f"Warning: Split {split_name} not found in dataset")
                continue

            for item in self._raw_dataset[split_name]:
                task_id = f"level{level}_{item['problem_id']}_{item['name']}"

                code = item['code']
                if self.config.enlarge_tensor_dims:
                    code = enlarge_tensor_dimensions(
                        code,
                        min_elements=self.config.min_tensor_elements
                    )

                components = parse_pytorch_code(code)

                task = KernelTask(
                    task_id=task_id,
                    name=item['name'],
                    level=level,
                    pytorch_code=code,
                    get_inputs_code=components['get_inputs'],
                    get_init_inputs_code=components['get_init_inputs'],
                    model_class_code=components['model_class']
                )
                self._tasks[task_id] = task

    def get_train_test_split(
        self,
        seed: int = 42
    ) -> Tuple[List[KernelTask], List[KernelTask]]:
        """
        Split tasks into train and test sets.
        Following Kevin-32B: 180 training, 20 holdout from levels 1-2.
        """
        all_tasks = list(self._tasks.values())
        random.seed(seed)
        random.shuffle(all_tasks)

        split_idx = int(len(all_tasks) * self.config.train_split_ratio)
        train_tasks = all_tasks[:split_idx]
        test_tasks = all_tasks[split_idx:]

        return train_tasks, test_tasks

    def get_task(self, task_id: str) -> Optional[KernelTask]:
        """Get a specific task by ID."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[KernelTask]:
        """Get all tasks."""
        return list(self._tasks.values())

    def get_tasks_by_level(self, level: int) -> List[KernelTask]:
        """Get all tasks for a specific difficulty level."""
        return [t for t in self._tasks.values() if t.level == level]

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks.values())


def create_training_dataset(
    tasks: List[KernelTask],
    tokenizer: Any,
    prompt_config: Any,
    max_length: int = 4096
) -> Dataset:
    """
    Create a HuggingFace Dataset for training from KernelTask list.
    """
    data = {
        'task_id': [],
        'prompt': [],
        'pytorch_code': [],
        'level': []
    }

    for task in tasks:
        prompt = prompt_config.task_prompt_template.format(
            pytorch_code=task.pytorch_code,
            gpu_type="GPU"  # Will be filled at runtime
        )

        data['task_id'].append(task.task_id)
        data['prompt'].append(prompt)
        data['pytorch_code'].append(task.pytorch_code)
        data['level'].append(task.level)

    return Dataset.from_dict(data)


def collate_kernel_tasks(
    tasks: List[KernelTask],
    batch_size: int
) -> List[List[KernelTask]]:
    """
    Collate tasks into batches.
    """
    batches = []
    for i in range(0, len(tasks), batch_size):
        batches.append(tasks[i:i + batch_size])
    return batches


if __name__ == "__main__":
    # Test dataset loading
    config = DatasetConfig()
    dataset = KernelBenchDataset(config)
    dataset.load()

    train_tasks, test_tasks = dataset.get_train_test_split()
    print(f"Train tasks: {len(train_tasks)}")
    print(f"Test tasks: {len(test_tasks)}")

    # Show sample task
    if train_tasks:
        sample = train_tasks[0]
        print(f"\nSample task: {sample.name}")
        print(f"Level: {sample.level}")
        print(f"Code preview:\n{sample.pytorch_code[:500]}...")
