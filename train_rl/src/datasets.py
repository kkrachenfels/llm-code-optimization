"""Dataset classes for loading C/C++ programs for RL training."""

from pathlib import Path
from typing import List, Dict, Optional
import random
import logging

logger = logging.getLogger(__name__)


class CodeDataset:
    """Base class for code datasets."""

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the dataset.

        Args:
            seed: Random seed for reproducibility
        """
        self.programs: List[Dict[str, str]] = []
        self.current_index = 0
        if seed is not None:
            random.seed(seed)

    def __len__(self) -> int:
        """Return number of programs in dataset."""
        return len(self.programs)

    def get_program(self, index: int) -> Dict[str, str]:
        """
        Get a specific program by index.

        Returns:
            Dict with keys: 'code', 'name', 'path'
        """
        if index < 0 or index >= len(self.programs):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self)}")
        return self.programs[index]

    def get_random_program(self) -> Dict[str, str]:
        """Get a random program from the dataset."""
        return random.choice(self.programs)

    def get_next_program(self) -> Dict[str, str]:
        """Get the next program in sequence (cycles through dataset)."""
        program = self.programs[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.programs)
        return program


class PolybenchDataset(CodeDataset):
    """Dataset loader for PolybenchC benchmarks."""

    def __init__(self, polybench_dir: str, seed: Optional[int] = None):
        """
        Initialize PolybenchC dataset.

        Args:
            polybench_dir: Path to PolybenchC directory
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.polybench_dir = Path(polybench_dir)
        self.utilities_dir = self.polybench_dir / "utilities"
        self.polybench_c = self.utilities_dir / "polybench.c"
        self._load_programs()

    def _load_programs(self):
        """Load all PolybenchC programs."""
        if not self.polybench_dir.exists():
            raise FileNotFoundError(
                f"PolybenchC directory not found: {self.polybench_dir}\n"
                f"Please run: python scripts/download_polybench.py"
            )

        # PolybenchC has subdirectories for different categories
        # (linear-algebra, datamining, stencils, etc.)
        categories = [
            'linear-algebra/blas',
            'linear-algebra/kernels',
            'linear-algebra/solvers',
            'datamining',
            'stencils',
            'medley'
        ]

        for category in categories:
            category_path = self.polybench_dir / category
            if not category_path.exists():
                continue

            # Each benchmark has its own directory with a .c file
            for benchmark_dir in category_path.iterdir():
                if not benchmark_dir.is_dir():
                    continue

                # Look for the main .c file (usually same name as directory)
                c_files = list(benchmark_dir.glob('*.c'))
                if not c_files:
                    continue

                # Use the first .c file found
                c_file = c_files[0]

                try:
                    with open(c_file, 'r') as f:
                        code = f.read()

                    # PolybenchC requires special compiler configuration
                    compiler_config = {
                        'include_paths': [
                            str(self.utilities_dir),
                            str(benchmark_dir)  # Include benchmark's own directory for .h files
                        ],
                        'additional_sources': [str(self.polybench_c)],
                        'defines': {'POLYBENCH_TIME': None},
                        'output_is_seconds': True
                    }

                    self.programs.append({
                        'code': code,
                        'name': benchmark_dir.name,
                        'path': str(c_file),
                        'category': category,
                        'compiler_config': compiler_config
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {c_file}: {e}")

        if not self.programs:
            raise ValueError(
                f"No programs found in {self.polybench_dir}. "
                f"Directory structure may be incorrect."
            )

        logger.info(f"Loaded {len(self.programs)} programs from PolybenchC")


class DirectoryDataset(CodeDataset):
    """Load C/C++ programs from a directory."""

    def __init__(
        self,
        directory: str,
        extensions: List[str] = ['.c', '.cpp', '.cc'],
        recursive: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize directory dataset.

        Args:
            directory: Path to directory containing C/C++ files
            extensions: List of file extensions to include
            recursive: Whether to search recursively
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.directory = Path(directory)
        self.extensions = extensions
        self.recursive = recursive
        self._load_programs()

    def _load_programs(self):
        """Load all C/C++ programs from directory."""
        if not self.directory.exists():
            raise FileNotFoundError(f"Directory not found: {self.directory}")

        # Find all matching files
        files = []
        if self.recursive:
            for ext in self.extensions:
                files.extend(self.directory.rglob(f'*{ext}'))
        else:
            for ext in self.extensions:
                files.extend(self.directory.glob(f'*{ext}'))

        # Load each file
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    code = f.read()

                self.programs.append({
                    'code': code,
                    'name': file_path.stem,
                    'path': str(file_path),
                    'compiler_config': {}  # Use default compiler settings
                })
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not self.programs:
            raise ValueError(f"No programs found in {self.directory}")

        logger.info(f"Loaded {len(self.programs)} programs from {self.directory}")


class SingleProgramDataset(CodeDataset):
    """Wrapper to use a single program as a dataset (for backwards compatibility)."""

    def __init__(self, program_path: str):
        """
        Initialize single program dataset.

        Args:
            program_path: Path to the program file
        """
        super().__init__()
        self.program_path = Path(program_path)
        self._load_program()

    def _load_program(self):
        """Load the single program."""
        if not self.program_path.exists():
            raise FileNotFoundError(f"Program not found: {self.program_path}")

        with open(self.program_path, 'r') as f:
            code = f.read()

        self.programs.append({
            'code': code,
            'name': self.program_path.stem,
            'path': str(self.program_path),
            'compiler_config': {}  # Use default compiler settings
        })


def create_dataset(dataset_type: str, **kwargs) -> CodeDataset:
    """
    Factory function to create datasets.

    Args:
        dataset_type: Type of dataset ('polybench', 'directory', 'single')
        **kwargs: Arguments passed to dataset constructor

    Returns:
        CodeDataset instance
    """
    if dataset_type == 'polybench':
        return PolybenchDataset(**kwargs)
    elif dataset_type == 'directory':
        return DirectoryDataset(**kwargs)
    elif dataset_type == 'single':
        return SingleProgramDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
