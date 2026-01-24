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

                # Look for the main .c file (same name as directory)
                c_files = list(benchmark_dir.glob('*.c'))
                if not c_files:
                    continue

                # Prefer the file matching the directory name
                c_file = None
                for f in c_files:
                    if f.stem == benchmark_dir.name:
                        c_file = f
                        break
                if c_file is None:
                    c_file = c_files[0]

                try:
                    with open(c_file, 'r') as f:
                        code = f.read()

                    # PolybenchC requires special compiler configuration
                    compiler_config = {
                        'compiler': 'gcc',
                        'flags': ['-O2', '-std=c11'],
                        'include_paths': [
                            str(self.utilities_dir),
                            str(benchmark_dir)  # Include benchmark's own directory for .h files
                        ],
                        'additional_sources': [str(self.polybench_c)],
                        'defines': {'POLYBENCH_TIME': None},
                        'output_is_seconds': True,
                        'linker_flags': ['-lm'],
                        'kernel_markers': ('#pragma scop', '#pragma endscop'),
                        'correctness_config': {
                            'compiler': 'gcc',
                            'flags': ['-O2', '-std=c11'],
                            'include_paths': [
                                str(self.utilities_dir),
                                str(benchmark_dir),
                            ],
                            'additional_sources': [str(self.polybench_c)],
                            'defines': {'POLYBENCH_DUMP_ARRAYS': None, 'MINI_DATASET': None},
                            'linker_flags': ['-lm'],
                        },
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


class SVCompDataset(CodeDataset):
    """Dataset loader for SV-COMP benchmarks (simple C verification programs)."""

    # Stub implementations for __VERIFIER_* functions and timing wrapper
    VERIFIER_STUBS = '''
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

// Verifier stubs - provide deterministic values for training
int __VERIFIER_nondet_int(void) { return 42; }
unsigned int __VERIFIER_nondet_uint(void) { return 42u; }
long __VERIFIER_nondet_long(void) { return 42L; }
unsigned long __VERIFIER_nondet_ulong(void) { return 42UL; }
short __VERIFIER_nondet_short(void) { return 42; }
unsigned short __VERIFIER_nondet_ushort(void) { return 42; }
char __VERIFIER_nondet_char(void) { return 'A'; }
unsigned char __VERIFIER_nondet_uchar(void) { return 65; }
_Bool __VERIFIER_nondet_bool(void) { return 1; }
float __VERIFIER_nondet_float(void) { return 1.0f; }
double __VERIFIER_nondet_double(void) { return 1.0; }
void *__VERIFIER_nondet_pointer(void) { return NULL; }

void __VERIFIER_assume(int cond) { if (!cond) exit(0); }
void __VERIFIER_error(void) { exit(1); }
void __VERIFIER_assert(int cond) { if (!cond) exit(1); }

// Stub for reach_error (used in many SV-COMP programs)
void reach_error(void) { exit(1); }

// Stub for assume_abort_if_not
void assume_abort_if_not(int cond) { if (!cond) exit(0); }
'''

    TIMING_WRAPPER_START = '''
// Timing wrapper
static struct timespec __start_time, __end_time;
static void __attribute__((constructor)) __start_timer(void) {
    clock_gettime(CLOCK_MONOTONIC, &__start_time);
}
static void __attribute__((destructor)) __end_timer(void) {
    clock_gettime(CLOCK_MONOTONIC, &__end_time);
    double elapsed = (__end_time.tv_sec - __start_time.tv_sec) * 1e6 +
                     (__end_time.tv_nsec - __start_time.tv_nsec) / 1e3;
    printf("%f\\n", elapsed);
}
'''

    def __init__(
        self,
        svcomp_dir: str,
        categories: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize SV-COMP dataset.

        Args:
            svcomp_dir: Path to sv-benchmarks/c directory
            categories: List of category subdirectories to load (default: simple ones)
            seed: Random seed for reproducibility
        """
        super().__init__(seed)
        self.svcomp_dir = Path(svcomp_dir)

        # Default to simpler categories if not specified
        self.categories = categories or [
            'loop-simple',
            'recursive-simple',
            'loops',
            'nla-digbench',
        ]
        self._load_programs()

    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess SV-COMP code to make it compilable and runnable.

        - Remove existing extern declarations for __VERIFIER_* functions
        - Add our stubs and timing wrapper
        """
        lines = code.split('\n')
        filtered_lines = []
        skip_patterns = [
            'extern void __assert_fail',
            'extern int __VERIFIER_nondet',
            'extern unsigned int __VERIFIER_nondet',
            'extern long __VERIFIER_nondet',
            'extern unsigned long __VERIFIER_nondet',
            'extern short __VERIFIER_nondet',
            'extern char __VERIFIER_nondet',
            'extern _Bool __VERIFIER_nondet',
            'extern float __VERIFIER_nondet',
            'extern double __VERIFIER_nondet',
            'extern void *__VERIFIER_nondet',
            'extern void abort',
            'extern void reach_error',
            'extern void __VERIFIER_error',
            'extern void __VERIFIER_assume',
            'void reach_error()',
            'void assume_abort_if_not(',
            'void __VERIFIER_assert(',
            '#include <assert.h>',
        ]

        in_reach_error_def = False
        in_assume_abort_def = False
        in_verifier_assert_def = False
        brace_count = 0

        for line in lines:
            # Skip extern declarations we're replacing
            skip = False
            for pattern in skip_patterns:
                if pattern in line:
                    # Check if this is a function definition we need to skip entirely
                    if 'void reach_error()' in line or 'void assume_abort_if_not(' in line or 'void __VERIFIER_assert(' in line:
                        if '{' in line:
                            brace_count = line.count('{') - line.count('}')
                            if brace_count > 0:
                                if 'reach_error' in line:
                                    in_reach_error_def = True
                                elif 'assume_abort_if_not' in line:
                                    in_assume_abort_def = True
                                elif '__VERIFIER_assert' in line:
                                    in_verifier_assert_def = True
                    skip = True
                    break

            # Skip function body if we're inside a function definition to skip
            if in_reach_error_def or in_assume_abort_def or in_verifier_assert_def:
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    in_reach_error_def = False
                    in_assume_abort_def = False
                    in_verifier_assert_def = False
                continue

            if not skip:
                filtered_lines.append(line)

        # Combine with stubs and timing
        processed_code = self.VERIFIER_STUBS + self.TIMING_WRAPPER_START + '\n' + '\n'.join(filtered_lines)
        return processed_code

    def _load_programs(self):
        """Load all SV-COMP programs from selected categories."""
        if not self.svcomp_dir.exists():
            raise FileNotFoundError(
                f"SV-COMP directory not found: {self.svcomp_dir}\n"
                f"Please clone: git clone https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks.git"
            )

        for category in self.categories:
            category_path = self.svcomp_dir / category
            if not category_path.exists():
                logger.warning(f"Category not found: {category_path}")
                continue

            # Load all .c files in the category
            for c_file in category_path.glob('*.c'):
                try:
                    with open(c_file, 'r') as f:
                        raw_code = f.read()

                    # Preprocess to add stubs and timing
                    processed_code = self._preprocess_code(raw_code)

                    # Use gcc for C files with appropriate flags
                    compiler_config = {
                        'compiler': 'gcc',
                        'flags': ['-O2', '-std=c11'],
                        'linker_flags': ['-lm', '-lrt'],
                    }

                    self.programs.append({
                        'code': processed_code,
                        'name': c_file.stem,
                        'path': str(c_file),
                        'category': category,
                        'compiler_config': compiler_config,
                        'raw_code': raw_code,  # Keep original for reference
                    })
                except Exception as e:
                    logger.warning(f"Failed to load {c_file}: {e}")

        if not self.programs:
            raise ValueError(
                f"No programs found in {self.svcomp_dir} for categories: {self.categories}"
            )

        logger.info(f"Loaded {len(self.programs)} programs from SV-COMP ({', '.join(self.categories)})")


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
        dataset_type: Type of dataset ('polybench', 'directory', 'svcomp', 'single')
        **kwargs: Arguments passed to dataset constructor

    Returns:
        CodeDataset instance
    """
    if dataset_type == 'polybench':
        return PolybenchDataset(**kwargs)
    elif dataset_type == 'directory':
        return DirectoryDataset(**kwargs)
    elif dataset_type == 'svcomp':
        return SVCompDataset(**kwargs)
    elif dataset_type == 'single':
        return SingleProgramDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
