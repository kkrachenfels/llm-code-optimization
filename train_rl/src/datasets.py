"""Dataset classes for loading C/C++ programs for RL training."""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import random
import logging
import re

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


class TSVCDataset(CodeDataset):
    """Dataset loader for TSVC (Test Suite for Vectorizing Compilers) kernels."""

    PROGRAM_TEMPLATE = '''\
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define LEN 32000
#define LEN2 256
typedef double TYPE;

{array_declarations}

volatile TYPE vol_sink;  // prevents dead code elimination

{helper_functions}

static void init_arrays(void) {{
{init_code}
}}

{checksum_function}

int main(void) {{
    init_arrays();
    struct timespec start, end;

    // Outer timing loop for more stable measurements
    int timing_runs = {timing_runs};
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int run = 0; run < timing_runs; run++) {{
        int ntimes = {ntimes};
        for (int nl = 0; nl < {nl_bound}; nl++) {{
{loop_body}
            vol_sink = a[0];
        }}
    }}

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) * 1e6 +
                     (end.tv_nsec - start.tv_nsec) / 1e3;
    // Report average time per run
    printf("%f\\n", elapsed / timing_runs);

    // Print checksums to stderr for correctness verification
    print_checksums();
    return 0;
}}
'''

    # All 1D arrays used in TSVC
    ARRAYS_1D = {'a', 'b', 'c', 'd', 'e', 'x'}
    # All 2D arrays used in TSVC
    ARRAYS_2D = {'aa', 'bb', 'cc', 'tt'}
    # Other arrays
    ARRAYS_SPECIAL = {'array', 'indx'}

    # Kernels known to not compile (skip these instead of compile-checking each run)
    KNOWN_UNCOMPILABLE_KERNELS = {
        's000', 's121', 's122', 's123', 's124', 's125', 's126', 's127', 's128',
        's131', 's132', 's141', 's151', 's152', 's173', 's174', 's176', 's231',
        's233', 's2233', 's235', 's251', 's1251', 's252', 's253', 's254', 's255',
        's258', 's261', 's2275', 's276', 's2710', 's281', 's1281', 's291', 's292',
        's311', 's31111', 's312', 's313', 's314', 's315', 's316', 's317', 's319',
        's3110', 's13110', 's3111', 's3112', 's3113', 's331', 's341', 's342',
        's343', 's351', 's1351', 's352', 's421', 's1421', 's422', 's423', 's424',
        's431', 's451', 's453', 's471',
    }

    def __init__(
        self,
        tsvc_dir: str,
        seed: Optional[int] = None,
        ntimes: int = 50000,
        timing_runs: int = 5,
        categories: Optional[List[str]] = None,
        skip_compile_check: bool = True
    ):
        """
        Initialize TSVC dataset.

        Args:
            tsvc_dir: Path to TSVC directory (or llvm-test-suite root)
            seed: Random seed for reproducibility
            ntimes: Iteration count per timing run (inner loop, default: 50000)
            timing_runs: Number of times to run the kernel for timing (outer loop, default: 5)
            categories: Optional filter for kernel categories
            skip_compile_check: Skip compile-checking kernels (uses KNOWN_UNCOMPILABLE_KERNELS only)
        """
        super().__init__(seed)
        self.ntimes = ntimes
        self.timing_runs = timing_runs
        self.categories = categories
        self.skip_compile_check = skip_compile_check

        # Handle both direct TSVC dir and llvm-test-suite root
        tsvc_path = Path(tsvc_dir)
        if (tsvc_path / 'tsc.inc').exists():
            self.tsvc_dir = tsvc_path
        elif (tsvc_path / 'MultiSource' / 'Benchmarks' / 'TSVC' / 'tsc.inc').exists():
            self.tsvc_dir = tsvc_path / 'MultiSource' / 'Benchmarks' / 'TSVC'
        else:
            raise FileNotFoundError(
                f"Cannot find tsc.inc in {tsvc_dir}.\n"
                f"Expected at {tsvc_path}/tsc.inc or "
                f"{tsvc_path}/MultiSource/Benchmarks/TSVC/tsc.inc"
            )

        self._load_programs()

    def _parse_init_function(self, content: str) -> Dict[str, List[Tuple]]:
        """
        Parse the init() function to build a map of kernel_name -> init specs.

        Returns:
            Dict mapping kernel name to list of (func, array, value, stride) tuples
        """
        init_map = {}

        # Find the init function
        init_match = re.search(r'int init\(char\* name\)\s*\{', content)
        if not init_match:
            return init_map

        # Extract init function body
        start = init_match.end()
        brace_depth = 1
        pos = start
        while pos < len(content) and brace_depth > 0:
            if content[pos] == '{':
                brace_depth += 1
            elif content[pos] == '}':
                brace_depth -= 1
            pos += 1
        init_body = content[start:pos-1]

        # Parse each if/else-if block
        # Pattern: if (!strcmp(name, "sNNN ")) { set1d(...); ... }
        block_pattern = re.compile(
            r'if\s*\(\s*!strcmp\s*\(\s*name\s*,\s*"([^"]+)"\s*\)\s*\)\s*\{([^}]*)\}',
            re.DOTALL
        )

        # Value name mapping from init() local vars
        value_map = {
            'any': 0.0, 'zero': 0.0, 'half': 0.5, 'one': 1.0,
            'two': 2.0, 'small': 0.000001,
        }
        stride_map = {
            'unit': 1, 'frac': -1, 'frac2': -2,
        }

        for match in block_pattern.finditer(init_body):
            kernel_name = match.group(1).strip()
            block_body = match.group(2)

            specs = []
            # Parse set1d/set2d/set1ds calls
            set_pattern = re.compile(
                r'(set1d|set2d|set1ds)\s*\(([^)]+)\)'
            )
            for set_match in set_pattern.finditer(block_body):
                func = set_match.group(1)
                args_str = set_match.group(2)
                args = [a.strip() for a in args_str.split(',')]

                if func == 'set1ds':
                    # set1ds(n, arr, value, stride)
                    if len(args) >= 4:
                        arr = args[1].lstrip('&').split('[')[0]
                        val = value_map.get(args[2], 0.0)
                        stride = stride_map.get(args[3], 1)
                        specs.append((func, arr, val, stride))
                else:
                    # set1d(arr, value, stride) or set2d(arr, value, stride)
                    if len(args) >= 3:
                        arr = args[0].lstrip('&').split('[')[0]
                        val = value_map.get(args[1], 0.0)
                        stride = stride_map.get(args[2], 1)
                        specs.append((func, arr, val, stride))

            init_map[kernel_name] = specs

        return init_map

    def _generate_init_code(self, specs: List[Tuple], used_arrays: set) -> str:
        """Generate initialization code from parsed specs."""
        lines = []
        for func, arr, val, stride in specs:
            if arr not in used_arrays:
                continue

            if arr in self.ARRAYS_2D:
                # 2D array initialization
                if stride == -1:
                    lines.append(f'    for (int i = 0; i < LEN2; i++)')
                    lines.append(f'        for (int j = 0; j < LEN2; j++)')
                    lines.append(f'            {arr}[i][j] = 1.0 / (TYPE)(i + 1);')
                elif stride == -2:
                    lines.append(f'    for (int i = 0; i < LEN2; i++)')
                    lines.append(f'        for (int j = 0; j < LEN2; j++)')
                    lines.append(f'            {arr}[i][j] = 1.0 / (TYPE)((i + 1) * (i + 1));')
                else:
                    lines.append(f'    for (int i = 0; i < LEN2; i++)')
                    lines.append(f'        for (int j = 0; j < LEN2; j += {stride})')
                    lines.append(f'            {arr}[i][j] = {val};')
            elif arr == 'array':
                if stride == -1:
                    lines.append(f'    for (int i = 0; i < LEN2*LEN2; i++)')
                    lines.append(f'        array[i] = 1.0 / (TYPE)(i + 1);')
                elif stride == -2:
                    lines.append(f'    for (int i = 0; i < LEN2*LEN2; i++)')
                    lines.append(f'        array[i] = 1.0 / (TYPE)((i + 1) * (i + 1));')
                else:
                    lines.append(f'    for (int i = 0; i < LEN2*LEN2; i += {stride})')
                    lines.append(f'        array[i] = {val};')
            else:
                # 1D array
                if stride == -1:
                    lines.append(f'    for (int i = 0; i < LEN; i++)')
                    lines.append(f'        {arr}[i] = 1.0 / (TYPE)(i + 1);')
                elif stride == -2:
                    lines.append(f'    for (int i = 0; i < LEN; i++)')
                    lines.append(f'        {arr}[i] = 1.0 / (TYPE)((i + 1) * (i + 1));')
                else:
                    lines.append(f'    for (int i = 0; i < LEN; i += {stride})')
                    lines.append(f'        {arr}[i] = {val};')

        if not lines:
            lines.append('    for (int i = 0; i < LEN; i++) a[i] = 1.0 / (TYPE)(i + 1);')

        return '\n'.join(lines)

    def _detect_used_arrays(self, body: str) -> set:
        """Detect which arrays are referenced in the loop body."""
        used = set()
        # Check 1D arrays
        for arr in self.ARRAYS_1D:
            if re.search(rf'\b{arr}\s*\[', body):
                used.add(arr)
        # Check 2D arrays
        for arr in self.ARRAYS_2D:
            if re.search(rf'\b{arr}\s*\[', body):
                used.add(arr)
        # Check special arrays
        if re.search(r'\barray\s*\[', body):
            used.add('array')
        if re.search(r'\bindx\s*\[', body):
            used.add('indx')
        # Always include 'a' for vol_sink
        used.add('a')
        return used

    def _generate_array_declarations(self, used_arrays: set) -> str:
        """Generate array declarations for used arrays."""
        decls = []
        for arr in sorted(used_arrays):
            if arr in self.ARRAYS_1D:
                decls.append(f'static TYPE {arr}[LEN] __attribute__((aligned(32)));')
            elif arr in self.ARRAYS_2D:
                decls.append(f'static TYPE {arr}[LEN2][LEN2] __attribute__((aligned(32)));')
            elif arr == 'array':
                decls.append(f'static TYPE array[LEN2*LEN2] __attribute__((aligned(32)));')
            elif arr == 'indx':
                decls.append(f'static int indx[LEN] __attribute__((aligned(32)));')
        return '\n'.join(decls)

    def _generate_checksum_function(self, used_arrays: set) -> str:
        """Generate a function that prints checksums of all used arrays."""
        lines = ['static void print_checksums(void) {']

        for arr in sorted(used_arrays):
            if arr in self.ARRAYS_1D:
                lines.append(f'    {{ double sum = 0.0; for (int i = 0; i < LEN; i++) sum += {arr}[i]; fprintf(stderr, "{arr}=%.10e\\n", sum); }}')
            elif arr in self.ARRAYS_2D:
                lines.append(f'    {{ double sum = 0.0; for (int i = 0; i < LEN2; i++) for (int j = 0; j < LEN2; j++) sum += {arr}[i][j]; fprintf(stderr, "{arr}=%.10e\\n", sum); }}')
            elif arr == 'array':
                lines.append(f'    {{ double sum = 0.0; for (int i = 0; i < LEN2*LEN2; i++) sum += array[i]; fprintf(stderr, "array=%.10e\\n", sum); }}')
            elif arr == 'indx':
                lines.append(f'    {{ long sum = 0; for (int i = 0; i < LEN; i++) sum += indx[i]; fprintf(stderr, "indx=%ld\\n", sum); }}')

        lines.append('}')
        return '\n'.join(lines)

    def _extract_kernels(self, content: str) -> List[Dict]:
        """
        Extract kernel functions from tsc.inc content.

        Returns:
            List of dicts with: name, comment, body, nl_bound, params
        """
        kernels = []
        lines = content.split('\n')

        # Find all kernel function starts
        func_pattern = re.compile(r'^int (s\d+)\s*\(([^)]*)\)\s*\{?\s*$')

        i = 0
        while i < len(lines):
            match = func_pattern.match(lines[i])
            if not match:
                i += 1
                continue

            name = match.group(1)
            params_str = match.group(2).strip()

            # Skip kernels with pointer parameters
            if '*' in params_str:
                i += 1
                continue

            # Extract the function body by tracking braces
            func_start = i
            # Find opening brace
            brace_line = i
            while brace_line < len(lines) and '{' not in lines[brace_line]:
                brace_line += 1
            if '{' in lines[i]:
                brace_line = i

            brace_depth = 0
            func_body_lines = []
            j = brace_line
            while j < len(lines):
                line = lines[j]
                brace_depth += line.count('{') - line.count('}')
                func_body_lines.append(line)
                if brace_depth <= 0:
                    break
                j += 1

            func_body = '\n'.join(func_body_lines)

            # Skip kernels with goto or exit
            if 'goto ' in func_body or 'exit(' in func_body:
                i = j + 1
                continue

            # Skip kernels known to not compile
            if name in self.KNOWN_UNCOMPILABLE_KERNELS:
                i = j + 1
                continue

            # Extract comment above function
            comment_lines = []
            ci = func_start - 1
            while ci >= 0 and (lines[ci].strip().startswith('//') or lines[ci].strip() == ''):
                if lines[ci].strip().startswith('//'):
                    comment_lines.insert(0, lines[ci].strip().lstrip('/ '))
                ci -= 1

            comment = ' '.join(comment_lines).strip()

            # Find the outer nl loop and extract the body
            nl_match = re.search(
                r'for\s*\(\s*int\s+nl\s*=\s*0\s*;\s*nl\s*<\s*([^;]+);\s*nl\+\+\s*\)\s*\{',
                func_body
            )
            if not nl_match:
                i = j + 1
                continue

            nl_bound_expr = nl_match.group(1).strip()

            # Find the nl loop body
            nl_loop_start = nl_match.end()
            # Find the position in func_body
            nl_body_start = nl_loop_start
            nl_brace_depth = 1
            pos = nl_body_start
            while pos < len(func_body) and nl_brace_depth > 0:
                if func_body[pos] == '{':
                    nl_brace_depth += 1
                elif func_body[pos] == '}':
                    nl_brace_depth -= 1
                pos += 1
            nl_body = func_body[nl_body_start:pos-1]

            # Remove the dummy() call and everything after it within the loop
            dummy_match = re.search(r'\n\s*dummy\s*\([^)]*\)\s*;', nl_body)
            if dummy_match:
                nl_body = nl_body[:dummy_match.start()]

            # Clean up the body - strip leading/trailing whitespace
            body_lines = nl_body.strip().split('\n')
            # Normalize indentation to 8 spaces (2 levels inside main+for)
            cleaned_lines = []
            for line in body_lines:
                stripped = line.strip()
                if stripped:
                    cleaned_lines.append('        ' + stripped)
                else:
                    cleaned_lines.append('')

            if not cleaned_lines:
                i = j + 1
                continue

            kernels.append({
                'name': name,
                'comment': comment,
                'body': '\n'.join(cleaned_lines),
                'nl_bound': nl_bound_expr,
                'params': params_str,
            })

            i = j + 1

        return kernels

    def _load_programs(self):
        """Load and parse TSVC kernels into standalone programs."""
        tsc_path = self.tsvc_dir / 'tsc.inc'
        if not tsc_path.exists():
            raise FileNotFoundError(f"tsc.inc not found at {tsc_path}")

        with open(tsc_path, 'r') as f:
            content = f.read()

        # Parse initialization specs
        init_map = self._parse_init_function(content)

        # Extract kernel functions
        kernels = self._extract_kernels(content)

        for kernel in kernels:
            name = kernel['name']
            body = kernel['body']
            nl_bound = kernel['nl_bound']
            params = kernel['params']

            # Detect used arrays
            used_arrays = self._detect_used_arrays(body)

            # Get init specs for this kernel
            init_key = name + ' ' * (5 - len(name)) if len(name) < 5 else name
            # Try various key formats
            specs = init_map.get(init_key) or init_map.get(name + ' ') or init_map.get(name) or []

            # Generate init code
            init_code = self._generate_init_code(specs, used_arrays)

            # Generate array declarations
            array_decls = self._generate_array_declarations(used_arrays)

            # Handle parameters by substituting hardcoded values
            body_substituted = body
            if params:
                # Replace simple scalar params with constants
                param_values = {
                    'n1': '1', 'n3': '1', 's1': '1.0', 's2': '2.0',
                    'k': '1', 'inc': '1', 'M': 'LEN',
                    't': '1.0', 's': '1.0',
                }
                for pname, pval in param_values.items():
                    body_substituted = re.sub(
                        rf'\b{pname}\b', pval, body_substituted
                    )
                nl_bound = re.sub(r'\b(n1|n3|k|inc|M)\b',
                                  lambda m: param_values.get(m.group(1), m.group(1)),
                                  nl_bound)

            # Handle nl_bound expression - substitute ntimes
            nl_bound_code = nl_bound.replace('ntimes', 'ntimes')

            # Check for helper function calls (s151s, s152s, s471s, f())
            helper_functions = ''
            if 's151s' in body_substituted:
                helper_functions += '''static int s151s(TYPE a_arr[LEN], TYPE b_arr[LEN], int m) {
    for (int i = 0; i < LEN-1; i++) {
        a_arr[i] = a_arr[i + m] + b_arr[i];
    }
    return 0;
}

'''
            if 's152s' in body_substituted:
                helper_functions += '''static int s152s(TYPE a_arr[LEN], TYPE b_arr[LEN], TYPE c_arr[LEN], int i) {
    a_arr[i] += b_arr[i] * c_arr[i];
    return 0;
}

'''
            if 's471s' in body_substituted:
                helper_functions += '''static inline int s471s(void) {
    return 0;
}

'''
            if re.search(r'\bf\s*\(', body_substituted):
                helper_functions += '''static inline TYPE f(TYPE fa, TYPE fb) {
    return fa * fb;
}

'''

            # Generate checksum function for correctness checking
            checksum_function = self._generate_checksum_function(used_arrays)

            # Generate the standalone program (includes timing and checksums)
            program = self.PROGRAM_TEMPLATE.format(
                array_declarations=array_decls,
                helper_functions=helper_functions,
                init_code=init_code,
                checksum_function=checksum_function,
                timing_runs=self.timing_runs,
                ntimes=self.ntimes,
                nl_bound=nl_bound_code,
                loop_body=body_substituted,
            )

            compiler_config = {
                'compiler': 'gcc',
                'flags': ['-O2', '-std=c11', '-lm', '-lrt'],
                # Correctness checking uses the same config - checksums are printed to stderr
                'correctness_config': {
                    'compiler': 'gcc',
                    'flags': ['-O2', '-std=c11', '-lm', '-lrt'],
                },
            }

            self.programs.append({
                'code': program,
                'name': name,
                'path': str(tsc_path),
                'category': 'tsvc',
                'comment': kernel['comment'],
                'compiler_config': compiler_config,
            })

        if not self.programs:
            raise ValueError(f"No kernels extracted from {tsc_path}")

        # Filter out kernels that fail to compile (skips known failures)
        if not self.skip_compile_check:
            self._filter_compilable_kernels()

        logger.info(f"Loaded {len(self.programs)} TSVC kernels")

    def _filter_compilable_kernels(self):
        """Filter out kernels that fail to compile.

        Kernels in KNOWN_UNCOMPILABLE_KERNELS are skipped during extraction.
        This method compile-checks the remaining kernels and logs any new
        failures so they can be added to KNOWN_UNCOMPILABLE_KERNELS.
        """
        from .compiler import CppCompiler

        valid_programs = []
        failed_names = []

        for prog in self.programs:
            config = prog.get('compiler_config', {})
            compiler_kwargs = {k: v for k, v in config.items()
                              if k not in ('kernel_markers', 'correctness_config')}
            compiler = CppCompiler(**compiler_kwargs)

            # Just test compilation, not runtime
            success, _, error = compiler.compile_and_run(prog['code'], num_runs=1)
            if success:
                valid_programs.append(prog)
            else:
                failed_names.append(prog['name'])

        if failed_names:
            logger.warning(
                f"Filtered out {len(failed_names)} non-compilable TSVC kernels: "
                f"{', '.join(failed_names)}"
            )
            logger.info(
                f"Add these to KNOWN_UNCOMPILABLE_KERNELS to skip compile-checking: "
                f"{set(failed_names)!r}"
            )

        self.programs = valid_programs

        if not self.programs:
            raise ValueError("No TSVC kernels passed compilation filter")


class CBenchDataset(CodeDataset):
    """Dataset loader for cBench benchmarks."""

    TIMING_PREFIX = '''\
#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

static struct timespec __cbench_start, __cbench_end;
static int __cbench_saved_stdout;

static void __attribute__((constructor)) __cbench_start_timer(void) {
    clock_gettime(CLOCK_MONOTONIC, &__cbench_start);
    // Redirect stdout to /dev/null to suppress benchmark output
    __cbench_saved_stdout = dup(STDOUT_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
        dup2(devnull, STDOUT_FILENO);
        close(devnull);
    }
}

static void __attribute__((destructor)) __cbench_end_timer(void) {
    clock_gettime(CLOCK_MONOTONIC, &__cbench_end);
    // Restore stdout
    dup2(__cbench_saved_stdout, STDOUT_FILENO);
    close(__cbench_saved_stdout);
    double elapsed = (__cbench_end.tv_sec - __cbench_start.tv_sec) * 1e6 +
                     (__cbench_end.tv_nsec - __cbench_start.tv_nsec) / 1e3;
    printf("%f\\n", elapsed);
}

'''

    # Benchmark configurations: benchmark_name -> config dict
    BENCHMARK_CONFIGS = {
        'automotive_bitcount': {
            'main_file': 'bitcnts.c',
            'description': 'Bit counting algorithms',
            'run_args': ['1000000'],
        },
        'automotive_basicmath': {
            'main_file': 'basicmath_small.c',
            'description': 'Basic math operations',
            'run_args': [],
        },
        'network_dijkstra': {
            'main_file': 'dijkstra_small.c',
            'description': 'Dijkstra shortest path algorithm',
            'run_args': [],  # uses input file, may need adjustment
        },
        'telecomm_CRC32': {
            'main_file': 'crc_32.c',
            'description': 'CRC32 checksum computation',
            'run_args': [],
        },
        'automotive_qsort1': {
            'main_file': 'qsort_small.c',
            'description': 'Quick sort algorithm',
            'run_args': [],
        },
        'security_sha': {
            'main_file': 'sha.c',
            'description': 'SHA hash computation',
            'run_args': [],
        },
        'security_blowfish': {
            'main_file': 'bf.c',
            'description': 'Blowfish encryption',
            'run_args': [],
        },
        'telecomm_adpcm': {
            'main_file': 'adpcm.c',
            'description': 'ADPCM audio compression',
            'run_args': [],
        },
    }

    def __init__(
        self,
        cbench_dir: str,
        seed: Optional[int] = None,
        benchmarks: Optional[List[str]] = None
    ):
        """
        Initialize cBench dataset.

        Args:
            cbench_dir: Path to cBench directory
            seed: Random seed for reproducibility
            benchmarks: Optional list of benchmark names to load (default: all available)
        """
        super().__init__(seed)
        self.cbench_dir = Path(cbench_dir)
        self.benchmarks = benchmarks
        self._load_programs()

    def _find_source_files(self, bench_dir: Path, main_file: str) -> Tuple[Optional[Path], List[str]]:
        """
        Find the main source file and additional sources in a benchmark directory.

        Returns:
            Tuple of (main_path, list of additional source paths)
        """
        # Look for source directory
        src_dir = bench_dir / 'src'
        if not src_dir.exists():
            src_dir = bench_dir

        main_path = src_dir / main_file
        if not main_path.exists():
            # Try searching recursively
            matches = list(bench_dir.rglob(main_file))
            if matches:
                main_path = matches[0]
                src_dir = main_path.parent
            else:
                return None, []

        # Find all other .c files in the same directory
        additional = []
        for c_file in src_dir.glob('*.c'):
            if c_file.name != main_file:
                additional.append(str(c_file))

        return main_path, additional

    def _load_programs(self):
        """Load cBench programs."""
        if not self.cbench_dir.exists():
            raise FileNotFoundError(
                f"cBench directory not found: {self.cbench_dir}\n"
                f"Please run: python scripts/download_cbench.py"
            )

        target_benchmarks = self.benchmarks or list(self.BENCHMARK_CONFIGS.keys())

        for bench_name in target_benchmarks:
            if bench_name not in self.BENCHMARK_CONFIGS:
                logger.warning(f"Unknown benchmark: {bench_name}, skipping")
                continue

            config = self.BENCHMARK_CONFIGS[bench_name]
            bench_dir = self.cbench_dir / bench_name

            if not bench_dir.exists():
                logger.warning(f"Benchmark directory not found: {bench_dir}")
                continue

            main_file = config['main_file']
            main_path, additional_sources = self._find_source_files(bench_dir, main_file)

            if main_path is None:
                logger.warning(f"Main file {main_file} not found for {bench_name}")
                continue

            try:
                with open(main_path, 'r') as f:
                    code = f.read()

                # Prepend timing code to the main source
                timed_code = self.TIMING_PREFIX + code

                src_dir = main_path.parent
                compiler_config = {
                    'compiler': 'gcc',
                    'flags': ['-O2', '-std=c11'],
                    'include_paths': [str(src_dir)],
                    'additional_sources': additional_sources,
                    'linker_flags': ['-lm', '-lrt'],
                    'run_args': config.get('run_args', []),
                }

                self.programs.append({
                    'code': timed_code,
                    'name': bench_name,
                    'path': str(main_path),
                    'category': 'cbench',
                    'description': config['description'],
                    'compiler_config': compiler_config,
                    'raw_code': code,
                })
            except Exception as e:
                logger.warning(f"Failed to load {bench_name}: {e}")

        if not self.programs:
            raise ValueError(
                f"No programs loaded from {self.cbench_dir}. "
                f"Check that benchmark directories exist."
            )

        logger.info(f"Loaded {len(self.programs)} cBench programs")


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
        dataset_type: Type of dataset ('polybench', 'directory', 'svcomp', 'tsvc', 'cbench', 'single')
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
    elif dataset_type == 'tsvc':
        return TSVCDataset(**kwargs)
    elif dataset_type == 'cbench':
        return CBenchDataset(**kwargs)
    elif dataset_type == 'single':
        return SingleProgramDataset(**kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
