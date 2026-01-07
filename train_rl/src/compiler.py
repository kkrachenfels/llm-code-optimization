import subprocess
import tempfile
import os
import re
from pathlib import Path
from typing import Tuple, Optional


class CppCompiler:
    """Handles C++ code compilation and execution."""

    def __init__(self, compiler: str = "g++", flags: Optional[list] = None):
        self.compiler = compiler
        self.flags = flags or ["-O2", "-std=c++17"]

    def compile_and_run(self, code: str, timeout: int = 30, num_runs: int = 3) -> Tuple[bool, Optional[float], str]:
        """
        Compile and run C++ code, measuring execution time.

        Args:
            code: C++ source code as string
            timeout: Maximum execution time in seconds
            num_runs: Number of times to run for averaging

        Returns:
            Tuple of (success, average_runtime_microseconds, error_message)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = os.path.join(tmpdir, "program.cpp")
            binary_path = os.path.join(tmpdir, "program")

            # Write source code
            try:
                with open(source_path, 'w') as f:
                    f.write(code)
            except Exception as e:
                return False, None, f"Failed to write source: {str(e)}"

            # Compile
            compile_cmd = [self.compiler] + self.flags + [source_path, "-o", binary_path]
            try:
                result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode != 0:
                    return False, None, f"Compilation failed: {result.stderr}"
            except subprocess.TimeoutExpired:
                return False, None, "Compilation timeout"
            except Exception as e:
                return False, None, f"Compilation error: {str(e)}"

            # Run multiple times and average
            runtimes = []
            for _ in range(num_runs):
                try:
                    result = subprocess.run(
                        [binary_path],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )

                    if result.returncode != 0:
                        return False, None, f"Runtime error: {result.stderr}"

                    # Parse runtime from stdout (expecting microseconds)
                    try:
                        runtime = float(result.stdout.strip())
                        runtimes.append(runtime)
                    except ValueError:
                        return False, None, f"Failed to parse runtime: {result.stdout}"

                except subprocess.TimeoutExpired:
                    return False, None, "Execution timeout"
                except Exception as e:
                    return False, None, f"Execution error: {str(e)}"

            avg_runtime = sum(runtimes) / len(runtimes)
            return True, avg_runtime, ""

    def extract_code_from_llm_output(self, text: str) -> Optional[str]:
        """
        Extract C++ code from LLM output, handling markdown code blocks.

        Args:
            text: LLM output text

        Returns:
            Extracted C++ code or None if not found
        """
        # Try to find code blocks with ```cpp or ```c++ (flexible with whitespace)
        cpp_pattern = r"```(?:cpp|c\+\+|C\+\+|CPP)\s*(.*?)```"
        matches = re.findall(cpp_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()

        # Try generic code blocks with newline
        generic_pattern = r"```\s*\n(.*?)```"
        matches = re.findall(generic_pattern, text, re.DOTALL)
        if matches:
            # Check if it looks like C++ code
            code = matches[0].strip()
            if "#include" in code or "int main" in code or "std::" in code:
                return code

        # Try any triple backticks
        any_backticks_pattern = r"```(.*?)```"
        matches = re.findall(any_backticks_pattern, text, re.DOTALL)
        for match in matches:
            code = match.strip()
            # Remove language identifier if present at start
            if code.startswith(('cpp', 'c++', 'C++', 'CPP')):
                code = code.split('\n', 1)[1] if '\n' in code else code
                code = code.strip()
            # Check if it looks like C++ code
            if "#include" in code or "int main" in code or "std::" in code:
                return code

        # If no code blocks, check if the entire text looks like C++ code
        if "#include" in text and "int main" in text:
            # Try to extract just the C++ part, removing any markdown
            lines = text.split('\n')
            code_lines = []
            in_code = False
            for line in lines:
                if '#include' in line and not in_code:
                    in_code = True
                if in_code:
                    # Stop at common markdown patterns
                    if line.strip().startswith(('**', '##', '```', 'Note:', 'Optimized')):
                        break
                    code_lines.append(line)
            if code_lines:
                return '\n'.join(code_lines).strip()

        return None


def get_baseline_runtime(program_path: str, compiler: CppCompiler, num_runs: int = 5) -> float:
    """
    Get baseline runtime for an original program.

    Args:
        program_path: Path to the C++ source file
        compiler: CppCompiler instance
        num_runs: Number of runs to average

    Returns:
        Average baseline runtime in microseconds
    """
    with open(program_path, 'r') as f:
        code = f.read()

    success, runtime, error = compiler.compile_and_run(code, num_runs=num_runs)

    if not success:
        raise RuntimeError(f"Failed to benchmark baseline: {error}")

    return runtime
