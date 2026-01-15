import subprocess
import tempfile
import os
import re
import logging
from pathlib import Path
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CppCompiler:
    """Handles C++ code compilation and execution."""

    # Local temp directory for compilation artifacts
    LOCAL_TMP_DIR = Path(__file__).parent.parent / "tmp"

    def __init__(
        self,
        compiler: str = "g++",
        flags: Optional[list] = None,
        include_paths: Optional[list] = None,
        additional_sources: Optional[list] = None,
        defines: Optional[dict] = None,
        output_is_seconds: bool = False,
        source_extension: Optional[str] = None
    ):
        """
        Initialize the C++ compiler.

        Args:
            compiler: Compiler command (e.g., 'g++', 'gcc')
            flags: Compilation flags (default: ["-O2", "-std=c++17"])
            include_paths: List of include directory paths (for -I flags)
            additional_sources: List of additional source files to compile
            defines: Dictionary of preprocessor defines (for -D flags)
            output_is_seconds: If True, expect output in seconds and convert to microseconds
            source_extension: File extension for source files (default: '.cpp' for g++, '.c' for gcc)
        """
        self.compiler = compiler
        self.flags = flags or ["-O2", "-std=c++17"]
        self.include_paths = include_paths or []
        self.additional_sources = additional_sources or []
        self.defines = defines or {}
        self.output_is_seconds = output_is_seconds
        # Auto-detect extension based on compiler if not specified
        if source_extension is None:
            self.source_extension = '.c' if compiler in ('gcc', 'cc', 'clang') else '.cpp'
        else:
            self.source_extension = source_extension

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
        # Ensure local tmp directory exists
        self.LOCAL_TMP_DIR.mkdir(exist_ok=True)

        with tempfile.TemporaryDirectory(dir=self.LOCAL_TMP_DIR) as tmpdir:
            source_path = os.path.join(tmpdir, f"program{self.source_extension}")
            binary_path = os.path.join(tmpdir, "program")

            # Write source code
            try:
                with open(source_path, 'w') as f:
                    f.write(code)
            except Exception as e:
                return False, None, f"Failed to write source: {str(e)}"

            # Build compile command
            compile_cmd = [self.compiler] + self.flags

            # Add include paths
            for include_path in self.include_paths:
                compile_cmd.extend(["-I", include_path])

            # Add defines
            for key, value in self.defines.items():
                if value is None or value == "":
                    compile_cmd.append(f"-D{key}")
                else:
                    compile_cmd.append(f"-D{key}={value}")

            # Add source files
            compile_cmd.append(source_path)
            compile_cmd.extend(self.additional_sources)

            # Add output file
            compile_cmd.extend(["-o", binary_path])

            # Log the compile command
            logger.debug(f"Compile command: {' '.join(compile_cmd)}")

            # Compile
            try:
                result = subprocess.run(
                    compile_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                if result.returncode != 0:
                    logger.debug(f"Compilation failed with return code {result.returncode}")
                    logger.debug(f"Compiler stderr:\n{result.stderr}")
                    logger.debug(f"Failed source code:\n{code[:1000]}...")
                    return False, None, "Compilation failed"
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

                    # Parse runtime from stdout
                    try:
                        runtime = float(result.stdout.strip())
                        # Convert seconds to microseconds if needed
                        if self.output_is_seconds:
                            runtime = runtime * 1_000_000
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
        Extract C++ code from LLM output, handling both raw code and markdown blocks.

        Args:
            text: LLM output text

        Returns:
            Extracted C++ code or None if not found
        """
        logger.debug(f"=== EXTRACT CODE START ===")
        logger.debug(f"Raw LLM output ({len(text)} chars):\n{text}")
        logger.debug(f"=== END RAW OUTPUT ===")

        original_text = text

        # Clean up the text - remove any trailing special tokens
        text = re.sub(r'<\|im_end\|>.*', '', text, flags=re.DOTALL).strip()
        text = re.sub(r'<\|endoftext\|>.*', '', text, flags=re.DOTALL).strip()

        if text != original_text:
            logger.debug(f"After cleanup ({len(text)} chars, removed {len(original_text) - len(text)} chars):\n{text}")
        else:
            logger.debug(f"No special tokens removed during cleanup")

        # PRIORITY 1: Check if text starts with raw C/C++ code (no markdown)
        # This is the expected format from our prompt
        stripped = text.strip()
        logger.debug(f"PRIORITY 1: Checking if text starts with raw C/C++ code...")
        logger.debug(f"First 50 chars: {repr(stripped[:50])}")
        if stripped.startswith('#include') or stripped.startswith('//') or stripped.startswith('/*'):
            logger.debug(f"Text starts with C/C++ code marker, extracting raw code...")
            # Looks like raw code - extract until we hit non-code content
            lines = stripped.split('\n')
            code_lines = []
            for i, line in enumerate(lines):
                # Stop at markdown or explanation patterns
                if line.strip().startswith(('**', '##', '```', 'Note:', 'Explanation:', 'Changes:', 'This ')):
                    logger.debug(f"Stopping at line {i}: {repr(line[:50])}")
                    break
                code_lines.append(line)
            if code_lines:
                code = '\n'.join(code_lines).strip()
                # Validate it has some C/C++ content
                if '#include' in code or 'int ' in code or 'void ' in code or 'double ' in code:
                    logger.debug(f"SUCCESS: Extracted raw code, {len(code_lines)} lines, {len(code)} chars")
                    return code
                else:
                    logger.debug(f"FAILED: Raw code didn't contain expected C/C++ markers")
        else:
            logger.debug(f"Text doesn't start with #include, //, or /* - trying other methods")

        # PRIORITY 2: Try to find code blocks with ```cpp or ```c++
        logger.debug(f"PRIORITY 2: Looking for ```cpp code blocks...")
        cpp_pattern = r"```(?:cpp|c\+\+|C\+\+|CPP|c)\s*(.*?)```"
        matches = re.findall(cpp_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            logger.debug(f"SUCCESS: Found code block with cpp marker, extracted {len(matches[0])} chars")
            return matches[0].strip()
        else:
            logger.debug(f"No ```cpp code blocks found")

        # PRIORITY 3: Try generic code blocks
        logger.debug(f"PRIORITY 3: Looking for generic ``` code blocks...")
        generic_pattern = r"```\s*\n(.*?)```"
        matches = re.findall(generic_pattern, text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            if "#include" in code or "int " in code or "void " in code:
                logger.debug(f"SUCCESS: Found generic code block, {len(code)} chars")
                return code
            else:
                logger.debug(f"Generic code block found but doesn't look like C/C++")
        else:
            logger.debug(f"No generic code blocks found")

        # PRIORITY 4: Try any triple backticks
        logger.debug(f"PRIORITY 4: Looking for any ``` blocks...")
        any_backticks_pattern = r"```(.*?)```"
        matches = re.findall(any_backticks_pattern, text, re.DOTALL)
        if matches:
            logger.debug(f"Found {len(matches)} backtick blocks, checking each...")
        for match in matches:
            code = match.strip()
            # Remove language identifier if present at start
            if code.split('\n')[0].strip().lower() in ('cpp', 'c++', 'c', 'cxx'):
                code = code.split('\n', 1)[1] if '\n' in code else code
                code = code.strip()
            if "#include" in code or "int " in code or "void " in code:
                logger.debug(f"SUCCESS: Found code in backticks, {len(code)} chars")
                return code
        if matches:
            logger.debug(f"Backtick blocks found but none contained valid C/C++ code")

        # PRIORITY 5: Look for #include anywhere and extract from there
        logger.debug(f"PRIORITY 5: Looking for #include anywhere in text...")
        if "#include" in text:
            lines = text.split('\n')
            code_lines = []
            in_code = False
            brace_count = 0
            for line in lines:
                if '#include' in line and not in_code:
                    in_code = True
                if in_code:
                    # Stop at markdown patterns
                    if line.strip().startswith(('**', '##', '```', 'Note:', 'Explanation:')):
                        break
                    code_lines.append(line)
                    brace_count += line.count('{') - line.count('}')
            if code_lines:
                logger.debug(f"SUCCESS: Extracted code from #include marker, {len(code_lines)} lines")
                return '\n'.join(code_lines).strip()
            else:
                logger.debug(f"#include found but couldn't extract valid code lines")
        else:
            logger.debug(f"No #include found in text")

        logger.debug(f"=== EXTRACTION FAILED ===")
        logger.debug(f"All 5 extraction methods failed. Full response:\n{text}")
        logger.debug(f"=== END FAILED RESPONSE ===")
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
