"""
Extract and validate CUDA code from model outputs.
"""

import re
import tempfile
import os
from typing import Optional, Tuple
from pathlib import Path


class CUDACodeExtractor:
    """Extracts and validates CUDA code from model outputs."""

    # Patterns for extracting code blocks
    CUDA_BLOCK_PATTERN = r"```(?:cuda|cpp|c\+\+|cu)\s*\n(.*?)```"
    GENERIC_BLOCK_PATTERN = r"```\s*\n(.*?)```"

    # Required components for a valid CUDA kernel
    REQUIRED_PATTERNS = [
        r"__global__\s+void",  # Kernel function
        r"<<<.*>>>",  # Kernel launch
    ]

    # Patterns that indicate PyTorch usage (which we want to avoid)
    PYTORCH_PATTERNS = [
        r"torch::",
        r"at::",
        r"\.to\(.*device.*\)",
        r"torch\.ops\.",
    ]

    # Patterns for proper pybind11 module definition
    PYBIND_PATTERNS = [
        r"PYBIND11_MODULE",
        r"TORCH_EXTENSION_NAME",
    ]

    def __init__(
        self,
        temp_dir: Optional[str] = None,
        strict_validation: bool = False,
    ):
        """
        Initialize the code extractor.

        Args:
            temp_dir: Directory for temporary CUDA files
            strict_validation: If True, reject code with PyTorch operations
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="cuda_kernels_")
        self.strict_validation = strict_validation
        os.makedirs(self.temp_dir, exist_ok=True)

    def extract(self, model_output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract CUDA code from model output.

        Args:
            model_output: The raw model output string

        Returns:
            Tuple of (cuda_code, error_message)
            If extraction succeeds, error_message is None
            If extraction fails, cuda_code is None
        """
        # Try to find CUDA/C++ code blocks first
        matches = re.findall(self.CUDA_BLOCK_PATTERN, model_output, re.DOTALL)

        if not matches:
            # Try generic code blocks
            matches = re.findall(self.GENERIC_BLOCK_PATTERN, model_output, re.DOTALL)

        if not matches:
            return None, "No code block found in model output"

        # If multiple code blocks, try to find the most complete one
        best_code = None
        best_score = -1

        for code in matches:
            score = self._score_code(code)
            if score > best_score:
                best_score = score
                best_code = code

        if best_code is None:
            return None, "No valid CUDA code found in code blocks"

        # Basic validation
        validation_error = self._validate_code(best_code)
        if validation_error:
            return None, validation_error

        return best_code.strip(), None

    def _score_code(self, code: str) -> int:
        """
        Score a code block based on how complete it looks.

        Higher score = more likely to be the actual implementation.
        """
        score = 0

        # Has kernel function
        if re.search(r"__global__\s+void", code):
            score += 10

        # Has kernel launch
        if re.search(r"<<<.*>>>", code):
            score += 10

        # Has includes
        if "#include" in code:
            score += 5

        # Has pybind11 module
        if re.search(r"PYBIND11_MODULE|TORCH_EXTENSION_NAME", code):
            score += 10

        # Penalize very short code
        if len(code) < 200:
            score -= 5

        # Penalize if it looks like just a snippet
        if code.count("...") > 2:
            score -= 10

        return score

    def _validate_code(self, code: str) -> Optional[str]:
        """
        Validate CUDA code for basic requirements.

        Returns error message if validation fails, None if OK.
        """
        # Check for required patterns
        has_kernel = re.search(r"__global__\s+void", code)
        has_launch = re.search(r"<<<.*>>>", code)

        if not has_kernel:
            return "No __global__ kernel function found"

        if not has_launch:
            return "No kernel launch (<<<...>>>) found"

        # Check for PyTorch usage if strict validation is enabled
        if self.strict_validation:
            for pattern in self.PYTORCH_PATTERNS:
                if re.search(pattern, code):
                    return f"Code contains PyTorch operations (pattern: {pattern})"

        # Check for pybind11 module (warning, not error)
        has_pybind = any(re.search(p, code) for p in self.PYBIND_PATTERNS)
        if not has_pybind:
            # This is a warning - the code might still work if we add the module
            pass

        return None

    def save_to_file(
        self,
        cuda_code: str,
        task_name: str,
        turn: int = 0,
        trajectory: int = 0,
        forward: bool = True,
    ) -> str:
        """
        Save CUDA code to a temporary file.

        Args:
            cuda_code: The CUDA code to save
            task_name: Name of the task
            turn: Turn number (for multi-turn)
            trajectory: Trajectory number (for parallel sampling)
            forward: Whether this is a forward pass (True) or backward pass (False).
                     Affects pybind11 module entry point name.

        Returns:
            Path to the saved file
        """
        # Create subdirectory for this task
        task_dir = Path(self.temp_dir) / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        filename = f"kernel_t{turn}_traj{trajectory}.cu"
        filepath = task_dir / filename

        # Ensure the code has proper includes and module definition
        complete_code = self._ensure_complete_code(cuda_code, forward=forward)

        with open(filepath, "w") as f:
            f.write(complete_code)

        return str(filepath)

    def _ensure_complete_code(self, code: str, forward: bool = True) -> str:
        """
        Ensure the code has all necessary components.

        Adds missing includes and pybind11 module if needed.

        Args:
            code: The CUDA code to complete
            forward: Whether this is a forward pass (True) or backward pass (False).
                     Determines the pybind11 entry point name (.forward vs .backward).
        """
        lines = []

        # Check if includes are present
        has_cuda_include = "#include <cuda_runtime.h>" in code
        has_torch_include = "#include <torch/extension.h>" in code

        # Add missing includes at the top
        if not has_cuda_include:
            lines.append("#include <cuda_runtime.h>")
        if not has_torch_include:
            lines.append("#include <torch/extension.h>")

        if lines:
            lines.append("")  # Empty line after includes

        lines.append(code)

        # Check if pybind11 module is present
        has_pybind = re.search(r"PYBIND11_MODULE", code)

        if not has_pybind:
            # Determine the entry point name based on forward/backward
            entry_point = "forward" if forward else "backward"
            entry_desc = "Forward pass" if forward else "Backward pass"

            # Try to find the main function name to expose
            # Look for functions that take and return torch::Tensor
            func_pattern = r"torch::Tensor\s+(\w+)\s*\("
            matches = re.findall(func_pattern, code)

            if matches:
                # Use the last matching function (usually the main wrapper)
                func_name = matches[-1]
                lines.append("")
                lines.append(f'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{')
                lines.append(f'    m.def("{entry_point}", &{func_name}, "{entry_desc}");')
                lines.append("}")

        return "\n".join(lines)

    def cleanup(self):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
