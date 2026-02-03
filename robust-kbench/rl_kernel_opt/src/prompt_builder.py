"""
Prompt builder for CUDA kernel optimization tasks.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from .task_sampler import KernelTask


@dataclass
class TurnFeedback:
    """Feedback from a previous turn for multi-turn refinement."""

    compiled: bool
    compile_error: Optional[str]
    correct: bool
    max_diff: Optional[float]
    speedup: Optional[float]
    torch_time_ms: Optional[float]
    cuda_time_ms: Optional[float]
    profile_info: Optional[Dict[str, Any]]

    def to_feedback_string(self) -> str:
        """Convert feedback to a string for the prompt."""
        lines = []

        if not self.compiled:
            lines.append("COMPILATION FAILED")
            if self.compile_error:
                # Truncate long error messages
                error = self.compile_error
                if len(error) > 1000:
                    error = error[:1000] + "\n... (error truncated)"
                lines.append(f"Error:\n{error}")
            return "\n".join(lines)

        lines.append("COMPILATION SUCCEEDED")

        if not self.correct:
            lines.append("CORRECTNESS CHECK FAILED")
            if self.max_diff is not None:
                lines.append(f"Maximum difference from reference: {self.max_diff:.6e}")
            return "\n".join(lines)

        lines.append("CORRECTNESS CHECK PASSED")

        if self.speedup is not None:
            lines.append(f"Speedup over PyTorch: {self.speedup:.2f}x")
            lines.append(
                f"PyTorch time: {self.torch_time_ms:.3f}ms, CUDA time: {self.cuda_time_ms:.3f}ms"
            )

        if self.profile_info:
            lines.append("\nProfile Information:")
            for key, value in self.profile_info.items():
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)


class PromptBuilder:
    """Builds prompts for CUDA kernel optimization."""

    # ---- Forward kernel prompts (following robust-kbench paper Appendix G.1) ----

    FORWARD_SYSTEM_PROMPT = """You are a CUDA engineer tasked with translating PyTorch code into CUDA kernel code.
The CUDA code you generate will be saved to a file and loaded using torch.utils.cpp_extension.load():
```python
cuda_fn = load(name=task_name, sources=[cuda_fname], extra_cuda_cflags=["-O3", "--use_fast_math"], with_cuda=True, verbose=True)
```
Later, the function will be called via `cuda_fn.forward(...)` and thoroughly tested.

When writing CUDA code:
- Use appropriate thread block sizes (typically 256 or 512 threads)
- Coalesce memory accesses where possible
- Use shared memory for data reuse
- Minimize thread divergence
- Consider using warp-level primitives when appropriate

REQUIRED CODE STRUCTURE:
```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void myKernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // kernel logic
    }
}

// C++ wrapper function
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    myKernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Forward pass");
}
```

CRITICAL RULES:
- Use TORCH_CHECK() for assertions, NOT AT_CHECK (deprecated)
- Use standard C++ types (int, float, etc.), NOT dim3_t or other invented types
- Use data_ptr<float>() to get raw pointers from tensors
- Always check bounds in kernel: if (idx < size)
- Launch kernel with <<<blocks, threads>>>
- Include the required pybind11 cuda module name in the code

IMPORTANT: Your implementation must produce EXACTLY the same numerical outputs as the PyTorch reference.
Do not use PyTorch operations in your CUDA kernel - implement the computation directly in CUDA."""

    # ---- Backward kernel prompts (following robust-kbench paper Appendix G.1) ----

    BACKWARD_SYSTEM_PROMPT = """You are a CUDA engineer tasked with writing efficient backward kernels for PyTorch code.
The CUDA code you generate will be saved to a file and loaded using torch.utils.cpp_extension.load():
```python
backward_fn = load(name=task_name, sources=[cuda_fname], extra_cuda_cflags=["-O3", "--use_fast_math"], with_cuda=True, verbose=True)
```
Later, the function will be called via `backward_fn.backward(...)` and thoroughly tested.

Write the corresponding backward CUDA kernel for the given Autograd function.

When writing CUDA code:
- Use appropriate thread block sizes (typically 256 or 512 threads)
- Coalesce memory accesses where possible
- Use shared memory for data reuse
- Minimize thread divergence
- Consider using warp-level primitives when appropriate

REQUIRED CODE STRUCTURE:
```cuda
#include <torch/extension.h>
#include <cuda_runtime.h>

// CUDA kernel for backward pass
__global__ void myBackwardKernel(const float* grad_output, const float* input,
                                  float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // backward kernel logic - compute gradients
    }
}

// C++ wrapper function
torch::Tensor backward(torch::Tensor grad_output, torch::Tensor input) {
    auto grad_input = torch::empty_like(input);
    int size = input.numel();
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    myBackwardKernel<<<blocks, threads>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size
    );

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward", &backward, "Backward pass");
}
```

CRITICAL RULES:
- Use TORCH_CHECK() for assertions, NOT AT_CHECK (deprecated)
- Use standard C++ types (int, float, etc.), NOT dim3_t or other invented types
- Use data_ptr<float>() to get raw pointers from tensors
- Always check bounds in kernel: if (idx < size)
- Launch kernel with <<<blocks, threads>>>
- Include the required pybind11 cuda module name in the code
- Try to minimize the usage of torch functions in the CUDA kernel. Write custom CUDA kernels with the highest possible performance.

IMPORTANT: Your implementation must produce EXACTLY the same backward operation (gradient computation) as the PyTorch reference.
Do not use PyTorch operations in your CUDA kernel - implement the gradient computation directly in CUDA."""

    # ---- Kept for backwards compatibility ----
    SYSTEM_PROMPT = FORWARD_SYSTEM_PROMPT

    # ---- Forward task template ----

    FORWARD_TASK_TEMPLATE = """## Task: {task_name}

### Description
{docstring}

### PyTorch Reference Implementation
```python
{pytorch_code}
```

### Input Specification
{input_specs}

### Output Specification
{output_spec}

### Requirements
Translate the PyTorch code into a working forward CUDA kernel. Your code should:
1. Match the PyTorch reference output within numerical tolerance (atol=1e-5, rtol=1e-5)
2. Be as fast as possible
3. Handle the input shapes specified above

Provide your complete CUDA implementation in a single code block."""

    # ---- Backward task template ----

    BACKWARD_TASK_TEMPLATE = """## Task: {task_name}

### Description
{docstring}

### PyTorch Autograd Function Reference
```python
{pytorch_code}
```

### Input Specification
{input_specs}

### Output Specification
{output_spec}

### Requirements
Write a backward CUDA kernel that computes the gradient of the computation shown in the Autograd function above. Your code should:
1. Match the PyTorch reference backward output within numerical tolerance (atol=1e-5, rtol=1e-5)
2. Be as fast as possible - minimize usage of torch functions and implement gradient computation directly in CUDA
3. Handle the input shapes specified above

Provide your complete CUDA implementation in a single code block."""

    # ---- Kept for backwards compatibility ----
    TASK_TEMPLATE = FORWARD_TASK_TEMPLATE

    REFINEMENT_TEMPLATE = """## Previous Attempt Feedback

{feedback}

## Instructions

Based on the feedback above, please provide an improved CUDA implementation that addresses the issues.
{specific_instructions}

Provide your complete revised CUDA implementation in a single code block."""

    def __init__(self, include_system_prompt: bool = True):
        """
        Initialize the prompt builder.

        Args:
            include_system_prompt: Whether to include system prompt in messages
        """
        self.include_system_prompt = include_system_prompt

    def build_initial_prompt(self, task: KernelTask) -> List[Dict[str, str]]:
        """
        Build the initial prompt for a task.

        Uses forward-specific or backward-specific prompts based on task.forward.

        Args:
            task: The kernel task

        Returns:
            List of message dictionaries for chat format
        """
        messages = []

        # Select system prompt based on forward/backward
        if self.include_system_prompt:
            system_prompt = self.FORWARD_SYSTEM_PROMPT if task.forward else self.BACKWARD_SYSTEM_PROMPT
            messages.append({"role": "system", "content": system_prompt})

        # Build input specs string
        input_specs_str = self._format_input_specs(task.input_specs)

        # Select task template based on forward/backward
        task_template = self.FORWARD_TASK_TEMPLATE if task.forward else self.BACKWARD_TASK_TEMPLATE

        # Build user message
        user_content = task_template.format(
            task_name=task.name,
            docstring=task.docstring.strip(),
            pytorch_code=task.pytorch_code.strip(),
            input_specs=input_specs_str,
            output_spec=task.output_spec,
        )

        messages.append({"role": "user", "content": user_content})

        return messages

    def build_refinement_prompt(
        self,
        task: KernelTask,
        previous_messages: List[Dict[str, str]],
        previous_response: str,
        feedback: TurnFeedback,
    ) -> List[Dict[str, str]]:
        """
        Build a refinement prompt based on previous attempt feedback.

        Args:
            task: The kernel task
            previous_messages: Previous conversation messages
            previous_response: The previous model response
            feedback: Feedback from evaluating the previous attempt

        Returns:
            Updated list of message dictionaries
        """
        messages = previous_messages.copy()

        # Add previous assistant response (summarized to save context)
        # Following Kevin-32B, we don't include full chain of thought
        messages.append(
            {
                "role": "assistant",
                "content": f"[Previous CUDA implementation attempt for {task.name}]",
            }
        )

        # Build specific instructions based on feedback
        specific_instructions = self._get_specific_instructions(feedback)

        # Build refinement message
        refinement_content = self.REFINEMENT_TEMPLATE.format(
            feedback=feedback.to_feedback_string(),
            specific_instructions=specific_instructions,
        )

        messages.append({"role": "user", "content": refinement_content})

        return messages

    def _format_input_specs(self, input_specs: Dict[str, Any]) -> str:
        """Format input specifications as a readable string."""
        lines = []

        if "input_names" in input_specs and input_specs["input_names"]:
            lines.append(f"Input tensor names: {', '.join(input_specs['input_names'])}")

        if "example_config" in input_specs:
            lines.append("\nExample configuration:")
            for key, value in input_specs["example_config"].items():
                lines.append(f"  {key}: {value}")

        if "shared_config" in input_specs:
            lines.append("\nShared configuration:")
            for key, value in input_specs["shared_config"].items():
                lines.append(f"  {key}: {value}")

        if not lines:
            return "See PyTorch reference for input specifications."

        return "\n".join(lines)

    def _get_specific_instructions(self, feedback: TurnFeedback) -> str:
        """Get specific instructions based on feedback type."""
        if not feedback.compiled:
            return """
Focus on fixing the compilation errors. Common issues include:
- Missing includes (cuda_runtime.h, torch/extension.h)
- Incorrect kernel launch syntax
- Type mismatches
- Missing pybind11 module definition"""

        if not feedback.correct:
            return """
Focus on fixing correctness issues. Common causes include:
- Incorrect indexing or bounds checking
- Numerical precision issues (use appropriate data types)
- Race conditions or synchronization issues
- Incorrect handling of edge cases (padding, boundary conditions)"""

        # If correct but slow
        if feedback.speedup is not None and feedback.speedup < 1.0:
            return """
Your implementation is slower than PyTorch. Consider:
- Better memory coalescing
- Using shared memory for data reuse
- Optimizing thread block configuration
- Reducing thread divergence
- Using vectorized memory operations"""

        # If already fast, try to improve further
        return """
Try to further optimize your implementation:
- Consider using warp-level primitives
- Explore different thread block configurations
- Consider loop unrolling
- Profile memory access patterns"""

    def format_for_generation(
        self, messages: List[Dict[str, str]], tokenizer
    ) -> str:
        """
        Format messages using the tokenizer's chat template.

        Args:
            messages: List of message dictionaries
            tokenizer: HuggingFace tokenizer with chat template

        Returns:
            Formatted prompt string
        """
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
