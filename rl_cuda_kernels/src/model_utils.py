"""
Model loading utilities for Qwen2.5-Coder series.
Supports LoRA fine-tuning for memory efficiency.
"""

import os
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

from .config import ModelConfig


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'fp32': torch.float32,
        'fp16': torch.float16,
        'bf16': torch.bfloat16
    }
    return dtype_map.get(dtype_str, torch.bfloat16)


def load_model_and_tokenizer(
    config: ModelConfig,
    device_map: str = "auto",
    use_quantization: bool = False,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a Qwen model and tokenizer with optional LoRA and quantization.

    Args:
        config: Model configuration
        device_map: Device placement strategy
        use_quantization: Enable quantization
        load_in_8bit: Use 8-bit quantization
        load_in_4bit: Use 4-bit quantization

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {config.model_name}")

    # Set up quantization config if needed
    quantization_config = None
    if use_quantization or load_in_4bit or load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=get_torch_dtype(config.torch_dtype),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    # Model loading kwargs
    model_kwargs = {
        'pretrained_model_name_or_path': config.model_name,
        'trust_remote_code': config.trust_remote_code,
        'device_map': device_map,
    }

    # Add dtype if not quantizing
    if quantization_config is None:
        model_kwargs['torch_dtype'] = get_torch_dtype(config.torch_dtype)
    else:
        model_kwargs['quantization_config'] = quantization_config

    # Add attention implementation if flash attention
    if config.attn_implementation == 'flash_attention_2':
        try:
            model_kwargs['attn_implementation'] = 'flash_attention_2'
        except Exception:
            print("Flash Attention 2 not available, using default attention")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=config.trust_remote_code,
        padding_side='left'  # For generation
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model loaded. Parameters: {model.num_parameters():,}")

    return model, tokenizer


def apply_lora(
    model: AutoModelForCausalLM,
    config: ModelConfig
) -> PeftModel:
    """
    Apply LoRA adapters to the model.

    Args:
        model: Base model
        config: Model configuration with LoRA settings

    Returns:
        Model with LoRA adapters
    """
    if not config.use_lora:
        return model

    print("Applying LoRA adapters...")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Prepare for training if quantized
    if hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit:
        model = prepare_model_for_kbit_training(model)
    elif hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit:
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied. Trainable: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return model


def load_model_for_training(
    config: ModelConfig,
    use_4bit: bool = True
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model optimized for training (with LoRA and optional quantization).
    """
    model, tokenizer = load_model_and_tokenizer(
        config,
        device_map="auto",
        load_in_4bit=use_4bit
    )

    if config.use_lora:
        model = apply_lora(model, config)

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    return model, tokenizer


def load_model_for_inference(
    config: ModelConfig,
    checkpoint_path: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load model optimized for inference.
    """
    model, tokenizer = load_model_and_tokenizer(
        config,
        device_map="auto",
        use_quantization=False
    )

    if checkpoint_path:
        # Load LoRA weights if available
        if os.path.exists(os.path.join(checkpoint_path, 'adapter_config.json')):
            print(f"Loading LoRA weights from {checkpoint_path}")
            model = PeftModel.from_pretrained(model, checkpoint_path)

    model.eval()
    return model, tokenizer


def merge_lora_weights(
    model: PeftModel,
    output_path: str
):
    """
    Merge LoRA weights into base model and save.
    """
    print("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")


def setup_vllm_engine(
    config: ModelConfig,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8
):
    """
    Set up vLLM engine for high-throughput inference.
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM not installed. Install with: pip install vllm")

    engine = LLM(
        model=config.model_name,
        trust_remote_code=config.trust_remote_code,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=config.max_seq_length
    )

    return engine


def get_model_memory_footprint(model: AutoModelForCausalLM) -> Dict[str, float]:
    """
    Get model memory footprint in GB.
    """
    param_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )
    buffer_bytes = sum(
        b.numel() * b.element_size()
        for b in model.buffers()
    )

    return {
        'parameters_gb': param_bytes / (1024 ** 3),
        'buffers_gb': buffer_bytes / (1024 ** 3),
        'total_gb': (param_bytes + buffer_bytes) / (1024 ** 3)
    }


def estimate_batch_memory(
    model: AutoModelForCausalLM,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype = torch.bfloat16
) -> float:
    """
    Estimate GPU memory needed for a batch (in GB).
    Rough estimate including activations for training.
    """
    # Model parameters
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradients (same size as parameters)
    grad_bytes = param_bytes

    # Optimizer states (Adam: 2x parameters for momentum and variance)
    optimizer_bytes = 2 * param_bytes

    # Activations (rough estimate: ~2x sequence memory per layer)
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    dtype_size = 2 if dtype in [torch.float16, torch.bfloat16] else 4

    activation_bytes = (
        batch_size * seq_length * hidden_size * dtype_size * num_layers * 2
    )

    total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes
    return total_bytes / (1024 ** 3)


if __name__ == "__main__":
    # Test model loading
    config = ModelConfig()
    print(f"Testing model loading: {config.model_name}")

    model, tokenizer = load_model_and_tokenizer(
        config,
        device_map="auto",
        load_in_4bit=True
    )

    print("\nModel info:")
    memory = get_model_memory_footprint(model)
    for k, v in memory.items():
        print(f"  {k}: {v:.2f} GB")

    # Test generation
    messages = [
        {"role": "user", "content": "Write a simple CUDA kernel for vector addition."}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nGenerated response:\n{response}")
