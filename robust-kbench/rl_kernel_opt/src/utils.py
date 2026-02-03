"""
Utility functions for RL kernel optimization.
"""

import os
import yaml
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base_config: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config dictionaries."""
    result = base_config.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_available_gpus() -> list:
    """Get list of available GPU indices."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def setup_logging(log_dir: str, use_wandb: bool = False, **wandb_kwargs):
    """Setup logging directory and optionally wandb."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if use_wandb:
        try:
            import wandb
            wandb.init(**wandb_kwargs)
        except ImportError:
            print("Warning: wandb not installed, skipping wandb logging")

    return log_dir


def format_speedup(speedup: Optional[float]) -> str:
    """Format speedup value for display."""
    if speedup is None:
        return "N/A"
    elif speedup >= 1.0:
        return f"{speedup:.2f}x faster"
    else:
        return f"{1/speedup:.2f}x slower"


def format_time_ms(time_ms: Optional[float]) -> str:
    """Format time in milliseconds."""
    if time_ms is None:
        return "N/A"
    elif time_ms < 0.001:
        return f"{time_ms * 1e6:.2f} ns"
    elif time_ms < 1.0:
        return f"{time_ms * 1e3:.2f} μs"
    elif time_ms < 1000:
        return f"{time_ms:.2f} ms"
    else:
        return f"{time_ms / 1000:.2f} s"


def count_parameters(model) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def estimate_memory_usage(model, batch_size: int = 1, seq_length: int = 4096) -> Dict[str, float]:
    """Estimate GPU memory usage in GB."""
    # Model parameters
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradients (same size as parameters)
    grad_bytes = param_bytes

    # Optimizer states (Adam: 2x parameters for momentum and variance)
    optimizer_bytes = 2 * param_bytes

    # Activations (rough estimate)
    hidden_size = getattr(model.config, "hidden_size", 2048)
    num_layers = getattr(model.config, "num_hidden_layers", 24)
    activation_bytes = batch_size * seq_length * hidden_size * num_layers * 4  # float32

    total_bytes = param_bytes + grad_bytes + optimizer_bytes + activation_bytes

    return {
        "parameters_gb": param_bytes / 1e9,
        "gradients_gb": grad_bytes / 1e9,
        "optimizer_gb": optimizer_bytes / 1e9,
        "activations_gb": activation_bytes / 1e9,
        "total_gb": total_bytes / 1e9,
    }


class EarlyStopping:
    """Early stopping helper."""

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "max",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False

        if self.mode == "max":
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class MetricsTracker:
    """Track and aggregate training metrics."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, list] = {}

    def update(self, metrics: Dict[str, float]):
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

    def get_average(self, key: str, window: Optional[int] = None) -> float:
        if key not in self.history:
            return 0.0

        window = window or self.window_size
        values = self.history[key][-window:]
        return sum(values) / len(values) if values else 0.0

    def get_latest(self, key: str) -> float:
        if key not in self.history or not self.history[key]:
            return 0.0
        return self.history[key][-1]

    def get_summary(self) -> Dict[str, float]:
        return {f"{key}_avg": self.get_average(key) for key in self.history}
