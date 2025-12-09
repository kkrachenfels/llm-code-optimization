import logging
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trl import PPOConfig
from trl.models import AutoModelForCausalLMWithValueHead


_mps_backend = getattr(torch.backends, "mps", None)
if _mps_backend and not hasattr(_mps_backend, "is_macos_or_newer"):
    has_legacy = hasattr(_mps_backend, "is_macos13_or_newer")

    def _is_macos_or_newer(major: int, minor: int) -> bool:
        """Fallback shim mirroring the newer helper for transformers."""
        if major <= 13:
            return False
        return has_legacy and _mps_backend.is_macos13_or_newer()

    setattr(_mps_backend, "is_macos_or_newer", _is_macos_or_newer)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def measure_runtime(source_dir: Path, binary_name: str = "toy_prog") -> float:
    """Compile and run the toy C++ program, returning the elapsed runtime."""
    source_file = source_dir / "main.cpp"
    if not source_file.exists():
        raise FileNotFoundError(f"Unable to find {source_file}")

    binary = source_dir / binary_name
    compile_cmd = ["g++", "-O2", "-std=c++11", str(source_file), "-o", str(binary)]
    subprocess.run(compile_cmd, check=True, cwd=source_dir)

    start = time.perf_counter()
    subprocess.run([str(binary)], check=True, cwd=source_dir)
    elapsed = time.perf_counter() - start
    LOGGER.info("Runtime for %s: %.5f seconds", binary_name, elapsed)
    return elapsed


class ToyPPOAgent:
    """Minimal PPO-ish agent wrapping TRL primitives without the heavy Trainer API."""

    def __init__(self, model_name: str = "distilgpt2") -> None:
        self.config = PPOConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1.41e-5)

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Generate a completion for the prompt using the current policy."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def step(self, prompt: str, response: str, reward: float) -> float:
        """Perform a single policy/value update driven by the reward signal."""
        prompt_tensor = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        response_tensor = self.tokenizer(
            response,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device)

        combined = torch.cat([prompt_tensor, response_tensor], dim=-1)
        logits, loss, value = self.model(input_ids=combined, labels=combined)

        value_preds = value[:, -1]
        advantage = torch.tensor([reward], device=self.device) - value_preds.detach()
        policy_loss = -(advantage * loss)
        value_loss = F.mse_loss(value_preds, torch.tensor([reward], device=self.device))

        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def main():
    test_dir = Path.cwd() / "test_program"
    if not test_dir.exists():
        raise FileNotFoundError("Create a `test_program` directory with a `main.cpp` toy program.")

    agent = ToyPPOAgent()
    baseline_runtime = measure_runtime(test_dir)
    prompts = [
        "How can we shave runtime off the following tiny program?",
        "Suggest a tweak that would speed up the toy workload.",
        "What compiler flags or code hints could make this small binary faster?",
    ]

    for step, prompt in enumerate(prompts, start=1):
        LOGGER.info("Starting PPO step %d with prompt: %s", step, prompt)
        response_text = agent.generate(prompt, max_new_tokens=32)
        reward_runtime = measure_runtime(test_dir)
        reward = (baseline_runtime - reward_runtime) / baseline_runtime
        loss = agent.step(prompt, response_text, reward)
        LOGGER.info(
            "Step %d response %r reward %.4f loss %.5f (baseline %.4fs vs %.4fs)",
            step,
            response_text,
            reward,
            loss,
            baseline_runtime,
            reward_runtime,
        )


if __name__ == "__main__":
    main()

