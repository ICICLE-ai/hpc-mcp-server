import os
from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        ...


def _model_id() -> str:
    return os.environ.get("MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")


def _torch_dtype():
    import torch

    dtype = os.environ.get("TORCH_DTYPE", "float32").lower()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16"}:
        return torch.float16
    if dtype in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported TORCH_DTYPE={dtype!r}")


def _local_files_only() -> bool:
    value = os.environ.get("HF_LOCAL_FILES_ONLY", "false").lower()
    return value in {"1", "true", "yes", "on"}


class TransformersCpuBackend:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = _model_id()
        local_files_only = _local_files_only()
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_torch_dtype(),
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        self.model.to(self.device)
        self.model.eval()
        self.torch = torch

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_backend() -> LLMBackend:
    backend = os.environ.get("LLM_BACKEND", "transformers").lower()
    if backend != "transformers":
        raise ValueError("CPU server requires LLM_BACKEND=transformers")
    return TransformersCpuBackend()
