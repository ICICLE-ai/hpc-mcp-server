import os
from typing import Protocol


class LLMBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        ...


def _model_id() -> str:
    return os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")


def _torch_dtype():
    import torch

    dtype = os.environ.get("TORCH_DTYPE", "bfloat16").lower()
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


class DeepSpeedLlamaBackend:
    def __init__(self) -> None:
        import deepspeed
        import torch
        from deepspeed.accelerator import get_accelerator
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch

        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")

        deepspeed.init_distributed()
        self.device = torch.device(get_accelerator().current_device_name())

        model_id = _model_id()
        torch_dtype = _torch_dtype()

        local_files_only = _local_files_only()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )

        self.model = deepspeed.init_inference(
            model,
            tensor_parallel={"tp_size": int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))},
            dtype=torch_dtype,
        )

        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with self.torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class TransformersLlamaBackend:
    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = _model_id()
        local_files_only = _local_files_only()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=_torch_dtype(),
            device_map=os.environ.get("DEVICE_MAP", "auto"),
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
        )
        self.model.eval()
        self.torch = torch

    def generate(self, prompt: str, max_new_tokens: int = 300) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        device = next(self.model.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

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
    backend = os.environ.get("LLM_BACKEND", "deepspeed").lower()
    if backend == "deepspeed":
        return DeepSpeedLlamaBackend()
    if backend == "transformers":
        return TransformersLlamaBackend()
    raise ValueError("LLM_BACKEND must be 'deepspeed' or 'transformers'")
