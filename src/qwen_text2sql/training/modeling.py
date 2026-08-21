"""Lazy model-loading helpers for Qwen3.5 LoRA/QLoRA."""

from __future__ import annotations

from typing import Any

from qwen_text2sql.config import ExperimentConfig


def _torch_dtype(name: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return mapping[name]


def load_qwen_text_model(config: ExperimentConfig) -> tuple[Any, Any]:
    """Load the text-only Qwen3.5 model and tokenizer for training.

    Heavy ML imports are intentionally local so data/evaluation tests remain CPU-light.
    """
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_id,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.quantization.enabled:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA 4-bit training requires a CUDA-capable environment")
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.quantization.quant_type,
            bnb_4bit_use_double_quant=config.quantization.double_quant,
            bnb_4bit_compute_dtype=_torch_dtype(config.quantization.compute_dtype),
        )
    else:
        if config.training.bf16:
            load_kwargs["dtype"] = torch.bfloat16
        elif config.training.fp16:
            load_kwargs["dtype"] = torch.float16
        else:
            load_kwargs["dtype"] = torch.float32

    model = Qwen3_5ForCausalLM.from_pretrained(config.model.model_id, **load_kwargs)
    if config.quantization.enabled:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, tokenizer
