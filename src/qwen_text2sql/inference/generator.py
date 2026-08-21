"""Greedy text-to-SQL generation for base models and PEFT adapters."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from qwen_text2sql.config import ModelConfig
from qwen_text2sql.sql.extract import extract_sql
from qwen_text2sql.training.formatting import conversation_prompt
from qwen_text2sql.types import PreparedExample


def load_inference_model(model_id: str, adapter_path: str | Path | None = None) -> tuple[Any, Any]:
    """Load a Qwen3.5 text model, optionally attaching a PEFT adapter."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    tokenizer_source = str(adapter_path) if adapter_path is not None else model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    model = Qwen3_5ForCausalLM.from_pretrained(model_id, device_map="auto", dtype=dtype)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def generate_sql(
    model: Any,
    tokenizer: Any,
    example: PreparedExample,
    config: ModelConfig,
) -> tuple[str, float]:
    """Generate one SQL statement with deterministic decoding when temperature is zero."""
    import torch

    messages = conversation_prompt(example)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(
        rendered, return_tensors="pt", truncation=True, max_length=config.max_seq_length
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    do_sample = config.temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs.update({"temperature": config.temperature, "top_p": config.top_p})
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    prompt_length = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return extract_sql(decoded), elapsed_ms
