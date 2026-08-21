"""Supervised LoRA/QLoRA training entry point."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qwen_text2sql.config import ExperimentConfig, TrainingConfig
from qwen_text2sql.io import read_jsonl, sha256_file
from qwen_text2sql.pipeline import prepared_from_mapping
from qwen_text2sql.training.formatting import sft_record
from qwen_text2sql.training.modeling import load_qwen_text_model


def warmup_steps_for(n_examples: int, training: TrainingConfig, *, world_size: int = 1) -> int:
    """Convert a warmup *ratio* into the absolute step count the trainer wants.

    transformers 5 removed ``warmup_ratio`` from ``TrainingArguments``, leaving
    only ``warmup_steps``. The ratio is worth preserving rather than replacing:
    the learning-curve sweep varies the training set by a factor of twenty, and a
    fixed step count would leave the smallest cell spending most of its training
    in warmup while the largest barely warms up at all. That is a schedule
    difference masquerading as a data-quantity effect, which is precisely the
    confound the sweep exists to avoid.
    """
    if n_examples <= 0:
        raise ValueError("n_examples must be positive")
    effective_batch = (
        training.per_device_train_batch_size * training.gradient_accumulation_steps * world_size
    )
    if effective_batch <= 0:
        raise ValueError("Effective batch size must be positive")
    steps_per_epoch = math.ceil(n_examples / effective_batch)
    total_steps = math.ceil(steps_per_epoch * training.epochs)
    return max(0, round(total_steps * training.warmup_ratio))


def sft_config_kwargs(
    config: ExperimentConfig,
    *,
    output_dir: Path,
    n_train_examples: int,
    has_eval_dataset: bool,
) -> dict[str, Any]:
    """Build the arguments for ``trl.SFTConfig``.

    Separated from :func:`train_adapter` so the argument names can be checked
    against the installed trl without a GPU or a model download. They were not,
    once: ``warmup_ratio`` was passed for a release of transformers that had
    removed it, and every training run failed with a TypeError before a single
    step. Nothing caught it because trl was absent from every environment the
    type checker ran in.
    """
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": config.training.epochs,
        "learning_rate": config.training.learning_rate,
        "per_device_train_batch_size": config.training.per_device_train_batch_size,
        "per_device_eval_batch_size": config.training.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.training.gradient_accumulation_steps,
        "warmup_steps": warmup_steps_for(n_train_examples, config.training),
        "weight_decay": config.training.weight_decay,
        "logging_steps": config.training.logging_steps,
        "eval_strategy": "steps" if has_eval_dataset else "no",
        "eval_steps": config.training.eval_steps if has_eval_dataset else None,
        "save_steps": config.training.save_steps,
        "save_strategy": "steps",
        "seed": config.training.seed,
        "gradient_checkpointing": config.training.gradient_checkpointing,
        "bf16": config.training.bf16,
        "fp16": config.training.fp16,
        "max_length": config.model.max_seq_length,
        "completion_only_loss": config.training.completion_only_loss,
        "report_to": "none",
    }


def train_adapter(
    config: ExperimentConfig,
    train_path: str | Path,
    validation_path: str | Path | None = None,
    *,
    max_train_examples: int | None = None,
) -> Path:
    """Fine-tune Qwen3.5 with LoRA or QLoRA and save only adapter artifacts."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    train_examples = [prepared_from_mapping(row) for row in read_jsonl(train_path)]
    if max_train_examples is not None:
        if max_train_examples <= 0:
            raise ValueError("max_train_examples must be positive")
        train_examples = train_examples[:max_train_examples]
    if not train_examples:
        raise ValueError("Training dataset is empty")
    eval_examples = (
        [prepared_from_mapping(row) for row in read_jsonl(validation_path)]
        if validation_path is not None
        else []
    )

    train_dataset = Dataset.from_list([sft_record(row) for row in train_examples])
    eval_dataset = (
        Dataset.from_list([sft_record(row) for row in eval_examples]) if eval_examples else None
    )
    model, tokenizer = load_qwen_text_model(config)
    peft_config = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=config.lora.use_rslora,
    )
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args = SFTConfig(
        **sft_config_kwargs(
            config,
            output_dir=output_dir,
            n_train_examples=len(train_examples),
            has_eval_dataset=eval_dataset is not None,
        )
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    metadata = {
        "model": config.model.model_id,
        "quantized": config.quantization.enabled,
        "train_examples": len(train_examples),
        "validation_examples": len(eval_examples),
        "train_data_sha256": sha256_file(train_path),
        "validation_data_sha256": sha256_file(validation_path) if validation_path else None,
        "config": asdict(config),
        "metrics": train_result.metrics,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return adapter_dir
