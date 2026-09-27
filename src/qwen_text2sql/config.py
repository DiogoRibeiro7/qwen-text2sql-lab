"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dataexcept import DataLoadingError, FileReadError, wrapping


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Foundation-model and generation configuration."""

    model_id: str
    max_seq_length: int = 4096
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    trust_remote_code: bool = False


@dataclass(frozen=True, slots=True)
class LoraConfigData:
    """PEFT LoRA hyperparameters."""

    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"
    use_rslora: bool = False


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Supervised fine-tuning configuration."""

    output_dir: str
    epochs: float = 2.0
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    seed: int = 42
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    completion_only_loss: bool = True


@dataclass(frozen=True, slots=True)
class QuantizationConfig:
    """4-bit quantization parameters for QLoRA."""

    enabled: bool = False
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete experiment configuration."""

    model: ModelConfig
    lora: LoraConfigData
    training: TrainingConfig
    quantization: QuantizationConfig


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}")
    return value


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a YAML experiment file into typed configuration objects."""
    config_path = Path(path)
    with wrapping((OSError, UnicodeError), FileReadError, path=str(config_path)):
        config_text = config_path.read_text(encoding="utf-8")
    with wrapping(yaml.YAMLError, DataLoadingError, source=str(config_path)):
        payload = yaml.safe_load(config_text)
    root = _require_mapping(payload, "config")
    model = ModelConfig(**_require_mapping(root.get("model", {}), "model"))
    lora = LoraConfigData(**_require_mapping(root.get("lora", {}), "lora"))
    training = TrainingConfig(**_require_mapping(root.get("training", {}), "training"))
    quantization = QuantizationConfig(
        **_require_mapping(root.get("quantization", {}), "quantization")
    )
    if model.max_seq_length <= 0:
        raise ValueError("model.max_seq_length must be positive")
    if lora.rank <= 0 or lora.alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if not 0.0 <= lora.dropout < 1.0:
        raise ValueError("LoRA dropout must be in [0, 1)")
    if training.learning_rate <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if training.bf16 and training.fp16:
        raise ValueError("bf16 and fp16 cannot both be enabled")
    return ExperimentConfig(model=model, lora=lora, training=training, quantization=quantization)
