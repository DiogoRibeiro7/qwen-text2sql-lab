"""Guards for the arguments handed to the trainer.

`train_adapter` passed `warmup_ratio` to `trl.SFTConfig` for a release of
transformers that had removed it, so every training run raised a TypeError
before a single step. Nothing caught it: the heavy ML stack was absent from
every environment the type checker ran in, so the call was never checked against
the real signature.

The first test below is the one that would have caught it, and it needs neither
a GPU nor a model download — only trl installed.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from qwen_text2sql.config import (
    ExperimentConfig,
    LoraConfigData,
    ModelConfig,
    QuantizationConfig,
    TrainingConfig,
)
from qwen_text2sql.training.train import sft_config_kwargs, warmup_steps_for


def _cpu_safe(kwargs: dict[str, object]) -> dict[str, object]:
    """Drop the mixed-precision flags so construction works on a CPU-only host.

    The shipped configs set bf16 for GPU training, and transformers refuses to
    build a config that asks for bf16 on hardware without it. Precision is a
    hardware concern; these tests are about argument compatibility.
    """
    return {**kwargs, "bf16": False, "fp16": False}


def _config(**training: object) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(model_id="Qwen/Qwen3.5-4B"),
        lora=LoraConfigData(),
        training=TrainingConfig(output_dir="results/checkpoints/demo", **training),  # type: ignore[arg-type]
        quantization=QuantizationConfig(),
    )


# --------------------------------------------------------------------------
# The regression guard
# --------------------------------------------------------------------------


def test_every_trainer_argument_exists_in_the_installed_trl() -> None:
    """The check that was missing when `warmup_ratio` was removed upstream."""
    trl = pytest.importorskip("trl", reason="trl is only present in a full environment")

    kwargs = sft_config_kwargs(
        _config(), output_dir=Path("out"), n_train_examples=100, has_eval_dataset=True
    )
    accepted = set(inspect.signature(trl.SFTConfig.__init__).parameters)
    unknown = sorted(set(kwargs) - accepted)
    assert not unknown, (
        f"trl {trl.__version__} does not accept {unknown}. Training would raise "
        "TypeError before its first step."
    )


def test_the_arguments_actually_construct_an_sft_config() -> None:
    """Names being accepted is necessary but not sufficient; values must work too."""
    trl = pytest.importorskip("trl", reason="trl is only present in a full environment")

    kwargs = sft_config_kwargs(
        _config(), output_dir=Path("out"), n_train_examples=100, has_eval_dataset=False
    )
    config = trl.SFTConfig(**_cpu_safe(kwargs))
    assert config.num_train_epochs == 2.0
    assert config.seed == 42


def test_warmup_ratio_is_not_passed_through_verbatim() -> None:
    """The project keeps a ratio; the trainer is given absolute steps."""
    kwargs = sft_config_kwargs(
        _config(), output_dir=Path("out"), n_train_examples=100, has_eval_dataset=False
    )
    assert "warmup_ratio" not in kwargs
    assert "warmup_steps" in kwargs


def test_evaluation_is_disabled_without_an_eval_dataset() -> None:
    """Asking for step-wise evaluation with no eval set is a runtime error."""
    without = sft_config_kwargs(
        _config(), output_dir=Path("out"), n_train_examples=100, has_eval_dataset=False
    )
    with_eval = sft_config_kwargs(
        _config(), output_dir=Path("out"), n_train_examples=100, has_eval_dataset=True
    )
    assert without["eval_strategy"] == "no"
    assert without["eval_steps"] is None
    assert with_eval["eval_strategy"] == "steps"
    assert with_eval["eval_steps"] == 100


# --------------------------------------------------------------------------
# warmup_steps_for
# --------------------------------------------------------------------------


def test_warmup_scales_with_the_training_set() -> None:
    """The reason the ratio is preserved rather than replaced by a step count.

    The learning curve varies the training set by a factor of twenty. A fixed
    step count would leave the smallest cell spending most of training in warmup
    and the largest barely warming up — a schedule difference that would read as
    a data-quantity effect.
    """
    training = _config().training
    small = warmup_steps_for(250, training)
    large = warmup_steps_for(5000, training)
    assert small < large
    # Both are the same fraction of their own run.
    assert small / 250 == pytest.approx(large / 5000, rel=0.2)


def test_warmup_is_the_configured_fraction_of_total_steps() -> None:
    training = _config(
        epochs=2.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        warmup_ratio=0.1,
    ).training
    # 1600 examples / batch 16 = 100 steps per epoch, x2 epochs = 200 steps.
    assert warmup_steps_for(1600, training) == 20


def test_a_zero_ratio_disables_warmup() -> None:
    """configs/baseline.yaml sets warmup_ratio: 0.0 and must produce no warmup."""
    assert warmup_steps_for(1000, _config(warmup_ratio=0.0).training) == 0


def test_warmup_accounts_for_distributed_world_size() -> None:
    """Data parallelism multiplies the effective batch, so there are fewer steps."""
    training = _config().training
    assert warmup_steps_for(4000, training, world_size=4) < warmup_steps_for(4000, training)


def test_warmup_never_goes_negative() -> None:
    assert warmup_steps_for(1, _config(warmup_ratio=0.0).training) >= 0


@pytest.mark.parametrize("n_examples", [0, -1])
def test_warmup_rejects_an_empty_training_set(n_examples: int) -> None:
    with pytest.raises(ValueError, match="n_examples must be positive"):
        warmup_steps_for(n_examples, _config().training)


def test_warmup_rejects_a_degenerate_batch_size() -> None:
    training = _config(per_device_train_batch_size=1, gradient_accumulation_steps=0).training
    with pytest.raises(ValueError, match="Effective batch size must be positive"):
        warmup_steps_for(100, training)


# --------------------------------------------------------------------------
# The shipped configurations
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["baseline", "qwen35_4b_lora", "qwen35_4b_qlora", "qwen35_4b_base_lora"],
)
def test_shipped_configs_produce_usable_trainer_arguments(name: str) -> None:
    """Every config in the repository must yield arguments trl will accept."""
    trl = pytest.importorskip("trl", reason="trl is only present in a full environment")

    from qwen_text2sql.config import load_config
    from qwen_text2sql.reporting import project_root

    config = load_config(project_root() / "configs" / f"{name}.yaml")
    kwargs = sft_config_kwargs(
        config, output_dir=Path("out"), n_train_examples=1000, has_eval_dataset=True
    )
    trl.SFTConfig(**_cpu_safe(kwargs))
