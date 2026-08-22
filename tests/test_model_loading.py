"""Tests for the arguments that decide what actually gets trained.

`model_load_kwargs` chooses the precision and whether the base model is
quantized. Those choices decide what a run *is*: a config that silently loaded
in the wrong precision, or that quietly skipped quantization, would produce a run
labelled QLoRA that was nothing of the sort — and condition C in the experiment
matrix exists precisely to isolate the effect of quantization.

The 4-bit branch cannot run here, so CUDA availability is simulated to exercise
the argument construction. What that cannot cover is whether the weights load
correctly on a real device, which needs a GPU.
"""

from __future__ import annotations

import pytest

from qwen_text2sql.config import (
    ExperimentConfig,
    LoraConfigData,
    ModelConfig,
    QuantizationConfig,
    TrainingConfig,
)

torch = pytest.importorskip("torch", reason="only present in a full environment")
pytest.importorskip("transformers", reason="only present in a full environment")

from qwen_text2sql.training.modeling import _torch_dtype, model_load_kwargs  # noqa: E402


def _config(
    *, quantization: QuantizationConfig | None = None, bf16: bool = True, fp16: bool = False
) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(model_id="Qwen/Qwen3.5-4B"),
        lora=LoraConfigData(),
        training=TrainingConfig(output_dir="out", bf16=bf16, fp16=fp16),
        quantization=quantization or QuantizationConfig(),
    )


@pytest.fixture()
def pretend_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let the 4-bit branch run on a host without a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


# --------------------------------------------------------------------------
# Precision
# --------------------------------------------------------------------------


def test_bf16_is_requested_when_configured() -> None:
    assert model_load_kwargs(_config(bf16=True))["dtype"] is torch.bfloat16


def test_fp16_is_requested_when_bf16_is_off() -> None:
    assert model_load_kwargs(_config(bf16=False, fp16=True))["dtype"] is torch.float16


def test_float32_is_the_fallback() -> None:
    """Neither flag set means full precision, not an unset dtype."""
    kwargs = model_load_kwargs(_config(bf16=False, fp16=False))
    assert kwargs["dtype"] is torch.float32


def test_bf16_wins_over_fp16_if_both_somehow_reach_here() -> None:
    """load_config rejects both, so this pins the order rather than the policy."""
    assert model_load_kwargs(_config(bf16=True, fp16=True))["dtype"] is torch.bfloat16


def test_the_unquantized_path_asks_for_no_quantization_config() -> None:
    assert "quantization_config" not in model_load_kwargs(_config())


# --------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------


def test_four_bit_without_cuda_raises_rather_than_falling_back() -> None:
    """A silent fallback would label a bf16 run as QLoRA and confound condition C."""
    if torch.cuda.is_available():  # pragma: no cover - depends on the host
        pytest.skip("this host has CUDA, so the guard cannot fire")
    config = _config(quantization=QuantizationConfig(enabled=True))
    with pytest.raises(RuntimeError, match="requires a CUDA-capable environment"):
        model_load_kwargs(config)


def test_the_quantization_config_carries_the_configured_settings(pretend_cuda: None) -> None:
    config = _config(
        quantization=QuantizationConfig(
            enabled=True, quant_type="nf4", double_quant=True, compute_dtype="bfloat16"
        )
    )
    quantization = model_load_kwargs(config)["quantization_config"]
    assert quantization.load_in_4bit is True
    assert quantization.bnb_4bit_quant_type == "nf4"
    assert quantization.bnb_4bit_use_double_quant is True
    assert quantization.bnb_4bit_compute_dtype is torch.bfloat16


def test_non_default_quantization_settings_are_honoured(pretend_cuda: None) -> None:
    """Reading these from the config is the point; hard-coding them would hide it."""
    config = _config(
        quantization=QuantizationConfig(
            enabled=True, quant_type="fp4", double_quant=False, compute_dtype="float16"
        )
    )
    quantization = model_load_kwargs(config)["quantization_config"]
    assert quantization.bnb_4bit_quant_type == "fp4"
    assert quantization.bnb_4bit_use_double_quant is False
    assert quantization.bnb_4bit_compute_dtype is torch.float16


def test_quantization_suppresses_the_plain_dtype(pretend_cuda: None) -> None:
    """Passing both would have the loader arguing with itself about precision."""
    config = _config(quantization=QuantizationConfig(enabled=True), bf16=True)
    kwargs = model_load_kwargs(config)
    assert "quantization_config" in kwargs
    assert "dtype" not in kwargs


# --------------------------------------------------------------------------
# Everything the loader always sends
# --------------------------------------------------------------------------


@pytest.mark.parametrize("enabled", [False, True])
def test_device_map_and_trust_flag_are_always_present(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    kwargs = model_load_kwargs(_config(quantization=QuantizationConfig(enabled=enabled)))
    assert kwargs["device_map"] == "auto"
    assert kwargs["trust_remote_code"] is False


def test_the_shipped_configs_all_produce_loadable_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every config in the repository must yield arguments the loader accepts."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    from qwen_text2sql.config import load_config
    from qwen_text2sql.reporting import project_root

    for name in ("baseline", "qwen35_4b_lora", "qwen35_4b_qlora", "qwen35_4b_base_lora"):
        config = load_config(project_root() / "configs" / f"{name}.yaml")
        kwargs = model_load_kwargs(config)
        assert kwargs["device_map"] == "auto"
        assert ("quantization_config" in kwargs) == config.quantization.enabled


# --------------------------------------------------------------------------
# _torch_dtype
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("bfloat16", "bfloat16"),
        ("float16", "float16"),
        ("float32", "float32"),
    ],
)
def test_dtype_names_map_to_torch_dtypes(name: str, expected: str) -> None:
    assert _torch_dtype(name) is getattr(torch, expected)


@pytest.mark.parametrize("name", ["float8", "bf16", "int8", "", "BFLOAT16"])
def test_an_unknown_dtype_is_refused(name: str) -> None:
    """Falling back to a default would silently train at a precision nobody chose."""
    with pytest.raises(ValueError, match="Unsupported torch dtype"):
        _torch_dtype(name)
