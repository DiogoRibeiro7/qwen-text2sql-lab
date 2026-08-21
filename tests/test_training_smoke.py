"""An end-to-end LoRA training run, on CPU, with no download.

Everything else in the suite tests a piece of training. This runs the whole
thing: `train_adapter` against a real Qwen3.5 checkpoint, a real tokenizer, a
real TRL trainer and real PEFT, from prepared JSONL through to a saved adapter
that is then loaded back and used for generation.

It is only possible because the checkpoint is built here rather than downloaded —
about ninety thousand random weights, saved to a temporary directory and loaded
by the ordinary `from_pretrained` path. Nothing is faked.

This is the test that would have caught the `warmup_ratio` regression, where
`train_adapter` raised a TypeError before its first step for want of an argument
transformers had removed. Signature checks confirm the names; only running it
confirms the run.

Marked `slow`, though the work itself is about 2.3 seconds: building and saving
the checkpoint is 1.8s and the training run 0.4s. The marker is really about the
twelve seconds of importing torch in a bare process, which is what makes it
unwelcome in the `make test-fast` inner loop. `make test`, and therefore
`make check`, runs it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from qwen_text2sql.io import write_jsonl

torch = pytest.importorskip("torch", reason="only present in a full environment")
pytest.importorskip("transformers", reason="only present in a full environment")
pytest.importorskip("trl", reason="only present in a full environment")
pytest.importorskip("peft", reason="only present in a full environment")

from qwen_text2sql.config import load_config  # noqa: E402
from qwen_text2sql.training.train import train_adapter  # noqa: E402

pytestmark = pytest.mark.slow

WORDS = [
    "select",
    "name",
    "from",
    "customers",
    "who",
    "are",
    "the",
    "?",
    "sql",
    "schema",
    "system",
    "user",
    "assistant",
    "create",
    "table",
    "text",
    "(",
    ")",
    ";",
    ",",
    ":",
]
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} "
    "{% endfor %}{% if add_generation_prompt %}assistant : {% endif %}"
)
# Full model weights, which this repository promises never to write.
FULL_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save a tiny but genuine Qwen3.5 checkpoint that from_pretrained can load."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    vocab: dict[str, int] = {f"<tok{i}>": i for i in range(8)}
    for token in [*WORDS, *"abcdefghijklmnopqrstuvwxyz0123456789"]:
        vocab.setdefault(token, len(vocab))

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<tok0>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<tok0>",
        pad_token="<tok1>",
        eos_token="<tok2>",
        bos_token="<tok3>",
        chat_template=CHAT_TEMPLATE,
    )

    config = Qwen3_5TextConfig(
        vocab_size=len(vocab),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        # Qwen3.5 interleaves linear and full attention; a two-layer model left
        # to the default layout gets no full-attention layer, which its cache
        # implementation rejects.
        layer_types=["full_attention", "full_attention"],
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    directory = tmp_path_factory.mktemp("checkpoint")
    Qwen3_5ForCausalLM(config).save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    return directory


def _dataset(path: Path, n: int = 8) -> Path:
    write_jsonl(
        path,
        [
            {
                "example_id": f"e{i}",
                "db_id": "shop",
                "question": "who are the customers ?",
                "evidence": "",
                "gold_sql": "select name from customers ;",
                "schema": "create table customers ( name text )",
                "db_path": "unused.sqlite",
                "difficulty": "simple",
                "source_id": str(i),
            }
            for i in range(n)
        ],
    )
    return path


def _config(tmp_path: Path, checkpoint: Path, **training: Any) -> Path:
    payload = {
        "model": {"model_id": str(checkpoint), "max_seq_length": 128, "max_new_tokens": 8},
        "lora": {"rank": 4, "alpha": 8, "target_modules": ["q_proj", "v_proj"]},
        "training": {
            "output_dir": str(tmp_path / "out"),
            "epochs": 1.0,
            "learning_rate": 0.001,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.1,
            "logging_steps": 1,
            "save_steps": 100,
            "gradient_checkpointing": False,
            # CPU-only host: transformers refuses a config asking for bf16 here.
            "bf16": False,
            "fp16": False,
            **training,
        },
        "quantization": {"enabled": False},
    }
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, checkpoint: Path) -> tuple[Path, Path]:
    """Run training once; the assertions below all read its output."""
    work = tmp_path_factory.mktemp("run")
    adapter = train_adapter(load_config(_config(work, checkpoint)), _dataset(work / "train.jsonl"))
    return adapter, work / "out"


def test_training_completes_and_produces_an_adapter(trained: tuple[Path, Path]) -> None:
    """The regression that motivated this: the run has to actually finish."""
    adapter, _ = trained
    assert adapter.is_dir()
    assert (adapter / "adapter_model.safetensors").is_file()
    assert (adapter / "adapter_config.json").is_file()


def test_only_adapter_artifacts_are_written(trained: tuple[Path, Path]) -> None:
    """The repository promises adapter-only checkpoints; this holds it to that.

    Writing full weights would put multi-gigabyte files under results/ on every
    sweep cell, which .gitignore is the last defence against rather than the
    first.
    """
    _, output_dir = trained
    written = {path.name for path in output_dir.rglob("*") if path.is_file()}
    assert not written & set(FULL_WEIGHT_NAMES), f"full model weights written: {written}"


def test_the_adapter_config_records_the_requested_rank(trained: tuple[Path, Path]) -> None:
    adapter, _ = trained
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["r"] == 4
    assert config["lora_alpha"] == 8
    assert set(config["target_modules"]) == {"q_proj", "v_proj"}


def test_run_metadata_makes_the_run_reproducible(trained: tuple[Path, Path]) -> None:
    """Rule 3: configuration, dataset hash, model id and example count, every run."""
    _, output_dir = trained
    metadata = json.loads((output_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_examples"] == 8
    assert metadata["quantized"] is False
    assert len(metadata["train_data_sha256"]) == 64
    assert metadata["validation_data_sha256"] is None
    assert metadata["config"]["lora"]["rank"] == 4
    assert "train_loss" in metadata["metrics"]


def test_the_saved_adapter_loads_and_generates(
    trained: tuple[Path, Path], checkpoint: Path
) -> None:
    """Close the loop: an adapter that cannot be loaded back is not a result."""
    from peft import PeftModel
    from transformers import AutoTokenizer, Qwen3_5ForCausalLM

    adapter, _ = trained
    base = Qwen3_5ForCausalLM.from_pretrained(checkpoint)
    model = PeftModel.from_pretrained(base, str(adapter)).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    inputs = tokenizer("who are the customers ?", return_tensors="pt")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=4, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
    assert outputs.shape[1] > inputs["input_ids"].shape[1]


def test_max_train_examples_truncates_the_run(tmp_path: Path, checkpoint: Path) -> None:
    """Every learning-curve cell depends on this actually limiting the data."""
    adapter = train_adapter(
        load_config(_config(tmp_path, checkpoint)),
        _dataset(tmp_path / "train.jsonl", n=8),
        max_train_examples=4,
    )
    metadata = json.loads((adapter.parent / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["train_examples"] == 4


def test_an_empty_training_set_is_refused(tmp_path: Path, checkpoint: Path) -> None:
    write_jsonl(tmp_path / "empty.jsonl", [])
    with pytest.raises(ValueError, match="Training dataset is empty"):
        train_adapter(load_config(_config(tmp_path, checkpoint)), tmp_path / "empty.jsonl")


def test_a_non_positive_example_limit_is_refused(tmp_path: Path, checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="max_train_examples must be positive"):
        train_adapter(
            load_config(_config(tmp_path, checkpoint)),
            _dataset(tmp_path / "train.jsonl"),
            max_train_examples=0,
        )
