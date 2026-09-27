from pathlib import Path

import pytest
from dataexcept import DataLoadingError, FileReadError

from qwen_text2sql.config import load_config


def test_load_valid_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  model_id: Qwen/Qwen3.5-4B
lora:
  rank: 8
  alpha: 16
training:
  output_dir: results/example
  bf16: true
  fp16: false
quantization:
  enabled: false
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.model.model_id == "Qwen/Qwen3.5-4B"
    assert config.lora.rank == 8


def test_reject_conflicting_precision(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model: {model_id: x}
lora: {}
training: {output_dir: x, bf16: true, fp16: true}
quantization: {}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot both"):
        load_config(path)


def test_config_file_and_yaml_errors_have_context(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileReadError) as missing_error:
        load_config(missing)
    assert missing_error.value.path == str(missing)
    assert missing_error.value.original is missing_error.value.__cause__

    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("model: [\n", encoding="utf-8")
    with pytest.raises(DataLoadingError) as malformed_error:
        load_config(malformed)
    assert malformed_error.value.source == str(malformed)
    assert malformed_error.value.original is malformed_error.value.__cause__
