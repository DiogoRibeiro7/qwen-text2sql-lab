from pathlib import Path

import pytest
from dataexcept import DataLoadingError, FileReadError, FileWriteError

from qwen_text2sql.io import read_jsonl, sha256_file, write_jsonl


def test_jsonl_round_trip_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"b": 2, "a": 1}, {"a": 3}])
    assert list(read_jsonl(path)) == [{"a": 1, "b": 2}, {"a": 3}]
    digest = sha256_file(path)
    assert len(digest) == 64


def test_jsonl_read_wraps_late_parse_errors(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"ok": true}\nnot-json\n', encoding="utf-8")

    rows = read_jsonl(path)
    assert next(rows) == {"ok": True}
    with pytest.raises(DataLoadingError) as raised:
        next(rows)
    assert raised.value.source == str(path)
    assert raised.value.original is raised.value.__cause__


def test_file_access_errors_retain_paths_and_causes(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    with pytest.raises(FileReadError) as raised_read:
        list(read_jsonl(missing))
    assert raised_read.value.path == str(missing)
    assert isinstance(raised_read.value.original, FileNotFoundError)
    assert raised_read.value.original is raised_read.value.__cause__

    with pytest.raises(FileReadError) as raised_hash:
        sha256_file(missing)
    assert raised_hash.value.path == str(missing)

    parent_file = tmp_path / "not-a-directory"
    parent_file.write_text("content", encoding="utf-8")
    with pytest.raises(FileWriteError) as raised_write:
        write_jsonl(parent_file / "rows.jsonl", [{"ok": True}])
    assert raised_write.value.path == str(parent_file)
    assert isinstance(raised_write.value.original, OSError)
    assert raised_write.value.original is raised_write.value.__cause__


def test_jsonl_object_validation_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "not-an-object.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(TypeError, match="Line 1"):
        list(read_jsonl(path))
