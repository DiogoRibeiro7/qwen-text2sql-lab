"""Small, deterministic JSON/JSONL I/O helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from dataexcept import DataLoadingError, FileReadError, FileWriteError, wrapping


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield dictionaries from a UTF-8 JSONL file."""
    input_path = Path(path)
    with (
        wrapping((OSError, UnicodeError), FileReadError, path=str(input_path)),
        wrapping(json.JSONDecodeError, DataLoadingError, source=str(input_path)),
        input_path.open("r", encoding="utf-8") as handle,
    ):
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Line {line_number} in {input_path} is not a JSON object")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write mappings as deterministic UTF-8 JSONL."""
    output_path = Path(path)
    with wrapping(OSError, FileWriteError, path=str(output_path.parent)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        wrapping((OSError, UnicodeError), FileWriteError, path=str(output_path)),
        output_path.open("w", encoding="utf-8") as handle,
    ):
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 digest of a file without loading it fully in memory."""
    digest = hashlib.sha256()
    input_path = Path(path)
    with (
        wrapping(OSError, FileReadError, path=str(input_path)),
        input_path.open("rb") as handle,
    ):
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
