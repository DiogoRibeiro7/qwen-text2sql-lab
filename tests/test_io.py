from pathlib import Path

from qwen_text2sql.io import read_jsonl, sha256_file, write_jsonl


def test_jsonl_round_trip_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"b": 2, "a": 1}, {"a": 3}])
    assert list(read_jsonl(path)) == [{"a": 1, "b": 2}, {"a": 3}]
    digest = sha256_file(path)
    assert len(digest) == 64
