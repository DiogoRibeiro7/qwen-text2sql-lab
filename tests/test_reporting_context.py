from __future__ import annotations

from pathlib import Path

import pytest

from qwen_text2sql.io import write_jsonl
from qwen_text2sql.reporting import context


def test_project_root_is_the_repository_not_the_working_directory() -> None:
    """Notebooks used Path(".."), which resolves differently per launch directory."""
    root = context.project_root()
    assert (root / "pyproject.toml").is_file()
    assert (root / "src" / "qwen_text2sql").is_dir()


def test_every_registered_artifact_is_under_a_gitignored_tree() -> None:
    """Artifacts are generated outputs; none should be a tracked source path."""
    for entry in context.ARTIFACTS.values():
        assert entry.relative_path.startswith(("data/", "results/")), entry.key


def test_missing_artifact_names_the_command_that_produces_it() -> None:
    """The whole point: a reader must never have to guess which step they skipped."""
    entry = context.Artifact(
        key="demo",
        relative_path="results/definitely_absent.jsonl",
        description="A demo artifact",
        produced_by="poetry run make-the-thing --flag value",
    )
    with pytest.raises(context.MissingArtifact) as excinfo:
        entry.require()
    message = str(excinfo.value)
    assert "A demo artifact" in message
    assert "results/definitely_absent.jsonl" in message
    assert "poetry run make-the-thing --flag value" in message


def test_missing_artifact_is_a_file_not_found_error() -> None:
    """So `except FileNotFoundError` in existing code keeps working."""
    assert issubclass(context.MissingArtifact, FileNotFoundError)


def test_require_returns_the_path_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(context, "project_root", lambda: tmp_path)
    target = tmp_path / "results" / "present.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")
    entry = context.Artifact("demo", "results/present.jsonl", "Demo", "n/a")
    assert entry.exists
    assert entry.require() == target


def test_artifact_lookup_lists_known_keys_on_a_typo() -> None:
    with pytest.raises(KeyError) as excinfo:
        context.artifact("lora_recrods")
    assert "lora_records" in str(excinfo.value)


def test_preflight_reports_presence_for_the_requested_artifacts() -> None:
    rows = context.preflight("bird_train", "lora_records")
    assert [row["artifact"] for row in rows] == ["bird_train", "lora_records"]
    assert all(isinstance(row["present"], bool) for row in rows)


def test_preflight_defaults_to_every_artifact() -> None:
    assert len(context.preflight()) == len(context.ARTIFACTS)


def test_environment_report_captures_what_a_result_depends_on() -> None:
    report = context.environment_report()
    assert set(report) >= {
        "generated_at",
        "git_commit",
        "python",
        "platform",
        "packages",
        "missing_packages",
    }
    assert isinstance(report["packages"], dict)
    # pandas is a hard dependency, so it must always be resolvable.
    assert "pandas" in report["packages"]


def test_dataset_fingerprint_distinguishes_two_datasets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reporting accuracy "on the validation set" is meaningless if the file changed."""
    monkeypatch.setattr(context, "project_root", lambda: tmp_path)
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_jsonl(first, [{"example_id": "1"}, {"example_id": "2"}])
    write_jsonl(second, [{"example_id": "1"}, {"example_id": "3"}])

    left = context.dataset_fingerprint(first)
    right = context.dataset_fingerprint(second)
    assert left["rows"] == right["rows"] == 2
    assert left["sha256"] != right["sha256"]
    assert len(str(left["sha256"])) == 64


def test_dataset_fingerprint_ignores_blank_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(context, "project_root", lambda: tmp_path)
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n\n', encoding="utf-8")
    assert context.dataset_fingerprint(path)["rows"] == 2


def test_dataset_fingerprint_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(context.MissingArtifact):
        context.dataset_fingerprint(tmp_path / "absent.jsonl")
