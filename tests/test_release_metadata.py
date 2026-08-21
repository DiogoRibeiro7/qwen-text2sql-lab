"""Guards for the metadata Zenodo archives permanently.

Zenodo reads `.zenodo.json` at the moment a GitHub release is published and mints
a DOI from it. A version stale by one bump is archived that way for good and
cannot be edited into agreement afterwards, so the disagreement has to be caught
before the tag rather than after the deposit.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_release_metadata", REPO / "scripts" / "check_release_metadata.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_release_metadata"] = module
    spec.loader.exec_module(module)
    return module


checker = _load()


@pytest.fixture()
def repo_copy(tmp_path: Path) -> Path:
    """A copy of the four metadata files, so tests can corrupt one safely."""
    for name in ("pyproject.toml", "CITATION.cff", ".zenodo.json", "CHANGELOG.md"):
        shutil.copy(REPO / name, tmp_path / name)
    return tmp_path


def _rewrite(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path.name}: {old!r} not found"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


# --------------------------------------------------------------------------
# The repository as it stands
# --------------------------------------------------------------------------


def test_the_shipped_metadata_is_consistent() -> None:
    """`make release-check` must pass on main, or it is not a usable gate."""
    assert checker.check(REPO) == []


def test_zenodo_json_is_valid_json_with_the_fields_zenodo_requires() -> None:
    payload = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    for field in checker.REQUIRED_ZENODO_FIELDS:
        assert payload.get(field), f"missing {field}"
    assert payload["upload_type"] == "software"
    assert payload["access_right"] == "open"
    assert all(entry.get("name") for entry in payload["creators"])


def test_the_deposit_does_not_claim_to_contain_data_or_weights() -> None:
    """The archive is source only; BIRD and Qwen are not ours to redistribute."""
    payload = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    notes = payload["notes"].lower()
    assert "redistributed in this archive" in notes
    assert "cc by-sa" in notes and "apache" in notes


# --------------------------------------------------------------------------
# What the checker catches
# --------------------------------------------------------------------------


def test_a_version_bumped_in_only_one_place_is_caught(repo_copy: Path) -> None:
    """The classic release mistake, and the one Zenodo makes permanent."""
    _rewrite(repo_copy / "pyproject.toml", 'version = "0.1.0"', 'version = "0.2.0"')
    problems = checker.check(repo_copy)
    assert any("version disagreement" in problem for problem in problems)


def test_a_stale_zenodo_version_is_caught(repo_copy: Path) -> None:
    _rewrite(repo_copy / ".zenodo.json", '"version": "0.1.0"', '"version": "0.0.9"')
    assert any("version disagreement" in p for p in checker.check(repo_copy))


def test_a_stale_citation_version_is_caught(repo_copy: Path) -> None:
    _rewrite(repo_copy / "CITATION.cff", "version: 0.1.0", "version: 0.0.9")
    assert any("version disagreement" in p for p in checker.check(repo_copy))


def test_a_licence_disagreement_is_caught(repo_copy: Path) -> None:
    """A wrong licence in the deposit misstates the terms of a citable artifact."""
    _rewrite(repo_copy / ".zenodo.json", '"license": "mit"', '"license": "apache-2.0"')
    assert any("licence disagreement" in p for p in checker.check(repo_copy))


def test_a_title_disagreement_is_caught(repo_copy: Path) -> None:
    _rewrite(
        repo_copy / ".zenodo.json", '"title": "Qwen Text-to-SQL Lab"', '"title": "Something Else"'
    )
    assert any("title disagreement" in p for p in checker.check(repo_copy))


def test_a_missing_required_zenodo_field_is_caught(repo_copy: Path) -> None:
    payload = json.loads((repo_copy / ".zenodo.json").read_text(encoding="utf-8"))
    del payload["description"]
    (repo_copy / ".zenodo.json").write_text(json.dumps(payload), encoding="utf-8")
    assert any("missing required fields" in p for p in checker.check(repo_copy))


def test_a_creator_without_a_name_is_caught(repo_copy: Path) -> None:
    payload = json.loads((repo_copy / ".zenodo.json").read_text(encoding="utf-8"))
    payload["creators"] = [{"affiliation": "Somewhere"}]
    (repo_copy / ".zenodo.json").write_text(json.dumps(payload), encoding="utf-8")
    problems = checker.check(repo_copy)
    assert any("creator" in p or "missing required fields" in p for p in problems)


def test_a_non_semver_version_is_caught(repo_copy: Path) -> None:
    for name, old, new in (
        ("pyproject.toml", 'version = "0.1.0"', 'version = "0.1"'),
        ("CITATION.cff", "version: 0.1.0", "version: 0.1"),
        (".zenodo.json", '"version": "0.1.0"', '"version": "0.1"'),
    ):
        _rewrite(repo_copy / name, old, new)
    assert any("MAJOR.MINOR.PATCH" in p for p in checker.check(repo_copy))


def test_a_changelog_with_nowhere_to_put_the_release_is_caught(repo_copy: Path) -> None:
    (repo_copy / "CHANGELOG.md").write_text("# Changelog\n\nNothing here.\n", encoding="utf-8")
    assert any("CHANGELOG" in p for p in checker.check(repo_copy))


def test_the_checker_exits_non_zero_on_a_problem(
    repo_copy: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`make release-check` has to fail the build, not merely mention it."""
    _rewrite(repo_copy / "pyproject.toml", 'version = "0.1.0"', 'version = "9.9.9"')
    monkeypatch.setattr(sys, "argv", ["check_release_metadata.py", "--root", str(repo_copy)])
    assert checker.main() == 1


def test_the_checker_exits_zero_when_consistent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["check_release_metadata.py", "--root", str(REPO)])
    assert checker.main() == 0


# --------------------------------------------------------------------------
# ORCID
# --------------------------------------------------------------------------


def test_the_shipped_orcid_passes_its_check_digit() -> None:
    payload = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    orcid = payload["creators"][0]["orcid"]
    assert orcid == "0009-0001-2022-7072"
    assert checker.orcid_checksum_ok(orcid)


def test_the_two_files_state_the_same_orcid() -> None:
    """Zenodo wants the bare identifier, CFF the resolvable URL; same person."""
    payload = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    citation = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    orcid = payload["creators"][0]["orcid"]
    assert f"https://orcid.org/{orcid}" in citation


def test_the_creator_carries_an_affiliation() -> None:
    payload = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    assert payload["creators"][0]["affiliation"] == "ESMAD - Instituto Politécnico do Porto"


def test_creators_carry_no_contributor_role() -> None:
    """`type` is a `contributors` field in Zenodo's schema, not a `creators` one.

    Left on a creator it is ignored, so the role would silently not be recorded.
    """
    payload = json.loads((REPO / ".zenodo.json").read_text(encoding="utf-8"))
    for creator in payload["creators"]:
        assert "type" not in creator
        assert set(creator) <= {"name", "affiliation", "orcid", "gnd"}


@pytest.mark.parametrize(
    "orcid",
    [
        pytest.param("0009-0001-2022-7073", id="wrong-check-digit"),
        pytest.param("0009-0001-2202-7072", id="transposed-digits"),
        pytest.param("0009-0001-2022", id="too-short"),
        pytest.param("0009000120227072", id="no-hyphens"),
        pytest.param("not-an-orcid-here", id="not-numeric"),
    ],
)
def test_a_bad_orcid_is_rejected(orcid: str) -> None:
    """A transposed digit is a valid-looking identifier belonging to someone else."""
    assert not checker.orcid_checksum_ok(orcid)


def test_an_orcid_ending_in_x_is_accepted() -> None:
    """The check digit is base-11, so X is a legitimate final character.

    Computed rather than recalled: the well-known example ORCID
    (0000-0002-1825-0097) ends in 7, and assuming otherwise would make this test
    assert the checker is broken.
    """
    assert checker.ORCID.match("0000-0002-1800-008X")
    assert checker.orcid_checksum_ok("0000-0002-1800-008X")


def test_a_bad_orcid_in_the_deposit_is_caught(repo_copy: Path) -> None:
    _rewrite(repo_copy / ".zenodo.json", "0009-0001-2022-7072", "0009-0001-2022-7073")
    problems = checker.check(repo_copy)
    assert any("check digit" in p for p in problems)


def test_an_orcid_disagreement_between_the_files_is_caught(repo_copy: Path) -> None:
    _rewrite(repo_copy / "CITATION.cff", "0009-0001-2022-7072", "0000-0002-1825-009X")
    assert any("ORCID disagreement" in p for p in checker.check(repo_copy))


def test_a_contributor_role_on_a_creator_is_caught(repo_copy: Path) -> None:
    """Zenodo ignores `type` on a creator, so the role would vanish silently."""
    payload = json.loads((repo_copy / ".zenodo.json").read_text(encoding="utf-8"))
    payload["creators"][0]["type"] = "ProjectLeader"
    (repo_copy / ".zenodo.json").write_text(json.dumps(payload), encoding="utf-8")
    problems = checker.check(repo_copy)
    assert any("contributors" in p for p in problems)
