#!/usr/bin/env python3
"""Check that the release metadata agrees with itself.

The version of this project is written in three places — ``pyproject.toml``,
``CITATION.cff`` and ``.zenodo.json`` — and a release that disagrees with itself
is worse than one that is merely late. Zenodo mints a DOI from ``.zenodo.json``
at the moment a GitHub release is published, so a stale version there is
archived permanently and cannot be edited into agreement afterwards.

Run this before tagging. See docs/releasing.md.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

# Zenodo rejects a deposit without these, and silently defaults several others.
REQUIRED_ZENODO_FIELDS = ("title", "upload_type", "creators", "description", "license")
# Zenodo's `creators` take these and nothing else. A role such as ProjectLeader
# belongs to `contributors`; left on a creator it is ignored rather than
# rejected, so the role would silently fail to be recorded.
ALLOWED_CREATOR_KEYS = {"name", "affiliation", "orcid", "gnd"}
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
ORCID = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")


def orcid_checksum_ok(orcid: str) -> bool:
    """Verify an ORCID's ISO 7064 MOD 11-2 check digit.

    A transposed digit yields a syntactically valid identifier belonging to
    somebody else, and the deposit attributing work to them is permanent.
    """
    digits = orcid.replace("-", "")
    if not ORCID.match(orcid):
        return False
    total = 0
    for char in digits[:15]:
        total = (total + int(char)) * 2
    remainder = (12 - total % 11) % 11
    return ("X" if remainder == 10 else str(remainder)) == digits[15]


def _read(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    zenodo = json.loads((root / ".zenodo.json").read_text(encoding="utf-8"))
    citation: dict[str, Any] = {}
    for line in (root / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        match = re.match(r'^(version|title|license|date-released):\s*"?([^"]*)"?\s*$', line)
        if match:
            citation[match.group(1)] = match.group(2).strip()
        orcid_match = re.match(r'^\s*orcid:\s*"?([^"]*)"?\s*$', line)
        if orcid_match:
            citation.setdefault("orcids", []).append(orcid_match.group(1).strip())
    return pyproject, zenodo, citation


def check(root: Path) -> list[str]:
    """Return a list of problems; empty means the metadata is consistent."""
    problems: list[str] = []
    pyproject, zenodo, citation = _read(root)

    project_version = str(pyproject["project"]["version"])
    if not SEMVER.match(project_version):
        problems.append(f"pyproject version {project_version!r} is not MAJOR.MINOR.PATCH")

    versions = {
        "pyproject.toml": project_version,
        "CITATION.cff": citation.get("version", "<missing>"),
        ".zenodo.json": str(zenodo.get("version", "<missing>")),
    }
    if len(set(versions.values())) != 1:
        rendered = ", ".join(f"{name}={value}" for name, value in versions.items())
        problems.append(f"version disagreement: {rendered}")

    titles = {
        "CITATION.cff": citation.get("title", "<missing>"),
        ".zenodo.json": zenodo.get("title", "<missing>"),
    }
    if len(set(titles.values())) != 1:
        problems.append(f"title disagreement: {titles}")

    licences = {
        "CITATION.cff": citation.get("license", "<missing>").lower(),
        ".zenodo.json": str(zenodo.get("license", "<missing>")).lower(),
        "pyproject.toml": str(pyproject["project"].get("license", "<missing>")).lower(),
    }
    if len(set(licences.values())) != 1:
        problems.append(f"licence disagreement: {licences}")

    missing = [field for field in REQUIRED_ZENODO_FIELDS if not zenodo.get(field)]
    if missing:
        problems.append(f".zenodo.json is missing required fields: {missing}")

    if not zenodo.get("creators") or not all(
        entry.get("name") for entry in zenodo.get("creators", [])
    ):
        problems.append(".zenodo.json has a creator without a name")

    for creator in zenodo.get("creators", []):
        unknown = sorted(set(creator) - ALLOWED_CREATOR_KEYS)
        if unknown:
            problems.append(
                f".zenodo.json creator {creator.get('name', '?')!r} has keys Zenodo "
                f"ignores on a creator: {unknown}. Roles belong in `contributors`."
            )

    zenodo_orcids = [c["orcid"] for c in zenodo.get("creators", []) if c.get("orcid")]
    for orcid in zenodo_orcids:
        if not ORCID.match(orcid):
            problems.append(f".zenodo.json ORCID {orcid!r} is not NNNN-NNNN-NNNN-NNNX")
        elif not orcid_checksum_ok(orcid):
            problems.append(f".zenodo.json ORCID {orcid!r} fails its check digit")
    # Zenodo wants the bare identifier; CFF wants the resolvable URL. Compare the
    # identifier so the two cannot drift apart.
    citation_orcids = [value.rsplit("/", 1)[-1] for value in citation.get("orcids", [])]
    if sorted(zenodo_orcids) != sorted(citation_orcids):
        problems.append(
            f"ORCID disagreement: .zenodo.json={sorted(zenodo_orcids)} "
            f"CITATION.cff={sorted(citation_orcids)}"
        )

    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"[{project_version}]" not in changelog and "## [Unreleased]" not in changelog:
        problems.append(f"CHANGELOG.md has neither an Unreleased section nor [{project_version}]")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    problems = check(args.root)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(f"\n{len(problems)} metadata problem(s); see docs/releasing.md", file=sys.stderr)
        return 1

    _, zenodo, _ = _read(args.root)
    print(f"Release metadata is consistent at version {zenodo['version']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
