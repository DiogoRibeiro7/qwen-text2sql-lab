# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Because this is a research repository, changes that alter a **reported metric**
or the **experimental protocol** are always listed, even when no public API
changes.

## [Unreleased]

### Added

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md` and this changelog.
- GitHub issue templates for bug reports, feature requests and experiment
  proposals, plus a pull-request template carrying a research-integrity
  checklist.
- `CODEOWNERS` and a Dependabot configuration grouping dev tooling and the ML
  stack into separate weekly update batches.
- `py.typed` marker, so downstream consumers get the package's type information
  (PEP 561).
- `.editorconfig` and `.gitattributes` for consistent whitespace and line
  endings across editors and platforms.
- `docs/README.md` index with a suggested reading order.
- `make format`, `make format-check`, `make build` and `make clean` targets.
- Committed `poetry.lock` resolving all 109 transitive dependencies, so
  `poetry install` reproduces an identical environment. CI verifies the lock
  stays in sync with `pyproject.toml`.

### Changed

- Packaging metadata migrated to PEP 621 (`[project]`), adding classifiers,
  keywords and project URLs. The Poetry build backend is unchanged.
- CI now runs on Python 3.11, 3.12 and 3.13, checks formatting in addition to
  lint, pins least-privilege permissions, cancels superseded runs, and caches
  pip downloads.
- Ruff and mypy pinned to matching versions across `.pre-commit-config.yaml`,
  the dev dependencies and CI, so a new tool release cannot turn the build red
  without a deliberate bump.
- Ruff now lints and formats `notebooks/` alongside `src`, `tests` and
  `scripts`.
- `scripts/validate_notebooks.py` additionally rejects committed notebook
  outputs and execution counts; `nbstripout` enforces this on commit.
- Documented commands no longer need the `PYTHONPATH=src` prefix, since
  `poetry install` installs the package itself.
- Expanded pre-commit hooks: JSON checks, merge-conflict and case-conflict
  detection, private-key detection, debug-statement detection and line-ending
  normalisation.

### Fixed

- Repository-wide lint and formatting violations that made the CI `ruff` step
  fail (23 lint errors across 13 unformatted files).
- `mypy --strict` failing locally while passing in CI: `peft` re-exports are now
  handled explicitly rather than depending on whether the package is installed.

## [0.1.0] - 2026-08-19

### Added

- Initial project scaffold: schema-grounded BIRD data preparation,
  database-disjoint splitting, LoRA and QLoRA training entry points, generation,
  read-only execution evaluation, metrics, paired bootstrap comparison,
  experiment planning and sweeps.
- Notebook sequence `00`–`09` mirroring the experimental protocol.
- Research protocol, architecture and data-contract documentation.

[Unreleased]: https://github.com/DiogoRibeiro7/qwen-text2sql-lab/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/qwen-text2sql-lab/releases/tag/v0.1.0
