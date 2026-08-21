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
- Tests for `cli.py` and `pipeline.prepared_from_mapping`, which had no coverage
  at all despite being pure Python. Total coverage rises from 56% to 71%, and
  `cli.py` from 0% to 87%. The suite now covers argument defaults that encode the
  research protocol (seed 42, 15% validation), the database-disjoint split
  written by `qwen-text2sql split`, evaluation of correct, wrong-but-runnable and
  unrunnable predictions, the legacy `predicted_sql` prediction key, and the
  guard that refuses a prediction whose `example_id` is not in the dataset.
- Tests for the read-only execution sandbox, taking
  `evaluation/execution.py` to 100% coverage and total coverage to 74%. These
  assert the guarantees SECURITY.md makes: mutating, DDL, `ATTACH` and `PRAGMA`
  statements are refused; the prefix check is not fooled by leading whitespace
  or casing; statement chaining and extension loading are blocked; the database
  file is byte-identical after every attack; a runaway query is interrupted and
  reported as a timeout; and every SQLite error is classified into the
  `error_kind` the metrics report.
- `qwen_text2sql.reporting`, a tested analysis layer for the notebooks:
  `context` (prerequisite handling, provenance capture, dataset fingerprints),
  `tables` (tidy analysis frames), `style` (one house style, colourblind-safe
  palette) and `figures` (publication-quality matplotlib figures).
- `qwen_text2sql.evaluation.intervals` with the Wilson score interval and a
  McNemar-based minimum-detectable-effect calculation, so a proportion is never
  reported as a bare point estimate and an evaluation can be sized before it is
  run.
- `matplotlib` and `ipykernel` declared in a `notebooks` dependency group. The
  notebooks imported matplotlib without it ever being declared; it resolved only
  as a transitive dependency, so a clean install broke notebooks 07 and 08.
- Tests for BIRD preparation and for the input guards across the package,
  taking total coverage to 87% with 21 modules at complete coverage. Preparation
  is the pipeline's entry point, where a defect changes what every later number
  describes rather than crashing: `example_id` is now pinned against a known
  digest because it is the join key between predictions and examples, ambiguous
  database resolution is asserted to raise rather than guess, and duplicate
  suppression is covered. The guards — malformed configuration, single-database
  splits, empty evaluations, a reference query that does not run, an empty
  schema, non-object JSONL lines — were the least-covered code in the package.
- Tests for the experiment orchestration — `pipeline.generate_and_evaluate` and
  the `prepare-bird` and `train` CLI subcommands — taking total coverage to 90%,
  with `pipeline.py` complete and `cli.py` at 98%. Only the model call is faked;
  the loop, the limit handling, the three output files and the metrics run for
  real against a real SQLite database. `train_adapter` is deliberately not
  covered: faking `datasets`, `peft`, `trl` and the model would leave a test
  that exercises only the mocks.
- Contract tests driving `generate_sql` through a real `Qwen3_5ForCausalLM`,
  built from a config in memory so it needs no download and no GPU. The model is
  real, which is where API drift lives; only the tokenizer is faked. This is the
  shape of test that would have caught the `warmup_ratio` breakage. An audit of
  every transformers, torch and peft call in the inference and model-loading
  paths found no further drift.
- `make format`, `make format-check`, `make build` and `make clean` targets.
- Committed `poetry.lock` resolving all 109 transitive dependencies, so
  `poetry install` reproduces an identical environment. CI verifies the lock
  stays in sync with `pyproject.toml`.

### Changed

- CI's automatic triggers disabled, leaving the workflow manual-only. The
  repository is private, so Actions minutes are billed; with no billing
  configured every run failed before executing a step, and a permanently red
  history hides real failures. Restoring the triggers is a two-line change and
  becomes free if the repository is made public.
- The CI status badge removed from the README, since it reported a billing
  condition rather than the state of the code.
- Added a `make hooks-push` pre-push gate running mypy, the test suite and
  notebook validation, so the checks CI would have run still happen before code
  leaves the machine.

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
- All ten notebooks rewritten. Each now states the question it answers, what it
  decides, and what its result does *not* show; declares its prerequisites and
  fails with the command that produces a missing one instead of rendering an
  empty table; captures commit, interpreter, platform and package versions as
  provenance; and reports proportions with confidence intervals. Notebooks 00,
  03 and 04 run end to end with no data present.
- `scripts/validate_notebooks.py` additionally rejects committed notebook
  outputs and execution counts; `nbstripout` enforces this on commit.
- Documented commands no longer need the `PYTHONPATH=src` prefix, since
  `poetry install` installs the package itself.
- Expanded pre-commit hooks: JSON checks, merge-conflict and case-conflict
  detection, private-key detection, debug-statement detection and line-ending
  normalisation.

### Fixed

- **Training could not run at all.** `train_adapter` passed `warmup_ratio` to
  `trl.SFTConfig`, but transformers 5 removed that argument from
  `TrainingArguments`, so every run raised
  `TypeError: SFTConfig.__init__() got an unexpected keyword argument
  'warmup_ratio'` before its first step. The project keeps expressing warmup as a
  ratio — a fixed step count would make the smallest learning-curve cell spend
  most of training in warmup and the largest barely warm up, confounding the
  sweep — and now converts it to `warmup_steps` from the training-set size.
  The trainer arguments moved into `sft_config_kwargs`, and a test checks every
  one of them against the installed trl.
- `mypy --strict` errors that were invisible without the ML stack installed:
  `PeftModel` assigned to a variable typed as the base model, an untyped
  `.eval()` call, matplotlib's `rcParams` key typing, and missing `pandas-stubs`.

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
