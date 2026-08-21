# Contributing

Thanks for your interest in this project. It is a research repository, so
contributions are judged on two axes: normal software quality, and whether the
change preserves the scientific claims the repository is able to make.

Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting set up

Python 3.11–3.13 and [Poetry](https://python-poetry.org/) are required.

```bash
git clone https://github.com/DiogoRibeiro7/qwen-text2sql-lab.git
cd qwen-text2sql-lab
poetry install --with dev,notebooks,quantization   # drop ",quantization" without a CUDA GPU
poetry run pre-commit install
```

Nothing in the default developer setup downloads model weights or BIRD
databases. The unit tests build small temporary SQLite databases instead.

**On Windows**, `poetry install` can fail with
`[WinError 206] The filename or extension is too long` while unpacking torch,
whose `dist-info` bundles deeply nested third-party licence files. Windows caps
*directory* creation at 248 characters unless long-path support is enabled, and
the default Poetry virtualenv cache path is long enough to push those licence
directories over the limit. Either shorten the virtualenv path:

```bash
poetry config virtualenvs.in-project true   # creates ./.venv instead
```

or enable long paths system-wide (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem`,
`LongPathsEnabled = 1`, then reboot).

`poetry.lock` is committed and resolves the full dependency graph, so every
machine installs byte-identical versions. If you change a dependency constraint
in `pyproject.toml`, run `poetry lock` and commit the result in the same change
— CI fails if the two drift apart.

## Quality gates

Every change must pass:

```bash
make check      # lint + format-check + typecheck + test
make notebooks  # static notebook validation, if notebooks changed
```

Individually:

| Command | What it enforces |
|---|---|
| `make lint` | Ruff lint rules (`E`, `F`, `I`, `UP`, `B`, `SIM`, `RUF`), notebooks included |
| `make format-check` | Ruff formatting (run `make format` to fix) |
| `make typecheck` | `mypy --strict` over `src/qwen_text2sql` |
| `make test` | pytest with branch coverage |
| `make notebooks` | notebooks parse, and carry no stale execution state |

### There is no CI safety net

The workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) is
complete and runs every gate above across Python 3.11-3.13, but **its automatic
triggers are disabled**. This repository is private, so Actions minutes are
metered and then billed; with no billing configured, every run failed before
executing a single step, turning the history red for a reason unrelated to the
code. A permanently red history hides real failures, so the workflow is
manual-only (`workflow_dispatch`) until the repository is made public — which
makes Actions free — or an Actions spending limit is set. Restoring it is a
two-line change, documented at the top of the file.

Until then, **nothing checks your work except you**. Install both hook stages:

```bash
make hooks-push
```

`make hooks` alone installs the fast commit-stage hooks. `make hooks-push` adds a
pre-push gate running mypy, the test suite and notebook validation — the checks
CI would otherwise have caught before the code left your machine. They run in
whatever environment is active, so activate the project virtualenv first, or
invoke them as `poetry run pre-commit`.

Run `make check` before opening a pull request regardless.

### A note on typing

The package is typed and ships a `py.typed` marker. `mypy --strict` is enforced
over `src/qwen_text2sql`. The heavy ML dependencies (`torch`, `transformers`,
`trl`, `peft`) are **not** installed in the lightweight environment, so type
errors that only appear with those packages present are invisible there.

This is not hypothetical. `train_adapter` passed `warmup_ratio` to
`trl.SFTConfig` long after transformers removed it, so every training run raised
a `TypeError` before its first step — and nothing noticed, because trl had never
been installed anywhere the type checker ran. **Run `make install` and then
`make typecheck` in a full environment before touching training or inference
code.**

## Code conventions

- **Reusable logic lives in `src/`.** Notebooks orchestrate, inspect and
  visualise; they do not define the implementation. A pull request that moves
  logic into a notebook will be asked to move it back out. Analysis helpers
  belong in `src/qwen_text2sql/reporting/`, where they can be tested.
- **A notebook never renders an empty result.** Declare prerequisites through
  `qwen_text2sql.reporting.artifact(...)` and call `.require()`; a missing input
  must fail with the command that produces it, not with a blank table.
- **Figures use the house style.** Call `use_house_style()` and the helpers in
  `reporting/figures.py` rather than restyling matplotlib per notebook, and
  report a proportion with an interval rather than as a bare point estimate.
- **ML imports stay lazy.** `torch`, `transformers`, `trl` and `peft` are
  imported inside the functions that need them, so data preparation, evaluation
  and the whole QA suite run without a GPU or a model download. Please preserve
  this.
- **Public functions are typed and documented.** One-line docstrings are fine;
  say what the function guarantees, not how it loops.
- **New behaviour comes with a test.** Prefer tests that build a temporary
  SQLite database over tests that mock the database away.

## Contributing an experiment

Changes to configs, evaluation logic, splits or metrics change what the
repository is allowed to claim. Open an
[experiment proposal](https://github.com/DiogoRibeiro7/qwen-text2sql-lab/issues/new?template=experiment_proposal.yml)
before doing the work, and keep to the rules in
[`docs/research_protocol.md`](docs/research_protocol.md):

1. Never optimise a decoding or training choice on the final evaluation set.
2. Keep schema-identical databases out of both sides of a train/validation split.
3. Save configuration, dataset hash, model ID and training-set size with every run.
4. Keep per-example predictions and evaluation records, not only aggregate scores.
5. Compare models on the same examples, and report uncertainty for differences.
6. Do not claim semantic correctness from SQL text similarity alone.
7. Treat benchmark annotation errors separately from model errors.

If a change alters a reported number, say so explicitly in the pull request and
include the before/after metrics.

## What must never be committed

Model weights, BIRD databases, training checkpoints, prediction dumps, and any
other large artifact. `.gitignore` and the `check-added-large-files` pre-commit
hook cover the common cases, but they are a safety net, not a policy.

## Pull requests

- Branch from `main`; keep one logical change per pull request.
- Write commit messages in the imperative mood ("add rank ablation sweep").
- Fill in the pull request template, including the research integrity checklist.
- Update `CHANGELOG.md` under `## [Unreleased]` for anything user-visible.

## Reporting bugs and vulnerabilities

Bugs go to the [issue tracker](https://github.com/DiogoRibeiro7/qwen-text2sql-lab/issues).
Security vulnerabilities must **not** be filed as public issues — see
[SECURITY.md](SECURITY.md).

## Licence

By contributing, you agree that your contributions are licensed under the
[MIT Licence](LICENSE).
