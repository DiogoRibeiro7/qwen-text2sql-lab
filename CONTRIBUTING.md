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
poetry install --with dev,quantization   # omit ",quantization" without a CUDA GPU
poetry run pre-commit install
```

Nothing in the default developer setup downloads model weights or BIRD
databases. The unit tests build small temporary SQLite databases instead.

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

`pre-commit` runs the fast subset automatically on commit. CI runs the full set
on Python 3.11, 3.12 and 3.13.

### A note on typing

The package is typed and ships a `py.typed` marker. `mypy --strict` is enforced
over `src/qwen_text2sql`. The heavy ML dependencies (`torch`, `transformers`,
`trl`, `peft`) are deliberately **not** installed in CI, so type errors that
only appear with those packages present will not be caught there. Run
`make typecheck` in a full environment before touching training or inference
code.

## Code conventions

- **Reusable logic lives in `src/`.** Notebooks orchestrate, inspect and
  visualise; they do not define the implementation. A pull request that moves
  logic into a notebook will be asked to move it back out.
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
