<p align="center">
  <img src="assets/project-avatar.png" alt="qwen-text2sql-lab project logo" width="160" height="160">
</p>

# Qwen Text-to-SQL Lab

[![CI](https://github.com/DiogoRibeiro7/qwen-text2sql-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/qwen-text2sql-lab/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Typed: mypy strict](https://img.shields.io/badge/mypy-strict-2a6db2.svg)](https://mypy-lang.org/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A reproducible research repository for adapting **Qwen3.5-4B** to text-to-SQL with **LoRA** and **QLoRA**, then evaluating whether task-specific fine-tuning improves executable SQL generation.

The repository is built around one primary question:

> How much task-specific data and adapter capacity are required before parameter-efficient fine-tuning produces a measurable improvement over the pretrained/post-trained foundation model?

The primary endpoint is **execution accuracy**. Generated SQL is executed against the target SQLite database and compared with the result of the verified reference query. Textual SQL equality is retained only as a secondary diagnostic.

## Research design

The core comparisons are:

| ID | Model | Adaptation | Purpose |
|---|---|---|---|
| A | `Qwen/Qwen3.5-4B` | none | post-trained foundation-model baseline |
| B | `Qwen/Qwen3.5-4B` | LoRA SFT | full-precision adapter training |
| C | `Qwen/Qwen3.5-4B` | QLoRA SFT | memory-efficient 4-bit adapter training |
| D | `Qwen/Qwen3.5-4B-Base` | LoRA SFT | isolate task adaptation from general post-training |

Two planned ablations are first-class parts of the project:

- training-set size: `250, 500, 1000, 2500, 5000, full` when those sizes are available;
- LoRA rank: `4, 8, 16, 32, 64`.

All development splits are made **by database**, not by row, to avoid placing identical database schemas in both train and validation partitions.

## Metrics

For example \(i\), let \(\hat q_i\) be the generated query, \(q_i\) the reference query, and \(D_i\) the corresponding database. Execution accuracy is

\[
\operatorname{EX}=\frac{1}{N}\sum_{i=1}^{N}
\mathbf{1}\left[R(\hat q_i,D_i)=R(q_i,D_i)\right].
\]

The implementation also records:

- valid SQL rate;
- normalized exact-match rate;
- error category (`syntax_error`, `missing_table`, `missing_column`, timeout, etc.);
- per-query generation and execution latency;
- paired bootstrap confidence intervals for differences in execution accuracy.

The local execution comparator treats rows as a multiset and uses a numerical tolerance for floating-point outputs. This is intentionally transparent and inspectable. For leaderboard submissions, use the benchmark's current official evaluator as an additional external check.

## Data

Training metadata uses the filtered BIRD training release:

```text
birdsql/bird23-train-filtered
```

The current BIRD development metadata is also supported:

```text
birdsql/bird_sql_dev_20251106
```

The Hugging Face datasets contain question/SQL metadata; SQLite databases must also be available locally so schemas can be extracted and predictions can be executed.

Download the official BIRD training database archive:

```bash
python scripts/download_bird_train.py
unzip data/raw/bird_train.zip -d data/raw/bird_train
```

For the revised BIRD dev databases, use the complete database package linked from the official `birdsql/bird_sql_dev_20251106` dataset card and extract it under `data/raw/`.

The project does not redistribute BIRD databases or model weights.

## Installation

Python 3.11+ is recommended. The project uses Poetry.

```bash
poetry install --with dev,quantization
```

`poetry.lock` is committed, so this resolves to the same versions on every
machine. For environments without 4-bit CUDA training, omit the quantization
group:

```bash
poetry install --with dev
```

Qwen3.5 text-only support is provided by Transformers through `Qwen3_5ForCausalLM`. The dependency floor in `pyproject.toml` is chosen to include released Qwen3.5 support.

JSONL reads, checksum failures, and configuration file reads raise
`dataexcept.FileReadError`. JSONL and output metadata writes raise
`dataexcept.FileWriteError`. Malformed JSONL and YAML raise
`dataexcept.DataLoadingError`. These errors retain the original exception in
`original` and as the exception cause. A valid JSONL value that is not an object
still raises `TypeError` with its line number; configuration validation errors
remain unchanged.

## Prepare BIRD training data

After extracting the databases, locate the directory that contains the database folders/files and run:

```bash
poetry run qwen-text2sql prepare-bird \
  --dataset birdsql/bird23-train-filtered \
  --split train \
  --db-root data/raw/bird_train/train_databases \
  --output data/processed/bird_train_all.jsonl
```

Then create a database-disjoint development split:

```bash
poetry run qwen-text2sql split \
  --input data/processed/bird_train_all.jsonl \
  --train-output data/processed/bird_train.jsonl \
  --validation-output data/processed/bird_validation.jsonl \
  --validation-fraction 0.15 \
  --seed 42
```

Each prepared record includes a stable example ID, question, optional evidence, reference SQL, schema text and local database path.


## Prepare the revised BIRD development set

After downloading and extracting the database package linked from the official revised BIRD development dataset card:

```bash
poetry run qwen-text2sql prepare-bird \
  --dataset birdsql/bird_sql_dev_20251106 \
  --split dev_20251106 \
  --db-root data/raw/bird_dev/dev_databases \
  --output data/processed/bird_dev_20251106.jsonl
```

Treat this as an external evaluation set: do not choose learning rates, LoRA ranks, decoding settings or stopping rules from its results.

## Foundation-model baseline

Generate SQL before fine-tuning:

```bash
poetry run python scripts/generate_predictions.py \
  --config configs/baseline.yaml \
  --data data/processed/bird_validation.jsonl \
  --output results/predictions/baseline_validation.jsonl
```

Evaluate by execution:

```bash
poetry run python scripts/evaluate_predictions.py \
  --data data/processed/bird_validation.jsonl \
  --predictions results/predictions/baseline_validation.jsonl \
  --records-output results/baseline_validation_records.jsonl \
  --metrics-output results/baseline_validation_metrics.json
```

## LoRA fine-tuning

```bash
poetry run python scripts/train_adapter.py \
  --config configs/qwen35_4b_lora.yaml \
  --train-data data/processed/bird_train.jsonl \
  --validation-data data/processed/bird_validation.jsonl
```

Only PEFT adapter artifacts are written to the configured checkpoint directory.

Generate adapted predictions:

```bash
poetry run python scripts/generate_predictions.py \
  --config configs/qwen35_4b_lora.yaml \
  --adapter results/checkpoints/qwen35_4b_lora/adapter \
  --data data/processed/bird_validation.jsonl \
  --output results/predictions/lora_validation.jsonl
```

## QLoRA fine-tuning

The QLoRA configuration loads the text model in 4-bit NF4, prepares it for k-bit training, then trains LoRA parameters over linear modules.

```bash
poetry run python scripts/train_adapter.py \
  --config configs/qwen35_4b_qlora.yaml \
  --train-data data/processed/bird_train.jsonl \
  --validation-data data/processed/bird_validation.jsonl
```

This path requires a compatible CUDA environment and `bitsandbytes`.

## Base vs post-trained comparison

To measure how much of the task comes from general post-training versus task-specific adaptation, repeat the LoRA experiment with:

```text
configs/qwen35_4b_base_lora.yaml
```

This gives the decomposition:

\[
\text{pretraining}\rightarrow\text{general post-training}\rightarrow\text{text-to-SQL adaptation}.
\]

## Learning curves and rank ablations

Create the planned experiment matrix:

```bash
poetry run python scripts/plan_experiments.py \
  --train-data data/processed/bird_train.jsonl \
  --output results/experiment_plan.csv
```

The machine-readable plan prevents silently dropping a training-set size or LoRA rank when results are later aggregated.

Run either sweep end to end with:

```bash
poetry run python scripts/run_sweep.py \
  --kind learning_curve \
  --config configs/qwen35_4b_qlora.yaml \
  --train-data data/processed/bird_train.jsonl \
  --validation-data data/processed/bird_validation.jsonl
```

Use `--kind rank_ablation` for the adapter-rank experiment. The sweep writes a per-cell checkpoint, predictions, evaluation records, metrics and an incrementally updated summary CSV.

## Statistical model comparison

After evaluating two models on the same examples:

```bash
poetry run python scripts/compare_models.py \
  --first results/baseline_validation_records.jsonl \
  --second results/lora_validation_records.jsonl \
  --n-bootstrap 10000 \
  --seed 42
```

The paired bootstrap operates on per-example execution success, preserving the paired nature of the benchmark.

## Notebooks

The notebook sequence mirrors the experiment rather than hiding logic inside
notebooks. Each one states the question it answers, what it decides, and — in a
closing section — what its result does **not** show.

| Notebook | Decides |
|---|---|
| [`00_research_protocol`](notebooks/00_research_protocol.ipynb) | The estimand, the comparison matrix, and whether the evaluation is large enough to detect the effect |
| [`01_data_audit`](notebooks/01_data_audit.ipynb) | Whether the data is fit to train on: split integrity, record integrity, prompt budget, composition |
| [`02_foundation_model_baseline`](notebooks/02_foundation_model_baseline.ipynb) | The reference point every later claim is measured against |
| [`03_lora_finetuning`](notebooks/03_lora_finetuning.ipynb) | The LoRA recipe, and what to watch while it trains |
| [`04_qlora_finetuning`](notebooks/04_qlora_finetuning.ipynb) | The QLoRA recipe, and that it differs from LoRA in exactly one respect |
| [`05_execution_evaluation`](notebooks/05_execution_evaluation.ipynb) | Whether fine-tuning helped, as a paired difference with an interval |
| [`06_error_analysis`](notebooks/06_error_analysis.ipynb) | What to build next, from the structure of the failures |
| [`07_learning_curves`](notebooks/07_learning_curves.ipynb) | Whether labelling more data is worth it |
| [`08_adapter_rank_ablation`](notebooks/08_adapter_rank_ablation.ipynb) | Whether adapter capacity is the binding constraint |
| [`09_base_vs_posttrained`](notebooks/09_base_vs_posttrained.ipynb) | How much capability comes from post-training vs task adaptation |

Three conventions keep them trustworthy:

- **Reusable logic lives in `src/qwen_text2sql/reporting/`**, which is type
  checked and unit tested. Notebooks orchestrate and interpret; they do not
  define the analysis.
- **A missing input fails loudly.** Every notebook declares its prerequisites and
  raises `MissingArtifact` naming the exact command that produces the file,
  rather than rendering an empty table that looks like a result.
- **No output is committed.** Notebooks are stored without outputs or execution
  counts, enforced by `nbstripout` and `make notebooks`.

Notebooks 00, 03 and 04 run end to end with no data at all. The rest stop at
their first missing prerequisite with instructions.

To read a results tree that is not this repository — an archived run, or a
colleague's — set `QWEN_TEXT2SQL_ARTIFACT_ROOT` to a directory containing `data/`
and `results/`. Only artifacts move; code and configuration stay put.

```bash
poetry install --with dev,notebooks
poetry run jupyter lab
```

## Quality gates

```bash
make check      # lint + format-check + typecheck + test
make notebooks  # static notebook validation
```

Individual targets are listed by `make help`. Install the git hooks once with
`make hooks` so formatting and the large-file guard run on every commit.

| Gate | Enforces |
|---|---|
| `make lint` | Ruff rules `E`, `F`, `I`, `UP`, `B`, `SIM`, `RUF`, notebooks included |
| `make format-check` | Ruff formatting (`make format` fixes) |
| `make typecheck` | `mypy --strict` over `src/qwen_text2sql` |
| `make test` | pytest with branch coverage |
| `make notebooks` | every notebook cell compiles and carries no committed outputs |

CI runs all of these on Python 3.11, 3.12 and 3.13 for every push and pull
request, and additionally builds and metadata-checks the distribution. Install
the hooks with `make hooks-push` so the same checks run before you push rather
than after.

Neither the local gates nor CI download model weights or BIRD databases. Unit tests build small temporary SQLite databases and verify schema extraction, read-only execution, result equivalence, splitting, formatting, metrics and bootstrap logic.

## Repository structure

```text
configs/             experiment configurations
data/                local raw/intermediate/processed data
notebooks/           notebook-first experiment walkthrough
results/             metrics, evaluated records and generated predictions
scripts/             reproducible command-line experiment entry points
src/qwen_text2sql/   reusable implementation
tests/               unit and regression tests
```

## Reproducibility rules

1. Never optimize a decoding or training choice on the final evaluation set.
2. Keep schema-identical databases out of both sides of an internal train/validation split.
3. Save configuration, dataset hash, model ID and number of training examples with every training run.
4. Keep generated predictions and per-example evaluation records, not only aggregate scores.
5. Compare models on the same examples and report uncertainty for accuracy differences.
6. Do not claim semantic correctness from SQL text similarity alone.
7. Treat benchmark annotation errors separately from model errors.
8. Do not commit downloaded model weights, BIRD databases, generated checkpoints or prediction dumps.

## External references

- Qwen3.5 model: https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3.5 base model: https://huggingface.co/Qwen/Qwen3.5-4B-Base
- Transformers Qwen3.5 implementation: https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md
- TRL SFTTrainer: https://huggingface.co/docs/trl/en/sft_trainer
- PEFT quantization guide: https://huggingface.co/docs/peft/developer_guides/quantization
- BIRD filtered training data: https://huggingface.co/datasets/birdsql/bird23-train-filtered
- BIRD revised development data: https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), which
covers the development setup, the quality gates, and the research-integrity
rules a change to the experimental protocol must respect. Participation is
governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

Security vulnerabilities must be reported privately — see
[SECURITY.md](SECURITY.md). Note that this project executes model-generated SQL
in order to score it; run evaluation against copies of benchmark databases, never
against a database holding real data.

Released changes are recorded in [CHANGELOG.md](CHANGELOG.md), and planned work
in [docs/roadmap.md](docs/roadmap.md).

## Citation

If you use this repository, cite it using the metadata in
[CITATION.cff](CITATION.cff), or via the "Cite this repository" button on GitHub.

## License

Project code is MIT licensed. Qwen model checkpoints and BIRD data remain governed by their own licenses and terms. The BIRD Hugging Face releases used here are published under CC BY-SA 4.0; Qwen3.5 model cards specify Apache 2.0 for the model artifacts.
