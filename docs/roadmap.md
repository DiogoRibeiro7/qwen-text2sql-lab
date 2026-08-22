# Roadmap

What is left to do, in the order the dependencies allow. Items are grouped by
what they unblock rather than by how interesting they are.

Two rules shape the ordering:

1. **Nothing is claimed without evidence in `results/`.** Every capability item
   below is worthless until the pipeline that measures it is trustworthy, which
   is why Phase 1 comes before Phase 2 even though Phase 2 is the interesting
   part.
2. **No result exists yet.** The repository has never been run against real BIRD
   data on a GPU. Every number in every notebook is currently a placeholder or a
   synthetic fixture, and the first real run is Phase 0.

---

## Phase 0 — Unblock

Nothing downstream can be validated until these are done. None of them are
research; all of them are currently blocking.

| # | Item | Why it blocks |
|---|---|---|
| 0.1 | **Make the repository public, or fund Actions minutes** | CI has never executed a single step. Every run has failed on billing before starting, so the workflow itself is unverified. Actions is free on public repositories. |
| 0.2 | **Re-enable the CI triggers** | A two-line edit in `.github/workflows/ci.yml`, once 0.1 is resolved. Until then the only gate is `make check` on a developer's machine. |
| 0.3 | **Enable the Zenodo toggle, then cut `v0.1.0`** | Zenodo cannot archive a private repository, and does not backfill: the toggle must precede the release. See [`releasing.md`](releasing.md). |
| 0.4 | **First GPU run** | `make install`, `make check`, then a short `train_adapter` on a few hundred examples. The training path is verified end to end on CPU with a synthetic checkpoint, but never against a real one. |
| 0.5 | **First real BIRD run** | `prepare-bird`, `split`, baseline generation and evaluation. This produces the first genuine numbers and exercises the data contract against real schemas. |

Going public also exposes the commit author email across the whole history,
which is worth deciding on deliberately rather than discovering.

---

## Phase 1 — Make the primary result trustworthy

The repository's claim is a paired difference in execution accuracy. These items
harden that claim. They come before any new capability, because a capability
measured with an untrustworthy instrument tells you nothing.

### 1.1 Integrate the official BIRD evaluator

The local comparator treats rows as a multiset with a numerical tolerance. It is
deliberately transparent and inspectable, but it is *ours*. Run the benchmark's
official evaluator alongside it as an external check, pinning the exact
version or commit used. Report both, and treat a disagreement as a finding.

### 1.2 Database-clustered bootstrap

**The current interval is optimistic and we know it.** BIRD examples cluster by
database: errors within one schema are correlated, so the effective sample size
is nearer the number of *databases* than the number of questions. The
example-level paired bootstrap in `evaluation/bootstrap.py` ignores this, and the
minimum-detectable-effect table in notebook 00 inherits the same optimism — it
says so, but saying so is not fixing it.

Add a clustered bootstrap that resamples databases and then examples within
them, report both intervals side by side, and document the estimand difference.

### 1.3 Repeated nested learning curves

Every learning-curve point is currently one seed. Two seeds at the same size can
differ by more than two adjacent points do, so a kink in the curve is
uninterpretable. Repeat each cell across seeds with nested subsets and report
dispersion, not just a line.

### 1.4 Decoding-hyperparameter protocol

Greedy decoding is fixed for the principal comparison, which is correct. But
there is no protocol for the decoding choices themselves, and rule 1 forbids
tuning them on the final evaluation set. Define where those choices are allowed
to be made and record them.

### 1.5 Difficulty and structural performance analysis

`accuracy_by_difficulty` exists and is tested, but nothing yet analyses
performance by *query structure* — joins, aggregation, nesting. An aggregate
number that hides "fails on every multi-join query" is not actionable.

### 1.6 Result registry and provenance graph

`run_metadata.json` records configuration, dataset digest and model id per run.
There is no index across runs, so answering "which config produced this number"
means reading directories. Build the registry and make `validate_results.py`
refuse a partial run that could be mistaken for a complete one.

### 1.7 Publication evidence pipeline

One command that regenerates every table and figure in a write-up from
`results/`, so a paper cannot drift from the evidence behind it.

---

## Phase 2 — Capability

Things that might actually move execution accuracy. Each needs Phase 1 in place
to be measurable.

| # | Item | Hypothesis it tests |
|---|---|---|
| 2.1 | **Schema linking** | Failures are dominated by referencing the wrong tables and columns, not by SQL syntax |
| 2.2 | **Database value retrieval** | Questions referencing literal values fail because the model cannot see them |
| 2.3 | **Schema-constrained decoding** | Invalid SQL can be prevented at decode time rather than detected afterwards |
| 2.4 | **Execution-guided self-correction** | A model shown its own execution error can repair the query |
| 2.5 | **SQL AST diagnostics** | Structural error categories are more actionable than SQLite's error strings |

Which of these to do first should be decided by
[`06_error_analysis`](../notebooks/06_error_analysis.ipynb) on real data, not in
advance. Its first split — invalid SQL versus valid-but-wrong — points directly
at 2.3 or at 2.1/2.2 respectively.

### 2.6 Factorial and target-module ablations

The current sweeps vary one factor at a time. A factorial design over rank,
learning rate and dataset size would detect interactions that one-at-a-time
sweeps cannot. Separately, ablate *which* modules the adapter targets, which is
currently `all-linear` and never questioned.

---

## Phase 3 — Beyond supervised fine-tuning

Only worth starting once Phase 1 can measure a difference reliably, since these
methods are noisier than SFT.

- **Rejection-sampling self-training** — the cheapest of the three, and a
  natural first step: sample, keep what executes correctly, retrain.
- **Preference optimization from execution outcomes** — pairs derived from
  verifiable success rather than human preference.
- **Reinforcement learning with a verifiable execution reward** — the most
  expensive and the most likely to reward-hack. Execution equivalence is a
  proxy; a policy optimised against a proxy will find its gaps.

---

## Phase 4 — Generalisation

The current design measures one model family on one benchmark. These test
whether anything transfers.

- **LiveSQLBench out-of-distribution evaluation** — the honest external check.
  Treat it as a final evaluation set: no tuning against it, ever.
- **Foundation model family comparison** — fair comparison needs identical
  prompts, adapters and decoding, which is harder than it sounds.
- **Qwen Coder baseline** — is a code-pretrained model of the same size better?
- **Multiple SQL dialects** — the execution sandbox is SQLite-specific, so this
  is a larger change than it appears.

---

## Cross-cutting

### Sandbox hardening (partially done)

Already in place: the read-only prefix check, `mode=ro`, `PRAGMA query_only`, a
wall-clock progress handler, and adversarial tests covering mutations, DDL,
`ATTACH`, `PRAGMA`, statement chaining, extension loading and CTE-hidden writes.
A test asserts the database file is byte-identical after every attack, and both
sandbox layers are pinned independently.

Still missing: a **SQLite authorizer callback**, a **row-count cap**, and
**memory limits**. The authorizer is the strongest of the three, because it
refuses categories of operation rather than pattern-matching statements.

### Efficiency instrumentation

Latency is recorded per query. Nothing records training time, peak memory or
energy, so the QLoRA memory claim in notebook 04 is arithmetic rather than
measurement.

### Reproducible GPU container

The lock file pins Python dependencies but not CUDA, drivers or system
libraries. A container closes that gap and makes 0.4 reproducible by someone
else.

### Adapter model card and release

If an adapter is published, it needs a model card stating training data,
intended use, evaluation results and limitations — and the licence boundaries
between code (MIT), BIRD (CC BY-SA 4.0) and Qwen (Apache 2.0).

---

## Engineering debt

Small, known, and none of it blocking.

| Item | Note |
|---|---|
| CUDA-only paths untested | `from_pretrained` on a real device, `prepare_model_for_kbit_training`, gradient checkpointing. Argument construction is covered; the calls themselves are not. |
| `CHANGELOG.md` conflicts on every PR | Every change edits the same region. A `changelog.d/` fragment directory assembled at release time would end it. |
| Notebook outputs are stripped | Deliberate — diffs stay readable and no stale number can masquerade as current. If rendered results are wanted on GitHub, export them separately rather than committing outputs. |
| `prompts/` is untracked | The research backlog lives in `.git/info/exclude`, so it is invisible to collaborators. This document is the tracked synthesis; consider tracking the source too. |

---

## What "done" means here

Taken from the guardrails this project already holds itself to, and worth
restating because every item above inherits them:

- No claim without generated evidence in `results/`.
- No tuning on the final evaluation set.
- Database-disjoint validation wherever schema generalisation is measured.
- Per-example outputs retained, not only aggregates.
- Every new public behaviour tested; every fixed bug given a regression test.
- No fabricated results when the accelerator or benchmark assets are absent.
