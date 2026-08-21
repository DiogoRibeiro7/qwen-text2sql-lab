# Research Protocol

## Objective

Estimate the incremental effect of parameter-efficient task adaptation on executable text-to-SQL performance for Qwen3.5-4B.

## Primary estimand

For a fixed evaluation set, the primary contrast is the paired difference in execution accuracy between an adapted model and its unadapted foundation-model baseline.

## Primary endpoint

Execution equivalence between generated and reference SQL results on the same SQLite database.

## Secondary endpoints

- SQL execution validity;
- normalized textual exact match;
- generation latency;
- execution error category;
- performance stratified by database and available difficulty labels.

## Experimental controls

- fixed random seeds;
- identical evaluation examples across compared models;
- deterministic greedy decoding for the principal comparison;
- database-disjoint internal validation;
- no final-set tuning;
- adapter-only checkpoints;
- stored per-example predictions and evaluations.

## Planned ablations

### Training-set size

Use 250, 500, 1000, 2500, 5000 examples when available, plus the full training set. The same seed and sampling procedure must be used when nested subsets are required.

### Adapter rank

Use ranks 4, 8, 16, 32 and 64 while holding other training settings fixed.

### Foundation checkpoint

Compare the post-trained checkpoint with the pretraining-only Base checkpoint to separate general post-training from domain/task adaptation.

## Statistical analysis

Use paired bootstrap resampling over evaluation examples for differences in binary execution success. Report the observed difference, percentile interval and bootstrap probability that the difference is positive.

## Failure conditions

The core hypothesis is not supported if fine-tuning does not improve held-out execution accuracy beyond sampling uncertainty, if gains disappear on unseen database schemas, or if gains are explained by evaluation leakage.
