## Summary

<!-- What does this change do, and why? -->

## Type of change

- [ ] Bug fix
- [ ] New feature or experiment capability
- [ ] Refactor / internal cleanup
- [ ] Documentation
- [ ] Experiment configuration or protocol change

## Research integrity checklist

Tick every box that applies, or explain why it does not.

- [ ] No decoding, training or stopping rule was tuned on a final evaluation set.
- [ ] Train/validation splits remain database-disjoint.
- [ ] Any reported accuracy change is accompanied by an uncertainty estimate (paired bootstrap).
- [ ] Per-example prediction and evaluation records are preserved, not only aggregates.
- [ ] No model weights, BIRD databases, checkpoints or prediction dumps are committed.

## Verification

- [ ] `make check` passes locally (`lint`, `format-check`, `typecheck`, `test`).
- [ ] `make notebooks` passes if notebooks changed.
- [ ] New or changed behaviour is covered by tests.

<!-- Paste relevant command output or metrics below. -->
