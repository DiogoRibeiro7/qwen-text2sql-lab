# Documentation

| Document | Read it when you need to know |
|---|---|
| [research_protocol.md](research_protocol.md) | What is being estimated, the primary and secondary endpoints, the experimental controls, the planned ablations, and the conditions under which the hypothesis is *not* supported. |
| [architecture.md](architecture.md) | How data flows from BIRD metadata through schema extraction, training, generation, execution and statistics — and why ML imports are lazy. |
| [roadmap.md](roadmap.md) | What is left to do and in what order — the blockers, the work that makes the primary result trustworthy, and the research directions behind it. |
| [releasing.md](releasing.md) | How to cut a release and archive it with Zenodo for a DOI, and why that needs a public repository. |
| [data_contract.md](data_contract.md) | The exact fields of a prepared JSONL record, their types and their meaning. |

Operational documentation lives outside this directory:

- [../README.md](../README.md) — installation, and the end-to-end command sequence for every experiment.
- [../CONTRIBUTING.md](../CONTRIBUTING.md) — development setup, quality gates and the rules a protocol change must respect.
- [../SECURITY.md](../SECURITY.md) — the threat model around executing model-generated SQL, and how to report a vulnerability.
- [../CHANGELOG.md](../CHANGELOG.md) — released changes, including any that alter a reported metric.
- [../notebooks/](../notebooks/) — the experiment walkthrough; `00_research_protocol.ipynb` is the executable form of `research_protocol.md`.

## Reading order for a new contributor

1. `research_protocol.md` — the repository exists to answer one question; start there.
2. `architecture.md` — how the code is arranged to answer it.
3. `data_contract.md` — the record format everything else assumes.
4. `../CONTRIBUTING.md` — how to land a change without weakening a claim.
