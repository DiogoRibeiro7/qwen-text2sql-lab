# Architecture

```text
BIRD metadata + SQLite databases
          |
          v
  schema extraction
          |
          v
 schema-grounded JSONL
      /           \\
     v             v
baseline         SFT data
inference        formatting
     |             |
     |       LoRA / QLoRA
     |             |
     +-------> prediction JSONL
                    |
                    v
            read-only execution
                    |
                    v
       per-example evaluation JSONL
                    |
                    v
       metrics + paired bootstrap
```

The package intentionally separates deterministic data/evaluation code from GPU-dependent training code. ML libraries are imported lazily inside training and inference functions so repository QA can run without model downloads.
