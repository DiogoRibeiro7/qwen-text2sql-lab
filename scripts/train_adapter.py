#!/usr/bin/env python3
"""Train one configured LoRA/QLoRA adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from qwen_text2sql.config import load_config
from qwen_text2sql.training.train import train_adapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--max-train-examples", type=int)
    args = parser.parse_args()
    adapter = train_adapter(
        load_config(args.config),
        args.train_data,
        args.validation_data,
        max_train_examples=args.max_train_examples,
    )
    print(adapter)


if __name__ == "__main__":
    main()
