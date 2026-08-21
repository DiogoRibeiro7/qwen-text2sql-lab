"""Deterministic development splits."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

from qwen_text2sql.types import PreparedExample


def split_by_database(
    rows: Sequence[PreparedExample], validation_fraction: float = 0.15, seed: int = 42
) -> tuple[list[PreparedExample], list[PreparedExample]]:
    """Create a database-disjoint train/validation split.

    Database-level splitting avoids leaking an identical schema into both subsets.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")
    by_db: dict[str, list[PreparedExample]] = defaultdict(list)
    for row in rows:
        by_db[row.db_id].append(row)
    if len(by_db) < 2:
        raise ValueError("At least two databases are required for a database-disjoint split")
    ranked = sorted(
        by_db,
        key=lambda db_id: hashlib.sha256(f"{seed}:{db_id}".encode()).hexdigest(),
    )
    n_validation = max(1, round(len(ranked) * validation_fraction))
    validation_ids = set(ranked[:n_validation])
    train = [row for row in rows if row.db_id not in validation_ids]
    validation = [row for row in rows if row.db_id in validation_ids]
    if not train or not validation:
        raise RuntimeError("Split unexpectedly produced an empty partition")
    return train, validation
