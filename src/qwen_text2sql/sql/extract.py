"""Utilities for turning raw model text into one SQL statement."""

from __future__ import annotations

import re

_CODE_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", flags=re.IGNORECASE | re.DOTALL)
_SQL_START_RE = re.compile(r"\b(SELECT|WITH)\b", flags=re.IGNORECASE)


def extract_sql(text: str) -> str:
    """Extract the first plausible read-only SQL statement from model output."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = text.strip()
    if not value:
        return ""
    fenced = _CODE_FENCE_RE.search(value)
    if fenced:
        value = fenced.group(1).strip()
    start = _SQL_START_RE.search(value)
    if start:
        value = value[start.start() :]
    # Text-to-SQL answers are single statements. The first semicolon terminates it.
    semicolon = value.find(";")
    if semicolon >= 0:
        value = value[: semicolon + 1]
    return value.strip()


def normalize_sql(sql: str) -> str:
    """Normalize whitespace and a terminal semicolon for strict textual comparison."""
    value = sql.strip().rstrip(";").strip()
    return re.sub(r"\s+", " ", value).casefold()
