"""Schema-grounded conversational formatting for SFT and inference."""

from __future__ import annotations

from typing import TypedDict

from qwen_text2sql.types import PreparedExample

SYSTEM_MESSAGE = (
    "You are a text-to-SQL system. Generate exactly one read-only SQLite query that answers "
    "the user's question using only the supplied schema. Return SQL only. Do not explain it."
)


class Message(TypedDict):
    """Minimal chat-message schema understood by Hugging Face chat templates."""

    role: str
    content: str


def user_content(example: PreparedExample) -> str:
    """Build a deterministic user message from a prepared example."""
    evidence = example.evidence.strip()
    evidence_block = f"\n\nEvidence:\n{evidence}" if evidence else ""
    return (
        f"Database schema:\n{example.schema}"
        f"{evidence_block}\n\nQuestion:\n{example.question}\n\nSQL:"
    )


def conversation_prompt(example: PreparedExample) -> list[Message]:
    """Return the system+user prompt for inference."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content(example)},
    ]


def sft_record(example: PreparedExample) -> dict[str, list[Message]]:
    """Return a conversational prompt-completion record for TRL SFTTrainer."""
    return {
        "prompt": conversation_prompt(example),
        "completion": [{"role": "assistant", "content": example.gold_sql.strip()}],
    }
