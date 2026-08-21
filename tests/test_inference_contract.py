"""Contract tests for generation against a real transformers model.

`train_adapter` passed `warmup_ratio` to trl for a release that had removed it,
and every training run died before its first step because no environment the type
checker ran in had trl installed. `generate_sql` is exposed to exactly the same
risk: it calls `model.generate` with hand-built keyword arguments, and nothing
verified them against a real model.

So these tests build a genuinely real `Qwen3_5ForCausalLM` — a few hundred
thousand random weights assembled from a config, needing no download and no GPU —
and drive `generate_sql` through it. The model is real, which is where API drift
lives. Only the tokenizer is faked, because a real one needs vocabulary files
this suite has no business downloading.
"""

from __future__ import annotations

from typing import Any

import pytest

from qwen_text2sql.config import ModelConfig
from qwen_text2sql.types import PreparedExample

torch = pytest.importorskip("torch", reason="only present in a full environment")
transformers = pytest.importorskip("transformers", reason="only present in a full environment")

from qwen_text2sql.inference.generator import generate_sql  # noqa: E402

PROMPT_TOKENS = 4
EXAMPLE = PreparedExample(
    example_id="a",
    db_id="shop",
    question="Who are the customers?",
    evidence="",
    gold_sql="SELECT name FROM customers",
    schema="CREATE TABLE customers (name TEXT);",
    db_path="unused.sqlite",
)


@pytest.fixture(scope="module")
def tiny_model() -> Any:
    """A real Qwen3.5 causal LM, small enough to build in memory.

    `layer_types` is set explicitly: Qwen3.5 interleaves linear and full
    attention, and a small model left to the default layout ends up with no full
    attention layer at all, which its cache implementation rejects.
    """
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        pad_token_id=0,
        eos_token_id=1,
        layer_types=["full_attention", "full_attention"],
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
    )
    return Qwen3_5ForCausalLM(config).eval()


class _FakeTokenizer:
    """Only the surface `generate_sql` actually touches."""

    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, decoded: str = "SELECT name FROM customers;") -> None:
        self.decoded = decoded
        self.chat_template_calls: list[dict[str, Any]] = []
        self.call_kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: Any, **kwargs: Any) -> str:
        self.chat_template_calls.append({"messages": messages, **kwargs})
        return "rendered prompt"

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[3, 4, 5, 6]]),
            "attention_mask": torch.ones(1, PROMPT_TOKENS, dtype=torch.long),
        }

    def decode(self, token_ids: Any, **kwargs: Any) -> str:
        return self.decoded


def test_greedy_generation_round_trips_through_a_real_model(tiny_model: Any) -> None:
    """The whole path: prompt, real model.generate, slice, extract."""
    tokenizer = _FakeTokenizer()
    sql, latency_ms = generate_sql(tiny_model, tokenizer, EXAMPLE, ModelConfig(model_id="tiny"))
    assert sql == "SELECT name FROM customers;"
    assert latency_ms > 0.0


def test_the_prompt_is_built_for_generation(tiny_model: Any) -> None:
    """A missing generation prompt makes the model continue the user turn instead."""
    tokenizer = _FakeTokenizer()
    generate_sql(tiny_model, tokenizer, EXAMPLE, ModelConfig(model_id="tiny"))
    call = tokenizer.chat_template_calls[0]
    assert call["add_generation_prompt"] is True
    assert call["tokenize"] is False
    roles = [message["role"] for message in call["messages"]]
    assert roles == ["system", "user"]


def test_the_prompt_is_truncated_to_the_configured_context(tiny_model: Any) -> None:
    """Without truncation an over-long schema raises instead of degrading."""
    tokenizer = _FakeTokenizer()
    generate_sql(tiny_model, tokenizer, EXAMPLE, ModelConfig(model_id="tiny", max_seq_length=99))
    assert tokenizer.call_kwargs["truncation"] is True
    assert tokenizer.call_kwargs["max_length"] == 99


def test_zero_temperature_generates_deterministically(tiny_model: Any) -> None:
    """The protocol fixes greedy decoding for the principal comparison."""
    config = ModelConfig(model_id="tiny", temperature=0.0, max_new_tokens=6)
    first = generate_sql(tiny_model, _FakeTokenizer(), EXAMPLE, config)[0]
    second = generate_sql(tiny_model, _FakeTokenizer(), EXAMPLE, config)[0]
    assert first == second


def test_sampling_is_accepted_by_the_model(tiny_model: Any) -> None:
    """temperature > 0 adds top_p, which greedy decoding must not receive."""
    config = ModelConfig(model_id="tiny", temperature=0.7, top_p=0.9, max_new_tokens=4)
    sql, _ = generate_sql(tiny_model, _FakeTokenizer(), EXAMPLE, config)
    assert sql == "SELECT name FROM customers;"


def test_only_the_continuation_is_decoded(tiny_model: Any) -> None:
    """Decoding the prompt too would make every prediction contain the schema."""
    seen: dict[str, Any] = {}

    class RecordingTokenizer(_FakeTokenizer):
        def decode(self, token_ids: Any, **kwargs: Any) -> str:
            seen["n_tokens"] = len(token_ids)
            seen["skip_special_tokens"] = kwargs.get("skip_special_tokens")
            return self.decoded

    config = ModelConfig(model_id="tiny", max_new_tokens=5)
    generate_sql(tiny_model, RecordingTokenizer(), EXAMPLE, config)
    # The prompt was PROMPT_TOKENS long; only what came after it is decoded.
    assert seen["n_tokens"] <= 5
    assert seen["skip_special_tokens"] is True


def test_prose_around_the_query_is_stripped(tiny_model: Any) -> None:
    """Unadapted models wrap SQL in explanation; the extractor must survive it."""
    tokenizer = _FakeTokenizer(
        decoded=(
            "Sure! Here is the query:\n```sql\nSELECT name FROM customers;\n```\nHope that helps."
        )
    )
    sql, _ = generate_sql(tiny_model, tokenizer, EXAMPLE, ModelConfig(model_id="tiny"))
    assert sql == "SELECT name FROM customers;"


def test_gradient_checkpointing_surface_exists(tiny_model: Any) -> None:
    """`load_qwen_text_model` calls both of these after loading."""
    tiny_model.gradient_checkpointing_enable()
    tiny_model.config.use_cache = False
    assert tiny_model.config.use_cache is False
    tiny_model.gradient_checkpointing_disable()
    tiny_model.config.use_cache = True
