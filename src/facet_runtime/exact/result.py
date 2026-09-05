"""The record a deterministic solver leaves behind when it answers.

Facet's exact solvers are not models, and this is deliberately not a model's
result type. It carries the same fields a consumer already reads off an
execution -- the text that was produced, and a summary of how it was produced
-- so that one reader can handle an exact answer and a reasoned one without
learning two vocabularies. What it does not carry is a model, a runtime, or a
device, because none of those took part.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExactDebugInfo:
    """How a deterministic solve ran. Every count here is zero on purpose."""

    prompt_char_length: int
    schema_top_level_keys: list[str]
    format_kind: str
    num_predict: int
    num_ctx: int
    response_summary: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExactCallResult:
    """One deterministic answer, with the working that produced it."""

    raw_prompt: str
    raw_response: str
    debug_info: ExactDebugInfo | None = field(default=None)
