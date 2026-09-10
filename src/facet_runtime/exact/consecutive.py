"""Consecutive integers determined by their stated sum.

For ``n`` ordinary consecutive integers beginning with ``x``, the question is
the single exact equation

    n*x + n*(n - 1)/2 = total.

The prose says what sequence is meant and which member or members are wanted;
an inline MathJax total may arrive separately in ``expressions``.  Neither is
useful without the other, so this solver reads both and refuses whenever that
pairing is not unique.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

CONSECUTIVE_METHOD = "exact consecutive-integer sum"

# Broad claim for this one closed family.  A claimed question is either solved
# here or refused here; it is never handed to a model to guess a sign or which
# integer was requested.
CONSECUTIVE_REQUEST = re.compile(
    r"(?=.*\bconsecutive\b[^.?!]{0,40}\b(?:integers?|numbers?)\b)"
    r"(?=.*\b(?:sum|total|add(?:s|ed|ing)?\s+up)\b)",
    re.IGNORECASE | re.DOTALL,
)

_COUNTS = {"two": 2, "three": 3, "four": 4, "five": 5}
_COUNTED_SEQUENCE = re.compile(
    r"\b(?P<count>two|three|four|five|[2-5])\s+consecutive\s+"
    r"(?:(?P<parity>even|odd)\s+)?(?:integers?|numbers?)\b",
    re.IGNORECASE,
)
_PARITY_SEQUENCE = re.compile(
    r"\bconsecutive\s+(?:even|odd)\s+(?:integers?|numbers?)\b",
    re.IGNORECASE,
)

_INTEGER = r"(?P<value>[+\-\N{MINUS SIGN}]?\d[\d,]*)"
_WRITTEN_TOTALS = (
    re.compile(
        rf"\b(?:sum|total)\b[^.?!]{{0,140}}?\b"
        rf"(?:is|equals?|is\s+equal\s+to)\s*{_INTEGER}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\badd(?:s|ed|ing)?\s+up\s+to\s*{_INTEGER}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:have|has|had)\s+(?:a\s+)?(?:sum|total)\s+of\s*{_INTEGER}\b",
        re.IGNORECASE,
    ),
)
_TOTAL_HOLES = (
    re.compile(
        r"\b(?:sum|total)\b[^.?!]{0,140}?\b"
        r"(?:is|equals?|is\s+equal\s+to)\s*(?=[.?!])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\badd(?:s|ed|ing)?\s+up\s+to\s*(?=[.?!])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:have|has|had)\s+(?:a\s+)?(?:sum|total)\s+of\s*(?=[.?!])",
        re.IGNORECASE,
    ),
)

_SCALAR_REQUEST = re.compile(
    r"\b(?:find|determine|identify|give|what\s+is|which\s+is)\b"
    r"[^.?!]{0,100}?\b"
    r"(?P<member>smallest|least|first|largest|greatest|last|middle)\b",
    re.IGNORECASE,
)
_ALL_REQUEST = re.compile(
    r"\b(?:find|determine|identify|list|give|what\s+are|which\s+are)\b"
    r"[^.?!]{0,100}?\b(?:integers?|numbers?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ConsecutiveAnswer:
    """The exact values requested, with enough working to audit the result."""

    values: tuple[str, ...]
    first: int
    count: int
    total: int


def _integer(text: str) -> int | None:
    """Read one standalone integer expression without evaluating syntax."""
    normalized = text.strip().replace("\N{MINUS SIGN}", "-").replace(",", "")
    if re.fullmatch(r"[+-]?\d+", normalized) is None:
        return None
    return int(normalized)


def _stated_total(instruction: str, expressions: list[str]) -> tuple[int | None, str]:
    """Read the one total attached to the sum relationship.

    MathJax is deliberately absent from the prose contract, leaving punctuation
    immediately after the relationship.  Only that explicit hole may consume a
    standalone integer expression; unrelated expressions are not guessed into
    the equation.
    """
    written = [
        value
        for pattern in _WRITTEN_TOTALS
        for match in pattern.finditer(instruction)
        if (value := _integer(match.group("value"))) is not None
    ]
    expression_values = [
        value
        for expression in expressions
        if (value := _integer(expression)) is not None
    ]

    if written:
        if len(written) != 1 or expression_values:
            return None, "the sum relationship does not state exactly one total"
        return written[0], ""

    if not any(pattern.search(instruction) for pattern in _TOTAL_HOLES):
        return None, "the sum relationship does not state a total"
    if len(expression_values) != 1:
        return None, (
            "the MathJax hole for the sum must supply exactly one integer total; "
            f"it supplies {len(expression_values)}"
        )
    return expression_values[0], ""


def _requested_values(
    instruction: str, sequence: tuple[int, ...]
) -> tuple[int, ...] | None:
    """Select exactly the member or members the question asks for."""
    scalar = list(_SCALAR_REQUEST.finditer(instruction))
    all_values = list(_ALL_REQUEST.finditer(instruction))
    if len(scalar) == 1:
        member = scalar[0].group("member").lower()
        if member in {"smallest", "least", "first"}:
            return (sequence[0],)
        if member in {"largest", "greatest", "last"}:
            return (sequence[-1],)
        if len(sequence) % 2 == 1:
            return (sequence[len(sequence) // 2],)
        return None
    if not scalar and len(all_values) == 1:
        return sequence
    return None


def solve_consecutive_sum(
    instruction: str, expressions: list[str], answer_parts: int
) -> tuple[ConsecutiveAnswer | None, str]:
    """Solve one ordinary consecutive-integer sum problem, or refuse it."""
    counted = list(_COUNTED_SEQUENCE.finditer(instruction))
    if _PARITY_SEQUENCE.search(instruction) or any(
        match.group("parity") for match in counted
    ):
        return None, (
            "consecutive even or odd integers are not the ordinary step-one "
            "consecutive-integer family"
        )
    counts = {
        _COUNTS.get(
            match.group("count").lower(),
            int(match.group("count")) if match.group("count").isdigit() else 0,
        )
        for match in counted
    }
    if len(counts) != 1:
        return None, "the question does not state exactly one sequence length"

    [count] = counts
    total, refusal = _stated_total(instruction, expressions)
    if total is None:
        return None, refusal

    first = (Fraction(total) - Fraction(count * (count - 1), 2)) / count
    if first.denominator != 1:
        return None, (
            f"{total} cannot be the sum of {count} ordinary consecutive integers"
        )
    start = int(first)
    sequence = tuple(start + offset for offset in range(count))
    if sum(sequence) != total:
        return None, "the computed integers do not satisfy the stated sum"

    wanted = _requested_values(instruction, sequence)
    if wanted is None:
        return None, "the question does not unambiguously say which integer(s) it wants"
    if len(wanted) != answer_parts:
        return None, (
            f"the question asks for {len(wanted)} integer(s), but the answer "
            f"surface requires {answer_parts}"
        )
    return (
        ConsecutiveAnswer(
            values=tuple(str(value) for value in wanted),
            first=start,
            count=count,
            total=total,
        ),
        "",
    )
