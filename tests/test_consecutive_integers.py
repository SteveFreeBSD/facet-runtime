"""Consecutive-integer sums are exact arithmetic, including negative totals."""

from __future__ import annotations

import pytest

from facet_runtime.solve import MathProblem, SolveRefused, solve_math


def no_reasoning(_prompt: str):
    raise AssertionError("a consecutive-integer sum reached the reasoning route")


def answer(
    instruction: str, expressions: tuple[str, ...] = (), *, answer_parts: int
) -> dict:
    return solve_math(
        MathProblem(
            instruction=instruction,
            expressions=expressions,
            answer_parts=answer_parts,
        ),
        reason=no_reasoning,
    )


@pytest.mark.parametrize(
    ("instruction", "expressions", "parts"),
    [
        (
            "Find three consecutive integers whose sum is .",
            ("42",),
            ["13", "14", "15"],
        ),
        (
            "Three consecutive integers add up to -24. Determine the integers.",
            (),
            ["-9", "-8", "-7"],
        ),
        (
            "The total of four consecutive numbers equals . List the numbers.",
            ("-10",),
            ["-4", "-3", "-2", "-1"],
        ),
        (
            (
                "The sum of three consecutive integers equals 96. "
                "Find the three consecutive integers."
            ),
            (),
            ["31", "32", "33"],
        ),
    ],
)
def test_regenerated_wording_and_mathjax_totals_route_exactly(
    instruction, expressions, parts
):
    result = answer(instruction, expressions, answer_parts=len(parts))

    assert result["answer"]["parts"] == parts
    assert result["route"] == "exact"
    assert result["provenance"]["source"] == "Facet Exact"
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_a_negative_sum_keeps_its_sign_and_sequence_order():
    result = answer(
        "The sum of three consecutive integers is -72. Find the integers.",
        answer_parts=3,
    )

    values = [int(value) for value in result["answer"]["parts"]]
    assert values == [-25, -24, -23]
    assert values != [23, 24, 25]
    assert sum(values) == -72


@pytest.mark.parametrize(
    ("instruction", "expected"),
    [
        (
            "Five consecutive integers have a sum of 35. What is the largest integer?",
            "9",
        ),
        ("Find the smallest of three consecutive integers whose sum is 30.", "9"),
    ],
)
def test_a_requested_member_is_returned_as_one_scalar(instruction, expected):
    result = answer(instruction, answer_parts=1)

    assert result["answer"]["entry"] == expected
    assert result["answer"]["parts"] == []


@pytest.mark.parametrize(
    ("instruction", "expressions", "answer_parts"),
    [
        (
            "Find three consecutive odd integers whose sum is 51.",
            (),
            3,
        ),
        (
            "Three consecutive integers have a sum of .",
            ("51",),
            3,
        ),
        (
            "Find three consecutive integers whose sum is .",
            ("51", "54"),
            3,
        ),
    ],
)
def test_unsupported_or_ambiguous_variants_fail_closed(
    instruction, expressions, answer_parts
):
    with pytest.raises(SolveRefused) as refused:
        answer(instruction, expressions, answer_parts=answer_parts)

    assert refused.value.kind == "unusable_result"
