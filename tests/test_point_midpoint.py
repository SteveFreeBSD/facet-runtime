"""The midpoint of two points, kept rational rather than rounded.

Live, repeatedly: no exact branch existed for this, the question went to
gpt-oss:20b, and the answer came back as `(8.5,-0.5)`. Hawkes refused it. The
midpoint of `(7,-4)` and `(10,3)` is `(17/2,-1/2)`, and the decimal is that
same value written in a notation the answer box has no key for.

Halving the sum of two rationals is exact. These hold that it stays exact, that
the family reaches no model at all, and that a question which merely mentions a
midpoint on the way to asking something else is declined rather than claimed.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact.midpoint import (
    MIDPOINT_METHOD,
    MIDPOINT_REQUEST,
    solve_midpoint,
)
from facet_runtime.solve import MathProblem, solve_math

INSTRUCTION = "Find the midpoint between the following pair of points."

#: The live question, in the shape the page states it.
LIVE_POINTS = ["(7,-4)", "(10,3)"]


def refuses(prompt: str):
    raise AssertionError("a midpoint question reached the reasoning route")


def problem(**changes) -> MathProblem:
    fields = {
        "instruction": INSTRUCTION,
        "expressions": tuple(LIVE_POINTS),
        "answer_parts": 1,
    }
    fields.update(changes)
    return MathProblem(**fields)


# --- the arithmetic stays exact --------------------------------------------


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        # The live question. A half is a half.
        (["(7,-4)", "(10,3)"], "(17/2,-1/2)"),
        # Both coordinates land on integers, and are written as integers.
        (["(1,2)", "(3,4)"], "(2,3)"),
        (["(-3,5)", "(2,-8)"], "(-1/2,-3/2)"),
        # Negative halves keep their sign in the numerator, as SymPy writes it.
        (["(-7,-4)", "(-10,-3)"], "(-17/2,-7/2)"),
        # A point stated with a decimal is read as the rational it is, and the
        # midpoint of two of them is exact -- not a float that happens to print
        # short.
        (["(1.5,2.5)", "(2.5,3.5)"], "(2,3)"),
        (["(0.5,0.25)", "(1.5,0.75)"], "(1,1/2)"),
        # Fractions, written as Hawkes writes them and as a person types them.
        ([r"(\frac{1}{3},2)", r"(1,\frac{7}{2})"], "(2/3,11/4)"),
        (["(1/3,2)", "(1,7/2)"], "(2/3,11/4)"),
        # A point at the origin is a point.
        (["(0,0)", "(5,7)"], "(5/2,7/2)"),
    ],
)
def test_the_midpoint_is_exact_and_never_a_decimal(points, expected):
    answer, refusal = solve_midpoint(INSTRUCTION, points)

    assert refusal == ""
    assert answer == expected
    assert "." not in answer


def test_the_decimal_the_model_produced_is_the_same_value_written_wrongly():
    """`(8.5,-0.5)` is not wrong arithmetic; it is unenterable notation."""
    answer, _ = solve_midpoint(INSTRUCTION, LIVE_POINTS)

    assert answer == "(17/2,-1/2)"
    assert answer != "(8.5,-0.5)"


def test_the_pair_is_the_canonical_ordered_pair_form():
    """The same form every other exact solver answering with a point returns:
    parentheses part of the answer, one comma, no spaces."""
    answer, _ = solve_midpoint(INSTRUCTION, LIVE_POINTS)

    assert answer.startswith("(") and answer.endswith(")")
    assert answer.count(",") == 1
    assert " " not in answer


# --- it is read from wherever the question states it -----------------------


@pytest.mark.parametrize(
    "instruction",
    [
        "Find the midpoint of the line segment joining (1,2) and (3,4).",
        "Determine the mid-point of the segment with endpoints (1,2) and (3,4).",
        "What is the midpoint of the segment between (1,2) and (3,4)?",
        "Compute the mid point of (1,2) and (3,4).",
    ],
)
def test_points_stated_in_the_instruction_are_read(instruction):
    assert MIDPOINT_REQUEST.search(instruction)
    answer, refusal = solve_midpoint(instruction, [])

    assert refusal == "" and answer == "(2,3)"


# --- and declined cleanly otherwise ----------------------------------------


@pytest.mark.parametrize(
    "instruction",
    [
        "Using the midpoint formula, find the distance between the two points.",
        "Find the midpoint and the slope of the line through the points.",
        "Find the midpoint and the length of the segment.",
        "Find the equation of the line through the midpoint of the segment.",
    ],
)
def test_a_question_asking_for_more_than_a_midpoint_is_declined(instruction):
    answer, refusal = solve_midpoint(instruction, LIVE_POINTS)

    assert answer is None
    assert "more than the midpoint" in refusal


@pytest.mark.parametrize(
    "instruction",
    [
        "The midpoint formula is useful here.",
        "Simplify the expression.",
        "Plot the following points in the Cartesian plane.",
    ],
)
def test_wording_that_does_not_ask_for_a_midpoint_is_not_claimed(instruction):
    assert MIDPOINT_REQUEST.search(instruction) is None


@pytest.mark.parametrize(
    "points",
    [[], ["(1,2)"], ["(1,2)", "(3,4)", "(5,6)"]],
)
def test_anything_but_exactly_two_points_is_declined(points):
    answer, refusal = solve_midpoint(INSTRUCTION, points)

    assert answer is None
    assert "needs two points" in refusal


def test_a_pair_that_cannot_be_read_exactly_is_declined():
    answer, refusal = solve_midpoint(INSTRUCTION, ["(a,2)", "(3,4)"])

    assert answer is None
    assert refusal


# --- through the router, with no model at all ------------------------------


def test_the_family_is_answered_exactly_and_asks_no_model():
    result = solve_math(problem(), reason=refuses)

    assert result["route"] == "exact"
    assert result["answer"]["display"] == "(17/2,-1/2)"
    assert result["answer"]["entry"] == "(17/2,-1/2)"
    assert result["answer"]["entry_mode"] == "verbatim"
    assert result["answer"]["parts"] == []
    assert result["provenance"]["router"] == "solved"
    assert result["provenance"]["method"] == MIDPOINT_METHOD
    assert result["provenance"]["model"] is None
    assert result["provenance"]["device"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_the_distance_between_the_same_two_points_is_still_its_own_answer():
    """Neither solver may claim the other's question."""
    from facet_runtime.exact.distance import solve_point_distance

    distance, refusal = solve_point_distance(
        "Find the distance between the following pair of points.", LIVE_POINTS
    )

    assert refusal == "" and distance == "sqrt(58)"
    assert solve_midpoint(
        "Find the distance between the following pair of points.", LIVE_POINTS
    ) == (None, "the question asks for more than the midpoint")
