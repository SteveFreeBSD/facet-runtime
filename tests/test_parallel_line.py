"""A parallel-line construction outranks generic output-format wording."""

from __future__ import annotations

import pytest

from facet_runtime.exact import (
    RELATION,
    Relation,
    line_request_intent,
    solve_exact,
)
from facet_runtime.solve import MathProblem, solve_math

LIVE = (
    "Find the equation of the line which passes through the point (−8,11) "
    "and is parallel to the given line. Express your answer in "
    "slope-intercept form. Simplify your answer."
)


@pytest.mark.parametrize(
    ("instruction", "equation", "expected"),
    [
        pytest.param(LIVE, "4x+8y=19", "y=-1/2*x+7", id="live-rational-negative"),
        pytest.param(
            "Find the equation which goes through (3,-4), is parallel to the "
            "given equation, and write it in slope-intercept form.",
            "2x+y=1",
            "y=-2x+2",
            id="integer-negative-slope",
        ),
        pytest.param(
            "Express the equation of the line containing the point (-2,1) and "
            "parallel to the given line in slope-intercept form.",
            "3x-2y=5",
            "y=3/2*x+4",
            id="positive-rational-slope-negative-coordinate",
        ),
        pytest.param(
            "Write the equation of a line that passes through (3,-2), is "
            "parallel to the given equation, in slope intercept form.",
            "4x+6y=7",
            "y=-2/3*x",
            id="zero-new-intercept",
        ),
        pytest.param(
            "Find the line through (1,2) parallel to the given line. Express "
            "your answer in slope-intercept form.",
            "2x+3y=4",
            "y=-2/3*x+8/3",
            id="fractional-intercept",
        ),
        pytest.param(
            "Find the equation of the line passing through (1,3) and parallel "
            "to the given line. Write in slope-intercept form.",
            "2x+y=5",
            "y=-2x+5",
            id="point-already-on-source-line",
        ),
    ],
)
def test_parallel_line_uses_source_slope_and_stated_point(
    instruction, equation, expected
):
    solution, decline = solve_exact(instruction, [equation])

    assert decline == ""
    assert solution is not None
    assert solution.entry == expected
    assert solution.display == expected
    assert solution.form == RELATION
    subject, value = expected.split("=", 1)
    assert solution.relation == Relation(subject=subject, value=value)


def test_live_parallel_construction_uses_no_model_calls():
    calls = []

    def reason(prompt):
        calls.append(prompt)
        raise AssertionError("parallel construction reached a model")

    result = solve_math(
        MathProblem(instruction=LIVE, expressions=("4x+8y=19",)), reason=reason
    )

    assert calls == []
    assert result["answer"]["entry"] == "y=-1/2*x+7"
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["evidence"]["computation"] == {
        "source_equation": "4x+8y=19",
        "source_slope": "-1/2",
        "stated_point": "(-8,11)",
        "parallel_slope": "-1/2",
        "y_intercept": "7",
        "verification": "f(-8)=11",
    }


def test_live_reader_shape_keeps_point_as_a_separate_expression():
    solution, refusal = solve_exact(
        "Find the equation of the line which passes through the point and is "
        "parallel to the given line. Express your answer in slope-intercept form.",
        ["4*x+8*y=19", "(−8,11)"],
    )

    assert refusal == ""
    assert solution is not None
    assert solution.entry == "y=-1/2*x+7"


def test_vertical_source_line_fails_closed():
    solution, decline = solve_exact(
        "Find the equation of the line through (2,5) parallel to the given "
        "line. Express your answer in slope-intercept form.",
        ["x=3"],
    )

    assert solution is None
    assert "cannot be isolated as a linear y equation" in decline


@pytest.mark.parametrize(
    ("instruction", "expressions", "expected"),
    [
        (
            "Convert the given equation to slope-intercept form.",
            ["2x+y=3"],
            "rewrite",
        ),
        (
            (
                "Find a line through (1,2) parallel to the given line. Express "
                "your answer in slope-intercept form."
            ),
            ["2x+y=3"],
            "parallel",
        ),
        (
            (
                "Find a line through (1,2) perpendicular to the given equation. "
                "Write your answer in slope-intercept form."
            ),
            ["2x+y=3"],
            "perpendicular",
        ),
        (
            "Find the equation through (1,2) and (3,4) in slope-intercept form.",
            ["(1,2)", "(3,4)"],
            "two-points",
        ),
        (
            "Write the equation through (1,2) in point-slope form.",
            ["slope=3"],
            "point-slope",
        ),
    ],
)
def test_richer_line_intent_precedes_output_format(instruction, expressions, expected):
    assert line_request_intent(instruction, expressions) == expected
