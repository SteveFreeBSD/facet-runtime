"""Exact, structured axis-intercept answers."""

import pytest
from test_solve_math import Reasoner

from facet_runtime.exact import AXIS_INTERCEPTS, ExactlyRefused, solve_exact
from facet_runtime.solve import MathProblem, solve_math

ASK = "Find the x- and y-intercepts, if possible."


@pytest.mark.parametrize(
    ("equation", "x_point", "y_point"),
    [
        ("4y=8", None, ("0", "2")),
        ("3x=12", ("4", "0"), None),
        ("2x+3y=6", ("3", "0"), ("0", "2")),
        ("2x-3y=0", ("0", "0"), ("0", "0")),
        ("2x+4y=1", ("1/2", "0"), ("0", "1/4")),
    ],
)
def test_axis_intercepts_are_exact_and_named(equation, x_point, y_point):
    solution, decline = solve_exact(ASK, [equation], answer_parts=2)

    assert solution is not None, decline
    assert solution.form == AXIS_INTERCEPTS
    assert solution.entry == ""
    assert solution.parts == ()
    assert solution.intercepts is not None
    actual_x = (
        None
        if solution.intercepts.x is None
        else (
            solution.intercepts.x.x,
            solution.intercepts.x.y,
        )
    )
    actual_y = (
        None
        if solution.intercepts.y is None
        else (
            solution.intercepts.y.x,
            solution.intercepts.y.y,
        )
    )
    assert (actual_x, actual_y) == (x_point, y_point)


def test_live_equation_crosses_as_structured_exact_answer_with_no_model_call():
    reason = Reasoner()
    result = solve_math(
        MathProblem(instruction=ASK, expressions=("4y=8",), answer_parts=2),
        reason=reason,
    )

    assert result["route"] == "exact"
    assert result["answer"]["form"] == AXIS_INTERCEPTS
    assert result["answer"]["intercepts"] == {
        "x": None,
        "y": {"x": "0", "y": "2"},
    }
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert reason.calls == []


@pytest.mark.parametrize("equation", ["y=x^2", "y=0", "1=1"])
def test_non_affine_or_non_unique_intercepts_are_refused_not_reasoned(equation):
    with pytest.raises(ExactlyRefused):
        solve_exact(ASK, [equation], answer_parts=2)
