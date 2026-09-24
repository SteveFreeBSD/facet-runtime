"""A written linear equation is rearranged into slope-intercept form exactly."""

from __future__ import annotations

import pytest

from facet_runtime.exact import RELATION, Relation, solve_exact
from facet_runtime.solve import MathProblem, solve_math


@pytest.mark.parametrize(
    ("instruction", "equation", "expected"),
    [
        pytest.param(
            "Express the given equation in slope-intercept form. Simplify your answer.",
            "4x+7y=13",
            "y=-4/7*x+13/7",
            id="live-sign-regression",
        ),
        pytest.param(
            "Write the equation in slope-intercept form.",
            "-3x+2y=5",
            "y=3/2*x+5/2",
            id="negative-a",
        ),
        pytest.param(
            "Convert the equation to slope-intercept form.",
            "2x-4y=6",
            "y=1/2*x-3/2",
            id="negative-b",
        ),
        pytest.param(
            "Put the equation into slope intercept form.",
            "3x+6y=0",
            "y=-1/2*x",
            id="zero-intercept",
        ),
        pytest.param(
            "Rewrite the equation in slope-intercept form.",
            "2x+y=3",
            "y=-2x+3",
            id="integer-slope-and-intercept",
        ),
        pytest.param(
            "Express the equation in slope intercept form.",
            "6x+9y=12",
            "y=-2/3*x+4/3",
            id="reduced-rationals",
        ),
    ],
)
def test_written_linear_equation_is_rearranged_exactly(instruction, equation, expected):
    solution, decline = solve_exact(instruction, [equation])

    assert decline == ""
    assert solution is not None
    assert solution.entry == expected
    assert solution.display == expected
    assert solution.form == RELATION
    subject, value = expected.split("=", 1)
    assert solution.relation == Relation(subject=subject, value=value)
    assert solution.method == "SymPy exact symbolic"


def test_live_route_uses_no_model_calls():
    def model_must_not_run(_problem):
        raise AssertionError("reasoning model was called")

    result = solve_math(
        MathProblem(
            instruction=(
                "Express the given equation in slope-intercept form. "
                "Simplify your answer."
            ),
            expressions=("4x+7y=13",),
        ),
        reason=model_must_not_run,
    )

    assert result["answer"]["entry"] == "y=-4/7*x+13/7"
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["method"] == "SymPy exact symbolic"


@pytest.mark.parametrize(
    ("equation", "reason"),
    [
        ("4x=13", "cannot be isolated"),
        ("xy+y=3", "requires a linear equation"),
        ("ax+y=3", "unsupported symbol"),
        ("x+y<3", "requires one exact equation"),
    ],
)
def test_rewrite_route_fails_closed(equation, reason):
    solution, decline = solve_exact(
        "Express the given equation in slope-intercept form.", [equation]
    )

    assert solution is None
    assert reason in decline
