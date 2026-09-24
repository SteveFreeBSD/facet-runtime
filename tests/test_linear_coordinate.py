"""Affine coordinate steps are exact and bounded, never model guesses."""

import pytest

from facet_runtime.exact.linear_coordinate import parse_task, solve_coordinate
from facet_runtime.solve import parse_problem, solve_math


def domain(
    axis="x",
    given=None,
    bounds=("-4", "6", "-3", "5"),
    steps=("1", "1"),
    rational=False,
):
    return {
        "axis": axis,
        "given": given,
        "bounds": list(bounds),
        "steps": list(steps),
        "allow_rational": rational,
    }


def solve(instruction, equation, task=None):
    payload = {"instruction": instruction, "expressions": [equation]}
    if task is not None:
        payload["coordinate_task"] = task

    def no_model(*args, **kwargs):
        pytest.fail("coordinate step called a model")

    return solve_math(parse_problem(payload), reason=no_model)


@pytest.mark.parametrize(
    ("equation", "prompt", "expected"),
    [
        ("3x+2y=8", "Given x = 2, determine the value for y.", "1"),
        ("3x+2y=8", "Given y = -2, determine the value for x.", "4"),
        ("3x+2y=8", "When x = 1, calculate the value of y.", "5/2"),
        ("x=0", "Given y = 3, find the x-coordinate.", "0"),
        ("y=-4", "Given x = 0, find the y-coordinate.", "-4"),
        ("3x+2y=8", "Given x = 1/3, determine the value for y.", "7/2"),
    ],
)
def test_supplied_coordinate_is_exact_without_a_model(equation, prompt, expected):
    result = solve(prompt, equation)
    assert result["answer"]["entry"] == expected
    assert result["answer"]["form"] == "scalar"
    assert result["provenance"]["evidence"]["model_calls"] == 0


@pytest.mark.parametrize(
    ("equation", "task", "point"),
    [
        ("y=9x+2", domain(), ("0", "2")),
        ("y=9x+20", domain(bounds=("-5", "4", "-2", "3")), ("-2", "2")),
        ("2x+3y=7", domain(), ("2", "1")),
        ("x=0", domain(), ("0", "0")),
        ("y=-1", domain(axis="y"), ("0", "-1")),
        ("y=x-100", domain(bounds=("101", "105", "1", "3")), ("101", "1")),
        ("y=1/2", domain(steps=("1", "1/2"), rational=True), ("0", "1/2")),
    ],
)
def test_free_coordinate_chooses_a_complete_visible_point(equation, task, point):
    scalar, actual = solve_coordinate(
        [equation], task["axis"], task["given"], parse_task(task)
    )
    assert actual == point
    assert scalar == point[0 if task["axis"] == "x" else 1]
    result = solve("Choose a value for " + task["axis"] + ".", equation, task)
    assert result["answer"]["entry"] == scalar
    assert result["provenance"]["evidence"]["model_calls"] == 0


@pytest.mark.parametrize(
    ("prompt", "equation", "task"),
    [
        ("Choose a value for x.", "y=x+100", domain()),
        ("Choose a value for x.", "y=1/2", domain()),
        ("Choose a value for x.", "y=1/2", domain(steps=("1", "1/2"))),
        ("Given x = 5, determine the value for y.", "y=3x", domain("y", "5")),
        ("Given x = 2, determine the value for y.", "x=2", None),
        ("Given x = 2, determine the value for y.", "y=x^2", None),
        ("Given x = 2, determine the value for y.", "y=1/x", None),
        ("Given x = 2 and given x = 3, find the value for y.", "x+y=5", None),
        ("Choose a value for x.", "x+y=5", None),
    ],
)
def test_ambiguity_missing_bounds_and_unusable_geometry_never_reason(
    prompt, equation, task
):
    from facet_runtime.solve import SolveRefused

    with pytest.raises(SolveRefused):
        solve(prompt, equation, task)
