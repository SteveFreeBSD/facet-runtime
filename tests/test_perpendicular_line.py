"""Relative line construction owns the task before generic output wording."""

import pytest
import sympy

from facet_runtime.exact.linear import line_request_intent
from facet_runtime.exact.router import ExactlyRefused, solve_exact
from facet_runtime.solve import MathProblem, SolveRefused, solve_math


def instruction(point="", form="slope-intercept", relationship="perpendicular"):
    return (
        f"Find the equation of the line passing through the stated point {point} "
        f"and {relationship} to the given line. "
        + (f"Express your answer in {form} form. Simplify your answer." if form else "")
    )


@pytest.mark.parametrize(
    ("equation", "point", "source_slope", "new_slope", "intercept", "expected"),
    [
        ("2x+y=5", "(3,-4)", "-2", "1/2", "-11/2", "y=1/2*x-11/2"),
        ("3x-2y=7", "(-2,1)", "3/2", "-2/3", "-1/3", "y=-2/3*x-1/3"),
        ("4x+6y=9", "(-3,2)", "-2/3", "3/2", "13/2", "y=3/2*x+13/2"),
        ("2y=8x+1", "(-1,-2)", "4", "-1/4", "-9/4", "y=-1/4*x-9/4"),
        ("x+y=3", "(0,0)", "-1", "1", "0", "y=x"),
        ("y=x+8", "(-4,-1)", "1", "-1", "-5", "y=-x-5"),
        ("2*(x+1)+5*y=3*y-7", "(-2,4)", "-1", "1", "6", "y=x+6"),
        ("y=0.5x+0.2", "(1/2,-3/4)", "1/2", "-2", "1/4", "y=-2x+1/4"),
        ("y=-2x+1", r"(-\frac{3}{2},\frac{1}{4})", "-2", "1/2", "1", "y=1/2*x+1"),
        ("2x+3y=3y-8", "(-7,-5/2)", "undefined", "0", "-5/2", "y=-5/2"),
    ],
)
def test_new_line_is_exact_through_point_and_orthogonal(
    equation, point, source_slope, new_slope, intercept, expected
):
    def no_model(*args, **kwargs):
        pytest.fail("perpendicular construction reached a model")

    result = solve_math(
        MathProblem(instruction=instruction(), expressions=(equation, point)),
        reason=no_model,
    )
    assert result["route"] == "exact"
    assert result["answer"]["entry"] == expected
    assert result["answer"]["form"] == "relation"
    assert result["provenance"]["source"] == "Facet Exact"
    assert result["provenance"]["method"] == "Exact perpendicular line through a point"
    proof = result["provenance"]["evidence"]
    assert proof["model_calls"] == 0
    assert proof["computation"]["source_slope"] == source_slope
    assert proof["computation"]["perpendicular_slope"] == new_slope
    assert proof["computation"]["y_intercept"] == intercept
    px, py = map(
        sympy.Rational, proof["computation"]["stated_point"].strip("()").split(",")
    )
    assert sympy.Rational(new_slope) * px + sympy.Rational(intercept) == py
    if source_slope != "undefined":
        assert sympy.Rational(source_slope) * sympy.Rational(new_slope) == -1


def test_horizontal_source_yields_vertical_when_form_allows_it():
    solved, _ = solve_exact(instruction("(-2,5)", form=""), ["3y+4=10"])
    assert solved.entry == "x=-2"
    assert solved.relation.subject == "x"
    assert solved.evidence["source_slope"] == "0"
    assert solved.evidence["perpendicular_slope"] == "undefined"


@pytest.mark.parametrize(
    ("prompt", "expressions"),
    [
        (instruction("(2,3)"), ["y=5"]),
        (instruction("(2,3)", form="point-slope"), ["y=x"]),
        (instruction("(2,3)", form="standard"), ["y=x"]),
        (instruction("(2,3) and (4,5)"), ["y=x"]),
        (instruction(), ["y=x"]),
        (instruction("(2,3)"), ["x^2+y=5"]),
        (instruction("(2,3)"), ["ax+y=5"]),
        (instruction("(2,3)"), [r"y=\sqrt{2}x+1"]),
        (instruction("(1/0,3)"), ["x+y=5"]),
        (instruction("(2,3)"), ["x=x"]),
        (instruction("(2,3)"), ["x=x+1"]),
        (instruction("(2,3)"), ["y=x", "y=-x"]),
        (instruction("(2,3)"), ["slope=2"]),
        (instruction("(2,3)"), ["y=x", "7"]),
        (instruction("(2,3)", relationship="parallel and perpendicular"), ["y=x"]),
    ],
)
def test_unsupported_and_ambiguous_constructions_stop_without_reasoning(
    prompt, expressions
):
    with pytest.raises(ExactlyRefused):
        solve_exact(prompt, expressions)

    def no_model(*args, **kwargs):
        pytest.fail("exact refusal fell back to a model")

    with pytest.raises(SolveRefused):
        solve_math(
            MathProblem(instruction=prompt, expressions=tuple(expressions)),
            reason=no_model,
        )


def test_perpendicular_precedence_does_not_rewrite_the_source():
    prompt = instruction("(4,1)")
    assert line_request_intent(prompt, ["2x+y=7"]) == "perpendicular"
    solved, _ = solve_exact(prompt, ["2x+y=7"])
    assert solved.entry == "y=1/2*x-1"


@pytest.mark.parametrize(("equation", "expected"), [("y=2", "y=-4"), ("x=2", "x=-3")])
def test_axis_aligned_parallel_sibling_uses_same_constructor(equation, expected):
    solved, _ = solve_exact(
        instruction("(-3,-4)", form="", relationship="parallel"), [equation]
    )
    assert solved.entry == expected
