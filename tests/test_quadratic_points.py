"""Exact points on a supplied quadratic, with named landmarks excluded."""

from __future__ import annotations

import pytest
import sympy

from facet_runtime.exact import solve_exact
from facet_runtime.solve import MathProblem, solve_math

ONE = "Find one point on the graph other than the vertex and the x-intercepts."
TWO = "Find two points on the parabola other than the vertex and the x-intercepts."


@pytest.mark.parametrize(
    ("expression", "instruction", "answer_parts", "entry", "parts"),
    [
        ("f(x)=(x-2)*(x+2)", ONE, 1, "(1,-3)", ()),
        ("y=x^2-4", ONE, 1, "(1,-3)", ()),
        (
            "p(x)=(x-2)*(x+2)",
            TWO,
            2,
            "",
            ("(1,-3)", "(-1,-3)"),
        ),
        ("y=x^2-4", TWO, 2, "", ("(1,-3)", "(-1,-3)")),
    ],
)
def test_factored_and_expanded_quadratics_return_the_requested_shape(
    expression, instruction, answer_parts, entry, parts
) -> None:
    solution, decline = solve_exact(
        instruction, [expression], answer_parts=answer_parts
    )

    assert decline == ""
    assert solution is not None
    assert solution.entry == entry
    assert solution.parts == parts
    assert len(solution.parts or (solution.entry,)) == answer_parts
    # x = 0 is the vertex and x = +/-2 are the intercepts. Neither appears.
    assert solution.evidence["vertex"] == "(0,-4)"
    assert solution.evidence["x_intercepts"] == "-2,2"
    assert solution.evidence["incidence"] == (
        "f(1)=-3" if answer_parts == 1 else "f(1)=-3; f(-1)=-3"
    )


def test_rational_values_and_irrational_intercepts_remain_exact() -> None:
    solution, decline = solve_exact(
        TWO,
        ["f(x)=1/2*x^2+1/3*x-1/6"],
        answer_parts=2,
    )

    assert decline == ""
    assert solution is not None
    assert solution.parts == ("(0,-1/6)", "(1,2/3)")
    assert solution.evidence == {
        "coefficients": "1/2,1/3,-1/6",
        "vertex": "(-1/3,-2/9)",
        "x_intercepts": "-1,1/3",
        "points": "(0,-1/6),(1,2/3)",
        "incidence": "f(0)=-1/6; f(1)=2/3",
    }

    irrational, decline = solve_exact(TWO, ["y=x^2-2"], answer_parts=2)
    assert decline == ""
    assert irrational is not None
    assert irrational.evidence["x_intercepts"] == "-sqrt(2),sqrt(2)"
    assert irrational.parts == ("(1,-1)", "(-1,-1)")


def test_every_returned_point_is_exactly_incident_and_not_excluded() -> None:
    solution, _ = solve_exact(TWO, ["f(x)=(x-3)*(x+1)"], answer_parts=2)

    assert solution is not None
    x = sympy.Symbol("x")
    function = (x - 3) * (x + 1)
    excluded_x = {sympy.Integer(1), sympy.Integer(3), sympy.Integer(-1)}
    for pair in solution.parts:
        point_x, point_y = map(sympy.sympify, pair.strip("()").split(","))
        assert point_x not in excluded_x
        assert sympy.expand(function.subs(x, point_x) - point_y) == 0


@pytest.mark.parametrize(
    "instruction",
    [
        "Find points on the graph other than the vertex and the x-intercepts.",
        "Find one or two points on the graph other than the vertex and the x-intercepts.",
        "Find two points on the graph other than the vertex.",
        "Find two points on the graph other than the vertex and the x-intercepts or y-intercept.",
        "Find one points on the graph other than the vertex and the x-intercepts.",
    ],
)
def test_malformed_or_ambiguous_point_requests_decline(instruction) -> None:
    solution, decline = solve_exact(instruction, ["f(x)=(x-3)*(x+1)"], answer_parts=2)

    assert solution is None
    assert decline == "no exact operation matched the instruction"


def test_count_expression_and_quadratic_ambiguity_decline() -> None:
    mismatched, mismatch = solve_exact(TWO, ["f(x)=(x-3)*(x+1)"], answer_parts=1)
    assert mismatched is None
    assert (
        mismatch == "point request asks for 2 answers but the answer shape requires 1"
    )

    multiple, multiple_reason = solve_exact(
        TWO, ["f(x)=x^2", "g(x)=x^2+1"], answer_parts=2
    )
    assert multiple is None
    assert multiple_reason == "points on a quadratic require one exact function"

    malformed, malformed_reason = solve_exact(TWO, ["f(x)=x^3"], answer_parts=2)
    assert malformed is None
    assert malformed_reason == "points on a quadratic require a rational quadratic"


def test_exact_route_makes_no_reasoning_model_call() -> None:
    calls = []

    def reason(prompt):
        calls.append(prompt)
        raise AssertionError("an exactly solvable point request reached a model")

    result = solve_math(
        MathProblem(
            instruction=TWO,
            expressions=("f(x)=(x-3)*(x+1)",),
            answer_parts=2,
            label="",
        ),
        reason=reason,
    )

    assert calls == []
    assert result["route"] == "exact"
    assert result["answer"]["parts"] == ["(0,-3)", "(2,-3)"]
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["evidence"]["computation"]["incidence"] == (
        "f(0)=-3; f(2)=-3"
    )
