"""Exact two-point plans for intercept-directed linear graphs."""

import pytest
from test_solve_math import Reasoner

from facet_runtime.graph import LINEAR_GRAPH_PLAN, PlanRefused, build_linear_graph_plan
from facet_runtime.solve import parse_problem, solve_math

ASK = (
    "Graph the equation by plotting the x- and y-intercepts. "
    "If an intercept does not exist, or is duplicated, use another point "
    "on the line to plot the graph."
)
GRAPH = {
    "family": "line",
    "orientation": "cartesian",
    "bounds": [-10, 10, -10, 10],
    "snap": [1, 1],
    "controls": "two-points",
}


def problem(equation: str):
    return parse_problem(
        {
            "result_kind": LINEAR_GRAPH_PLAN,
            "instruction": ASK,
            "expressions": [equation],
            "graph": GRAPH,
        }
    )


def test_live_horizontal_line_uses_the_intercept_and_a_simple_substitute():
    reason = Reasoner()

    result = solve_math(problem("4y=8"), reason=reason)

    assert result["route"] == "exact"
    assert result["answer"] == {
        "kind": LINEAR_GRAPH_PLAN,
        "plan": {
            "kind": "line",
            "coefficients": {"x": "0", "y": "1", "constant": "-2"},
            "points": [
                {"x": "0", "y": "2", "role": "y-intercept"},
                {"x": "1", "y": "2", "role": "substitute"},
            ],
        },
    }
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert reason.calls == []


@pytest.mark.parametrize(
    ("equation", "points"),
    [
        (
            "3x=12",
            [
                {"x": "4", "y": "0", "role": "x-intercept"},
                {"x": "4", "y": "1", "role": "substitute"},
            ],
        ),
        (
            "2x+3y=6",
            [
                {"x": "3", "y": "0", "role": "x-intercept"},
                {"x": "0", "y": "2", "role": "y-intercept"},
            ],
        ),
        (
            "y=x",
            [
                {"x": "0", "y": "0", "role": "x-intercept"},
                {"x": "1", "y": "1", "role": "substitute"},
            ],
        ),
    ],
)
def test_intercepts_and_substitutes_are_derived_generically(equation, points):
    result = solve_math(problem(equation), reason=Reasoner())

    assert result["answer"]["plan"]["points"] == points


def test_stated_point_plot_contract_does_not_become_a_line_deriver():
    with pytest.raises(PlanRefused, match="intercept-directed"):
        build_linear_graph_plan(
            "Plot the points (1,2) and (3,4).",
            ["y=x+1"],
            problem("y=x").graph,
        )


@pytest.mark.parametrize("equation", ["y=x^2", "1=1"])
def test_non_lines_are_refused_without_reasoning(equation):
    with pytest.raises(Exception, match="line"):
        solve_math(problem(equation), reason=Reasoner())
