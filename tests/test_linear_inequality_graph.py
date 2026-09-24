"""Exact plans for Cartesian linear-inequality graph answers."""

import pytest
from test_solve_math import Reasoner

from facet_runtime.graph import LINEAR_INEQUALITY_GRAPH_PLAN
from facet_runtime.solve import SolveRefused, parse_problem, solve_math

GRAPH = {
    "family": "linear-inequality",
    "orientation": "cartesian",
    "bounds": [-10, 10, -10, 10],
    "snap": [1, 1],
    "controls": "boundary-two-points-regions",
}


def problem(expression: str | list[str]):
    expressions = [expression] if isinstance(expression, str) else expression
    return parse_problem(
        {
            "result_kind": LINEAR_INEQUALITY_GRAPH_PLAN,
            "instruction": "Graph the solution set of the following linear inequality:",
            "expressions": expressions,
            "graph": GRAPH,
        }
    )


def test_live_strict_inequality_is_an_exact_dashed_plan_with_no_model():
    reason = Reasoner()

    result = solve_math(problem("2x+6y<6"), reason=reason)

    assert result["route"] == "exact"
    assert result["answer"] == {
        "kind": LINEAR_INEQUALITY_GRAPH_PLAN,
        "plan": {
            "kind": "linear-inequality",
            "coefficients": {"x": "1", "y": "3", "constant": "-3"},
            "relation": "<",
            "boundary": "dashed",
            "points": [{"x": "3", "y": "0"}, {"x": "0", "y": "1"}],
        },
    }
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert reason.calls == []


def test_answer_generated_boundary_equality_does_not_steal_partial_graph_resume():
    reason = Reasoner()

    result = solve_math(problem(["2x+6y<6", "y=-x/3+1"]), reason=reason)

    assert result["route"] == "exact"
    assert result["answer"]["plan"]["relation"] == "<"
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert reason.calls == []


def test_two_stated_inequalities_are_refused():
    with pytest.raises(SolveRefused):
        solve_math(problem(["x<2", "y>3"]), reason=Reasoner())


@pytest.mark.parametrize(
    ("expression", "relation", "boundary"),
    [
        ("-2x-6y>=-6", "<=", "solid"),
        ("x>4", ">", "dashed"),
        ("y<=-2", "<=", "solid"),
    ],
)
def test_relation_direction_and_boundary_style_are_normalized(
    expression, relation, boundary
):
    plan = solve_math(problem(expression), reason=Reasoner())["answer"]["plan"]

    assert plan["relation"] == relation
    assert plan["boundary"] == boundary
    assert len({(point["x"], point["y"]) for point in plan["points"]}) == 2


@pytest.mark.parametrize("expression", ["x^2+y<4", "1<2", "x<y<3"])
def test_non_affine_or_non_single_inequalities_are_refused(expression):
    with pytest.raises(SolveRefused):
        solve_math(problem(expression), reason=Reasoner())
