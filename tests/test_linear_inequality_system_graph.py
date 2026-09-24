"""Exact plans for mounted two-inequality union/intersection surfaces."""

import pytest
from test_solve_math import Reasoner

from facet_runtime.graph import LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN
from facet_runtime.solve import SolveRefused, parse_problem, solve_math


def problem(connector: str, expressions: list[str]):
    return parse_problem(
        {
            "result_kind": LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN,
            "instruction": "Solve the system of two linear inequalities graphically.",
            "expressions": expressions,
            "graph": {
                "family": "linear-inequality-system",
                "controls": "mounted-boundaries-combined-regions",
                "connector": connector,
            },
        }
    )


def test_or_is_an_exact_union_with_mixed_boundary_styles_and_no_model():
    reason = Reasoner()

    result = solve_math(problem("or", ["x>6", "y>=5"]), reason=reason)

    assert result["route"] == "exact"
    assert result["answer"] == {
        "kind": LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN,
        "plan": {
            "kind": "linear-inequality-system",
            "connector": "or",
            "operation": "union",
            "inequalities": [
                {
                    "coefficients": {"x": "1", "y": "0", "constant": "-6"},
                    "relation": ">",
                    "boundary": "dashed",
                },
                {
                    "coefficients": {"x": "0", "y": "1", "constant": "-5"},
                    "relation": ">=",
                    "boundary": "solid",
                },
            ],
        },
    }
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert reason.calls == []


def test_and_is_an_exact_intersection():
    result = solve_math(problem("and", ["x>6", "y>=5"]), reason=Reasoner())

    assert result["answer"]["plan"]["operation"] == "intersection"
    assert result["answer"]["plan"]["connector"] == "and"


@pytest.mark.parametrize(
    ("connector", "expressions"),
    [("xor", ["x>6", "y>=5"]), ("or", ["x>6"]), ("and", ["x>6", "y>=5", "y<9"])],
)
def test_invalid_connector_or_member_count_is_refused(connector, expressions):
    with pytest.raises(SolveRefused):
        solve_math(problem(connector, expressions), reason=Reasoner())
