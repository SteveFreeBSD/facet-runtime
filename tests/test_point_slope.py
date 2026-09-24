"""Exact slope from two structurally identified Cartesian points."""

from __future__ import annotations

import pytest

from facet_runtime.exact import ExactlyRefused, solve_exact
from facet_runtime.solve import parse_problem, solve_math


@pytest.mark.parametrize(
    ("points", "answer"),
    [
        ([(-5, -5), (3, -6)], "-1/8"),
        ([(1, 2), (4, 8)], "2"),
        ([(-3, 4), (5, 4)], "0"),
        ([(2, -7), (-4, 5)], "-2"),
    ],
)
def test_two_graph_points_produce_one_reduced_exact_slope(points, answer):
    solution, decline = solve_exact(
        "Find the slope of the line in the graph using the points on the graph.",
        [],
        points=[(str(x), str(y)) for x, y in points],
        answer_parts=1,
    )

    assert decline == ""
    assert solution is not None
    assert solution.display == answer
    assert solution.entry == answer
    assert solution.form == "scalar"
    assert solution.method == "Exact slope from two Cartesian points"


def test_vertical_line_uses_the_prompt_owned_dne_convention():
    solution, decline = solve_exact(
        "Find the slope. If the slope does not exist, write DNE for your answer.",
        [],
        points=[("4", "-2"), ("4", "7")],
        answer_parts=1,
    )

    assert decline == ""
    assert solution is not None
    assert solution.entry == "DNE"
    assert solution.form == "scalar"


def test_public_route_never_calls_a_model_for_two_exact_points():
    problem = parse_problem(
        {
            "instruction": "Find the slope of the line using the points shown.",
            "points": [{"x": "-5", "y": "-5"}, {"x": "3", "y": "-6"}],
            "answer_parts": 1,
        }
    )
    result = solve_math(
        problem,
        reason=lambda _prompt: pytest.fail("model called for exact graph points"),
    )

    assert result["route"] == "exact"
    assert result["answer"]["display"] == "-1/8"
    assert result["answer"]["form"] == "scalar"
    assert result["provenance"]["evidence"]["model_calls"] == 0


@pytest.mark.parametrize(
    ("points", "answer_parts", "reason"),
    [
        ([("1", "2")], 1, "exactly two"),
        ([("1", "2"), ("3", "4"), ("5", "6")], 1, "exactly two"),
        ([("1", "2"), ("3", "4")], 2, "one scalar"),
        ([("4", "-2"), ("4", "7")], 1, "does not state"),
    ],
)
def test_ambiguous_shape_or_vertical_notation_is_claimed_and_refused(
    points, answer_parts, reason
):
    with pytest.raises(ExactlyRefused, match=reason):
        solve_exact(
            "Find the slope of the line using the points shown.",
            [],
            points=points,
            answer_parts=answer_parts,
        )
