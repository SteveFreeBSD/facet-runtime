"""Plot the points a question states, without asking anything.

Live, on 2026-09-07: "plot the following points in the Cartesian plane", four
ordered pairs written into the prompt, no answer box, and four draggable point
controls on the graph. The pairs are the answer -- there is nothing to derive,
so no model is engaged at all.
"""

from __future__ import annotations

import pytest

from facet_runtime.graph import PlanRefused, build_point_plot_plan, read_plot_points
from facet_runtime.solve import parse_problem, solve_math

LIVE = "Plot the following points in the Cartesian plane: (1,2), (-3,4), (0,-5), (6,0)"


def refuses(_prompt):
    raise AssertionError("a stated-points question must never reach a model")


def test_the_stated_points_are_read_exactly() -> None:
    assert read_plot_points(LIVE, []) == [
        {"x": "1", "y": "2"},
        {"x": "-3", "y": "4"},
        {"x": "0", "y": "-5"},
        {"x": "6", "y": "0"},
    ]


def test_the_route_is_deterministic_and_engages_no_model() -> None:
    result = solve_math(
        parse_problem({"result_kind": "point_plot_plan", "instruction": LIVE}),
        reason=refuses,
    )

    assert result["route"] == "exact"
    assert result["answer"]["plan"]["kind"] == "points"
    assert len(result["answer"]["plan"]["points"]) == 4
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_a_repeated_point_is_kept_because_the_set_is_the_answer() -> None:
    plan = build_point_plot_plan("Plot the points (2,2) and (2,2).", [])

    assert len(plan["points"]) == 2


def test_exact_rationals_survive_and_decimals_are_not_invented() -> None:
    plan = build_point_plot_plan("Plot the points (1/2,-3/4) and (0,0).", [])

    assert plan["points"] == [{"x": "1/2", "y": "-3/4"}, {"x": "0", "y": "0"}]


@pytest.mark.parametrize(
    "instruction",
    [
        "Find the distance between the points (1,2) and (3,4).",
        "Simplify the following expression.",
    ],
)
def test_another_question_about_pairs_is_not_claimed(instruction) -> None:
    with pytest.raises(PlanRefused):
        build_point_plot_plan(instruction, [])


def test_a_question_stating_no_points_is_refused_by_name() -> None:
    with pytest.raises(PlanRefused, match="stated points"):
        build_point_plot_plan("Plot the following points in the plane.", [])
