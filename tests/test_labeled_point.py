from __future__ import annotations

import pytest

from facet_runtime.exact import ExactlyRefused, solve_exact
from facet_runtime.solve import SolveRefused, parse_problem, solve_math


@pytest.mark.parametrize(
    ("point", "written"),
    [
        (("4", "2"), "(4,2)"),
        (("-7", "5"), "(-7,5)"),
        (("0", "-3"), "(0,-3)"),
        (("6", "0"), "(6,0)"),
        (("0", "0"), "(0,0)"),
    ],
)
def test_a_structurally_read_labeled_point_is_an_exact_ordered_pair(point, written):
    solution, decline = solve_exact(
        "Identify the coordinates of the point Q on the graph.",
        [],
        points=[point],
        answer_parts=2,
    )

    assert decline == ""
    assert solution is not None
    assert solution.display == written
    assert solution.entry == ""
    assert solution.parts == point
    assert solution.form == "ordered-pair"
    assert solution.method == "Exact labeled Cartesian point"


def test_the_public_router_uses_no_model_for_the_page_owned_point():
    problem = parse_problem(
        {
            "instruction": "Identify the coordinates of the point Q on the graph.",
            "points": [{"x": "-4", "y": "0"}],
            "answer_parts": 2,
        }
    )

    def model_was_called(_prompt):  # pragma: no cover - failure sentinel
        raise AssertionError("a model was called for exact graph geometry")

    result = solve_math(problem, reason=model_was_called)

    assert result["route"] == "exact"
    assert result["answer"]["form"] == "ordered-pair"
    assert result["answer"]["parts"] == ["-4", "0"]
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_point_then_coordinates_wording_takes_the_same_exact_route():
    solution, decline = solve_exact(
        "State point B's coordinates.",
        [],
        points=[("0", "-5")],
        answer_parts=2,
    )

    assert decline == ""
    assert solution is not None
    assert solution.form == "ordered-pair"
    assert solution.parts == ("0", "-5")


@pytest.mark.parametrize(
    ("points", "answer_parts", "reason"),
    [
        ([("1", "2"), ("3", "4")], 2, "one identified point"),
        ([("1", "2")], 1, "two coordinate components"),
    ],
)
def test_ambiguous_identity_or_answer_shape_is_claimed_and_refused(
    points, answer_parts, reason
):
    with pytest.raises(ExactlyRefused, match=reason):
        solve_exact(
            "Determine the coordinates of point R.",
            [],
            points=points,
            answer_parts=answer_parts,
        )


def test_an_ambiguous_public_request_never_falls_through_to_reasoning():
    problem = parse_problem(
        {
            "instruction": "Identify the coordinates of point R.",
            "points": [{"x": "1", "y": "2"}, {"x": "3", "y": "4"}],
            "answer_parts": 2,
        }
    )

    with pytest.raises(SolveRefused, match="one identified point") as refused:
        solve_math(problem, reason=lambda _prompt: pytest.fail("model called"))

    assert refused.value.kind == "unusable_result"
