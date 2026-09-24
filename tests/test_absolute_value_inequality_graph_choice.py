"""Rendered graph alternatives are matched by exact inequality semantics."""

from __future__ import annotations

import pytest

from facet_runtime.exact import CHOICE, solve_exact
from facet_runtime.solve import MathProblem, solve_math

ASK = "Determine which graph represents the original inequality."


def no_model(_prompt):
    raise AssertionError("an exact graph choice reached a model")


def test_live_strict_absolute_value_graph_is_chosen_semantically():
    choices = (
        "Graph: x=-4 dashed; x=3 dashed; shade=outside",
        "Graph: x=-3 dashed; x=4 dashed; shade=between",
        "Graph: x=-4 dashed; x=3 dashed; shade=between",
        "Graph: x=-3 dashed; x=4 dashed; shade=outside",
    )

    result = solve_math(
        MathProblem(
            instruction=ASK,
            expressions=("|8x+4|<28",),
            answer_choices=choices,
        ),
        reason=no_model,
    )

    assert result["route"] == "exact"
    assert result["answer"]["form"] == CHOICE
    assert result["answer"]["entry"] == choices[2]
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert (
        result["provenance"]["evidence"]["computation"]
        | {
            "left_boundary": "-4",
            "right_boundary": "3",
            "left_style": "dashed",
            "right_style": "dashed",
            "shading": "between",
            "matching_choices": "1",
        }
        == result["provenance"]["evidence"]["computation"]
    )


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (r"|x|\leq3", "Graph: x=-3 solid; x=3 solid; shade=between"),
        (r"|x|\geq3", "Graph: x=-3 solid; x=3 solid; shade=outside"),
        ("|x|>3", "Graph: x=-3 dashed; x=3 dashed; shade=outside"),
    ],
)
def test_boundary_strictness_and_interior_or_exterior_are_preserved(
    expression, expected
):
    alternatives = [
        "Graph: x=-3 solid; x=3 solid; shade=between",
        "Graph: x=-3 solid; x=3 solid; shade=outside",
        "Graph: x=-3 dashed; x=3 dashed; shade=between",
        "Graph: x=-3 dashed; x=3 dashed; shade=outside",
    ]

    solved, refusal = solve_exact(ASK, [expression], choices=alternatives)

    assert solved is not None, refusal
    assert solved.form == CHOICE
    assert solved.entry == expected
