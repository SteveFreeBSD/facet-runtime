"""An absolute-value rewrite is two comparisons plus its logical connector."""

from __future__ import annotations

import pytest

from facet_runtime.exact import INEQUALITY_PAIR, solve_exact
from facet_runtime.exact.router import ExactlyRefused
from facet_runtime.solve import MathProblem, solve_math

ASK = "Rewrite the given inequality as two linear inequalities."


@pytest.mark.parametrize(
    ("expression", "left", "connector", "right"),
    [
        ("|2x+10|<6", "2x+10>-6", "and", "2x+10<6"),
        (r"|2x+10|\leq6", "2x+10>=-6", "and", "2x+10<=6"),
        ("|2x+10|>6", "2x+10<-6", "or", "2x+10>6"),
        (r"|2x+10|\geq6", "2x+10<=-6", "or", "2x+10>=6"),
    ],
)
def test_each_relation_preserves_its_connector_and_strictness(
    expression, left, connector, right
):
    solved, decline = solve_exact(ASK, [expression])

    assert solved is not None, decline
    assert solved.form == INEQUALITY_PAIR
    assert solved.entry == ""
    assert solved.parts == ()
    assert solved.inequality_pair.left.written == left
    assert solved.inequality_pair.connector == connector
    assert solved.inequality_pair.right.written == right


def test_the_live_family_is_exact_and_makes_no_model_call():
    def model(_prompt):
        raise AssertionError("the rewrite reached a model")

    result = solve_math(
        MathProblem(instruction=ASK, expressions=("|9x+1|<8",)), reason=model
    )

    assert result["route"] == "exact"
    assert result["answer"]["form"] == INEQUALITY_PAIR
    assert result["answer"]["inequality_pair"] == {
        "left": {"left": "9x+1", "relation": ">", "right": "-8"},
        "connector": "and",
        "right": {"left": "9x+1", "relation": "<", "right": "8"},
    }
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_rewrite_preserves_a_negative_leading_affine_expression_verbatim():
    solved, decline = solve_exact(ASK, [r"|-9y+10|\geq34"])

    assert solved is not None, decline
    assert solved.inequality_pair.left.written == "-9y+10<=-34"
    assert solved.inequality_pair.connector == "or"
    assert solved.inequality_pair.right.written == "-9y+10>=34"
    assert "9*y - 10" not in solved.display


def test_the_rewrite_is_not_silently_replaced_by_interval_notation():
    solved, _ = solve_exact(ASK, ["|x|<3"])

    assert solved.display == "x>-3 AND x<3"
    assert solved.inequality_pair is not None


@pytest.mark.parametrize("expression", ["|x|<0", "|x|>=0", "|x|<y", "|x|+x<3"])
def test_degenerate_or_non_affine_rewrites_are_refused(expression):
    with pytest.raises(ExactlyRefused):
        solve_exact(ASK, [expression])
