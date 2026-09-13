"""An absolute-value inequality is two linear ones, and is solved exactly.

Live, on 2026-09-12: `|z + 3| <= 1`, "express your answer in interval
notation". The exact inequality route claimed the question, met `Abs(z + 3)`,
declined it as "not polynomial", and a reasoning model answered `(-∞,∞)` --
repeatedly -- for a set that is `[-4,-2]`.

These hold the family generically: every comparison, a reversed comparison when
the absolute value is isolated by a negative, exact rational and decimal
endpoints, the empty set and the whole line where the bound says so, a union
where `>` or `>=` needs one, and a named decline for everything else.

Every answer these hold is read from the router. The consumer's capability gate
observes the answers this suite builds, and an answer held at the solver is one
it never sees: rational endpoints were once held there because the gate had not
declared them, and a live question was answered with one all the same.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact import SCALAR, solve_exact
from facet_runtime.exact.inequality import INEQUALITY_METHOD, solve_linear_inequality
from facet_runtime.solve import MathProblem, solve_math

ASK = "Solve the absolute value inequality. Write your answer in interval notation."
DECIMAL = f"{ASK} Use decimal form for numerical values."


def written(expression: str, instruction: str = ASK) -> str:
    solution, decline = solve_exact(instruction, [expression])
    assert solution is not None, decline
    assert solution.method == INEQUALITY_METHOD
    return solution.entry


def declined(expression: str, instruction: str = ASK) -> str:
    solved, decline = solve_linear_inequality(instruction, [expression])
    assert solved is None, solved
    return decline


def refuses(prompt: str):
    raise AssertionError("an absolute-value inequality reached the reasoning route")


# --- it reaches Facet Exact, with no model ------------------------------------


def test_the_live_question_is_answered_exactly_with_no_model():
    result = solve_math(
        MathProblem(instruction=ASK, expressions=[r"|z+3|\leq1"]), reason=refuses
    )

    assert result["route"] == "exact"
    assert result["answer"]["entry"] == "[-4,-2]"
    assert result["answer"]["form"] == "scalar"
    assert result["provenance"]["method"] == INEQUALITY_METHOD
    assert result["provenance"]["router"] == "solved"
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_a_union_reaches_facet_exact_too():
    result = solve_math(
        MathProblem(instruction=ASK, expressions=[r"|2w-5|\geq3"]), reason=refuses
    )

    assert result["route"] == "exact"
    assert result["answer"]["entry"] == "(-∞,1]∪[4,∞)"
    assert result["provenance"]["evidence"]["model_calls"] == 0


@pytest.mark.parametrize(
    "expression",
    [
        r"|z+3|\leq1",
        r"\left|z+3\right|\leq 1",
        "|z + 3| ≤ 1",
        r"1\geq|z+3|",
        # Multiplied through by -1, so the comparison turns round with it.
        "−1 ≤ −|z + 3|",
    ],
)
def test_every_spelling_of_the_same_inequality_is_the_same_set(expression):
    solution, decline = solve_exact(ASK, [expression])

    assert solution is not None, decline
    assert solution.entry == "[-4,-2]"
    assert solution.form == SCALAR
    assert solution.entry_mode == "math"


# --- each comparison ---------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "interval"),
    [
        ("|2x-5|<3", "(1,4)"),
        (r"|2x-5|\leq3", "[1,4]"),
        ("|2x-5|>3", "(-∞,1)∪(4,∞)"),
        (r"|2x-5|\geq3", "(-∞,1]∪[4,∞)"),
        ("3>|2x-5|", "(1,4)"),
        (r"3\leq|2x-5|", "(-∞,1]∪[4,∞)"),
    ],
)
def test_each_comparison_gives_its_own_set(expression, interval):
    assert written(expression) == interval


def test_a_negative_coefficient_inside_the_bars_is_the_same_distance():
    assert written(r"|3-2x|\leq5") == "[-1,4]"
    assert written("|-x-2|>1") == "(-∞,-3)∪(-1,∞)"


def test_the_absolute_value_is_isolated_before_it_is_split():
    assert written(r"2|x-1|+3\leq7") == "[-1,3]"


def test_negating_both_sides_without_turning_the_comparison_is_another_set():
    """`-1 >= -|z+3|` is `|z+3| >= 1`, not the live question."""
    assert written("−1 ≥ −|z + 3|") == "(-∞,-4]∪[-2,∞)"


def test_isolating_by_a_negative_turns_the_comparison_round():
    solution, _ = solve_exact(ASK, ["-3|x+2|<-6"])

    assert solution is not None
    assert solution.entry == "(-∞,-4)∪(0,∞)"
    assert "> 2" in solution.evidence["steps"]


def test_a_chain_around_an_absolute_value_is_both_comparisons():
    assert written("2<|x|<5") == "(-5,-2)∪(2,5)"


# --- exact endpoints ----------------------------------------------------------


def test_rational_coefficients_give_exact_rational_endpoints():
    assert written(r"|\frac{1}{2}x-1|<\frac{3}{4}") == "(1/2,7/2)"
    assert written(r"|3x+1|\geq2") == "(-∞,-1]∪[1/3,∞)"


def test_decimal_coefficients_are_exact_rationals():
    assert written(r"|0.5x+1|\leq2") == "[-6,2]"


def test_decimal_form_writes_terminating_endpoints_exactly():
    assert written("|2x-1|<2", DECIMAL) == "(-0.5,1.5)"
    assert written(r"|4x-1|\geq1", DECIMAL) == "(-∞,0]∪[0.5,∞)"


def test_decimal_form_declines_an_endpoint_no_decimal_writes():
    assert "no exact decimal form" in declined(r"|3x+1|\geq2", DECIMAL)


# --- the empty set and the whole line -----------------------------------------


@pytest.mark.parametrize(
    ("expression", "interval"),
    [
        ("|x-1|<0", "∅"),
        ("|x-1|<-2", "∅"),
        (r"|x-1|\leq-2", "∅"),
        ("|x-1|>-1", "(-∞,∞)"),
        (r"|x-1|\geq-1", "(-∞,∞)"),
        (r"|x-1|\geq0", "(-∞,∞)"),
        ("|x-1|>0", "(-∞,1)∪(1,∞)"),
    ],
)
def test_the_bound_decides_the_empty_set_and_the_whole_line(expression, interval):
    assert written(expression) == interval


def test_the_empty_set_and_the_line_reach_facet_exact():
    empty, _ = solve_exact(ASK, ["|x-1|<-2"])
    line, _ = solve_exact(ASK, ["|x-1|>-2"])

    assert empty is not None and empty.entry == "∅"
    assert line is not None and line.entry == "(-∞,∞)"


def test_a_single_point_is_not_written_as_an_interval():
    assert "not an interval" in declined(r"|x-1|\leq0")


# --- outside the family, fail-closed -------------------------------------------


@pytest.mark.parametrize(
    ("expression", "reason"),
    [
        ("|x^2-4|<1", "not linear"),
        # Refused as written, before a cancelled factor could hide the divisor.
        (r"|\frac{1}{x}|<2", "divides by its variable"),
        ("|x+y|<2", "exactly one variable"),
        ("|x-1|<x", "variable outside the bars"),
        ("|x|+|x-1|<3", "could not be read exactly"),
        ("||x|-1|<2", "could not be read exactly"),
        ("|x-1|<3 or |x+1|>5", "disjunction"),
    ],
)
def test_outside_the_family_is_declined_by_name(expression, reason):
    assert reason in declined(expression)


def test_a_declined_absolute_value_inequality_names_its_gap_to_the_router():
    solution, decline = solve_exact(ASK, ["|x^2-4|<1"])

    assert solution is None
    assert decline == "an absolute value that is not linear"


def test_set_builder_notation_is_still_declined():
    solution, decline = solve_exact(
        "Solve the absolute value inequality. Write the answer in set-builder notation.",
        [r"|z+3|\leq1"],
    )

    assert solution is None
    assert "interval notation" in decline
