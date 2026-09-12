"""A linear inequality in one variable is exact arithmetic, not a reasoning question.

Live, on 2026-09-12, lesson 1.7: a compound inequality, "solve the inequality
and express your answer in interval notation". No exact branch read an
inequality at all, so the router declined with "no exact operation matched the
instruction" and a model on a GPU answered it in nine seconds.

These hold the family generically: every comparison direction, the comparison
turning round under a negative coefficient, endpoints that stay exact, and a
named decline for everything the family does not claim.
"""

from __future__ import annotations

import pytest
import sympy

from facet_runtime.exact import NO_OPERATION_MATCHED, SCALAR, solve_exact
from facet_runtime.exact.inequality import (
    INEQUALITY_METHOD,
    solve_linear_inequality,
    write_interval,
)
from facet_runtime.result import ExecutionMetrics, RunResult
from facet_runtime.solve import MathProblem, solve_math

ASK = "Solve the inequality and express your answer in interval notation."
DECIMAL = f"{ASK} Use decimal form for numerical values."


def written(expressions: list[str], instruction: str = ASK) -> str:
    solved, decline = solve_linear_inequality(instruction, expressions)
    assert solved is not None, decline
    return solved.written


def declined(expressions: list[str], instruction: str = ASK) -> str:
    solved, decline = solve_linear_inequality(instruction, expressions)
    assert solved is None, solved
    assert decline
    return decline


def refuses(prompt: str):
    raise AssertionError("a linear inequality reached the reasoning route")


# --- recognition: the router claims it, and answers without a model ---------


@pytest.mark.parametrize(
    "instruction",
    [
        ASK,
        "Consider the following compound inequality. " + DECIMAL,
        "Solve the following linear inequalities. Write the solution in interval notation.",
    ],
)
def test_the_router_answers_an_inequality_exactly(instruction):
    solution, decline = solve_exact(instruction, [r"-12<4t+8\leq20"])

    assert solution is not None, decline
    assert solution.display == solution.entry == "(-5,3]"
    assert solution.form == SCALAR
    assert solution.entry_mode == "math"
    assert solution.method == INEQUALITY_METHOD
    assert solution.evidence["variable"] == "t"


def test_it_crosses_the_wire_as_an_exact_route_with_no_model():
    result = solve_math(
        MathProblem(instruction=DECIMAL, expressions=[r"-15\leq 5-4w<21"]),
        reason=refuses,
    )

    assert result["route"] == "exact"
    assert result["answer"]["entry"] == "(-4,5]"
    assert result["answer"]["form"] == "scalar"
    assert result["provenance"]["method"] == INEQUALITY_METHOD
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


@pytest.mark.parametrize(
    "expression",
    [
        r"-1<2x+3\leq9",
        "-1 < 2x + 3 ≤ 9",
        r"-1\lt 2x+3\le 9",
        r"-1<2\left(x+1\right)+1\leq9",
        "−1 < 2x + 3 ≤ 9",
    ],
)
def test_every_spelling_of_the_comparison_reads_the_same(expression):
    assert written([expression]) == "(-2,3]"


def test_a_question_about_an_equation_is_not_claimed():
    solution, _ = solve_exact("Solve the equation.", ["2x+3=9"])

    assert solution is not None
    assert solution.entry == "3"
    assert solution.method != INEQUALITY_METHOD


# --- solving: each direction, strict and not, simple and compound ------------


@pytest.mark.parametrize(
    ("expression", "interval"),
    [
        ("2x+1<7", "(-∞,3)"),
        (r"2x+1\leq7", "(-∞,3]"),
        ("2x+1>7", "(3,∞)"),
        (r"2x+1\geq7", "[3,∞)"),
        ("7>2x+1", "(-∞,3)"),
        (r"7\geq2x+1", "(-∞,3]"),
        ("1<2x+1<7", "(0,3)"),
        (r"1\leq2x+1\leq7", "[0,3]"),
        (r"1<2x+1\leq7", "(0,3]"),
        (r"1\leq2x+1<7", "[0,3)"),
        ("7>2x+1>1", "(0,3)"),
        (r"7\geq2x+1>1", "(0,3]"),
    ],
)
def test_each_comparison_gives_its_own_ends(expression, interval):
    assert written([expression]) == interval


@pytest.mark.parametrize(
    "expressions",
    [
        [r"x+2>0 \text{ and } 3x\leq12"],
        ["x+2>0 and 3x≤12"],
        ["x+2>0", r"3x\leq12"],
    ],
)
def test_a_conjunction_is_the_intersection(expressions):
    assert written(expressions) == "(-2,4]"


def test_variables_on_both_sides_are_collected():
    assert written([r"5x-3\geq2x+9"]) == "[4,∞)"


def test_a_conjunction_with_nothing_in_common_is_empty():
    assert written(["x>5", "x<2"]) == "∅"


def test_a_contradiction_and_an_identity_are_the_empty_set_and_the_line():
    assert written(["x+1<x"]) == "∅"
    assert written([r"2(x+1)\geq2x"]) == "(-∞,∞)"


# --- dividing by a negative turns the comparison round ----------------------


@pytest.mark.parametrize(
    ("expression", "interval"),
    [
        ("-2x<6", "(-3,∞)"),
        (r"-2x\leq6", "[-3,∞)"),
        ("-2x>6", "(-∞,-3)"),
        (r"-2x\geq6", "(-∞,-3]"),
        (r"4<-2x\leq10", "[-5,-2)"),
        (r"-\frac{1}{3}x+1>2", "(-∞,-3)"),
    ],
)
def test_a_negative_coefficient_reverses_the_comparison(expression, interval):
    assert written([expression]) == interval


def test_the_working_says_the_comparison_turned_round():
    solved, _ = solve_linear_inequality(ASK, ["5-4w<21"])

    assert solved is not None
    assert "turns round" in solved.evidence["steps"]
    assert solved.solution == sympy.Interval.open(-4, sympy.oo)


# --- exact endpoints --------------------------------------------------------


def test_a_rational_endpoint_stays_a_rational():
    assert written(["3x<1"]) == "(-∞,1/3)"
    assert written([r"-2x+1\geq6"]) == "(-∞,-5/2]"
    assert written([r"\frac{2}{3}<\frac{x}{4}\leq\frac{7}{2}"]) == "(8/3,14]"


def test_decimal_inputs_are_exact_rationals():
    assert written([r"0.1x\leq0.3"]) == "(-∞,3]"


def test_decimal_form_writes_a_terminating_endpoint_exactly():
    assert written(["2x<5"], DECIMAL) == "(-∞,2.5)"
    assert written([r"-8x\geq1"], DECIMAL) == "(-∞,-0.125]"
    assert written(["1<2x+1<7"], DECIMAL) == "(0,3)"


def test_decimal_form_declines_an_endpoint_no_decimal_writes_exactly():
    assert "no exact decimal form" in declined(["3x<1"], DECIMAL)


@pytest.mark.parametrize(
    ("solution", "text"),
    [
        (sympy.Interval.Lopen(-8, 7), "(-8,7]"),
        (sympy.Interval.Ropen(sympy.Rational(-5, 2), sympy.oo), "[-5/2,∞)"),
        (sympy.Interval(-sympy.oo, 0), "(-∞,0]"),
        (sympy.S.Reals, "(-∞,∞)"),
        (sympy.S.EmptySet, "∅"),
    ],
)
def test_infinity_is_never_a_closed_end(solution, text):
    assert write_interval(solution) == text


# --- declines, named, outside the family ------------------------------------


@pytest.mark.parametrize(
    ("expressions", "reason"),
    [
        (["x^2<4"], "not linear"),
        (["x+y<3"], "exactly one variable"),
        (["x<1 or x>3"], "disjunction"),
        (["1<2x+1<3<4"], "more than two comparisons"),
        ([r"x\leq3", r"x\geq3"], "not an interval"),
        ([r"1<x\neq4"], "only <, <=, > and >="),
        (["2<3"], "no variable"),
        (["<2x+1"], "missing one of its sides"),
        (["1<2x+"], "could not be read exactly"),
        ([r"\frac{1}{x}<2"], "not polynomial"),
    ],
)
def test_outside_the_family_is_declined_by_name(expressions, reason):
    assert reason in declined(expressions)


@pytest.mark.parametrize(
    "instruction",
    [
        "Solve the inequality. Write the answer in set-builder notation.",
        "Solve the inequality and write the answer in inequality notation.",
        "Graph the solution set of the inequality in set-builder notation.",
        "Solve the inequality.",
    ],
)
def test_a_notation_other_than_intervals_is_declined(instruction):
    solution, decline = solve_exact(instruction, ["2x+1<7"])

    assert solution is None
    assert "interval notation" in decline


@pytest.mark.parametrize(
    "instruction",
    [
        "Consider the following compound inequality. Graph the solution set.",
        "Solve the inequality and graph the solution on a number line.",
        "Plot the solution set of the inequality.",
    ],
)
def test_a_graphed_solution_set_is_the_same_exact_set(instruction):
    """Drawing the set is the consumer's; the set itself is this solver's."""
    solution, decline = solve_exact(instruction, [r"-24<3y-6\leq15"])

    assert solution is not None, decline
    assert solution.entry == "(-6,7]"
    assert solution.form == SCALAR
    assert solution.method == INEQUALITY_METHOD


def test_a_declined_inequality_still_reaches_the_reasoning_route():
    """A decline is not a refusal: a quadratic inequality is a real question."""
    asked = []

    def reason(prompt: str) -> RunResult:
        asked.append(prompt)
        return RunResult(
            text="FINAL ANSWER: (-2,2)",
            requested_backend="gpu",
            actual_backend="gpu",
            runtime="Ollama",
            model="gpt-oss:20b",
            device="GPU",
            elapsed_ms=1.0,
            fallback=False,
            metrics=ExecutionMetrics(),
            evidence={},
        )

    result = solve_math(
        MathProblem(instruction=ASK, expressions=["x^2<4"]), reason=reason
    )

    assert asked
    assert result["route"] == "reasoning"
    assert result["provenance"]["router_detail"] == "an inequality that is not linear"


def test_naming_an_inequality_without_writing_one_is_left_to_the_other_solvers():
    solution, decline = solve_exact(ASK, ["3x+1"])

    assert solution is None
    assert decline == NO_OPERATION_MATCHED
