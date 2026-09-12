"""The step decides which displayed inequality is solved.

Live, on 2026-09-12, lesson 1.7: a problem displayed as `-4(w-1) <= 28` and
`1+w < 9`, and step 1 of 4 said "Solve the first inequality and express your
answer in interval notation." The exact route intersected every comparison it
was given and answered `[-6,8)` -- the compound inequality, correct for a later
step and wrong for this one, whose answer is `[-6,∞)`.

These hold the scope generically: a singular ordinal selects one displayed
inequality before anything is combined, and a step that names none -- a
compound inequality, "the inequalities", a graph of the solution set -- still
intersects them all exactly as before.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact import solve_exact
from facet_runtime.exact.inequality import INEQUALITY_METHOD
from facet_runtime.solve import MathProblem, solve_math

DISPLAYED = [r"-4(w-1)\leq28", "1+w<9"]
TAIL = "and express your answer in interval notation. Use decimal form for numerical values."


def step(words: str) -> str:
    return f"Consider the following inequality problem. {words} {TAIL}"


def answer(instruction: str, expressions: list[str]) -> str:
    solution, decline = solve_exact(instruction, expressions)
    assert solution is not None, decline
    return solution.entry


def refuses(prompt: str):
    raise AssertionError("an inequality step reached the reasoning route")


# --- one displayed inequality, chosen by the step ------------------------------


def test_the_live_step_solves_only_the_first_inequality_exactly():
    result = solve_math(
        MathProblem(
            instruction=step("Solve the first inequality"), expressions=DISPLAYED
        ),
        reason=refuses,
    )

    assert result["route"] == "exact"
    assert result["answer"]["entry"] == "[-6,∞)"
    assert result["provenance"]["method"] == INEQUALITY_METHOD
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["evidence"]["computation"]["selected"] == "1 of 2"


@pytest.mark.parametrize(
    ("words", "entry"),
    [
        ("Solve the first inequality", "[-6,∞)"),
        ("Solve the 1st inequality", "[-6,∞)"),
        ("Step 1 of 4: Solve the first inequality", "[-6,∞)"),
        ("Solve the second inequality", "(-∞,8)"),
        ("Solve the 2nd inequality", "(-∞,8)"),
        ("Solve the last inequality", "(-∞,8)"),
    ],
)
def test_an_ordinal_selects_its_inequality(words, entry):
    assert answer(step(words), DISPLAYED) == entry


def test_inequalities_joined_inline_are_selected_the_same_way():
    joined = [r"-4(w-1)\leq28 \text{ and } 1+w<9"]

    assert answer(step("Solve the first inequality"), joined) == "[-6,∞)"
    assert answer(step("Solve the second inequality"), joined) == "(-∞,8)"


def test_a_chain_is_one_inequality_when_counting():
    displayed = [r"-3<2x+1\leq7", "x>0"]

    assert answer(step("Solve the first inequality"), displayed) == "(-2,3]"
    assert answer(step("Solve the second inequality"), displayed) == "(0,∞)"


def test_an_absolute_value_inequality_is_selected_like_any_other():
    displayed = [r"|x-1|\leq2", "x>0"]

    assert answer(step("Solve the first inequality"), displayed) == "[-1,3]"
    assert answer(step("Solve the second inequality"), displayed) == "(0,∞)"


def test_the_graph_of_one_inequality_is_that_inequality():
    assert (
        answer("Graph the solution set of the second inequality.", DISPLAYED)
        == "(-∞,8)"
    )


# --- a step about all of them still intersects ---------------------------------


@pytest.mark.parametrize(
    "instruction",
    [
        step("Solve the compound inequality"),
        step("Write the solution of the compound inequality"),
        "Solve the inequalities and express your answer in interval notation.",
        (
            "Consider the following inequality problem. Graph the solution set of "
            "the compound inequality."
        ),
        "Solve the inequality and express your answer in interval notation.",
    ],
)
def test_a_step_naming_no_single_inequality_intersects_them_as_before(instruction):
    solution, decline = solve_exact(instruction, DISPLAYED)

    assert solution is not None, decline
    assert solution.entry == "[-6,8)"
    assert "selected" not in solution.evidence


def test_a_displayed_compound_inequality_still_reaches_facet_exact():
    result = solve_math(
        MathProblem(
            instruction=step("Solve the compound inequality"), expressions=DISPLAYED
        ),
        reason=refuses,
    )

    assert result["answer"]["entry"] == "[-6,8)"
    assert result["provenance"]["evidence"]["model_calls"] == 0


# --- a scope that cannot be read is declined ------------------------------------


def test_a_position_past_the_last_inequality_is_declined():
    solution, decline = solve_exact(step("Solve the third inequality"), DISPLAYED)

    assert solution is None
    assert decline == "the step names inequality 3 of 2 written"


def test_a_step_naming_two_different_inequalities_is_declined():
    solution, decline = solve_exact(
        step("Compare the first inequality with the second inequality"), DISPLAYED
    )

    assert solution is None
    assert decline == "the step names more than one of the inequalities"


def test_the_last_inequality_and_the_second_of_two_are_one_position():
    assert (
        answer(
            step("Solve the second inequality, the last inequality shown"), DISPLAYED
        )
        == "(-∞,8)"
    )
