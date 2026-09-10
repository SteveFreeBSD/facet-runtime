"""How many solutions an equation has, answered in the page's own words.

Live, on 2026-09-10, lesson 1.6 question 1: `4t + 5 = 4(t + 3) - 7`. It reduces
to `0 = 0`, which is "Infinite Solutions" and is not hard. Facet classified it
exactly, engaging no model at all, and the answer was refused anyway:

    solve-refused {"status":"ambiguous","why":"facet-answer-unusable"}
    Facet returned no usable answer: this question is answered by choosing one
    of the alternatives it published, and the answer is not one of them

The page had published `No Solution (∅)`, `One Solution`, `Infinite Solutions
(ℝ)`. The classification is "Infinite Solutions". The contract that an answer to
a choice question must *be* one of the choices -- which is right, and is what
selects a control -- refused it over a notation the algebra has no business
knowing about. All three classifications failed this way, "One Solution"
included, whose display is `One Solution (t = 2)` and is no better a match.

The algebra does not learn Hawkes' notation: a classification carrying `(ℝ)`
would be wrong for every other consumer, and wrong in a way that spreads.
Instead the two are reconciled once, at the only boundary holding both -- the
classification on one side, the published alternatives on the other.

These hold that boundary: the exact wording for the two notated choices, the
revealed-textbox lifecycle that "One Solution" still depends on, and the two
directions this must fail closed in -- nothing recognisable, and more than one.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact.solution_kind import (
    INFINITE_SOLUTIONS,
    NO_SOLUTION,
    ONE_SOLUTION,
    SolutionKindRefused,
    kind_of,
    select,
)
from facet_runtime.solve import MathProblem, SolveRefused, solve_math

#: The live page's own three alternatives, character for character.
LIVE = ["No Solution (∅)", "One Solution", "Infinite Solutions (ℝ)"]

#: The live specimen and its two siblings, as the page states them.
INFINITE = "4t+5=4(t+3)-7"
NONE_AT_ALL = "4t+5=4t+9"
EXACTLY_ONE = "4t+5=2t+9"


def refuses(prompt: str):
    raise AssertionError("a solution-kind question reached the reasoning route")


def answer(equation: str, choices=LIVE) -> dict:
    return solve_math(
        MathProblem(
            instruction="Solve the equation.",
            expressions=[equation],
            answer_choices=tuple(choices),
        ),
        reason=refuses,
    )


# --- the live specimen ------------------------------------------------------


def test_infinite_solutions_is_answered_in_the_pages_own_words():
    """The defect, at the question that produced it."""
    result = answer(INFINITE)

    assert result["answer"]["display"] == "Infinite Solutions (ℝ)"
    # `entry` is what selects the control, so it is the published wording too,
    # and `verbatim` says it is already exactly what belongs in the answer.
    assert result["answer"]["entry"] == "Infinite Solutions (ℝ)"
    assert result["answer"]["entry_mode"] == "verbatim"
    assert result["answer"]["form"] == "choice"
    # Two sign-free reductions and no model, which is the other half of the
    # claim: this was never a question worth a GPU.
    assert result["route"] == "exact"
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_no_solution_is_answered_with_the_empty_set_notation_the_page_appended():
    result = answer(NONE_AT_ALL)

    assert result["answer"]["display"] == "No Solution (∅)"
    assert result["answer"]["entry"] == "No Solution (∅)"
    assert result["answer"]["form"] == "choice"


def test_one_solution_selects_its_own_choice_and_does_not_type_a_value():
    """On the choice surface the answer is the choice. Hawkes reveals the box
    only after it is made, so a value here would have nowhere to go."""
    result = answer(EXACTLY_ONE)

    assert result["answer"]["display"] == "One Solution"
    assert result["answer"]["entry"] == "One Solution"
    assert result["answer"]["form"] == "choice"


# --- the lifecycle the value depends on -------------------------------------


def test_the_revealed_textbox_still_receives_the_numeric_solution():
    """The second read of the same question: "One Solution" has been chosen,
    Hawkes has revealed its textbox, and that surface is a field publishing no
    choices at all. Nothing on this path may have changed -- it is where the
    answer to a One Solution question is actually entered."""
    result = solve_math(
        MathProblem(
            instruction="Solve the equation.",
            expressions=[EXACTLY_ONE],
            answer_choices=(),
        ),
        reason=refuses,
    )

    assert result["answer"]["entry"] == "2"
    assert result["answer"]["entry_mode"] == "math"
    assert result["answer"]["form"] == "scalar"
    assert result["answer"]["display"] == "One Solution (t = 2)"


def test_a_named_variable_still_solves_to_its_value_without_choices():
    """The other value-returning branch, equally untouched."""
    result = solve_math(
        MathProblem(
            instruction="Solve the equation for t.",
            expressions=[EXACTLY_ONE],
            answer_choices=(),
        ),
        reason=refuses,
    )

    assert result["answer"]["entry"] == "2"
    assert result["answer"]["entry_mode"] == "math"


# --- reading one published choice -------------------------------------------


@pytest.mark.parametrize(
    ("choice", "kind"),
    [
        ("No Solution (∅)", NO_SOLUTION),
        ("No Solution", NO_SOLUTION),
        ("no solutions", NO_SOLUTION),
        ("∅", NO_SOLUTION),
        ("The Empty Set", NO_SOLUTION),
        ("One Solution", ONE_SOLUTION),
        ("Exactly One Solution", ONE_SOLUTION),
        ("Unique Solution", ONE_SOLUTION),
        ("Infinite Solutions (ℝ)", INFINITE_SOLUTIONS),
        ("Infinite Solutions", INFINITE_SOLUTIONS),
        ("Infinitely Many Solutions", INFINITE_SOLUTIONS),
        ("All Real Numbers", INFINITE_SOLUTIONS),
        ("ℝ", INFINITE_SOLUTIONS),
        ("  infinite   solutions  (ℝ)  ", INFINITE_SOLUTIONS),
    ],
)
def test_the_ways_a_page_may_write_each_fact(choice, kind):
    assert kind_of(choice) == kind


@pytest.mark.parametrize(
    "choice",
    [
        "Cannot be determined",
        "Two Solutions",
        "More Than One Solution",
        "Quadrant I",
        "Not a Real Number",
        "t = 2",
        "",
        "   ",
        # A page saying two contradictory things. Believing either half of it
        # would be choosing which half to believe.
        "No Solution (ℝ)",
        "Infinite Solutions (∅)",
        # A notation this does not read. `R` is a variable far more often than
        # it is the reals.
        "Infinite Solutions (R)",
        "Infinite Solutions (all complex numbers)",
    ],
)
def test_unrelated_or_unreadable_choice_text_states_nothing(choice):
    assert kind_of(choice) is None


def test_more_than_one_solution_is_not_read_as_one_solution():
    """The pattern is anchored whole, so a different fact ending in the same
    two words is not this one."""
    assert kind_of("More Than One Solution") is None
    assert kind_of("No Solution") is NO_SOLUTION


# --- failing closed ---------------------------------------------------------


def test_no_compatible_choice_refuses_rather_than_taking_the_nearest():
    """Four alternatives about something else. The nearest one is not the
    right one, and there is nothing here for a model to add."""
    with pytest.raises(SolveRefused) as refusal:
        answer(INFINITE, ["Quadrant I", "Quadrant II", "Quadrant III", "Quadrant IV"])

    assert "no published choice states it" in str(refusal.value)


def test_two_compatible_choices_refuse_rather_than_picking_one():
    """The page said something this cannot read, and either pick would be a
    pick nobody could check afterwards."""
    with pytest.raises(SolveRefused) as refusal:
        answer(INFINITE, ["No Solution (∅)", "Infinite Solutions (ℝ)", "ℝ"])

    assert "2 published choices state infinite solutions" in str(refusal.value)


def test_selecting_is_refused_for_a_classification_outside_the_vocabulary():
    """ "Two Solutions" is a real classification and no page in this family
    offers it. It is refused, not mapped onto the nearest of the three."""
    with pytest.raises(SolutionKindRefused):
        select("Two Solutions", LIVE)


def test_a_choice_page_whose_answer_is_a_value_is_refused_not_typed():
    """A question published alternatives and reduced to a value: there is no
    box on that surface, so the value is not an answer to it."""
    with pytest.raises(SolveRefused) as refusal:
        solve_math(
            MathProblem(
                instruction="Solve the equation.",
                expressions=["x^2-5*x+6=0"],
                answer_choices=tuple(LIVE),
            ),
            reason=refuses,
        )

    assert "rather than one of them" in str(refusal.value)


def test_select_returns_the_published_string_and_never_its_own_wording():
    """Character for character, including a notation this module only reads."""
    assert select(INFINITE_SOLUTIONS, LIVE) == "Infinite Solutions (ℝ)"
    assert select(NO_SOLUTION, LIVE) == "No Solution (∅)"
    assert select(ONE_SOLUTION, LIVE) == "One Solution"
