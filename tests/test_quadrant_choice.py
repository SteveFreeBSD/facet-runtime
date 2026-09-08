"""A radio question with five choices is one answer, computed from two signs.

Live, on 2026-09-08, every one of these reached a reasoning model on a GPU and
came back unusable. The page publishes one radio group of five choices; the
editor probe counted five *controls* and called them five answers; Facet was
asked for five separate values to a question that has one; the model produced
five, and the last of them was the prompt's own description of a PART line.

Nothing about the mathematics was hard. Which quadrant a point is in is two
comparisons against zero. The deterministic stage simply had no branch for the
question, so it declined and the family fell through to a route it has no
business reaching.

These hold both halves of the Facet-side fix: the classification is exact and
engages no model, and the answer is one of the page's own published choices --
never Facet's own wording for the same idea, because the consumer selects a
control by the words the page printed.
"""

from __future__ import annotations

import pytest
from test_solve_math import Reasoner

from facet_runtime.exact.quadrant import (
    ORIGIN,
    QUADRANT_I,
    QUADRANT_II,
    QUADRANT_III,
    QUADRANT_IV,
    X_AXIS,
    Y_AXIS,
    classify,
    coverage,
    read_point,
    solve_quadrant,
)
from facet_runtime.solve import MathProblem, SolveRefused, parse_problem, solve_math

INSTRUCTION = "In which quadrant does the point lie?"

#: The shape the live question publishes: four quadrants and one axis choice.
FIVE = [
    "Quadrant I",
    "Quadrant II",
    "Quadrant III",
    "Quadrant IV",
    "The point is on an axis",
]
#: A page that names each axis separately.
NAMED = [
    "Quadrant I",
    "Quadrant II",
    "Quadrant III",
    "Quadrant IV",
    "x-axis",
    "y-axis",
    "Origin",
]
#: A page offering the four quadrants and nothing else.
FOUR = ["Quadrant I", "Quadrant II", "Quadrant III", "Quadrant IV"]


def refuses(prompt: str):
    raise AssertionError("a quadrant question reached the reasoning route")


def problem(**changes) -> MathProblem:
    fields = {
        "instruction": INSTRUCTION,
        "expressions": ("(3,-4)",),
        "answer_parts": 1,
        "answer_choices": tuple(FIVE),
    }
    fields.update(changes)
    return MathProblem(**fields)


# --- the classification itself ---------------------------------------------


@pytest.mark.parametrize(
    ("point", "region"),
    [
        ((4, 7), QUADRANT_I),
        ((-2, 5), QUADRANT_II),
        ((-3, -1), QUADRANT_III),
        ((3, -4), QUADRANT_IV),
        ((6, 0), X_AXIS),
        ((-6, 0), X_AXIS),
        ((0, 5), Y_AXIS),
        ((0, -5), Y_AXIS),
        ((0, 0), ORIGIN),
    ],
)
def test_every_region_of_the_plane_is_decided_by_two_comparisons(point, region):
    assert classify(point) == region


def test_a_coordinate_stays_exact_and_never_becomes_a_float():
    """A sign read off 0.30000000000000004 is right for a reason nobody can check."""
    x, y = read_point(INSTRUCTION, [r"(\frac{1}{3}, -0.5)"])

    assert x == pytest.approx(1 / 3) and x * 3 == 1
    assert y * 2 == -1


def test_two_points_are_a_different_question_and_are_declined():
    assert read_point(INSTRUCTION, ["(1,2) and (3,4)"]) is None


# --- the answer is one of the page's own words -----------------------------


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(4,7)", "Quadrant I"),
        ("(-2,5)", "Quadrant II"),
        ("(-3,-1)", "Quadrant III"),
        ("(3,-4)", "Quadrant IV"),
        ("(0,5)", "The point is on an axis"),
        ("(6,0)", "The point is on an axis"),
        ("(0,0)", "The point is on an axis"),
    ],
)
def test_the_live_five_choice_question_is_answered_from_its_own_choices(
    expression, expected
):
    choice, refusal = solve_quadrant(INSTRUCTION, [expression], FIVE)

    assert refusal == ""
    assert choice == expected
    assert choice in FIVE


@pytest.mark.parametrize(
    ("expression", "expected"),
    [("(6,0)", "x-axis"), ("(0,5)", "y-axis"), ("(0,0)", "Origin")],
)
def test_a_named_axis_is_preferred_over_a_general_one(expression, expected):
    """Specific before general: both are true and only one is the answer."""
    choice, refusal = solve_quadrant(INSTRUCTION, [expression], [*NAMED, "on an axis"])

    assert refusal == "" and choice == expected


def test_a_region_no_choice_names_is_refused_rather_than_approximated():
    """Four quadrant choices have no right answer for a point on an axis."""
    choice, refusal = solve_quadrant(INSTRUCTION, ["(0,5)"], FOUR)

    assert choice == ""
    assert "no published choice names it" in refusal


def test_a_choice_that_names_no_region_is_never_chosen():
    assert coverage("None of these") == frozenset()
    assert coverage("Quadrant I or the x-axis") == frozenset()


def test_two_choices_naming_one_region_is_refused():
    choice, refusal = solve_quadrant(
        INSTRUCTION, ["(4,7)"], ["Quadrant I", "quadrant 1", "Quadrant II"]
    )

    assert choice == ""
    assert "2 published choices name" in refusal


# --- through the router, with no model at all ------------------------------


def test_the_family_is_answered_exactly_and_asks_no_model():
    result = solve_math(problem(), reason=refuses)

    assert result["route"] == "exact"
    assert result["answer"]["kind"] == "value"
    assert result["answer"]["display"] == "Quadrant IV"
    assert result["answer"]["entry"] == "Quadrant IV"
    assert result["answer"]["parts"] == []
    assert result["answer"]["entry_mode"] == "verbatim"
    assert result["provenance"]["router"] == "solved"
    assert result["provenance"]["method"] == "exact quadrant classification"
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_a_quadrant_question_it_cannot_answer_is_refused_not_reasoned():
    """Falling through would ask a model to guess which choices the page showed."""
    with pytest.raises(SolveRefused) as raised:
        solve_math(
            problem(expressions=("(0,5)",), answer_choices=tuple(FOUR)), reason=refuses
        )

    assert "no published choice names it" in str(raised.value)


def test_a_quadrant_question_with_no_choices_is_refused():
    with pytest.raises(SolveRefused) as raised:
        solve_math(problem(answer_choices=()), reason=refuses)

    assert "answered by choosing" in str(raised.value)


# --- the contract, whichever route runs ------------------------------------


def test_a_reasoned_answer_that_is_not_a_published_choice_is_refused():
    """A model that writes its own wording has answered the mathematics, not
    the question: the consumer selects a control by the page's own words."""
    reasoner = Reasoner(text="FINAL ANSWER: the fourth quadrant")

    with pytest.raises(SolveRefused) as raised:
        solve_math(
            problem(instruction="Classify the point.", expressions=("(3,-4)",)),
            reason=reasoner,
        )

    assert "not one of them" in str(raised.value)


def test_a_reasoned_answer_that_is_a_published_choice_is_accepted():
    reasoner = Reasoner(text="FINAL ANSWER: Quadrant IV")

    result = solve_math(
        problem(instruction="Classify the point.", expressions=("(3,-4)",)),
        reason=reasoner,
    )

    assert result["route"] == "reasoning"
    assert result["answer"]["display"] == "Quadrant IV"


def test_the_reasoning_prompt_lists_the_choices_it_requires_back():
    from facet_runtime.solve import reasoning_prompt

    prompt = reasoning_prompt(problem(instruction="Classify the point."))

    for choice in FIVE:
        assert f"- {choice}" in prompt
    assert "copied word for word" in prompt


# --- the request that caused all of this -----------------------------------


def test_a_choice_question_may_not_ask_for_several_answers():
    """Five alternatives are not five answers. This is the live defect, refused
    at the boundary as well as fixed where it was made."""
    with pytest.raises(SolveRefused) as raised:
        parse_problem(
            {
                "instruction": INSTRUCTION,
                "expressions": ["(3,-4)"],
                "answer_parts": 5,
                "answer_choices": FIVE,
            }
        )

    assert "one answer, not 5" in str(raised.value)


@pytest.mark.parametrize(
    "choices",
    [["only one"], [], ["a", "a"], ["a", ""], ["a", 3], "not a list", ["x" * 200, "y"]],
)
def test_a_malformed_choice_list_is_refused(choices):
    payload = {
        "instruction": INSTRUCTION,
        "expressions": ["(3,-4)"],
        "answer_choices": choices,
    }
    if choices == []:
        # An empty list is "this question publishes no choices", which is the
        # ordinary case and not a fault.
        assert parse_problem(payload).answer_choices == ()
        return
    with pytest.raises(SolveRefused):
        parse_problem(payload)
