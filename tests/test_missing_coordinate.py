"""The missing half of an ordered pair that satisfies a stated equation.

Live, on 2026-09-07: `2x + y = 2`, "determine the missing coordinate in
(4, ?) so that it will satisfy the given equation". One box on screen, one
answer, and the answer is -6. The deterministic stage had no branch for it and
declined, so gpt-oss:20b answered -- and the browser was independently asking
for two values, so it answered twice: `10` and `10`.

This is one row of a table of values with one blank in it, and it is answered
by the same machinery a grid is: bind what the question states, solve for what
it left out, and prove the result by substituting it back into the relation the
question actually wrote.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact.coordinate import (
    COORDINATE_REQUEST,
    read_partial_pair,
    solve_missing_coordinate,
)
from facet_runtime.exact.router import solve_exact

#: The live instruction, and the equation it was asked about.
LIVE = (
    "Determine the missing coordinate in the ordered pair (4, ?) "
    "so that it will satisfy the given equation."
)
EQUATION = "2x+y=2"


@pytest.mark.parametrize(
    "pair",
    [
        pytest.param("(4,?)", id="a-question-mark"),
        pytest.param("(4,)", id="an-empty-slot"),
        pytest.param("(4, )", id="a-slot-of-whitespace"),
        pytest.param("(4,__)", id="an-underscore-rule"),
    ],
)
def test_the_live_question_is_answered_exactly(pair) -> None:
    """However the page typeset the box, the pair says the same thing."""
    assert solve_missing_coordinate(LIVE, [EQUATION, pair]) == ("-6", "")


def test_the_other_coordinate_can_be_the_missing_one() -> None:
    """`2x + 2 = 2` gives x = 0, and the blank is on the left."""
    assert solve_missing_coordinate(LIVE, [EQUATION, "(?,2)"]) == ("0", "")


def test_the_pair_may_be_written_into_the_sentence() -> None:
    answer, refusal = solve_missing_coordinate(f"{LIVE} (4, ?)", [EQUATION])

    assert (answer, refusal) == ("-6", "")


def test_a_coordinate_named_rather_than_positioned_is_read() -> None:
    """Some questions say "when x = 4" instead of drawing the pair."""
    answer, refusal = solve_missing_coordinate(
        "Determine the missing coordinate when x = 4.", [EQUATION]
    )

    assert (answer, refusal) == ("-6", "")


def test_the_router_answers_it_instead_of_declining() -> None:
    """The whole defect: nothing here claimed this question."""
    solution, refusal = solve_exact(LIVE, [EQUATION, "(4,?)"])

    assert refusal == ""
    assert solution is not None
    assert solution.display == "-6"
    assert solution.entry == "-6"
    assert solution.parts == ()
    assert solution.method == "SymPy exact ordered-pair completion"


def test_the_answer_is_one_value_and_never_two() -> None:
    """The page draws one box. Two parts is what went wrong, not the value."""
    solution, _ = solve_exact(LIVE, [EQUATION, "(4,?)"])

    assert solution is not None and solution.parts == ()


@pytest.mark.parametrize(
    ("equation", "pair", "expected"),
    [
        pytest.param("y=3x-1", "(2,?)", "5", id="slope-intercept"),
        pytest.param("x+y=0", "(?,7)", "-7", id="negative-result"),
        pytest.param("3x+2y=12", "(?,3)", "2", id="both-sides-with-coefficients"),
        pytest.param("2x+y=2", "(1/2,?)", "1", id="a-rational-coordinate"),
        pytest.param("4x-3y=1", "(1,?)", "1", id="a-rational-result-that-is-whole"),
    ],
)
def test_the_relation_is_solved_generally(equation, pair, expected) -> None:
    assert solve_missing_coordinate(LIVE, [equation, pair]) == (expected, "")


def test_an_exact_fraction_is_returned_as_one() -> None:
    """Never rounded, exactly as a table's own blanks are never rounded."""
    answer, refusal = solve_missing_coordinate(LIVE, ["3x+2y=4", "(1,?)"])

    assert refusal == ""
    assert answer == "1/2"


def test_a_pair_with_both_coordinates_written_is_a_different_question() -> None:
    """Checking a pair is not completing one, and has a different answer."""
    assert read_partial_pair("(4,-6)") is None


def test_more_than_one_missing_coordinate_is_declined_rather_than_picked() -> None:
    """A vertical line meets a parabola twice, and the box takes one value."""
    answer, refusal = solve_missing_coordinate(LIVE, ["y^2=x", "(4,?)"])

    assert answer is None
    assert "more than one" in refusal


@pytest.mark.parametrize(
    "instruction",
    [
        "Find the distance between the following pair of points.",
        "Simplify the following expression.",
        "Graph the linear function.",
    ],
)
def test_another_question_is_not_claimed(instruction) -> None:
    assert COORDINATE_REQUEST.search(instruction) is None


def test_two_equations_are_declined_by_name() -> None:
    """One pair, one relation; two is a system and a different question."""
    answer, refusal = solve_missing_coordinate(LIVE, [EQUATION, "x-y=1", "(4,?)"])

    assert answer is None
    assert "one stated equation" in refusal


def test_a_pair_that_cannot_be_read_is_declined_by_name() -> None:
    """Asked with no pair anywhere: not in the markup and not in the words.

    `LIVE` writes its own pair into the sentence, which is why it is read from
    there when the markup does not carry one.
    """
    answer, refusal = solve_missing_coordinate(
        "Determine the missing coordinate of the ordered pair "
        "so that it will satisfy the equation.",
        [EQUATION],
    )

    assert answer is None
    assert "could not be read" in refusal
