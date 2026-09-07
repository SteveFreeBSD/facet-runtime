"""The distance between two points, answered exactly.

Live, on 2026-09-07: the distance between `(7,0)` and `(-3,-1)`. The
deterministic stage had no branch for it, declined, and a reasoning model
answered with an approximation -- which the add-on could not place, because the
box Hawkes gives this question publishes `0123456789-` and a Radical template
and has no `.` in its character set at all. The answer it asks for is the exact
radical, and there was nothing wrong with the arithmetic that produced the
decimal; the question simply has an exact answer and was not being given one.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact.distance import DISTANCE_REQUEST, solve_point_distance
from facet_runtime.exact.router import solve_exact

#: The live instruction and its two points.
LIVE = "Find the distance between the following pair of points."
LIVE_POINTS = ["(7,0)", "(-3,-1)"]


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        pytest.param(LIVE_POINTS, "sqrt(101)", id="the-live-question"),
        pytest.param(["(0,0)", "(3,4)"], "5", id="a-whole-number-distance"),
        pytest.param(["(1,1)", "(11,11)"], "10*sqrt(2)", id="reduced-not-left-raw"),
        pytest.param(["(-3,-1)", "(7,0)"], "sqrt(101)", id="order-does-not-matter"),
        pytest.param(["(1/2,0)", "(0,1/2)"], "sqrt(2)/2", id="rational-coordinates"),
        pytest.param(["(2.5,0)", "(0,0)"], "5/2", id="a-decimal-stays-exact"),
    ],
)
def test_the_distance_is_computed_exactly(points, expected) -> None:
    assert solve_point_distance(LIVE, points) == (expected, "")


def test_an_irrational_distance_is_never_rounded() -> None:
    """A decimal is not a rounder answer here. It is an unenterable one."""
    answer, refusal = solve_point_distance(LIVE, LIVE_POINTS)

    assert refusal == ""
    assert "." not in answer
    assert answer == "sqrt(101)"


def test_the_router_answers_it_exactly_instead_of_declining() -> None:
    """The whole defect: this question reached a model because nothing here
    claimed it."""
    solution, refusal = solve_exact(LIVE, LIVE_POINTS)

    assert refusal == ""
    assert solution is not None
    assert solution.display == "sqrt(101)"
    assert solution.entry == "sqrt(101)"
    # Named for what it did, so a reader can see the formula behind the answer.
    assert solution.method == "SymPy exact distance between two points"


def test_the_points_may_arrive_in_the_instruction_instead() -> None:
    """Some questions typeset the pairs; others write them in the sentence.

    Asked of the solver rather than the router: the router declines a question
    with no expressions at all before any solver sees it, and that guard is
    older than this one and not this one's to move.
    """
    answer, refusal = solve_point_distance(
        "Find the distance between the points (7,0) and (-3,-1).", []
    )

    assert refusal == ""
    assert answer == "sqrt(101)"


def test_the_same_pair_written_twice_is_one_point_and_not_two() -> None:
    """Which is what makes reading both the markup and the prose safe.

    It also means a question about the distance from a point to itself cannot
    be asked here, and that is the right trade: Hawkes does not ask it, and
    reading one pair as two points would be the more damaging mistake.
    """
    answer, refusal = solve_point_distance(LIVE, ["(4,9)", "(4,9)"])

    assert answer is None
    assert "two points" in refusal


def test_a_pair_stated_twice_is_still_one_point() -> None:
    """Read from the markup and from the prose, and counted once."""
    solution, refusal = solve_exact(
        "Find the distance between the points (7,0) and (-3,-1).", LIVE_POINTS
    )

    assert refusal == ""
    assert solution is not None and solution.display == "sqrt(101)"


@pytest.mark.parametrize(
    "instruction",
    [
        "Find the midpoint between the following pair of points.",
        "Find the slope of the line through the following pair of points.",
        "Find the equation of the line through the following pair of points.",
    ],
)
def test_another_question_about_the_same_two_points_is_not_claimed(instruction):
    """Each of these has its own answer and none of them is a distance."""
    answer, refusal = solve_point_distance(instruction, LIVE_POINTS)

    assert answer is None
    assert refusal


def test_a_question_that_is_not_about_points_is_not_matched() -> None:
    """`distance` on its own is a word that other questions use too."""
    assert DISTANCE_REQUEST.search("A train travels a distance of 90 miles.") is None
    assert DISTANCE_REQUEST.search("Simplify the following expression.") is None


@pytest.mark.parametrize(
    ("points", "why"),
    [
        pytest.param(["(7,0)"], "two points", id="only-one-point"),
        pytest.param(["(1,1)", "(2,2)", "(3,3)"], "two points", id="three-points"),
        pytest.param([], "two points", id="no-points-at-all"),
    ],
)
def test_anything_but_two_points_is_declined_by_name(points, why) -> None:
    """A named gap, so a fallback can say which one it fell through."""
    answer, refusal = solve_point_distance("Find the distance between points.", points)

    assert answer is None
    assert why in refusal
