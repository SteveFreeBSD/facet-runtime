"""A line derived from its stated properties, and everything that must decline.

The live failure this covers: Hawkes lesson 3.2 states `f(0) = -3` and
`slope = -5`, no solver matched, and a reasoning model answered `-2x-3` -- the
intercept it was given and a slope it invented. Nothing here reaches a model.

The second one it covers is the shape of the answer rather than its value.
Lesson 2.4 asks for "the equation of the line in slope-intercept form" and
draws a bare answer box; this route returned `-2x+5`, which is not an equation,
and Hawkes refused it as a format error. The answer is the equation, and both
of its sides travel with it so that a consumer whose surface prints `f(x) =`
beside the box can still enter the value alone.
"""

from __future__ import annotations

import pytest
import sympy
from sympy.parsing.sympy_parser import (
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

from facet_runtime.exact import RELATION, Relation, solve_exact, subject
from facet_runtime.solve import MathProblem, solve_math

FIND = "Find the linear function with the given properties."
LINE = "Find the equation of the line through the points (1,2) and (3,8)."

#: The live question, exactly as the page's markup converts.
LIVE = (FIND, ["f(0)=-3", "slope=-5"])

#: The other live question, from lesson 2.4 question 4 of 9. The instruction is
#: the page's own, step marker included, and the two expressions are what its
#: two MathJax elements convert to -- "Point" and the semicolon between them
#: are ordinary text and never reach a solver.
#:
#: This is the question that was answered `-2x+5` and refused as a format
#: error. It names no function anywhere, and its answer box is bare.
LESSON_2_4 = (
    (
        "Step 1 of 1 Find the equation of the line in slope-intercept form "
        "that passes through the following point with the given slope. "
        "Simplify your answer."
    ),
    ["(0,5)", "Slope=-2"],
)


@pytest.mark.parametrize(
    ("instruction", "expressions", "subject", "value"),
    [
        pytest.param(*LIVE, "f(x)", "-5x-3", id="live-slope-and-value"),
        pytest.param(FIND, ["f(0)=-3", "m=-5"], "f(x)", "-5x-3", id="slope-as-m"),
        # What lesson 3.2 actually writes, through the real converter: the
        # slope names the function it belongs to, and MathJax's spacing is
        # gone by the time a solver sees it.
        pytest.param(
            FIND,
            ["f(0)=-3", "Slopeoff=-5"],
            "f(x)",
            "-5x-3",
            id="live-slope-of-f-collapsed",
        ),
        pytest.param(
            FIND,
            ["f(0)=-3", "Slope of f = -5"],
            "f(x)",
            "-5x-3",
            id="live-slope-of-f-spaced",
        ),
        pytest.param(
            FIND, ["f(0)=-3", "Slopeoff(x)=-5"], "f(x)", "-5x-3", id="slope-of-f-of-x"
        ),
        pytest.param(
            FIND, ["f(0)=-3", "slopeof-5"], "f(x)", "-5x-3", id="slope-of-a-value"
        ),
        pytest.param(
            FIND, ["slope=-5", "f(2)=7"], "f(x)", "-5x+17", id="slope-and-a-point"
        ),
        pytest.param(
            FIND,
            ["y-intercept=-3", "slope=-5"],
            "y",
            "-5x-3",
            id="slope-and-intercept",
        ),
        pytest.param(FIND, ["b=-3", "m=-5"], "y", "-5x-3", id="m-and-b"),
        pytest.param(FIND, ["(1,2)", "(3,8)"], "y", "3x-1", id="two-points"),
        pytest.param(LINE, ["(1,2)", "(3,8)"], "y", "3x-1", id="two-points-in-prose"),
        pytest.param(
            FIND, ["x-intercept=2", "slope=3"], "y", "3x-6", id="slope-and-x-intercept"
        ),
        pytest.param(
            FIND,
            ["f(-2)=7", "slope=\\frac{1}{2}"],
            "f(x)",
            "(1/2)x+8",
            id="fractional-slope-is-parenthesised",
        ),
        pytest.param(FIND, ["slope=1", "f(0)=4"], "f(x)", "x+4", id="unit-slope"),
        pytest.param(
            FIND, ["slope=-1", "f(0)=2"], "f(x)", "-x+2", id="negative-unit-slope"
        ),
        pytest.param(FIND, ["slope=0", "f(0)=4"], "f(x)", "4", id="constant-function"),
        pytest.param(
            FIND, ["slope=2", "f(0)=0"], "f(x)", "2x", id="through-the-origin"
        ),
        pytest.param(
            FIND,
            ["f(x)=", "f(0)=-3", "slope=-5"],
            "f(x)",
            "-5x-3",
            id="the-fields-own-label-states-nothing",
        ),
        pytest.param(
            FIND,
            ["f(0)=-3", "f(0)=-3", "slope=-5"],
            "f(x)",
            "-5x-3",
            id="the-same-property-twice",
        ),
    ],
)
def test_stated_properties_give_the_line_they_determine(
    instruction, expressions, subject, value
) -> None:
    solution, decline = solve_exact(instruction, expressions)

    assert decline == ""
    assert solution is not None
    # The answer to "find the equation of the line" is the equation. Returning
    # `mx + b` alone was an answer only on a page that prints the left side
    # beside its box, and this route cannot know whether a page does.
    assert solution.entry == f"{subject}={value}"
    assert solution.display == f"{subject}={value}"
    assert solution.form == RELATION
    # Both sides, so a consumer whose surface states the subject enters the
    # value without splitting the written form to find it.
    assert solution.relation == Relation(subject=subject, value=value)
    assert solution.parts == ()
    # `y = mx + b` is mathematics, not a phrase: a consumer renders it in
    # whatever entry syntax its own answer surface needs.
    assert solution.entry_mode == "math"


def test_the_bare_box_question_is_answered_with_its_whole_equation() -> None:
    """Lesson 2.4 question 4, which Hawkes refused as an incorrect format.

    The mathematics was never in doubt: the line through `(0,5)` with slope
    `-2` is `y = -2x + 5`, and this route derived it. What it returned was
    `-2x+5` -- the right-hand side, which is an expression and not an equation,
    and which answers the question only on a page that prints `y =` beside its
    answer box. That page prints nothing, so nothing supplied the half that was
    missing.
    """
    solution, decline = solve_exact(*LESSON_2_4)

    assert decline == ""
    assert solution is not None
    assert solution.display == "y=-2x+5"
    assert solution.entry == "y=-2x+5"
    assert solution.form == RELATION
    assert solution.relation == Relation(subject="y", value="-2x+5")


@pytest.mark.parametrize(
    ("instruction", "expressions", "expected"),
    [
        pytest.param(*LESSON_2_4, "y", id="slope-intercept-form-names-nothing"),
        pytest.param(LINE, ["(1,2)", "(3,8)"], "y", id="an-equation-of-a-line"),
        pytest.param(*LIVE, "f(x)", id="a-function-named-by-one-of-its-values"),
        pytest.param(
            FIND,
            ["f(x)=", "f(0)=-3", "slope=-5"],
            "f(x)",
            id="a-function-named-by-the-fields-own-label",
        ),
        pytest.param(
            FIND,
            ["g(0)=-3", "slope=-5"],
            "g(x)",
            id="the-name-is-the-questions-and-not-a-fixed-letter",
        ),
        pytest.param(
            FIND,
            ["f(0)=-3", "g(2)=7"],
            "y",
            id="two-named-functions-are-not-one-subject",
        ),
    ],
)
def test_the_subject_is_the_one_the_question_names(
    instruction, expressions, expected
) -> None:
    """What stands left of the equals sign is read, never assumed.

    `y` is what slope-intercept form means and is the answer to a question that
    names nothing. A question that names a function is answered under that
    name, whichever letter it used. Two names is neither, because a line stated
    under one of two functions is a guess about which.
    """
    assert subject(instruction, list(expressions)) == expected


def test_the_live_question_carries_its_whole_derivation() -> None:
    solution, _ = solve_exact(*LIVE)

    assert solution is not None
    assert solution.evidence == {
        "slope": "-5",
        "y_intercept": "-3",
        "function": "f(x)=-5x-3",
        "properties": "f(0)=-3; slope=-5",
        "verification": "f(0)=-3; slope=-5",
    }


def test_every_stated_property_holds_of_the_returned_function() -> None:
    """The check is redone here, against the answer and not against the working."""
    solution, _ = solve_exact(FIND, ["f(-2)=7", "slope=\\frac{1}{2}"])

    assert solution is not None
    x = sympy.Symbol("x")
    # Parsed the way a reader reads it, with the implicit multiplication left
    # implicit: what is proved is the answer string, not a restatement of it.
    function = parse_expr(
        solution.relation.value,
        transformations=standard_transformations + (implicit_multiplication,),
    )
    assert sympy.simplify(function.subs(x, -2) - 7) == 0
    assert sympy.simplify(sympy.diff(function, x) - sympy.Rational(1, 2)) == 0


@pytest.mark.parametrize(
    ("expressions", "reason"),
    [
        pytest.param(
            ["slope=-5"],
            "a slope alone does not determine a linear function",
            id="slope-alone",
        ),
        pytest.param(
            ["f(0)=-3"],
            "one point alone does not determine a linear function",
            id="one-point-alone",
        ),
        pytest.param(
            ["(1,2)", "(1,5)"],
            "a vertical line is not a linear function",
            id="vertical",
        ),
        pytest.param(
            ["slope=-5", "f(0)=-3", "f(1)=0"],
            "the stated properties have no common linear function",
            id="a-third-property-contradicts-the-first-two",
        ),
        pytest.param(
            ["slope=-5", "slope=2", "f(0)=1"],
            "the stated slopes disagree",
            id="two-slopes",
        ),
        pytest.param(
            ["f(0)=-3", "f=-5"],
            "a stated property could not be read exactly",
            id="a-bare-equality-is-not-a-slope",
        ),
        pytest.param(
            ["f(0)=-3", "f(0)=-5"],
            "a vertical line is not a linear function",
            id="one-input-two-values",
        ),
        pytest.param(
            ["slope=\\frac{1}{0}", "f(0)=1"],
            "a stated property could not be read exactly",
            id="one-property-read-and-one-not",
        ),
    ],
)
def test_underdetermined_contradictory_and_unreadable_properties_decline(
    expressions, reason
) -> None:
    solution, decline = solve_exact(FIND, expressions)

    assert solution is None
    assert decline == reason


@pytest.mark.parametrize(
    "instruction",
    [
        "Simplify the following expression.",
        "Graph the linear function.",
        "Solve the following equation.",
        "Find the vertex.",
    ],
)
def test_a_question_that_is_not_this_one_never_reaches_this_route(instruction) -> None:
    """The gate is the instruction. Properties on the page do not open it."""
    solution, decline = solve_exact(instruction, ["f(0)=-3", "slope=-5"])

    assert solution is None
    assert decline != "" and "linear function" not in decline


@pytest.mark.parametrize(
    "expressions",
    [
        pytest.param(["\\frac{x+1}{x-2}"], id="a-rational-expression"),
        pytest.param(["2x+3y=6"], id="an-equation-to-rearrange"),
        pytest.param(["f(x)=3x+2"], id="a-function-already-written-down"),
    ],
)
def test_naming_a_linear_function_does_not_claim_a_question_that_states_none(
    expressions,
) -> None:
    """No property, no claim: the remaining solvers still get their turn.

    Otherwise every future exact route for a question whose instruction happens
    to say "linear function" would be shadowed by this one's refusal.
    """
    solution, decline = solve_exact(FIND, expressions)

    assert solution is None
    assert decline == "no exact operation matched the instruction"


@pytest.mark.parametrize(
    "instruction",
    [
        (
            "Find the equation of the line perpendicular to a line with a "
            "slope of 2, through the point (1,3)."
        ),
        (
            "Find the equation of the line parallel to a line with a "
            "slope of 2, through the point (1,3)."
        ),
    ],
)
def test_a_line_stated_against_another_line_is_refused_whole(instruction) -> None:
    """The words are the same and mean something else.

    A perpendicular's slope is the negative reciprocal of the one written on
    the page. Read literally, this derives `2x+1` -- wrong, and confidently so.
    """
    solution, decline = solve_exact(instruction, ["(1,3)"])

    assert solution is None
    assert decline == (
        "a line stated by its relation to another line is not derived here"
    )


def test_a_decimal_property_stays_an_exact_rational() -> None:
    solution, _ = solve_exact(FIND, ["slope=0.5", "f(0)=1"])

    assert solution is not None
    assert solution.entry == "f(x)=(1/2)x+1"
    assert solution.relation.value == "(1/2)x+1"
    assert solution.evidence["slope"] == "1/2"


def test_a_third_point_neither_determines_nor_is_ignored() -> None:
    """Two points fix the line; the rest are checked against it, not dropped."""
    agrees, decline = solve_exact(FIND, ["f(0)=-3", "f(2)=7", "f(1)=2"])
    assert decline == ""
    assert agrees is not None
    assert agrees.entry == "f(x)=5x-3"
    assert agrees.evidence["verification"] == "f(0)=-3; f(2)=7; f(1)=2"

    disagrees, refusal = solve_exact(FIND, ["f(0)=-3", "f(2)=7", "f(1)=0"])
    assert disagrees is None
    assert refusal == "the stated properties have no common linear function"


def test_a_two_value_answer_shape_is_refused_rather_than_half_filled() -> None:
    solution, decline = solve_exact(*LIVE, answer_parts=2)

    assert solution is None
    assert decline == "a linear function is one answer but the answer shape requires 2"


def test_the_live_question_makes_no_reasoning_model_call() -> None:
    calls = []

    def reason(prompt):
        calls.append(prompt)
        raise AssertionError("an exactly derivable linear function reached a model")

    instruction, expressions = LIVE
    result = solve_math(
        MathProblem(
            instruction=instruction,
            expressions=tuple(expressions),
            answer_parts=1,
            label="",
        ),
        reason=reason,
    )

    assert calls == []
    assert result["route"] == "exact"
    assert result["answer"]["entry"] == "f(x)=-5x-3"
    assert result["answer"]["form"] == RELATION
    assert result["answer"]["relation"] == {"subject": "f(x)", "value": "-5x-3"}
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["evidence"]["computation"]["function"] == "f(x)=-5x-3"
