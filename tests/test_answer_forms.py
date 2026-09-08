"""Every exact answer says which family it belongs to.

The recurring failure this exists to end: a new exact solver is added, Facet
answers correctly, and only afterwards -- live, on somebody's coursework -- is
it discovered that the consumer cannot enter that shape of answer. It happened
to the vertex (a pair the page bracketed), to the distance (a radical), to the
quadrant (a chosen alternative), and most recently to the midpoint, which
returns a pair of *rationals* and had no insertion path for the composition.

Every one of those was invisible on this side, because an answer crossed as a
string and the consumer had to rediscover its structure by parsing it.

`form` is the smallest thing that makes it visible: which family, from a closed
set. It is not a description of the value -- what notation a value is written
in stays readable from the value -- it is the one fact a consumer needs before
it can say whether it has a path at all.

The rule these tests hold is: a new family cannot be added here without a
consumer-side row saying how it is entered, or saying explicitly that it is not.
That row lives in `docs/ANSWER_CAPABILITIES.md` in the facet-hawkes repository,
and the test that reads it lives beside it.
"""

from __future__ import annotations

import pytest
from test_solve_math import Reasoner

from facet_runtime.exact import (
    ANSWER_FORMS,
    CHOICE,
    ORDERED_PAIR,
    PARTS,
    SCALAR,
    solve_exact,
)
from facet_runtime.exact.quadrant import QUADRANT_METHOD
from facet_runtime.solve import MathProblem, solve_math

#: The realness question, in the words Hawkes asks it in. Pulled out because a
#: quoted phrase inside a quoted sentence reads badly inline.
REALNESS_INSTRUCTION = (
    'If the expression does not represent a real number, indicate "Not a Real Number".'
)

#: One question per exact family this repository can answer, and the form each
#: must report. Read as the enumeration it is: a solver added without a row
#: here is a solver whose answers nothing downstream has agreed to consume.
FAMILIES = [
    # (name, instruction, expressions, extras, expected form)
    (
        "scalar symbolic",
        "Simplify. Express your answer using rational exponents.",
        [r"y^{3/4} \cdot y^{2/5}"],
        {},
        SCALAR,
    ),
    (
        "scalar radical (distance)",
        "Find the distance between the following pair of points.",
        ["(7,0)", "(-3,-1)"],
        {},
        SCALAR,
    ),
    (
        "ordered pair (midpoint)",
        "Find the midpoint between the following pair of points.",
        ["(7,-4)", "(10,3)"],
        {},
        ORDERED_PAIR,
    ),
    (
        "ordered pair (vertex)",
        "Find the vertex of the parabola.",
        ["f(x)=(x-3)^2-1"],
        {},
        ORDERED_PAIR,
    ),
    (
        "parts (several roots)",
        "Solve the equation.",
        ["x^2-9=0"],
        {"answer_parts": 2},
        PARTS,
    ),
    (
        "choice (realness)",
        REALNESS_INSTRUCTION,
        [r"\sqrt{-100}"],
        {},
        CHOICE,
    ),
    (
        "choice (quadrant)",
        "In which quadrant does the point lie?",
        ["(3,-4)"],
        {
            "choices": [
                "Quadrant I",
                "Quadrant II",
                "Quadrant III",
                "Quadrant IV",
                "The point is on an axis",
            ]
        },
        CHOICE,
    ),
]


@pytest.mark.parametrize(
    ("name", "instruction", "expressions", "extras", "expected"),
    FAMILIES,
    ids=[case[0] for case in FAMILIES],
)
def test_every_exact_family_names_its_form(
    name, instruction, expressions, extras, expected
):
    solution, decline = solve_exact(instruction, expressions, **extras)

    assert solution is not None, decline
    assert solution.form == expected
    assert solution.form in ANSWER_FORMS


def test_the_form_set_is_closed():
    """Growing it is a protocol change, and one the consumer has to be told."""
    assert ANSWER_FORMS == (SCALAR, ORDERED_PAIR, PARTS, CHOICE)


def test_the_form_crosses_on_the_answer_object():
    """A consumer reads it off the answer, beside `entry_mode`."""
    result = solve_math(
        MathProblem(
            instruction="Find the midpoint between the following pair of points.",
            expressions=("(7,-4)", "(10,3)"),
        ),
        reason=Reasoner(),
    )

    assert result["answer"]["form"] == ORDERED_PAIR
    assert result["answer"]["display"] == "(17/2,-1/2)"
    assert result["provenance"]["method"] == "SymPy exact midpoint of two points"


@pytest.mark.parametrize(
    ("text", "parts_needed", "choices", "expected"),
    [
        ("FINAL ANSWER: 42", 1, (), SCALAR),
        ("FINAL ANSWER: -3, 3\nPART 1: -3\nPART 2: 3", 2, (), PARTS),
        ("FINAL ANSWER: Quadrant IV", 1, ("Quadrant IV", "Quadrant I"), CHOICE),
    ],
)
def test_a_reasoned_answer_names_the_form_it_actually_produced(
    text, parts_needed, choices, expected
):
    """Never `ordered-pair`: no reasoned answer is held to that structure, and
    claiming a family nothing verified would be worse than claiming none."""
    result = solve_math(
        MathProblem(
            instruction="Find the domain of the following function.",
            expressions=(r"f(x)=\frac{x+1}{x^2-9}",),
            answer_parts=parts_needed,
            answer_choices=choices,
        ),
        reason=Reasoner(text=text),
    )

    assert result["route"] == "reasoning"
    assert result["answer"]["form"] == expected


def test_the_quadrant_answer_is_a_choice_and_not_a_scalar():
    """The distinction that matters downstream: one is typed, one is clicked."""
    solution, _ = solve_exact(
        "In which quadrant does the point lie?",
        ["(3,-4)"],
        choices=["Quadrant I", "Quadrant II", "Quadrant III", "Quadrant IV"],
    )

    assert solution.form == CHOICE
    assert solution.method == QUADRANT_METHOD
