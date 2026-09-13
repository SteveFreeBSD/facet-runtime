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

import ast
from pathlib import Path

import pytest
from test_solve_math import Reasoner

from facet_runtime.exact import (
    ANSWER_FORMS,
    CHOICE,
    ORDERED_PAIR,
    PARTS,
    RELATION,
    SCALAR,
    solve_exact,
    solve_over_points,
)
from facet_runtime.exact.quadrant import QUADRANT_METHOD
from facet_runtime.solve import MathProblem, solve_math

#: Every module that may construct an exact answer. Listed as a directory walk
#: rather than by name, so a solver added in a new file is covered by the
#: structural check below without anyone remembering to add it here.
EXACT_SOURCES = tuple(
    (Path(__file__).resolve().parents[1] / "src" / "facet_runtime").rglob("*.py")
)

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
        "relation (a line in slope-intercept form)",
        (
            "Find the equation of the line in slope-intercept form that "
            "passes through the following point with the given slope."
        ),
        ["(0,5)", "Slope=-2"],
        {},
        RELATION,
    ),
    (
        "scalar interval (linear inequality)",
        "Solve the inequality and express your answer in interval notation.",
        [r"-33<3y-9\leq12"],
        {},
        SCALAR,
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


def test_a_two_value_regression_is_a_parts_answer():
    """Regression: it said `scalar` while carrying two values.

    `solve_over_points` splits on how many values the question takes, and the
    two-value branch was constructed without a `form`, so it took the dataclass
    default -- `scalar` -- while carrying `parts=("9", "36")`. Nothing caught
    it because `FAMILIES` above has no regression row and no test compared the
    two fields to each other.

    It was latent rather than live: the consumer still routes on `parts` and
    only validates `form`. But the whole reason `form` exists is so a consumer
    can stop parsing the string, and an answer that says `scalar` while holding
    two values is exactly the misreport it was added to prevent.
    """
    solution, decline = solve_over_points(
        "Treating revenue as a function of the number of photos sold, if she "
        "uses quadratic regression to fit a curve to the data, what number of "
        "photos sold and what price per photo will maximize her revenue?",
        [("4", "224"), ("5", "260"), ("12", "288")],
        2,
    )

    assert solution is not None, decline
    assert solution.parts == ("9", "36")
    assert solution.form == PARTS


def test_a_one_value_regression_is_still_a_scalar():
    """The other branch of the same split, so the fix cannot over-reach."""
    solution, decline = solve_over_points(
        "Treating revenue as a function of the number of photos sold, if she "
        "uses quadratic regression to fit a curve to the data, what number of "
        "photos sold will maximize her revenue?",
        [("4", "224"), ("5", "260"), ("12", "288")],
        1,
    )

    assert solution is not None, decline
    assert solution.parts == ()
    assert solution.entry == "9"
    assert solution.form == SCALAR


@pytest.mark.parametrize("path", sorted(EXACT_SOURCES))
def test_no_emitter_carries_parts_without_saying_so(path):
    """The structural invariant, held against the source rather than a corpus.

    Fixing the regression's own branch fixes one site. What stops the next one
    is this: every `ExactSolution(...)` that passes `parts` must also pass a
    `form`, because the default is `scalar` and a silent default is how the
    last one happened. A conditional form -- `SCALAR if single else PARTS` --
    satisfies it, which is the shape the table completion already uses.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "ExactSolution"
        ):
            continue
        named = {keyword.arg for keyword in node.keywords}
        if "parts" in named and "form" not in named:
            offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == [], (
        f"{offenders} construct an ExactSolution with parts and no form, so it "
        "defaults to scalar while carrying several values"
    )


@pytest.mark.parametrize(
    ("equation", "entry"),
    [
        ("2x-1=0", r"\frac{1}{2}"),
        ("4x=3", r"\frac{3}{4}"),
        ("|2x-3|=0", r"\frac{3}{2}"),
    ],
)
def test_a_solved_value_is_mathematics_and_not_a_literal(equation, entry):
    """Regression: a fractional root was unenterable while an integer one worked.

    `entry_mode` says how literally to take a value, and `verbatim` means "type
    exactly this". Two branches carried a *solved* value at the default
    `verbatim` -- the linear equation whose question did not name its variable,
    and the single-solution absolute value -- while the branch beside each said
    `math`. `_display` writes a rational as `\frac{1}{2}`, so the consumer took
    it literally and the keypad refused it on the backslash.

    Integers hid it: `3x+2=8` answers `2`, which is the same either way.
    """
    solution, decline = solve_exact("Solve the equation.", [equation])

    assert solution is not None, decline
    assert solution.entry == entry
    assert solution.entry_mode == "math"


def test_a_classification_is_still_a_literal():
    """The other half of the same branch, so the fix cannot over-reach: "No
    Solution" is a phrase and is typed, or selected, exactly as written."""
    solution, decline = solve_exact("Solve the equation.", ["|2x-1|=-4"])

    assert solution is not None, decline
    assert solution.entry == "No Solution"
    assert solution.entry_mode == "verbatim"


@pytest.mark.parametrize(
    ("instruction", "equation", "answer_parts", "roots"),
    [
        ("Solve the equation. Enter both solutions.", "|x|=2", 2, ("-2", "2")),
        ("Solve the equation.", "|2x-1|=4", 1, (r"\frac{-3}{2}", r"\frac{5}{2}")),
    ],
)
def test_two_absolute_value_roots_are_two_answers(
    instruction, equation, answer_parts, roots
):
    """Audit F08: `|x|=2`, asked for both solutions, answered "Two Solutions".

    Both roots were computed and shown, and neither could be entered. Several
    roots are several values, as the polynomial branch has always published
    them. A count of solutions is an answer only where the page offers it as a
    choice, and a question that publishes choices is answered before this.
    """
    solution, decline = solve_exact(instruction, [equation], answer_parts=answer_parts)

    assert solution is not None, decline
    assert solution.parts == roots
    assert solution.entry == ""
    assert solution.entry_mode == "math"
    assert solution.form == PARTS


def test_two_absolute_value_roots_cross_the_wire_as_parts():
    """The audit's own reproduction, at the boundary a consumer reads."""
    result = solve_math(
        MathProblem(
            instruction="Solve the equation. Enter both solutions.",
            expressions=("|x|=2",),
            answer_parts=2,
        ),
        reason=Reasoner(),
    )

    assert result["route"] == "exact"
    assert result["answer"]["parts"] == ["-2", "2"]
    assert result["answer"]["form"] == PARTS


def test_the_form_set_is_closed():
    """Growing it is a protocol change, and one the consumer has to be told."""
    assert ANSWER_FORMS == (SCALAR, ORDERED_PAIR, PARTS, CHOICE, RELATION)


def test_an_equation_carries_both_its_sides_and_nothing_else_does():
    """The one family whose answer is not a single value.

    A consumer has two ways to enter an equation and has to choose between
    them, so both sides cross rather than one written form somebody splits
    later. Every other family carries no relation at all, which is what makes
    the field's presence the signal.
    """
    line, decline = solve_exact(
        "Find the equation of the line in slope-intercept form that passes "
        "through the following point with the given slope.",
        ["(0,5)", "Slope=-2"],
    )

    assert line is not None, decline
    assert line.form == RELATION
    assert (line.relation.subject, line.relation.value) == ("y", "-2x+5")
    assert line.entry == line.relation.written == "y=-2x+5"

    scalar, decline = solve_exact("Simplify.", ["2+3"])
    assert scalar is not None, decline
    assert scalar.relation is None


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
    assert result["answer"]["relation"] is None
    assert result["provenance"]["method"] == "SymPy exact midpoint of two points"


def test_an_equations_two_sides_cross_on_the_answer_object():
    """The consumer reads them off the answer, beside `form`."""
    result = solve_math(
        MathProblem(
            instruction=(
                "Find the equation of the line in slope-intercept form that "
                "passes through the following point with the given slope."
            ),
            expressions=("(0,5)", "Slope=-2"),
        ),
        reason=Reasoner(),
    )

    assert result["answer"]["form"] == RELATION
    assert result["answer"]["entry"] == "y=-2x+5"
    assert result["answer"]["relation"] == {"subject": "y", "value": "-2x+5"}


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
