"""Which quadrant a point lies in, decided rather than reasoned about.

This is arithmetic on two signs. There is no expression to manipulate, no
approximation to make, and nothing a model can know about it that comparing a
number to zero does not -- and yet live, on 2026-09-08, every one of these went
to a reasoning model on a GPU and came back wrong-shaped, five times in an hour.

The reason was not the mathematics. The page publishes one radio group of five
choices, the editor probe counted five *controls* and called them five answers,
and Facet was asked for five separate values to a question with one. The model
duly produced five, the deterministic stage never saw the question because
nothing here claimed it, and the whole family fell through to a route it has no
business reaching.

Both halves are fixed. This is the half that answers it: read the point, compare
each coordinate to zero, and name the region. The other half is the browser
counting one radio group as one answer.

A question like this is answered by *choosing*, not by writing, so the answer is
not a value this module invents -- it is one of the choices the page published.
`select` takes those choices and returns the single one that names the region,
or declines. It never invents a label, never edits one, and never picks between
two that both match: an answer to a choice question that is not one of the
choices is not an answer to it at all.
"""

from __future__ import annotations

import re

import sympy

#: The name this solver answers to. A reader looking at "Quadrant III" should be
#: able to see that two sign comparisons produced it, and that nothing else did.
QUADRANT_METHOD = "exact quadrant classification"

#: The question this module answers.
#:
#: Both shapes Hawkes asks it in: "in which quadrant does the point lie" and
#: "determine the quadrant or axis". The word "axis" is admitted because a
#: five-choice version of this question is usually four quadrants and an axis,
#: and a point on an axis is in no quadrant at all -- which is the one case a
#: reader most needs the deterministic answer for.
QUADRANT_REQUEST = re.compile(
    r"\b(?:which|what|determine|identify|state|name|find)\b[^.?!]*?"
    r"\bquadrant\b|\bquadrant\b[^.?!]*?\b(?:lie|lies|located|located\s+in|falls)\b",
    re.IGNORECASE,
)

#: A coordinate: an integer, a decimal, `a/b`, or MathJax's `\frac{a}{b}`.
#: Never a float -- a coordinate read as 0.30000000000000004 has a sign, but a
#: reader cannot check the arithmetic that produced it.
_VALUE = r"(?:\\frac\{[^{}]+\}\{[^{}]+\}|[+-]?\s*\d+\s*/\s*\d+|[+-]?\s*\d*\.?\d+)"
_POINT = re.compile(rf"\(\s*(?P<x>{_VALUE})\s*,\s*(?P<y>{_VALUE})\s*\)")
_FRACTION = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")

#: Every region of the plane a point can be in, as this module names them.
#: Closed, and exhaustive over the two signs: there is no other answer to this
#: question, so a classification outside this set is a bug rather than a case.
QUADRANT_I = "quadrant-1"
QUADRANT_II = "quadrant-2"
QUADRANT_III = "quadrant-3"
QUADRANT_IV = "quadrant-4"
X_AXIS = "x-axis"
Y_AXIS = "y-axis"
ORIGIN = "origin"

REGIONS: tuple[str, ...] = (
    QUADRANT_I,
    QUADRANT_II,
    QUADRANT_III,
    QUADRANT_IV,
    X_AXIS,
    Y_AXIS,
    ORIGIN,
)

#: Which regions a published choice covers, by how it is written.
#:
#: Matched against the choice's own words, reduced to lower case with runs of
#: non-letters collapsed -- so "Quadrant III", "quadrant 3" and "QUADRANT-III"
#: are one thing, and the page's capitalisation and punctuation decide nothing.
#: Roman and arabic both, because Hawkes writes quadrants in Roman and a reader
#: writing a fixture will write them in either.
#:
#: A choice covers a *set* of regions rather than one, because a page may offer
#: "the point lies on an axis" without saying which -- one choice that is the
#: right answer for three different points. A choice that names the axis, or
#: names the origin, covers only that, and is preferred where both are offered.
_CHOICE_COVERAGE: tuple[tuple[frozenset[str], re.Pattern[str]], ...] = (
    (frozenset({QUADRANT_I}), re.compile(r"^quadrant (?:i|1)$")),
    (frozenset({QUADRANT_II}), re.compile(r"^quadrant (?:ii|2)$")),
    (frozenset({QUADRANT_III}), re.compile(r"^quadrant (?:iii|3)$")),
    (frozenset({QUADRANT_IV}), re.compile(r"^quadrant (?:iv|4)$")),
    # An axis choice is written many ways -- "x-axis", "on the x axis", "the
    # point lies on the x-axis" -- and every one of them says the same thing.
    # The axis letter beside the word is what identifies it.
    (frozenset({X_AXIS}), re.compile(r"\bx axis\b")),
    (frozenset({Y_AXIS}), re.compile(r"\by axis\b")),
    (frozenset({ORIGIN}), re.compile(r"\borigin\b")),
    # "On an axis", with no letter. The origin is on both axes, so it is
    # covered too: a page offering this one choice and no separate origin has
    # said that a point at (0,0) belongs here.
    (frozenset({X_AXIS, Y_AXIS, ORIGIN}), re.compile(r"\baxis\b|\baxes\b")),
)


class QuadrantRefused(Exception):
    """This question is not one this module can answer, and says why."""


def _rational(text: str) -> sympy.Rational:
    """One coordinate, exactly. Rationals stay rational; nothing becomes float."""
    cleaned = _FRACTION.sub(r"(\1)/(\2)", text.replace(" ", ""))
    value = sympy.Rational(sympy.nsimplify(cleaned, rational=True))
    return value


def read_point(instruction: str, expressions: list[str]) -> tuple | None:
    """The one point this question is about, or None.

    The markup is read before the prose, for the same reason every solver here
    does: a pair Hawkes typeset is the question's own data, while a pair inside
    a sentence may be an example. Exactly one point, because this question is
    about one -- two pairs is a different question and is declined rather than
    answered about whichever came first.
    """
    for source in (expressions, [instruction]):
        found = [match for text in source for match in _POINT.finditer(str(text))]
        if len(found) == 1:
            try:
                return (
                    _rational(found[0].group("x")),
                    _rational(found[0].group("y")),
                )
            except (TypeError, ValueError, ZeroDivisionError, sympy.SympifyError):
                return None
        if len(found) > 1:
            return None
    return None


def classify(point: tuple) -> str:
    """Which region of the plane the point is in. Two comparisons, exhaustive."""
    x, y = point
    if x == 0 and y == 0:
        return ORIGIN
    if y == 0:
        return X_AXIS
    if x == 0:
        return Y_AXIS
    if x > 0:
        return QUADRANT_I if y > 0 else QUADRANT_IV
    return QUADRANT_II if y > 0 else QUADRANT_III


def normalised(choice: str) -> str:
    """One published choice, reduced to the words and numbers in it.

    Digits are kept, because "Quadrant 1" is the same choice as "Quadrant I"
    and a rule that dropped the number would read it as the bare word.
    """
    return re.sub(r"[^a-z0-9]+", " ", choice.lower()).strip()


def coverage(choice: str) -> frozenset[str]:
    """Which regions a published choice is the right answer for.

    Empty when it names none -- "None of these" is a real choice and names no
    region, so a question is never answered with it on the strength of the
    others having been ruled out. That is an inference about the page's
    intentions, and this module makes none.
    """
    text = normalised(choice)
    found = [regions for regions, pattern in _CHOICE_COVERAGE if pattern.search(text)]
    if not found:
        return frozenset()
    # "x-axis" matches the rule for the x-axis and the rule for any axis, and
    # they do not disagree: one is contained in the other, and the contained
    # one is what the choice says. Where no match is contained in all the
    # others the choice genuinely names two things -- "Quadrant I or the
    # x-axis" is a real thing to print on a page -- and it names no one region.
    narrowest = min(found, key=len)
    if not all(narrowest <= regions for regions in found):
        return frozenset()
    # A choice that says "quadrant" and is not one of the four is not a choice
    # this module can read. "Quadrant I or the x-axis" would otherwise be taken
    # for the axis, on the strength of the half of it that parsed.
    if "quadrant" in text and not narrowest <= {
        QUADRANT_I,
        QUADRANT_II,
        QUADRANT_III,
        QUADRANT_IV,
    }:
        return frozenset()
    return narrowest


def select(region: str, choices: list[str]) -> str:
    """The one published choice that is the answer for this region, or raise.

    Specific before general: where a page offers both "the y-axis" and "on an
    axis", the one that names the axis is the answer and the other is merely
    also true. Only when no choice names the region exactly is a covering
    choice used, and then only if there is exactly one.

    Fail closed either way. A region no choice covers means this is not the
    question it looked like -- four quadrant choices and no axis option, asked
    about a point on an axis, has no right answer among them and must not be
    given the nearest one. Two equally specific choices means the page said
    something this cannot read, and picking either would be picking for a
    reason nobody could check.
    """
    covers = [(choice, coverage(choice)) for choice in choices]
    exact = [choice for choice, regions in covers if regions == {region}]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise QuadrantRefused(
            f"{len(exact)} published choices name the {region.replace('-', ' ')}"
        )
    general = [choice for choice, regions in covers if region in regions]
    if len(general) == 1:
        return general[0]
    if general:
        raise QuadrantRefused(
            f"{len(general)} published choices cover the {region.replace('-', ' ')}"
        )
    raise QuadrantRefused(
        f"the point lies on the {region.replace('-', ' ')} and no published "
        "choice names it"
    )


def solve_quadrant(
    instruction: str, expressions: list[str], choices: list[str]
) -> tuple[str, str]:
    """Answer a quadrant question with one of its own choices, or say why not.

    Returns `(choice, "")` or `("", refusal)`. The refusal is final and its
    caller must not fall through on it: there is nothing a model can add to two
    sign comparisons, and asked this question it would have to guess which
    alternatives the page offered before it could name one of them.
    """
    if not choices:
        return "", (
            "a quadrant question is answered by choosing, and this one "
            "published no choices"
        )
    point = read_point(instruction, expressions)
    if point is None:
        return "", "no single coordinate pair could be read for this quadrant question"
    try:
        return select(classify(point), choices), ""
    except QuadrantRefused as refusal:
        return "", str(refusal)
