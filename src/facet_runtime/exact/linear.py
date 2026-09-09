"""A line, from the properties that fix it. No search, no model, no guessing.

Hawkes lesson 3.2 states a linear function the way a textbook does: not as an
expression to rewrite, but as two facts about a line -- a slope and a value, a
value and a value, two points. Nothing on the page is an expression to simplify,
so every solver that reads a verb and manipulates what follows it declined, and
the question went to a reasoning model. Live, on `f(0) = -3` with `slope = -5`,
the model answered `-2x-3`: the intercept it was given, and a slope it invented.

Two facts determine a line and one does not. That is the whole of the
mathematics here, and it is arithmetic over exact rationals: solve for `m` and
`b`, then put every stated property back into `y = mx + b` and check it. A
property that does not check is a refusal, never a rounding.

What is deliberately absent is inference. This reads explicit statements of a
line's properties in the forms Hawkes writes them, and declines everything else
-- a word problem that implies a rate, a line described by its relationship to
another line, a vertical line, a single point on its own. A decline here costs
a reasoning call; a guess here costs a wrong answer typed into real coursework.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sympy

#: An exact rational, as any of the forms a Hawkes page writes one in: an
#: integer, a decimal, `a/b`, or MathJax's `\frac{a}{b}`. Never a float --
#: `0.1` is read as one tenth, and stays one tenth all the way to the answer.
_VALUE = r"(?:\\frac\{[^{}]+\}\{[^{}]+\}|[+-]?\s*\d+\s*/\s*\d+|[+-]?\s*\d*\.?\d+)"

#: The question this module answers. Both halves are required: a request verb,
#: and a line or linear function as the thing requested. "Graph the linear
#: function" is a different question and does not match; "write it in
#: slope-intercept form" does, and then declines for want of any property.
LINEAR_REQUEST = re.compile(
    r"\b(?:find|write|determine|give|state|express)\b[^.?!]*?"
    r"\b(?:linear\s+function|linear\s+equation|equation\s+of\s+(?:the|a)\s+line"
    r"|slope[-\s]intercept\s+form)\b",
    re.IGNORECASE,
)

# The properties, each in the spellings Hawkes uses. Every one of them names
# what it is: there is no pattern here that reads a bare number as a slope.
#: A slope, including the form Hawkes actually writes it in: "Slope of f = -5",
#: which names the function whose slope it is. Two things make that hard to
#: read. The name sits between the word and the value, and the MathML converter
#: strips every space -- `<mtext>Slope of </mtext>` arrives glued to what
#: follows it as `Slopeoff=-5`. So the spacing is optional throughout, and the
#: connector after the name needs no word boundary in front of it, because the
#: page's own spacing is gone by the time this reads it.
_SLOPE = re.compile(
    rf"(?:^|\b)(?:the\s*)?slope(?![-\s]?intercept)"
    rf"(?:\s*of\s*(?:the\s*)?(?:function\s*)?[A-Za-z](?:\s*\(\s*[A-Za-z]\s*\))?)?"
    rf"\s*(?:=|:|is\b|of\b)\s*(?P<value>{_VALUE})"
    rf"|(?:^|\b)m\s*(?:=|:|\bis\b)\s*(?P<mvalue>{_VALUE})",
    re.IGNORECASE,
)
_FUNCTION_VALUE = re.compile(
    rf"(?:^|[^A-Za-z])(?P<name>[A-Za-z])\s*\(\s*(?P<at>{_VALUE})\s*\)\s*=\s*"
    rf"(?P<value>{_VALUE})",
    re.IGNORECASE,
)
_Y_INTERCEPT = re.compile(
    rf"(?:^|\b)(?:the\s+)?(?:y[-\s]intercept|b)\s*(?:=|:|\bis\b|\bof\b|\bat\b)\s*"
    rf"(?:\(\s*0\s*,\s*(?P<pair>{_VALUE})\s*\)|(?P<value>{_VALUE}))",
    re.IGNORECASE,
)
_X_INTERCEPT = re.compile(
    rf"(?:^|\b)(?:the\s+)?x[-\s]intercept\s*(?:=|:|\bis\b|\bof\b|\bat\b)\s*"
    rf"(?:\(\s*(?P<pair>{_VALUE})\s*,\s*0\s*\)|(?P<value>{_VALUE}))",
    re.IGNORECASE,
)
_POINT = re.compile(rf"\(\s*(?P<x>{_VALUE})\s*,\s*(?P<y>{_VALUE})\s*\)")

#: Where a pair written in prose stops being scenery and starts being a point
#: the line goes through. Without an anchor, any parenthesised pair anywhere in
#: a sentence would be read as data.
_THROUGH = re.compile(
    r"\b(?:pass(?:es|ing)?\s+through|goes?\s+through|through|contain(?:s|ing)?"
    r"\s+the\s+points?|the\s+points?)\b",
    re.IGNORECASE,
)

#: A line described by its relationship to another line. The properties are
#: stated in the same words -- a slope, a point -- and mean something else: the
#: slope of a perpendicular is the negative reciprocal of the one written down.
#: Read literally, "perpendicular to a line with a slope of 2, through (1,3)"
#: derives `2x+1`, which is confidently and completely wrong, so the whole
#: family is refused rather than partly understood.
RELATIVE_TO_ANOTHER_LINE = re.compile(
    r"\b(?:parallel|perpendicular|orthogonal|normal\s+to)\b", re.IGNORECASE
)

#: An expression that states nothing: the answer field's own `f(x) =` label, or
#: a bare name. It is skipped rather than refused, because skipping a fragment
#: that carries no property discards no information. Anything else unreadable
#: is a refusal -- a property that cannot be read is not a property that can be
#: left out.
_STATES_NOTHING = re.compile(r"^[A-Za-z]\s*(?:\(\s*[A-Za-z]\s*\))?\s*=?$")

#: A function the question names, by writing something it is applied to: a
#: value of it, `f(0) = -3`, or the bare `f(x) =` an answer field is labelled
#: with. One letter, because that is what Hawkes names a function with, and an
#: argument that is a variable or a number, because those are the two things it
#: writes inside the brackets.
_NAMED_FUNCTION = re.compile(
    rf"(?:^|[^A-Za-z])(?P<name>[A-Za-z])\s*\(\s*(?:[A-Za-z]|{_VALUE})\s*\)\s*="
)

#: The variable a line is written in. `render` writes `mx + b` in it, so the
#: subject below is written as a function of the same one.
VARIABLE = "x"


def subject(instruction: str, expressions: list[str]) -> str:
    """The left side of the equation this question asks for.

    A line is two numbers and an equation stating them, and what stands on the
    left of that equation is the question's to say. Hawkes says it by naming a
    function -- writing one of its values, `f(0) = -3`, or labelling the answer
    field `f(x) =` -- and says nothing when it asks for slope-intercept form,
    which is written `y = mx + b` by definition.

    So one name is that name, and none is `y`. More than one is `y` as well:
    two named functions in one question is not a line stated under either of
    them, and picking whichever was written first would be a guess about which
    one the answer belongs to.
    """
    names = {
        match.group("name")
        for text in (instruction, *expressions)
        for match in _NAMED_FUNCTION.finditer(text)
    }
    return f"{names.pop()}({VARIABLE})" if len(names) == 1 else "y"


def rational(text: str) -> sympy.Rational | None:
    """One exact rational, or None when the text is not a number at all."""
    cleaned = text.replace("−", "-").replace(" ", "")
    fraction = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", cleaned)
    if fraction:
        top, bottom = (rational(part) for part in fraction.groups())
        if top is None or bottom is None or bottom == 0:
            return None
        return sympy.Rational(top, bottom)
    if not re.fullmatch(r"[+-]?(?:\d+/\d+|\d*\.?\d+)", cleaned):
        return None
    try:
        value = sympy.Rational(cleaned)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value


@dataclass(frozen=True, slots=True)
class Property:
    """One stated fact about the line: a slope, or a point it passes through.

    `text` is how the page said it and is deliberately outside the comparison.
    The same fact stated twice -- once in the markup and once in the words --
    is one fact, and `f(0) = -3` and a y-intercept of -3 are the same point.
    """

    kind: str
    slope: sympy.Rational | None = None
    x: sympy.Rational | None = None
    y: sympy.Rational | None = None
    text: str = field(default="", compare=False)


@dataclass(frozen=True, slots=True)
class DerivedLine:
    """The line the stated properties determine, and the working behind it."""

    slope: sympy.Rational
    intercept: sympy.Rational
    evidence: dict[str, str]


def _slope(value, text) -> Property:
    return Property("slope", slope=value, text=text)


def _point(x, y, text) -> Property:
    return Property("point", x=x, y=y, text=text)


def read_properties(fragment: str) -> list[Property] | None:
    """Every property one fragment states, or None if it states none.

    A fragment is one expression read off the page, or the instruction itself.
    The distinction between "states no property" and "states something this
    cannot read" is made by the caller, which knows which of the two matters.
    """
    found: list[Property] = []
    claimed: list[tuple[int, int]] = []

    def take(match) -> bool:
        """Reserve a span, so one substring is never read as two properties."""
        span = match.span()
        if any(start < span[1] and span[0] < end for start, end in claimed):
            return False
        claimed.append(span)
        return True

    for match in _FUNCTION_VALUE.finditer(fragment):
        at, value = rational(match.group("at")), rational(match.group("value"))
        if at is None or value is None or not take(match):
            continue
        found.append(_point(at, value, f"{match.group('name')}({at})={value}"))
    for match in _Y_INTERCEPT.finditer(fragment):
        value = rational(match.group("pair") or match.group("value"))
        if value is None or not take(match):
            continue
        found.append(_point(sympy.Integer(0), value, f"y-intercept={value}"))
    for match in _X_INTERCEPT.finditer(fragment):
        value = rational(match.group("pair") or match.group("value"))
        if value is None or not take(match):
            continue
        found.append(_point(value, sympy.Integer(0), f"x-intercept={value}"))
    for match in _SLOPE.finditer(fragment):
        value = rational(match.group("value") or match.group("mvalue") or "")
        if value is None or not take(match):
            continue
        found.append(_slope(value, f"slope={value}"))
    anchor = _THROUGH.search(fragment)
    for match in _POINT.finditer(fragment):
        if anchor is None and match.span()[0] != 0:
            # A bare pair is a point only when the fragment is that pair -- the
            # usual markup for "the line through (1,2) and (3,8)" -- or when
            # the words said the line passes through it.
            continue
        x, y = rational(match.group("x")), rational(match.group("y"))
        if x is None or y is None or not take(match):
            continue
        found.append(_point(x, y, f"({x},{y})"))
    return found or None


def collect(instruction: str, expressions: list[str]) -> tuple[list[Property], int]:
    """Read every stated property, and count the fragments that would not read.

    Both halves are returned because they answer different questions. The
    properties are the mathematics. The count is what stops a line from being
    derived out of the fragments that happened to parse, while a fragment that
    said something else about it was quietly dropped.
    """
    properties: list[Property] = []
    unreadable = 0

    def add(new: list[Property]) -> None:
        for item in new:
            if item not in properties:
                properties.append(item)

    for expression in expressions:
        fragment = expression.strip()
        stated = read_properties(fragment)
        if stated is None:
            if not _STATES_NOTHING.fullmatch(fragment):
                unreadable += 1
            continue
        add(stated)
    add(read_properties(instruction) or [])
    return properties, unreadable


def solve_linear_function(
    instruction: str, expressions: list[str], answer_parts: int = 1
) -> tuple[DerivedLine | None, str]:
    """Derive `y = mx + b` from the stated properties, or decline and say why.

    Returns the line and an empty reason, or None and the reason. The router
    turns the line into an answer; the mathematics and its proof are all here.

    An empty reason with no result is the third answer: the page stated no
    property this reads, so this is not the question after all and the router
    goes on looking. A named reason means it *was* this question and the line
    could not be derived from it, which is what a live fallback reports.
    """
    if answer_parts != 1:
        return None, (
            f"a linear function is one answer but the answer shape requires "
            f"{answer_parts}"
        )
    if RELATIVE_TO_ANOTHER_LINE.search(" ".join((instruction, *expressions))):
        return None, (
            "a line stated by its relation to another line is not derived here"
        )

    properties, unreadable = collect(instruction, expressions)
    if not properties:
        # Nothing here is a property, so nothing here is this question. Said
        # with an empty reason so the remaining exact solvers still get their
        # turn: an instruction naming a linear function is not a claim that
        # every such question belongs to this route.
        return None, ""
    if unreadable:
        return None, "a stated property could not be read exactly"

    slopes = [item.slope for item in properties if item.kind == "slope"]
    points = [(item.x, item.y) for item in properties if item.kind == "point"]
    if len(set(slopes)) > 1:
        return None, "the stated slopes disagree"

    if slopes:
        if not points:
            return None, "a slope alone does not determine a linear function"
        slope = slopes[0]
        intercept = points[0][1] - slope * points[0][0]
    elif len(points) >= 2:
        first_x, first_y = points[0]
        # Any second point at a different x fixes the slope; a later one that
        # disagrees is caught by the verification below rather than ignored.
        apart = next((point for point in points[1:] if point[0] != first_x), None)
        if apart is None:
            return None, "a vertical line is not a linear function"
        slope = sympy.Rational(apart[1] - first_y, apart[0] - first_x)
        intercept = first_y - slope * first_x
    else:
        return None, "one point alone does not determine a linear function"

    # Every property, checked against the function that came out. Deriving from
    # two of them and trusting the rest is how a third, contradictory, property
    # would be silently dropped.
    checks: list[str] = []
    for item in properties:
        if item.kind == "slope":
            if sympy.simplify(item.slope - slope) != 0:
                return None, "the stated properties have no common linear function"
            checks.append(f"slope={slope}")
        else:
            value = sympy.simplify(slope * item.x + intercept)
            if sympy.simplify(value - item.y) != 0:
                return None, "the stated properties have no common linear function"
            checks.append(f"f({item.x})={value}")

    evidence = {
        "slope": str(slope),
        "y_intercept": str(intercept),
        "function": f"f(x)={render(slope, intercept)}",
        "properties": "; ".join(item.text for item in properties),
        "verification": "; ".join(checks),
    }
    return DerivedLine(slope=slope, intercept=intercept, evidence=evidence), ""


def render(slope, intercept) -> str:
    """Write `mx + b` the way it is typed: exactly, and without ambiguity.

    A rational coefficient is parenthesised. `3/2x` reads as `3/(2x)` to a
    homework editor as easily as it reads as `(3/2)x` to a person, and the
    difference is a wrong answer.
    """
    if slope == 0:
        return str(intercept)
    if slope == 1:
        term = "x"
    elif slope == -1:
        term = "-x"
    elif slope.is_Integer:
        term = f"{slope}x"
    else:
        term = f"({slope})x"
    if intercept == 0:
        return term
    return f"{term}{'+' if intercept > 0 else '-'}{abs(intercept)}"
