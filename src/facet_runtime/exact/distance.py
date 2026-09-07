"""The distance between two points, computed rather than approximated.

Hawkes asks this with the two points written as coordinate pairs and answered
in one dynamic box whose published character set is digits and a minus sign,
with a Radical template beside it. So the answer it wants is the exact radical
-- `sqrt(101)` -- and a decimal is not a worse answer to this question, it is
an unenterable one: there is no `.` in that box's character set at all.

Live, on 2026-09-07, the deterministic stage had no branch for this question.
It declined, a reasoning model answered it with an approximation, and the
add-on could place none of it. Nothing was wrong with the model's arithmetic;
the question simply has an exact answer and was not being given one.

This module computes the same square root the student is asked for, keeps it
symbolic through SymPy so `sqrt(101)` stays `sqrt(101)` and `sqrt(200)` becomes
`10*sqrt(2)`, and declines anything it cannot read as exactly two points.
"""

from __future__ import annotations

import re

import sympy

from facet_runtime.exact.linear import rational

#: The name this solver answers to. A reader looking at `sqrt(101)` should be
#: able to see that the distance formula produced it.
DISTANCE_METHOD = "SymPy exact distance between two points"

#: The same value pattern the line solver reads a coordinate with: an integer,
#: a decimal, `a/b`, or MathJax's `\frac{a}{b}`. Never a float.
_VALUE = r"(?:\\frac\{[^{}]+\}\{[^{}]+\}|[+-]?\s*\d+\s*/\s*\d+|[+-]?\s*\d*\.?\d+)"

_POINT = re.compile(rf"\(\s*(?P<x>{_VALUE})\s*,\s*(?P<y>{_VALUE})\s*\)")

#: The question this module answers. "Distance" alone is not enough -- a rate
#: question says it too -- so a request verb and the pair being asked about are
#: both required.
DISTANCE_REQUEST = re.compile(
    r"\b(?:find|calculate|determine|compute|what\s+is)\b[^.?!]*?"
    r"\bdistance\b[^.?!]*?\b(?:points?|pair|coordinates)\b",
    re.IGNORECASE,
)

#: Asked about the same two points, and a different question. Each of these has
#: its own answer and none of them is a distance, so matching "distance" in a
#: sentence that also asks for one of them is not enough to claim the question.
_OTHER_ASK = re.compile(
    r"\b(?:midpoint|slope|equation\s+of\s+the\s+line|area|perimeter)\b",
    re.IGNORECASE,
)


def read_points(instruction: str, expressions: list[str]) -> list[tuple] | None:
    """The coordinate pairs this question is about, or None.

    The markup is read before the prose: a pair Hawkes typeset is the question's
    own data, while a pair in a sentence may be an example. Both are read,
    because the two points arrive in the markup on some questions and inside
    the instruction on others, and duplicates are dropped so a pair stated in
    both places is still one point.
    """
    found: list[tuple] = []
    for fragment in [*expressions, instruction]:
        for match in _POINT.finditer(fragment or ""):
            x = rational(match.group("x"))
            y = rational(match.group("y"))
            if x is None or y is None:
                return None  # a pair that cannot be read exactly is a refusal
            pair = (x, y)
            if pair not in found:
                found.append(pair)
    return found


def solve_point_distance(
    instruction: str, expressions: list[str]
) -> tuple[str | None, str]:
    """The exact distance between the two points, or a named refusal.

    Returns the answer in the same machine notation every other exact solver
    returns -- SymPy's own `sqrt(101)` -- and never a decimal. An irrational
    distance is irrational; rounding it here would produce something the answer
    box cannot even accept.
    """
    if _OTHER_ASK.search(instruction):
        return None, "the question asks for more than the distance"
    points = read_points(instruction, expressions)
    if points is None:
        return None, "a coordinate pair could not be read exactly"
    if len(points) != 2:
        return None, f"the distance needs two points and {len(points)} were read"

    (x1, y1), (x2, y2) = points
    squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    # `nsimplify` is not wanted and neither is `evalf`: the inputs are already
    # exact rationals, so the square root of their sum is exact too, and SymPy
    # reduces it as far as it goes -- `sqrt(200)` to `10*sqrt(2)`, `sqrt(25)`
    # to `5`. What comes out is the answer, not a rounding of it.
    value = sympy.sqrt(sympy.Rational(squared))
    return str(sympy.simplify(value)), ""
