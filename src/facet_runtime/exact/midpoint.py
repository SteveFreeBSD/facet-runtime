"""The midpoint of two points, kept rational.

Live, repeatedly: the deterministic stage had no branch for this, so the
question went to gpt-oss:20b, which answered `(8.5,-0.5)`. Hawkes refused it,
and was right to. The midpoint of `(7,-4)` and `(10,3)` is `(17/2,-1/2)`; the
decimal is a rendering of that answer in a notation the answer box does not
take, and on a box publishing `0123456789-` with a Fraction template there is
no `.` to type at all.

Nothing was wrong with the model's arithmetic. Halving the sum of two rationals
is exact, and asking a model to do it is asking it to render an exact value in
whatever notation it favours -- which for a half is a decimal about as often as
not. So the family is computed here instead, the way the distance between the
same two points already is: read the pair, average each coordinate, and let
SymPy keep the result rational.

This module never evaluates to a float. `Rational(17, 2)` prints as `17/2` and
an integer midpoint prints as an integer, so the exact answer and its written
form are the same object.
"""

from __future__ import annotations

import re

from facet_runtime.exact.distance import read_points

#: The name this solver answers to. A reader looking at `(17/2,-1/2)` should be
#: able to see that the midpoint formula produced it, and that no model did.
MIDPOINT_METHOD = "SymPy exact midpoint of two points"

#: The question this module answers.
#:
#: A request verb is required. "Midpoint" appears in the prose of questions
#: that ask for something else -- "using the midpoint formula, find the
#: distance" -- and claiming those on the strength of the word alone would
#: answer a question nobody asked. Hyphen and space are both admitted because
#: Hawkes writes it both ways.
MIDPOINT_REQUEST = re.compile(
    r"\b(?:find|calculate|determine|compute|give|state|identify|what\s+is)\b"
    r"[^.?!]*?\bmid[\s-]?point\b",
    re.IGNORECASE,
)

#: Asked about the same two points, and a different question. Each of these has
#: its own answer and none of them is a midpoint, so a sentence asking for one
#: of them as well is ambiguous rather than a midpoint question with extra
#: words -- and is declined rather than half-answered.
_OTHER_ASK = re.compile(
    r"\b(?:distance|slope|equation\s+of\s+the\s+line|length|area|perimeter)\b",
    re.IGNORECASE,
)


def solve_midpoint(instruction: str, expressions: list[str]) -> tuple[str | None, str]:
    """The exact midpoint of the two points, or a named refusal.

    Returned in the canonical ordered-pair form every other exact solver that
    answers with a point uses: the parentheses are part of the answer, and the
    coordinates are written the way SymPy writes an exact rational.
    """
    if _OTHER_ASK.search(instruction):
        return None, "the question asks for more than the midpoint"
    points = read_points(instruction, expressions)
    if points is None:
        return None, "a coordinate pair could not be read exactly"
    if len(points) != 2:
        return None, f"a midpoint needs two points and {len(points)} were read"

    (x1, y1), (x2, y2) = points
    # Both coordinates are exact rationals by the time they reach here, so the
    # halved sums are too. Nothing is evaluated, rounded or approximated: a
    # half stays a half, and an integer midpoint stays an integer.
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    return f"({x},{y})", ""
