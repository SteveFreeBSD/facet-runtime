"""The missing half of an ordered pair that satisfies a stated equation.

Hawkes writes the equation, draws the pair with one coordinate given and one
answer box, and asks for the value that makes the pair a solution. That is one
row of a table of values with one blank in it, and it is answered by the same
machinery `table.py` completes a grid with: bind what the question states,
solve for what it left out, and prove the result by substituting it back into
the relation the question actually wrote.

Live, on 2026-09-07: `2x + y = 2`, "determine the missing coordinate in (4, ?)".
The deterministic stage had no branch for it, declined, and a reasoning model
answered `10` -- twice, because the browser was also asking for two values --
against a question whose answer is `-6`.
"""

from __future__ import annotations

import re

import sympy

from facet_runtime.exact.linear import rational
from facet_runtime.exact.table import (
    TableRefused,
    _order,
    _relation,
    _solutions,
)

#: The name this solver answers to.
COORDINATE_METHOD = "SymPy exact ordered-pair completion"

#: The columns an ordered pair has, in the order it is written.
_AXES = ("x", "y")

#: The question this module answers. Hawkes phrases it as a missing coordinate,
#: a missing value, or completing the pair; all of them say what the pair must
#: do, which is satisfy the equation that was written down.
COORDINATE_REQUEST = re.compile(
    r"\b(?:find|determine|complete|give|state|what\s+is)\b[^.?!]*?"
    r"(?:"
    r"\bmissing\s+(?:coordinate|value|entry|number)\b"
    r"|\bordered\s+pair\b[^.?!]*?\bsatisf"
    r"|\bcoordinate\b[^.?!]*?\bsatisf"
    r")",
    re.IGNORECASE,
)

#: A pair with one side written and the other left for the student. The blank
#: reaches us as an empty slot, a question mark, or an underscore rule,
#: depending on how the page typeset the box.
_BLANK = re.compile(r"^[\s?_–—.…]*$")

_PAIR = re.compile(r"\(\s*(?P<first>[^,()]*?)\s*,\s*(?P<second>[^,()]*?)\s*\)")

#: The other way the known coordinate arrives: named rather than positioned.
_NAMED = re.compile(
    r"\b(?P<axis>[xy])\s*=\s*(?P<value>-?\s*(?:\d+\s*/\s*\d+|\d*\.?\d+))",
    re.IGNORECASE,
)


def read_partial_pair(text: str) -> tuple[int, object] | None:
    """`(axis index of the blank, the value that was given)`, or None.

    Exactly one coordinate readable and exactly one blank. A pair with both
    sides written is not this question -- it is a pair to be checked, which is
    a different answer -- and a pair with neither is not a question at all.
    """
    for match in _PAIR.finditer(text or ""):
        sides = (match.group("first"), match.group("second"))
        values = [rational(side) for side in sides]
        blanks = [_BLANK.fullmatch(side) is not None for side in sides]
        for index in (0, 1):
            other = 1 - index
            if blanks[index] and values[other] is not None:
                return index, values[other]
    return None


def _stated_axis(instruction: str) -> tuple[int, object] | None:
    """A coordinate named rather than positioned: "when x = 4"."""
    match = _NAMED.search(instruction or "")
    if match is None:
        return None
    value = rational(match.group("value"))
    if value is None:
        return None
    given = _AXES.index(match.group("axis").lower())
    return 1 - given, value


def solve_missing_coordinate(
    instruction: str, expressions: list[str]
) -> tuple[str | None, str]:
    """The coordinate that completes the pair, or a named refusal."""
    written = [item for item in expressions if item and item.strip()]
    relations = [item for item in written if "=" in item and _PAIR.search(item) is None]
    if len(relations) != 1:
        return None, "an ordered pair is completed from one stated equation"

    # The pair may be typeset beside the equation or written into the sentence,
    # and either way it says the same thing.
    partial = None
    for source in [item for item in written if item not in relations] + [instruction]:
        partial = read_partial_pair(source)
        if partial is not None:
            break
    if partial is None:
        partial = _stated_axis(instruction)
    if partial is None:
        return None, "the pair's given coordinate could not be read exactly"

    missing, given = partial
    try:
        residual = _relation(relations, set(_AXES))
    except TableRefused as refusal:
        return None, str(refusal)

    symbols = {name: sympy.Symbol(name, real=True) for name in _AXES}
    unknown = symbols[_AXES[missing]]
    bindings = {symbols[_AXES[1 - missing]]: given}
    if unknown not in residual.free_symbols:
        return None, "the equation does not constrain the missing coordinate"
    try:
        found = _solutions(residual, bindings, unknown, "the ordered pair")
    except TableRefused as refusal:
        return None, str(refusal)

    # Hawkes draws one box, so it wants one value. Where an equation admits
    # more than one -- a parabola crossed by a vertical line does -- the
    # question has not been answered by picking one, and saying so is the only
    # reading that is not a guess.
    if len(found) != 1:
        return None, "the equation admits more than one missing coordinate"
    return str(sorted(found, key=_order)[0]), ""
