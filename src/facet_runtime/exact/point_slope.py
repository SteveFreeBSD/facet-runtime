"""The exact slope of a line identified by two page-owned graph points."""

from __future__ import annotations

import re
from fractions import Fraction

POINT_SLOPE_METHOD = "Exact slope from two Cartesian points"

POINT_SLOPE_REQUEST = re.compile(
    r"\b(?:find|determine|calculate|compute)\b[^.?!]{0,160}\bslope\b"
    r"(?![-\s]?intercept)",
    re.IGNORECASE,
)

_DNE_CONVENTION = re.compile(
    r"\b(?:if|when)\b[^.?!]{0,120}"
    r"(?:slope[^.?!]{0,50}(?:does\s+not\s+exist|is\s+undefined)|vertical\s+line)"
    r"[^.?!]{0,80}\bDNE\b",
    re.IGNORECASE,
)


def solve_point_slope(
    instruction: str,
    points: list[tuple[str, str]],
    answer_parts: int,
) -> tuple[str | None, str]:
    """Compute rise over run, or claim and name why it cannot be answered."""
    if POINT_SLOPE_REQUEST.search(instruction) is None:
        return None, "the instruction does not ask for the slope from graph points"
    if len(points) != 2:
        return None, (
            "a graph-slope question needs exactly two identified points and "
            f"{len(points)} were supplied"
        )
    if answer_parts != 1:
        return None, (
            "a slope is one scalar answer and the answer contract requires "
            f"{answer_parts}"
        )

    try:
        (x1, y1), (x2, y2) = [(Fraction(x), Fraction(y)) for x, y in points]
    except (ValueError, ZeroDivisionError) as error:
        return None, f"the graph points are not exact rational coordinates: {error}"

    run = x2 - x1
    rise = y2 - y1
    if run == 0:
        if _DNE_CONVENTION.search(instruction):
            return "DNE", ""
        return None, (
            "the graph is vertical but the instruction does not state the required "
            "undefined-slope notation"
        )
    return str(rise / run), ""
