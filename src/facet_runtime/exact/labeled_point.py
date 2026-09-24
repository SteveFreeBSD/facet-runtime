"""The exact coordinates of one labeled point read from a Cartesian graph.

The consumer owns graph reading.  By the time this solver is called it has
already associated the label named by the instruction with one page-owned
point and normalised that point to exact coordinates.  Facet owns the answer:
it keeps the ordered-pair family and the two coordinate components explicit,
so a consumer with two fields never has to split display text.
"""

from __future__ import annotations

import re

LABELED_POINT_METHOD = "Exact labeled Cartesian point"

LABELED_POINT_REQUEST = re.compile(
    r"\b(?:identify|find|determine|give|state|read|what\s+are)\b"
    r"[^.?!]{0,200}(?:"
    r"\bcoordinates?\b[^.?!]{0,100}\bpoint\b"
    r"|\bpoint\b[^.?!]{0,100}\bcoordinates?\b"
    r")",
    re.IGNORECASE,
)


def solve_labeled_point(
    instruction: str,
    points: list[tuple[str, str]],
    answer_parts: int,
) -> tuple[tuple[str, str] | None, str]:
    """Return the one structurally identified point, or a named refusal."""
    if LABELED_POINT_REQUEST.search(instruction) is None:
        return None, "the instruction does not ask for a labeled point's coordinates"
    if len(points) != 1:
        return None, (
            "a labeled-point coordinate question needs one identified point and "
            f"{len(points)} were supplied"
        )
    if answer_parts != 2:
        return None, (
            "a labeled point has two coordinate components and the answer contract "
            f"requires {answer_parts}"
        )
    return points[0], ""
