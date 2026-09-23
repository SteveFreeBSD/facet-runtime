"""Exact x- and y-intercepts of one stated affine equation.

An axis intercept is not a scalar.  It is either a point on the named axis or
it is absent, and both axes are requested together on the Hawkes surface that
motivated this solver.  Keeping those alternatives structured is what lets a
consumer select an ``absent`` control for one row and fill a coordinate pair
for the other without recovering meaning from display prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy

from facet_runtime.exact.table import TableRefused, _relation

INTERCEPT_METHOD = "SymPy exact axis intercepts"

INTERCEPT_REQUEST = re.compile(
    r"\b(?:find|determine|identify|give|state)\b[^.?!]*?"
    r"(?:"
    r"\bx(?:[-\s]?intercepts?)?\s*-?\s*(?:and|&)\s*"
    r"y[-\s]?intercepts?\b"
    r"|\bx[-\s]?intercepts?\b[^.?!]*?\b(?:and|&)\b[^.?!]*?"
    r"\by[-\s]?intercepts?\b"
    r"|\by(?:[-\s]?intercepts?)?\s*-?\s*(?:and|&)\s*"
    r"x[-\s]?intercepts?\b"
    r"|\by[-\s]?intercepts?\b[^.?!]*?\b(?:and|&)\b[^.?!]*?"
    r"\bx[-\s]?intercepts?\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Coordinate:
    x: str
    y: str

    @property
    def written(self) -> str:
        return f"({self.x},{self.y})"


@dataclass(frozen=True, slots=True)
class AxisIntercepts:
    x: Coordinate | None
    y: Coordinate | None

    @property
    def display(self) -> str:
        x = self.x.written if self.x is not None else "absent"
        y = self.y.written if self.y is not None else "absent"
        return f"x-intercept: {x}; y-intercept: {y}"


def _one_intercept(
    residual: sympy.Expr,
    *,
    variable: sympy.Symbol,
    other: sympy.Symbol,
) -> sympy.Expr | None:
    """Return the unique intercept coordinate, or ``None`` when absent.

    A residual that becomes identically zero on an axis describes the whole
    axis and therefore has no *unique* intercept.  That is refused by the
    caller instead of being collapsed into either a point or ``absent``.
    """
    restricted = sympy.expand(residual.subs(other, 0))
    if restricted == 0:
        raise ValueError("the equation coincides with an axis")
    solutions = sympy.solve(restricted, variable)
    real = [value for value in solutions if value.is_real is not False]
    if not real:
        return None
    if len(real) != 1:
        raise ValueError("an axis has more than one intercept")
    value = sympy.factor(real[0])
    if value.free_symbols or value.is_real is not True:
        raise ValueError("an intercept was not proved real and exact")
    if sympy.simplify(restricted.subs(variable, value)) != 0:
        raise ValueError("an intercept failed substitution verification")
    return value


def solve_axis_intercepts(
    instruction: str, expressions: list[str]
) -> tuple[AxisIntercepts | None, str]:
    """Compute both axis intercepts of one rational affine equation."""
    if INTERCEPT_REQUEST.search(instruction or "") is None:
        return None, ""
    relations = [item for item in expressions if "=" in item]
    if len(relations) != 1 or len([item for item in expressions if item.strip()]) != 1:
        return None, "axis intercepts require one stated equation"

    x, y = sympy.symbols("x y", real=True)
    try:
        residual = sympy.expand(_relation(relations, {"x", "y"}))
        polynomial = sympy.Poly(residual, x, y)
    except (TableRefused, sympy.PolynomialError, TypeError, ValueError) as error:
        return None, f"axis intercepts require a rational affine equation: {error}"
    if polynomial.total_degree() > 1 or not all(
        coefficient.is_rational for coefficient in polynomial.coeffs()
    ):
        return None, "axis intercepts require a rational affine equation"
    if residual == 0 or not residual.free_symbols:
        return None, "the stated equation does not determine a line"

    try:
        x_value = _one_intercept(residual, variable=x, other=y)
        y_value = _one_intercept(residual, variable=y, other=x)
    except ValueError as error:
        return None, str(error)
    result = AxisIntercepts(
        x=None if x_value is None else Coordinate(str(x_value), "0"),
        y=None if y_value is None else Coordinate("0", str(y_value)),
    )
    return result, ""
