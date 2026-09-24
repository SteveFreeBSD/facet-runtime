"""Exact coordinate substitution and bounded choices on an affine line."""

import re
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, floor

import sympy

from facet_runtime.exact.table import _relation

METHOD = "Exact linear coordinate substitution"
REQUEST = re.compile(
    r"\b(?:determine|find|calculate|compute|choose|select)\s+(?:the\s+|a\s+|any\s+)?"
    r"(?:value\s+(?:for|of)\s+|[xy][- ]coordinate\b|coordinate\s+(?:for|of)\s+)[xy]?",
    re.IGNORECASE,
)
TARGET = re.compile(
    r"\b(?:value|coordinate)\s+(?:for|of)\s+([xy])\b|\b([xy])[- ]coordinate\b",
    re.IGNORECASE,
)
GIVEN = re.compile(
    r"\b(?:given|when|if)\s+([xy])\s*=\s*(-?\s*(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?))(?![\d./])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CoordinateTask:
    axis: str
    given: str | None
    bounds: tuple[Fraction, ...]
    steps: tuple[Fraction, ...]
    allow_rational: bool


def parse_task(value: dict) -> CoordinateTask:
    if not isinstance(value, dict) or set(value) != {
        "axis",
        "given",
        "bounds",
        "steps",
        "allow_rational",
    }:
        raise ValueError(
            "coordinate task requires axis, given, bounds, steps and rational permission"
        )
    if value["axis"] not in ("x", "y") or type(value["allow_rational"]) is not bool:
        raise ValueError("invalid coordinate axis or rational permission")

    def number(raw):
        if not isinstance(raw, str) or len(raw) > 64:
            raise ValueError("coordinates must be bounded exact strings")
        return Fraction(raw)

    if not isinstance(value["bounds"], list) or not isinstance(value["steps"], list):
        raise TypeError("coordinate bounds and steps must be lists")
    bounds = tuple(number(v) for v in value["bounds"])
    steps = tuple(number(v) for v in value["steps"])
    if len(bounds) != 4 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("coordinate bounds must be ordered")
    if len(steps) != 2 or min(steps) <= 0:
        raise ValueError("coordinate steps must be positive")
    given = value["given"]
    if given is not None:
        given = str(number(given))
    return CoordinateTask(value["axis"], given, bounds, steps, value["allow_rational"])


def named_task(instruction: str) -> tuple[str, str] | None:
    targets = {a or b for a, b in TARGET.findall(instruction.lower())}
    givens = {
        (axis.lower(), str(Fraction(value.replace(" ", ""))))
        for axis, value in GIVEN.findall(instruction)
    }
    if len(targets) != 1 or len(givens) != 1:
        return None
    axis = targets.pop()
    given_axis, given = givens.pop()
    return (axis, given) if axis != given_axis else None


def solve_coordinate(
    expressions: list[str],
    axis: str,
    given: str | None,
    task: CoordinateTask | None = None,
) -> tuple[str, tuple[str, str]]:
    """Return the requested scalar and its proved point, never a guessed point."""
    if axis not in ("x", "y"):
        raise ValueError("a coordinate request must name one axis")
    residual = sympy.expand(_relation(expressions, {"x", "y"}))
    x, y = sympy.symbols("x y", real=True)
    try:
        poly = sympy.Poly(residual, x, y)
    except sympy.PolynomialError as error:
        raise ValueError(
            "coordinate substitution requires a linear polynomial"
        ) from error
    if poly.total_degree() != 1 or not all(c.is_Rational for c in poly.coeffs()):
        raise ValueError(
            "coordinate substitution requires one rational linear equation"
        )
    a, b, c = (Fraction(str(poly.coeff_monomial(term))) for term in (x, y, 1))
    index = 0 if axis == "x" else 1
    coefficients = (a, b)

    def accepted(point):
        if a * point[0] + b * point[1] + c != 0:
            return False
        if task is None:
            return True
        xmin, xmax, ymin, ymax = task.bounds
        return (
            xmin <= point[0] <= xmax
            and ymin <= point[1] <= ymax
            and all((v / step).denominator == 1 for v, step in zip(point, task.steps))
            and (task.allow_rational or all(v.denominator == 1 for v in point))
        )

    if given is not None:
        if coefficients[index] == 0:
            raise ValueError(
                "the given coordinate does not determine the other coordinate uniquely"
            )
        point = [Fraction(0), Fraction(0)]
        point[1 - index] = Fraction(given)
        point[index] = (
            -(coefficients[1 - index] * point[1 - index] + c) / coefficients[index]
        )
        if not accepted(point):
            raise ValueError(
                "the supplied coordinate produces a point outside the permitted bounds or grid"
            )
    else:
        if task is None:
            raise ValueError("choosing a coordinate requires explicit bounds and grid")
        candidates = set()
        # Search both axes: this also finds vertical/horizontal lines and
        # integer points when the other axis's step is fractional.
        for free in (0, 1):
            dependent = 1 - free
            if coefficients[dependent] == 0:
                continue
            lower, upper = task.bounds[2 * free : 2 * free + 2]
            step = task.steps[free]
            start, end = ceil(lower / step), floor(upper / step)
            if end - start > 20000:
                raise ValueError("coordinate grid exceeds the bounded exact search")
            for n in range(start, end + 1):
                point = [Fraction(0), Fraction(0)]
                point[free] = n * step
                point[dependent] = (
                    -(coefficients[free] * point[free] + c) / coefficients[dependent]
                )
                if accepted(point):
                    candidates.add(tuple(point))
        if not candidates:
            raise ValueError(
                "no valid visible point is representable on the permitted grid"
            )
        point = min(
            candidates,
            key=lambda p: (
                any(v.denominator != 1 for v in p),
                sum(abs(v) for v in p),
                abs(p[index]),
                p[index] < 0,
                p[0],
                p[1],
            ),
        )
    return str(point[index]), tuple(str(v) for v in point)
