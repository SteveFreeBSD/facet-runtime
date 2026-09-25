"""The two graph specialists Facet routes to, and their strict reply parsing.

Most questions want a value. Two families want a *plan*: a vertical parabola
that a consumer will draw, and the coefficients of a quadratic regression over
points a consumer measured. Neither has a deterministic route here -- the exact
solvers answer expressions, not geometry -- so both are reasoned, and what
comes back is a structured proposal rather than an answer.

The distinction matters and is the whole reason these are separate operations.
Facet proposes; it does not prove. A plan leaves here having been parsed
strictly -- exact schema, no extra or duplicate keys, exact integer or rational
coordinates and never a decimal approximation -- and that is a check on the
*model*, not a warrant. Whoever owns the surface the plan will be drawn on
proves it against its own mathematics before anything is drawn, and is the
authority that matters. Two independent checks are the point of the split.

Nothing here knows what a page is, and that now includes the wording. A
specialist receives an instruction, the exact expressions it concerns, and
normalised geometry -- bounds, a snap grid, a family -- or normalised point
coordinates. There is no element, no handle, no picture, and no action.

The prompts name no consumer either. A model told which product is asking, or
what will be done with its reply afterwards, has been told something it cannot
act on and may reason about instead -- and that is not free. Telling this
specialist that its coefficients would be "rounded for display" while also
requiring exact ones set two of its instructions against each other, and it
spent its entire output budget weighing them and answered nothing at all.
Where an instruction and the schema really do disagree, the prompt now settles
it outright rather than leaving the model to.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import sympy

from facet_runtime.exact.intercepts import AffineLine, Coordinate, affine_line
from facet_runtime.exact.table import _safe_sympy_expression

#: The result kinds a consumer may ask for beyond an ordinary value.
PARABOLA_PLAN = "parabola_plan"
QUADRATIC_REGRESSION = "quadratic_regression"
POINT_PLOT_PLAN = "point_plot_plan"
LINEAR_GRAPH_PLAN = "linear_graph_plan"
LINEAR_INEQUALITY_GRAPH_PLAN = "linear_inequality_graph_plan"
LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN = "linear_inequality_system_graph_plan"

#: How many points a plotting question may ask for. Hawkes draws one draggable
#: control per point, and a question asking for none or for dozens is not the
#: one this reads.
MIN_PLOT_POINTS = 1
MAX_PLOT_POINTS = 12

#: An exact integer or rational, written as a string. A decimal is refused:
#: `0.333` is not a third, and a plan proved against exact mathematics cannot
#: be built out of values that were already rounded.
RATIONAL = re.compile(r"-?(?:0|[1-9][0-9]*)(?:/[1-9][0-9]*)?\Z")

#: The one graph family and control layout these specialists support. Growing
#: either is a new specialist, not a looser check on this one.
GRAPH_FAMILY = "parabola"
GRAPH_ORIENTATION = "vertical"
GRAPH_CONTROLS = "vertex-and-symmetric-points"
LINE_GRAPH_FAMILY = "line"
LINE_GRAPH_ORIENTATION = "cartesian"
LINE_GRAPH_CONTROLS = "two-points"
INEQUALITY_GRAPH_FAMILY = "linear-inequality"
INEQUALITY_GRAPH_CONTROLS = "boundary-two-points-regions"
INEQUALITY_SYSTEM_GRAPH_FAMILY = "linear-inequality-system"
INEQUALITY_SYSTEM_GRAPH_CONTROLS = "mounted-boundaries-combined-regions"

MIN_REGRESSION_POINTS = 3
MAX_REGRESSION_POINTS = 32


class PlanRefused(ValueError):
    """A specialist's input or its reply did not match the contract."""


@dataclass(frozen=True, slots=True)
class GraphContext:
    """Normalised geometry: what the grid is, never what the page is."""

    family: str
    orientation: str | None
    bounds: tuple[float, ...]
    snap: tuple[float, ...]
    controls: str
    connector: str | None = None

    def as_json(self) -> dict[str, Any]:
        result = {
            "family": self.family,
            "orientation": self.orientation,
            "bounds": list(self.bounds),
            "snap": list(self.snap),
            "controls": self.controls,
        }
        if self.connector is not None:
            result["connector"] = self.connector
        return result


@dataclass(frozen=True, slots=True)
class Point:
    """One measured coordinate, exact and already normalised."""

    x: str
    y: str

    def as_json(self) -> dict[str, str]:
        return {"x": self.x, "y": self.y}


def _number(value: Any, where: str) -> float:
    # `True` is an int in Python, so the type is checked before the value.
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise PlanRefused(f"{where} must be a number")
    if not math.isfinite(value):
        raise PlanRefused(f"{where} must be finite")
    return float(value)


def _rational(value: Any, where: str) -> str:
    if not isinstance(value, str) or not RATIONAL.fullmatch(value):
        raise PlanRefused(f"{where} must be an exact integer or rational string")
    return value


def _exact_keys(payload: Any, expected: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlanRefused(f"{where} must be an object")
    if set(payload) != set(expected):
        raise PlanRefused(f"{where} must carry exactly {', '.join(expected)}")
    return payload


def parse_graph_context(payload: Any) -> GraphContext:
    """Read the normalized geometry a function plan is drawn on."""
    if (
        isinstance(payload, dict)
        and payload.get("family") == INEQUALITY_SYSTEM_GRAPH_FAMILY
    ):
        fields = _exact_keys(payload, ("family", "controls", "connector"), "graph")
        if fields["controls"] != INEQUALITY_SYSTEM_GRAPH_CONTROLS:
            raise PlanRefused(
                "a linear inequality system needs its mounted-region controls"
            )
        if fields["connector"] not in {"and", "or"}:
            raise PlanRefused("a linear inequality system needs an and/or connector")
        return GraphContext(
            family=fields["family"],
            orientation=None,
            bounds=(),
            snap=(),
            controls=fields["controls"],
            connector=fields["connector"],
        )
    fields = _exact_keys(
        payload, ("family", "orientation", "bounds", "snap", "controls"), "graph"
    )
    supported = {
        (GRAPH_FAMILY, GRAPH_ORIENTATION, GRAPH_CONTROLS),
        (LINE_GRAPH_FAMILY, LINE_GRAPH_ORIENTATION, LINE_GRAPH_CONTROLS),
        (INEQUALITY_GRAPH_FAMILY, LINE_GRAPH_ORIENTATION, INEQUALITY_GRAPH_CONTROLS),
    }
    identity = (fields["family"], fields["orientation"], fields["controls"])
    if identity not in supported:
        raise PlanRefused("the graph family, orientation and controls do not agree")
    bounds, snap = fields["bounds"], fields["snap"]
    if not isinstance(bounds, list) or len(bounds) != 4:
        raise PlanRefused("graph bounds must be four numbers")
    if not isinstance(snap, list) or len(snap) != 2:
        raise PlanRefused("graph snap must be two numbers")
    edges = tuple(_number(value, "a graph bound") for value in bounds)
    spacing = tuple(_number(value, "a graph snap") for value in snap)
    if edges[0] >= edges[1] or edges[2] >= edges[3]:
        raise PlanRefused("graph bounds must be ordered")
    if min(spacing) <= 0:
        raise PlanRefused("graph snap must be positive")
    return GraphContext(
        family=fields["family"],
        orientation=fields["orientation"],
        bounds=edges,  # type: ignore[arg-type]
        snap=spacing,  # type: ignore[arg-type]
        controls=fields["controls"],
    )


def parse_points(
    payload: Any, *, minimum: int = MIN_REGRESSION_POINTS
) -> tuple[Point, ...]:
    """Read exact normalised coordinates, with the caller's required count."""
    if not isinstance(payload, list):
        raise PlanRefused("points must be a list")
    if not minimum <= len(payload) <= MAX_REGRESSION_POINTS:
        raise PlanRefused(
            f"this question takes {minimum} to {MAX_REGRESSION_POINTS} points"
        )
    read = []
    for item in payload:
        fields = _exact_keys(item, ("x", "y"), "a point")
        read.append(
            Point(
                _rational(fields["x"], "a point x"), _rational(fields["y"], "a point y")
            )
        )
    return tuple(read)


def parabola_prompt(
    instruction: str, expressions: tuple[str, ...], context: GraphContext
) -> str:
    """Ask for a parabola plan, in terms of the geometry and nothing else."""
    return (
        "Produce a graph plan for the exact function. Return ONLY one JSON object, "
        "no markdown, prose, code, or extra keys. Coordinates must be exact integer "
        'or rational STRINGS (for example "-3/2"). Required schema: '
        '{"kind":"parabola","orientation":"vertical","opening":"up or down",'
        '"vertex":{"x":"rational","y":"rational"},'
        '"points":[{"x":"rational","y":"rational"},'
        '{"x":"rational","y":"rational"}]}. '
        "Derive the vertex, opening and two symmetric defining points from the function. "
        "First point must be right of vertex, second left. Prefer one unit horizontal "
        "offset if it fits the bounds and snap grid. You have no actions available.\n"
        + json.dumps(
            {
                "instruction": instruction,
                "exact_expression": list(expressions),
                "graph_answer": context.as_json(),
            }
        )
    )


def regression_prompt(instruction: str, points: tuple[Point, ...]) -> str:
    """Ask for exact regression coefficients over the points exactly as given.

    A question of this shape usually carries a rounding instruction of its own,
    which contradicts the exact coefficients this schema requires. The prompt
    settles that here, in one clause, because a model left to settle it spends
    the budget doing so.
    """
    return (
        "Find the quadratic least-squares regression y=a*x^2+b*x+c for these exact "
        "point coordinates. Return ONLY JSON with exactly this schema: "
        '{"kind":"quadratic-regression","coefficients":["a","b","c"]}. '
        "Coefficients must be exact integer or rational strings, NOT decimal "
        "approximations: give the exact coefficients even where the instruction "
        "asks for rounded ones. No prose, markdown, or extra keys.\n"
        + json.dumps(
            {
                "instruction": instruction,
                "points": [point.as_json() for point in points],
            }
        )
    )


def strict_json(text: str) -> Any:
    """Parse one JSON object and refuse a repeated key.

    A duplicate key is not a formatting quirk. `json.loads` keeps the last one
    silently, so a reply naming a vertex twice would be read as whichever came
    last, and nothing downstream could tell that a choice had been made.
    """

    def unique(pairs):
        seen: dict[str, Any] = {}
        for key, value in pairs:
            if key in seen:
                raise PlanRefused(f"the reply names {key} more than once")
            seen[key] = value
        return seen

    try:
        return json.loads(text, object_pairs_hook=unique)
    except json.JSONDecodeError as error:
        raise PlanRefused(f"the reply was not one JSON object: {error}") from error


def parse_parabola_plan(text: str) -> dict[str, Any]:
    """Read a proposed parabola plan out of a reply, or refuse the reply."""
    fields = _exact_keys(
        strict_json(text),
        ("kind", "orientation", "opening", "vertex", "points"),
        "a parabola plan",
    )
    if fields["kind"] != GRAPH_FAMILY or fields["orientation"] != GRAPH_ORIENTATION:
        raise PlanRefused(f"a plan must be a {GRAPH_ORIENTATION} {GRAPH_FAMILY}")
    if fields["opening"] not in ("up", "down"):
        raise PlanRefused("a parabola opens up or down")
    points = fields["points"]
    if not isinstance(points, list) or len(points) != 2:
        raise PlanRefused("a parabola plan takes exactly two defining points")
    return {
        "kind": fields["kind"],
        "orientation": fields["orientation"],
        "opening": fields["opening"],
        "vertex": _plan_point(fields["vertex"], "the vertex"),
        "points": [_plan_point(point, "a defining point") for point in points],
    }


def _plan_point(payload: Any, where: str) -> dict[str, str]:
    fields = _exact_keys(payload, ("x", "y"), where)
    return {
        "x": _rational(fields["x"], f"{where} x"),
        "y": _rational(fields["y"], f"{where} y"),
    }


def parse_regression_plan(text: str) -> dict[str, Any]:
    """Read proposed regression coefficients out of a reply, or refuse it."""
    fields = _exact_keys(
        strict_json(text), ("kind", "coefficients"), "a regression plan"
    )
    if fields["kind"] != "quadratic-regression":
        raise PlanRefused("a regression plan must be a quadratic regression")
    coefficients = fields["coefficients"]
    if not isinstance(coefficients, list) or len(coefficients) != 3:
        raise PlanRefused("a quadratic regression has exactly three coefficients")
    return {
        "kind": fields["kind"],
        "coefficients": [_rational(value, "a coefficient") for value in coefficients],
    }


#: A literal ordered pair as a page writes one: integers or exact rationals,
#: never a decimal, and never an expression to be evaluated.
_PLOT_VALUE = r"(?:[+-]?\s*\d+\s*/\s*[1-9]\d*|[+-]?\s*\d+)"
_PLOT_PAIR = re.compile(rf"\(\s*(?P<x>{_PLOT_VALUE})\s*,\s*(?P<y>{_PLOT_VALUE})\s*\)")

#: The question this reads. "Plot the following points" states its own answer
#: -- the pairs are written down -- so there is nothing here for a model to
#: work out, and asking one would be inventing uncertainty.
PLOT_REQUEST = re.compile(
    r"\b(?:plot|graph|place|draw)\b[^.?!]*?\bpoints?\b",
    re.IGNORECASE,
)


def _plot_rational(text: str) -> str:
    """One coordinate, normalised to the exact form a plan carries."""
    cleaned = re.sub(r"\s+", "", text).replace("\u2212", "-")
    if "/" in cleaned:
        top, bottom = cleaned.split("/", 1)
        sign = "-" if top.startswith("-") else ""
        top = top.lstrip("+-")
        if int(top) == 0:
            return "0"
        return f"{sign}{int(top)}/{int(bottom)}"
    value = int(cleaned)
    return str(value)


def read_plot_points(instruction: str, expressions: list[str]) -> list[dict[str, str]]:
    """Every literal ordered pair this question writes down, in order.

    Deterministic and exact. The pairs are the question's own words; nothing is
    evaluated, rounded, reordered or deduplicated -- a question may legitimately
    ask for the same point twice, and the set it asks for is the answer.
    """
    found: list[dict[str, str]] = []
    for fragment in [instruction, *expressions]:
        for match in _PLOT_PAIR.finditer(fragment or ""):
            found.append(
                {
                    "x": _plot_rational(match.group("x")),
                    "y": _plot_rational(match.group("y")),
                }
            )
    return found


def build_point_plot_plan(
    instruction: str, expressions: list[str], context: GraphContext | None = None
) -> dict[str, Any]:
    """The plan for a "plot these points" question, or a refusal.

    A plan and not an answer: it says where the page's own controls must end
    up, and the browser proves every one of them against the live graph before
    a key is pressed.
    """
    if INTEGER_LINE_POINTS_REQUEST.search(instruction or ""):
        return build_integer_line_point_plan(instruction, expressions, context)
    if not PLOT_REQUEST.search(instruction or ""):
        raise PlanRefused("this is not a request to plot stated points")
    points = read_plot_points(instruction, expressions)
    if not MIN_PLOT_POINTS <= len(points) <= MAX_PLOT_POINTS:
        raise PlanRefused(
            f"a plotting plan needs between {MIN_PLOT_POINTS} and "
            f"{MAX_PLOT_POINTS} stated points, and {len(points)} were read"
        )
    return {"kind": "points", "points": points}


LINEAR_GRAPH_REQUEST = re.compile(
    r"\bgraph\b[^.?!]*\bequation\b[^.?!]*\b(?:x|y)[-\s]?intercepts?\b",
    re.IGNORECASE,
)

# A distinct point-plan operation: the pairs are chosen, not quoted, and are
# integer solutions of one equation, not necessarily its axis intercepts.
INTEGER_LINE_POINTS_REQUEST = re.compile(
    r"\b(?:graph|plot)\b[^.?!]*\b(?:any\s+)?(?:two|2)\s+"
    r"(?:ordered\s+pairs|points)\b[^.?!]*\binteger(?:[-\s]+valued?)?\s+coordinates\b"
    r"[^.?!]*\b(?:satisfy|satisfying)\b[^.?!]*\bequation\b",
    re.IGNORECASE,
)


def build_integer_line_point_plan(
    instruction: str, expressions: list[str], context: GraphContext | None
) -> dict[str, Any]:
    """Choose two small integer solutions in the exact visible snap lattice.

    The existing affine parser yields integer a,b,c. Intersect the full
    Diophantine solution parameter with both coordinate bounds, then minimize
    |x|+|y| at its breakpoints. No grid-size-dependent scan or approximation.
    """
    if not INTEGER_LINE_POINTS_REQUEST.search(instruction or ""):
        raise PlanRefused("this is not an integer line-point request")
    if context is None or (context.family, context.orientation, context.controls) != (
        LINE_GRAPH_FAMILY,
        LINE_GRAPH_ORIENTATION,
        LINE_GRAPH_CONTROLS,
    ):
        raise PlanRefused("integer line points require authoritative graph geometry")
    if re.search(r"\bintercepts?\b", instruction, re.IGNORECASE) or read_plot_points(
        instruction, expressions
    ):
        raise PlanRefused(
            "integer line-point selection conflicts with stated points or intercepts"
        )
    try:
        from facet_runtime.exact.intercepts import affine_coefficients

        coefficients = affine_coefficients(expressions)
    except ValueError as error:
        raise PlanRefused(str(error)) from error
    xmin, xmax, ymin, ymax = map(_fraction, context.bounds)
    sx, sy = map(_fraction, context.snap)
    # An integer coordinate on rational spacing p/q must be a multiple of p.
    px, py = sx.numerator, sy.numerator
    a, b, c = map(int, coefficients)
    aa, bb = a * px, b * py
    divisor = math.gcd(aa, bb)
    if not divisor or (-c) % divisor:
        raise PlanRefused("the equation has no integer solutions on this graph grid")
    u, v, _ = sympy.gcdex(aa, bb)
    x0, y0 = int(u) * (-c // divisor) * px, int(v) * (-c // divisor) * py
    dx, dy = (bb // divisor) * px, -(aa // divisor) * py
    lower, upper = None, None
    for base, step, minimum, maximum in ((x0, dx, xmin, xmax), (y0, dy, ymin, ymax)):
        if not step:
            if not minimum <= base <= maximum:
                raise PlanRefused("the line misses the visible graph range")
            continue
        limits = sorted(((minimum - base) / step, (maximum - base) / step))
        lo, hi = math.ceil(limits[0]), math.floor(limits[1])
        lower = lo if lower is None else max(lower, lo)
        upper = hi if upper is None else min(upper, hi)
    if lower is None or upper is None or upper - lower < 1:
        raise PlanRefused("fewer than two distinct in-bounds integer points exist")
    candidates = {lower, lower + 1, upper - 1, upper}
    # Axis roots minimize total size; x=+/-y also breaks ties by the largest
    # coordinate when the total is constant across an interval.
    for base, step in ((x0, dx), (y0, dy), (x0 - y0, dx - dy), (x0 + y0, dx + dy)):
        if step:
            root = Fraction(-base, step)
            for nearest in (math.floor(root), math.ceil(root)):
                candidates.update(
                    max(lower, min(upper, nearest + offset)) for offset in (-1, 0, 1)
                )
    points = sorted(
        {(x0 + dx * t, y0 + dy * t) for t in candidates},
        key=lambda p: (
            abs(p[0]) + abs(p[1]),
            max(abs(p[0]), abs(p[1])),
            abs(p[0]),
            p[0] < 0,
            p[1] < 0,
        ),
    )[:2]
    if len(points) != 2 or any(
        not (xmin <= x <= xmax and ymin <= y <= ymax)
        or a * x + b * y + c != 0
        or (Fraction(x) / sx).denominator != 1
        or (Fraction(y) / sy).denominator != 1
        for x, y in points
    ):
        raise PlanRefused("integer line-point verification failed")
    return {"kind": "points", "points": [{"x": str(x), "y": str(y)} for x, y in points]}


def _fraction(value: str | float) -> Fraction:
    return Fraction(str(value))


def _written(value: Fraction) -> str:
    return (
        str(value.numerator)
        if value.denominator == 1
        else f"{value.numerator}/{value.denominator}"
    )


def _point(coordinate: Coordinate, role: str) -> dict[str, str]:
    return {"x": coordinate.x, "y": coordinate.y, "role": role}


def _simple_lattice(limit: int) -> list[int]:
    values = [0]
    for value in range(1, limit + 1):
        values.extend((value, -value))
    return values


def _substitute_point(
    line: AffineLine,
    context: GraphContext,
    used: set[tuple[Fraction, Fraction]],
) -> dict[str, str]:
    """Choose the simplest distinct exact lattice point on a line."""
    xmin, xmax, ymin, ymax = map(_fraction, context.bounds)
    sx, sy = map(_fraction, context.snap)
    a, b, c = map(Fraction, (line.a, line.b, line.c))
    limit = 400

    def acceptable(x: Fraction, y: Fraction) -> bool:
        return (
            xmin <= x <= xmax
            and ymin <= y <= ymax
            and x / sx == int(x / sx)
            and y / sy == int(y / sy)
            and a * x + b * y + c == 0
            and (x, y) not in used
        )

    # Horizontal lines naturally use x=1 before x=-1; vertical lines use y=1.
    # The same ordering extends to every slope and is deterministic.
    if b:
        for step in _simple_lattice(limit):
            x = sx * step
            y = -(a * x + c) / b
            if acceptable(x, y):
                return {"x": _written(x), "y": _written(y), "role": "substitute"}
    if a:
        for step in _simple_lattice(limit):
            y = sy * step
            x = -(b * y + c) / a
            if acceptable(x, y):
                return {"x": _written(x), "y": _written(y), "role": "substitute"}
    raise PlanRefused("the graph offers no second exact lattice point on the line")


def build_linear_graph_plan(
    instruction: str, expressions: list[str], context: GraphContext
) -> dict[str, Any]:
    """Derive two exact defining points for an intercept-directed line graph."""
    if LINEAR_GRAPH_REQUEST.search(instruction or "") is None:
        raise PlanRefused("this is not an intercept-directed linear graph request")
    if (
        context.family != LINE_GRAPH_FAMILY
        or context.orientation != LINE_GRAPH_ORIENTATION
        or context.controls != LINE_GRAPH_CONTROLS
    ):
        raise PlanRefused("a linear graph needs a Cartesian two-point surface")
    try:
        line = affine_line(expressions)
    except ValueError as error:
        raise PlanRefused(str(error)) from error

    xmin, xmax, ymin, ymax = map(_fraction, context.bounds)
    sx, sy = map(_fraction, context.snap)

    def on_grid(coordinate: Coordinate) -> bool:
        x, y = Fraction(coordinate.x), Fraction(coordinate.y)
        return (
            xmin <= x <= xmax
            and ymin <= y <= ymax
            and x / sx == int(x / sx)
            and y / sy == int(y / sy)
        )

    points: list[dict[str, str]] = []
    used: set[tuple[Fraction, Fraction]] = set()
    for role, coordinate in (
        ("x-intercept", line.intercepts.x),
        ("y-intercept", line.intercepts.y),
    ):
        if coordinate is None:
            continue
        exact = (Fraction(coordinate.x), Fraction(coordinate.y))
        if exact in used:
            continue
        if not on_grid(coordinate):
            raise PlanRefused(f"the {role} is not representable on the graph grid")
        points.append(_point(coordinate, role))
        used.add(exact)
    while len(points) < 2:
        substitute = _substitute_point(line, context, used)
        exact = (Fraction(substitute["x"]), Fraction(substitute["y"]))
        points.append(substitute)
        used.add(exact)
    return {
        "kind": "line",
        "coefficients": {"x": line.a, "y": line.b, "constant": line.c},
        "points": points,
    }


_INEQUALITY = re.compile(r"(?P<relation><=|>=|≤|≥|<|>|\\leq?|\\geq?)")


def _linear_inequality_semantics(expression: str) -> tuple[list[int], str]:
    """One exact normalized affine boundary and its oriented relation."""
    match = _INEQUALITY.search(expression)
    if match is None or _INEQUALITY.search(expression, match.end()) is not None:
        raise PlanRefused("the expression is not one inequality")
    left_text, right_text = expression[: match.start()], expression[match.end() :]
    if not left_text.strip() or not right_text.strip():
        raise PlanRefused("the inequality has a missing side")
    try:
        residual = sympy.expand(
            _safe_sympy_expression(left_text) - _safe_sympy_expression(right_text)
        )
        x, y = sympy.symbols("x y", real=True)
        polynomial = sympy.Poly(residual, x, y)
    except (
        SyntaxError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        sympy.PolynomialError,
    ) as error:
        raise PlanRefused(
            "the inequality could not be read as an exact line"
        ) from error
    if (
        polynomial.total_degree() > 1
        or not residual.free_symbols
        or not residual.free_symbols <= {x, y}
        or not all(value.is_rational for value in polynomial.coeffs())
    ):
        raise PlanRefused("the inequality boundary is not a rational affine line")
    values = [
        sympy.Rational(polynomial.coeff_monomial(term))
        for term in (x, y, sympy.Integer(1))
    ]
    denominator = sympy.ilcm(*(value.q for value in values))
    integers = [int(value * denominator) for value in values]
    divisor = abs(sympy.igcd(*integers)) or 1
    integers = [value // divisor for value in integers]
    relation = {
        "≤": "<=",
        "≥": ">=",
        r"\le": "<=",
        r"\leq": "<=",
        r"\ge": ">=",
        r"\geq": ">=",
    }.get(match.group("relation"), match.group("relation"))
    first = next(value for value in integers if value)
    if first < 0:
        integers = [-value for value in integers]
        relation = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[relation]
    return integers, relation


def build_linear_inequality_graph_plan(
    instruction: str, expressions: list[str], context: GraphContext
) -> dict[str, Any]:
    """Derive the exact boundary and half-plane relation for one inequality."""
    if not re.search(
        r"\bgraph\b[^.?!]*\b(?:linear\s+)?inequalit", instruction, re.IGNORECASE
    ):
        raise PlanRefused("this is not a request to graph a linear inequality")
    if (
        context.family != INEQUALITY_GRAPH_FAMILY
        or context.orientation != LINE_GRAPH_ORIENTATION
        or context.controls != INEQUALITY_GRAPH_CONTROLS
    ):
        raise PlanRefused("a linear inequality needs its Cartesian composite surface")
    written = [item.strip() for item in expressions if item.strip()]
    inequalities = [item for item in written if _INEQUALITY.search(item)]
    if not inequalities:
        raise PlanRefused("a linear inequality graph requires one stated inequality")
    if len(inequalities) > 1:
        plans = [
            build_linear_inequality_graph_plan(instruction, [item], context)
            for item in inequalities
        ]
        if any(plan != plans[0] for plan in plans[1:]):
            raise PlanRefused(
                "a linear inequality graph requires one stated inequality"
            )
        return plans[0]
    expression = inequalities[0]
    integers, relation = _linear_inequality_semantics(expression)
    xmin, xmax, ymin, ymax = map(_fraction, context.bounds)
    a, b, c = map(Fraction, integers)

    def inside(point: tuple[Fraction, Fraction]) -> bool:
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    candidates: list[tuple[Fraction, Fraction]] = []
    if a:
        candidates.append((-c / a, Fraction(0)))
    if b:
        candidates.append((Fraction(0), -c / b))
    for step in _simple_lattice(400):
        if b:
            candidates.append((Fraction(step), -(a * step + c) / b))
        if a:
            candidates.append((-(b * step + c) / a, Fraction(step)))
    chosen: list[tuple[Fraction, Fraction]] = []
    for point in candidates:
        if inside(point) and point not in chosen:
            chosen.append(point)
        if len(chosen) == 2:
            break
    if len(chosen) != 2:
        raise PlanRefused(
            "the graph bounds contain fewer than two exact boundary points"
        )
    return {
        "kind": "linear-inequality",
        "coefficients": {
            "x": str(integers[0]),
            "y": str(integers[1]),
            "constant": str(integers[2]),
        },
        "relation": relation,
        "boundary": "dashed" if relation in {"<", ">"} else "solid",
        "points": [{"x": _written(px), "y": _written(py)} for px, py in chosen],
    }


def build_linear_inequality_system_graph_plan(
    instruction: str, expressions: list[str], context: GraphContext
) -> dict[str, Any]:
    """Preserve a two-inequality connector and derive both exact boundaries."""
    if not re.search(
        r"\bsystem\b[^.?!]*\blinear\s+inequalit", instruction, re.IGNORECASE
    ):
        raise PlanRefused("this is not a linear inequality system")
    if (
        context.family != INEQUALITY_SYSTEM_GRAPH_FAMILY
        or context.orientation is not None
        or context.controls != INEQUALITY_SYSTEM_GRAPH_CONTROLS
        or context.bounds
        or context.snap
        or context.connector not in {"and", "or"}
    ):
        raise PlanRefused("a linear inequality system needs its mounted-region surface")
    written = [item.strip() for item in expressions if item.strip()]
    if len(written) != 2 or any(_INEQUALITY.search(item) is None for item in written):
        raise PlanRefused("a linear inequality system requires two stated inequalities")
    members = []
    for item in written:
        coefficients, relation = _linear_inequality_semantics(item)
        members.append(
            {
                "coefficients": {
                    "x": str(coefficients[0]),
                    "y": str(coefficients[1]),
                    "constant": str(coefficients[2]),
                },
                "relation": relation,
                "boundary": "dashed" if relation in {"<", ">"} else "solid",
            }
        )
    connector = context.connector
    return {
        "kind": "linear-inequality-system",
        "connector": connector,
        "operation": "union" if connector == "or" else "intersection",
        "inequalities": members,
    }
