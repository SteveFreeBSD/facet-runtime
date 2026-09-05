"""Answering a question whose subject is measured points rather than an expression.

Three measurements and a quadratic through them is not an estimate. The normal
equations have one exact rational solution, its turning point is exact, and the
value there is exact -- so a question that fits a curve to data and then asks
where that curve is highest has a deterministic answer, computed here, with no
model and nothing rounded on the way.

The reason this is not merely `solve_vertex` with extra steps is that the
question is *about* the data. Nobody wrote the function down: it exists only as
the least-squares fit, and the instruction says what to do with it. So both the
fit and the reading of the instruction happen together, and everything that
cannot be read is declined rather than assumed.

Two shapes are answered. A question that asks only where the fitted curve turns
gets one value. A question that also asks for a *rate* -- a price per photo, a
cost per unit -- gets two, because the rate at the turning point is the optimal
total divided by the optimal input. That second reading is only taken when the
instruction names the same quantity the curve is a function of; a "per" whose
unit is something else is a different question, and a wrong reading of it would
be typed into a real answer box.
"""

from __future__ import annotations

import re
from typing import Literal

import sympy

#: The fit this module performs, named in the instruction. Nothing else is
#: attempted: a linear or exponential regression is a different computation.
QUADRATIC_REGRESSION = re.compile(r"\bquadratic\s+regression\b", re.IGNORECASE)

#: Which end of the fitted curve the question wants.
MAXIMISE = re.compile(r"\bmaximi[sz]e[sd]?\b|\bmaximum\b|\bgreatest\b", re.IGNORECASE)
MINIMISE = re.compile(r"\bminimi[sz]e[sd]?\b|\bminimum\b|\bleast\b", re.IGNORECASE)

#: What the fitted curve is a function of, in the question's own words:
#: "treating revenue as a function of the number of photos sold, ...". The
#: quantity named here is the one a rate may be quoted per.
FUNCTION_OF = re.compile(
    r"\bas\s+a\s+function\s+of\s+(?:the\s+)?(.{1,60}?)\s*(?:[,.;]|$)",
    re.IGNORECASE | re.DOTALL,
)

#: A rate the question asks for: "what price per photo". Only the unit is
#: taken; what the rate is called does not change how it is computed.
PER_UNIT = re.compile(r"\bper\s+([A-Za-z]+)\b", re.IGNORECASE)

Direction = Literal["maximum", "minimum"]

MIN_POINTS = 3


class NotThisQuestion(ValueError):
    """The instruction does not ask for what this module computes."""


def fit_quadratic(
    points: list[tuple[str, str]],
) -> tuple[sympy.Rational, sympy.Rational, sympy.Rational]:
    """The exact least-squares quadratic through these points.

    Solved as the normal equations over rationals, so three points give the
    interpolating parabola exactly and more than three give the true
    least-squares fit -- never a decimal approximation of either.

    Raises `NotThisQuestion` when the points do not determine one quadratic:
    fewer than three distinct inputs leaves the fit underdetermined, and an
    underdetermined fit has no single turning point to report.
    """
    if len(points) < MIN_POINTS:
        raise NotThisQuestion(
            f"a quadratic regression needs at least {MIN_POINTS} points"
        )
    try:
        xs = [sympy.Rational(x) for x, _ in points]
        ys = [sympy.Rational(y) for _, y in points]
    except (TypeError, ValueError, ZeroDivisionError) as error:
        raise NotThisQuestion(f"a point is not an exact number: {error}") from error
    design = sympy.Matrix([[x * x, x, 1] for x in xs])
    if design.rank() != 3:
        raise NotThisQuestion("the points do not determine one quadratic")
    solved = (design.T * design).solve(design.T * sympy.Matrix(ys))
    return tuple(sympy.Rational(value) for value in solved)  # type: ignore[return-value]


def _direction(instruction: str) -> Direction:
    """Which turning point the question asks for, or refuse to guess.

    A question naming both, or neither, is not answered. Choosing one of them
    for the reader would decide the sign of the answer on no evidence.
    """
    wants_maximum = bool(MAXIMISE.search(instruction))
    wants_minimum = bool(MINIMISE.search(instruction))
    if wants_maximum == wants_minimum:
        raise NotThisQuestion("the instruction does not ask for a maximum or a minimum")
    return "maximum" if wants_maximum else "minimum"


def _singular_forms(unit: str) -> tuple[str, ...]:
    """The spellings of a unit that should be recognised as the same word."""
    lowered = unit.lower()
    forms = {lowered, f"{lowered}s"}
    if lowered.endswith("s"):
        forms.add(lowered[:-1])
    if lowered.endswith("es"):
        forms.add(lowered[:-2])
    return tuple(sorted(forms))


def rate_unit(instruction: str) -> str | None:
    """The unit a second, per-unit answer is quoted in, or None.

    Returns a unit only when the instruction quotes a rate *per the same
    quantity the curve is a function of*: "revenue as a function of the number
    of photos sold ... what price per photo". That link is what makes the rate
    equal to the optimal total divided by the optimal input. A "per" naming
    anything else -- per event, per hour -- is a quantity this module was never
    given and must not invent.
    """
    unit = PER_UNIT.search(instruction)
    if unit is None:
        return None
    subject = FUNCTION_OF.search(instruction)
    if subject is None:
        return None
    described = subject.group(1).lower()
    return (
        unit.group(1)
        if any(form in described for form in _singular_forms(unit.group(1)))
        else None
    )


def _written(a, b, c) -> str:
    """The fitted curve as it would be written down, for a reader to check."""
    terms = []
    for value, power in ((a, "x^2"), (b, "x"), (c, "")):
        if value == 0:
            continue
        if not terms:
            sign = "-" if value < 0 else ""
        else:
            sign = " - " if value < 0 else " + "
        size = "" if power and abs(value) == 1 else str(abs(value))
        terms.append(f"{sign}{size}{power}")
    return "y = " + ("".join(terms) or "0")


def regression_optimum(
    instruction: str, points: list[tuple[str, str]], answer_parts: int
) -> tuple[list[str], dict[str, str]]:
    """Where the fitted curve turns, and what it is worth there.

    Returns the answer's separate values and the working behind them. The
    working crosses back as evidence so whoever asked can recompute the whole
    thing from the same points and compare, rather than take this on trust.
    """
    if not QUADRATIC_REGRESSION.search(instruction):
        raise NotThisQuestion("the instruction does not ask for a quadratic regression")
    direction = _direction(instruction)
    a, b, c = fit_quadratic(points)
    if a == 0:
        raise NotThisQuestion("the fitted curve is not a quadratic")
    # Fail closed on a curve that turns the wrong way. A parabola opening
    # upwards has no maximum at all, so its vertex is not the answer to a
    # question that asked for one -- it is the answer to the opposite question.
    if (direction == "maximum") != (a < 0):
        raise NotThisQuestion(f"the fitted curve has no {direction}")

    optimum = sympy.Rational(-b, 2 * a) if b else sympy.Integer(0)
    value = a * optimum**2 + b * optimum + c
    working = {
        "fit": _written(a, b, c),
        "coefficients": f"{a},{b},{c}",
        "direction": direction,
        "optimum_input": str(optimum),
        "optimum_value": str(value),
    }

    if answer_parts == 1:
        return [str(optimum)], working
    if answer_parts != 2:
        raise NotThisQuestion(
            f"a regression optimum is one or two values, not {answer_parts}"
        )

    unit = rate_unit(instruction)
    if unit is None:
        raise NotThisQuestion(
            "the second answer is not a rate per the quantity this curve is a "
            "function of"
        )
    if optimum == 0:
        raise NotThisQuestion("a rate at zero of the quantity is undefined")
    rate = sympy.Rational(value, optimum)
    working["rate_unit"] = unit
    working["rate"] = f"{value}/{optimum} = {rate}"
    return [str(optimum), str(rate)], working
