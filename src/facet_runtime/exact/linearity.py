"""Exact classification of a stated equation as linear or not linear.

Linearity belongs to the relation after both sides have been brought together
and simplified.  Looking for an ``xy`` token in either written side is not a
classification: the same term may occur on both sides and cancel exactly.

This solver deliberately owns only rational polynomial equations in one or two
variables.  An identity, a contradiction, a relation with a written variable
denominator, or an unfamiliar choice vocabulary is refused rather than forced
into either class.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy

from facet_runtime.exact.symbolic import (
    _safe_sympy_expression,
    _written_denominators,
)

LINEARITY_METHOD = "SymPy exact linearity classification"

LINEARITY_REQUEST = re.compile(
    r"\b(?:determine|decide|identify|state|classify)\b[^.?!]*?"
    r"\b(?:equation|relation)\b[^.?!]*?\blinear\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class LinearityClassification:
    choice: str
    simplified_relation: str
    total_degree: int


def _choice_for(linear: bool, choices: list[str]) -> str:
    """Return the page's own unique label for this classification."""

    def normalized(choice: str) -> str:
        return re.sub(r"[\s_-]+", " ", choice.strip().casefold())

    labels: dict[str, list[str]] = {"linear": [], "not linear": []}
    for choice in choices:
        label = normalized(choice)
        if label == "linear":
            labels["linear"].append(choice)
        elif label in {"not linear", "nonlinear", "non linear"}:
            labels["not linear"].append(choice)
        else:
            raise ValueError("the published choices are not a linearity classification")
    if len(labels["linear"]) != 1 or len(labels["not linear"]) != 1:
        raise ValueError(
            "the published choices do not uniquely name Linear and Not Linear"
        )
    return labels["linear" if linear else "not linear"][0]


def classify_equation_linearity(
    instruction: str, expressions: list[str], choices: list[str]
) -> LinearityClassification | None:
    """Classify one exact rational polynomial equation after cancellation.

    ``None`` means the instruction is not a linearity-classification request.
    A named request that cannot be proved raises ``ValueError`` so its caller
    can refuse it without asking a model to guess among page-owned choices.
    """
    if LINEARITY_REQUEST.search(instruction or "") is None:
        return None

    written = [item.strip() for item in expressions if item and item.strip()]
    if len(written) != 1 or written[0].count("=") != 1:
        raise ValueError("linearity classification requires one stated equation")
    left_text, right_text = written[0].split("=", 1)
    if not left_text.strip() or not right_text.strip():
        raise ValueError("the stated equation could not be read exactly")
    try:
        left = _safe_sympy_expression(left_text)
        right = _safe_sympy_expression(right_text)
        denominators = [
            *_written_denominators(left_text),
            *_written_denominators(right_text),
        ]
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError("the stated equation could not be read exactly") from error

    symbols = left.free_symbols | right.free_symbols
    if not 1 <= len(symbols) <= 2:
        raise ValueError("linearity classification requires one or two variables")
    if any(denominator.free_symbols for denominator in denominators):
        raise ValueError("a relation with a variable denominator is unsupported")

    residual = sympy.expand(left - right)
    if residual == 0:
        raise ValueError("the equation simplifies to an identity, not a line")
    if not residual.free_symbols:
        raise ValueError("the equation simplifies to a contradiction, not a line")
    try:
        polynomial = sympy.Poly(residual, *sorted(symbols, key=lambda item: item.name))
    except sympy.PolynomialError as error:
        raise ValueError("the simplified relation is not a polynomial") from error
    if not all(coefficient.is_rational for coefficient in polynomial.coeffs()):
        raise ValueError("the simplified relation is not a rational polynomial")

    degree = int(polynomial.total_degree())
    choice = _choice_for(degree == 1, choices)
    return LinearityClassification(
        choice=choice,
        simplified_relation=str(sympy.factor(residual)),
        total_degree=degree,
    )
