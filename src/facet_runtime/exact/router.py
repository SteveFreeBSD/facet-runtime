"""Facet's deterministic stage: answer it exactly, or say why not.

This is the first thing a solve request meets. It is given a question in words
and the exact expressions that question is about, and it either produces an
answer that was *computed* -- checkable, reproducible, and free -- or it
declines and names the gap. Nothing here is a model, and nothing here reaches a
network.

A decline is a first-class result. It is the only thing that sends a question
on to the reasoning path, and its reason is carried out to the caller so that a
live fallback can say which gap it fell through rather than merely that it fell.

What this module deliberately does not do is present an answer. It reports the
values and how literally to take them; turning those into keystrokes, fields,
or option clicks belongs to whoever owns the page, and Facet owns no page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from facet_runtime.exact.answer import extract_final_math
from facet_runtime.exact.polynomial import answer_polynomial_product
from facet_runtime.exact.symbolic import (
    AbsoluteValueEquationResult,
    LinearEquationResult,
    _requested_operation,
    _requested_variable,
    _safe_sympy_expression,
    answer_symbolic_math,
    quadratic_coefficients,
    solve_equation,
)

#: The name the deterministic stage answers to. It is a solver, and it is named
#: as one: a reader must never have to work out whether this was a model.
EXACT_METHOD = "SymPy exact symbolic"

#: Why the deterministic stage declined, when no operation matched at all.
#: Worded exactly as the router has always worded it, because a consumer shows
#: this to a person who is watching a question fall through.
NO_OPERATION_MATCHED = "no exact operation matched the instruction"

#: How literally a consumer should take a value.
#:
#: `verbatim` means the value is already exactly what belongs in an answer --
#: a coordinate pair, a classification, a phrase like "Not a Real Number" --
#: and rewriting it would damage it. `math` means the value is mathematics and
#: a consumer should render it in whatever entry syntax its own surface needs.
#: `auto` means the value is mathematics *unless* it is a plain phrase, which
#: is the one case a solver can produce either way.
EntryMode = Literal["verbatim", "math", "auto"]

_PROSE = re.compile(r"[A-Za-z][A-Za-z ]*")
_VERTEX = re.compile(
    r"\b(?:find|identify|determine)\s+(?:the\s+)?vertex\b", re.IGNORECASE
)
_POLYNOMIAL_CHOICE = re.compile(
    r"polynomial\s+or\s+a\s+non[- ]polynomial", re.IGNORECASE
)
_RADICAL = re.compile(r"\\sqrt(?:\[(\d+)\])?\{(.+)\}", re.DOTALL)


def is_prose(value: str) -> bool:
    """Whether a value is a phrase rather than mathematics."""
    return _PROSE.fullmatch(value) is not None


def entry_text(value: str, entry_mode: EntryMode) -> str | None:
    """Whether a consumer must take this value exactly as it stands.

    Returns the value when it must be used verbatim, and None when the consumer
    should render it in whatever entry syntax its own answer surface needs.
    Defined here, beside the mode itself, so there is one reading of it.
    """
    if entry_mode == "verbatim" or (entry_mode == "auto" and is_prose(value)):
        return value
    return None


@dataclass(frozen=True, slots=True)
class ExactSolution:
    """One exactly computed answer, in the shape the question asked for.

    `display` is the whole answer as it would be read. `entry` is the single
    value that belongs in a one-value answer, and `parts` carries the separate
    values when a question takes more than one. They are kept apart on purpose:
    recovering the boundary between two answers by splitting display prose
    later is guessing at mathematics after the fact.
    """

    display: str
    entry: str = ""
    parts: tuple[str, ...] = ()
    entry_mode: EntryMode = "verbatim"
    method: str = EXACT_METHOD


def solve_exact(
    instruction: str, expressions: list[str]
) -> tuple[ExactSolution | None, str]:
    """Answer the question exactly, or decline it and say why.

    Returns the solution and an empty reason, or None and the reason.
    """
    if not expressions:
        return None, "no exact expression was supplied"

    if _VERTEX.search(instruction):
        return solve_vertex(expressions)

    classification = classify_polynomial(instruction, expressions)
    if classification is not None:
        return classification, ""

    # "Is this a real number?" is decidable before any solver runs: an even
    # root of a negative number is not real, and every other case here is.
    # Observed repeatedly in lesson 1.2 -- the square root of -100 -- where the
    # symbolic solver produced `10*I`, which is not an answer to the question
    # asked, and the caller then reported the question unsupported.
    realness = evaluate_real_radical(instruction, expressions)
    if realness is not None:
        return realness, ""

    equation = solve_one_equation(instruction, expressions)
    if equation is not None:
        return equation, ""

    result = answer_symbolic_math(problem_text=instruction, expressions=expressions)
    if result is None:
        result = answer_polynomial_product(f"{instruction}\n{expressions[0]}")
    if result is None:
        # Only the exact solvers answer here. Handing these expressions to a
        # model at this point would produce an answer with no independent check
        # behind it at all, which is the reasoning route's decision to make and
        # not this one's.
        return None, NO_OPERATION_MATCHED

    final_math = extract_final_math(result.raw_response)
    if not final_math:
        return None, "the exact solver produced no final answer"
    return ExactSolution(display=final_math, entry=final_math, entry_mode="auto"), ""


def solve_vertex(expressions: list[str]) -> tuple[ExactSolution | None, str]:
    if len(expressions) != 1:
        return None, "vertex requires one exact function"
    try:
        a, b, c = quadratic_coefficients(expressions[0])
        h = -b / (2 * a)
        k = a * h * h + b * h + c
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError):
        return None, "vertex requires a rational quadratic"
    # A vertex is one ordered pair, and its parentheses are part of the answer.
    pair = f"({h},{k})"
    return ExactSolution(display=pair, entry=pair), ""


def classify_polynomial(
    instruction: str, expressions: list[str]
) -> ExactSolution | None:
    """Classify polynomial-choice questions without needing a picture."""
    if not _POLYNOMIAL_CHOICE.search(instruction):
        return None
    for expression in expressions:
        try:
            value = _safe_sympy_expression(expression)
            if not value.is_polynomial(*value.free_symbols):
                raise ValueError("fractional or negative exponent")
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError):
            return ExactSolution(display="Non-Polynomial", entry="Non-Polynomial")
        return ExactSolution(display="Polynomial", entry="Polynomial")
    return None


def evaluate_real_radical(
    instruction: str, expressions: list[str]
) -> ExactSolution | None:
    """Answer "determine if this is a real number", or None if not asked.

    The question offers two options -- "Real Number" and "Not a Real Number" --
    and asks for the value only when it is real. Selecting the option stays the
    reader's action; this only says which one is right.

    The test is the index's parity, not SymPy's principal branch: `(-27)**(1/3)`
    evaluates to a complex number, but the real cube root of -27 is -3, and the
    question means the real one. Only an *even* root of a negative fails to be
    real.
    """
    import sympy

    text = instruction.lower()
    if "real number" not in text:
        return None
    # Hawkes uses two prompt families for the same rule. Some ask whether the
    # radical is real; others say to evaluate it and explicitly direct the
    # student to indicate "Not a Real Number" when it is not. The latter is
    # what the live sqrt(-36) question used, so limiting this branch to
    # determine/decide/whether allowed SymPy's complex `6*I` to escape.
    if not any(
        phrase in text
        for phrase in ("determine", "decide", "whether", "not a real number")
    ):
        return None

    match = _RADICAL.fullmatch(expressions[0].strip())
    if match is None:
        return None
    index = int(match.group(1) or 2)
    try:
        radicand = _safe_sympy_expression(match.group(2), positive_symbols=False)
    except Exception:  # noqa: BLE001 - anything unparseable is simply not this question
        return None
    if radicand.free_symbols or not radicand.is_real:
        return None  # a value that depends on a variable is not this question

    if radicand.is_negative and index % 2 == 0:
        return ExactSolution(display="Not a Real Number", entry="Not a Real Number")
    value = sympy.real_root(radicand, index)
    exact = sympy.nsimplify(value, rational=True)
    if not exact.is_rational:
        return None  # an irrational value is not what this question asks for
    return ExactSolution(display=str(exact), entry=str(exact))


def solve_one_equation(
    instruction: str, expressions: list[str]
) -> ExactSolution | None:
    """Map one exact equation result onto the structured answer model."""
    if _requested_operation(instruction) != "solve" or len(expressions) != 1:
        return None
    target = _requested_variable(instruction)
    result = solve_equation(expressions[0], variable=target)
    if result is None:
        return None
    if isinstance(result, AbsoluteValueEquationResult):
        entry = result.classification
        if len(result.solutions) == 1:
            entry = result.solutions[0]
        return ExactSolution(display=result.display_text, entry=entry)
    if isinstance(result, LinearEquationResult):
        if result.solution is None:
            return ExactSolution(
                display=result.classification, entry=result.classification
            )
        if target is not None:
            return ExactSolution(
                display=f"{result.variable} = {result.solution}",
                entry=result.solution,
                entry_mode="math",
            )
        return ExactSolution(
            display=(
                f"{result.classification} ({result.variable} = {result.solution})"
            ),
            entry=result.solution,
        )
    if not result.solutions:
        return ExactSolution(display=result.classification, entry=result.classification)
    if len(result.solutions) == 1:
        return ExactSolution(
            display=result.display_text,
            entry=result.solutions[0],
            entry_mode="math",
        )
    # More than one root is more than one answer. They stay separate values.
    return ExactSolution(
        display=result.display_text,
        parts=tuple(result.solutions),
        entry_mode="math",
    )
