"""Deterministic exact mathematics, the route Facet takes before reasoning."""

from facet_runtime.exact.answer import extract_final_math
from facet_runtime.exact.result import ExactCallResult, ExactDebugInfo
from facet_runtime.exact.router import (
    EXACT_METHOD,
    NO_OPERATION_MATCHED,
    EntryMode,
    ExactSolution,
    classify_polynomial,
    entry_text,
    evaluate_real_radical,
    is_prose,
    solve_exact,
    solve_one_equation,
    solve_vertex,
)

__all__ = [
    "EXACT_METHOD",
    "NO_OPERATION_MATCHED",
    "EntryMode",
    "ExactCallResult",
    "ExactDebugInfo",
    "ExactSolution",
    "classify_polynomial",
    "entry_text",
    "evaluate_real_radical",
    "extract_final_math",
    "is_prose",
    "solve_exact",
    "solve_one_equation",
    "solve_vertex",
]
