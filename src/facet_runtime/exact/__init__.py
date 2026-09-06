"""Deterministic exact mathematics, the route Facet takes before reasoning."""

from facet_runtime.exact.answer import extract_final_math
from facet_runtime.exact.linear import solve_linear_function
from facet_runtime.exact.regression import (
    NotThisQuestion,
    fit_quadratic,
    rate_unit,
    regression_optimum,
)
from facet_runtime.exact.result import ExactCallResult, ExactDebugInfo
from facet_runtime.exact.router import (
    EXACT_METHOD,
    NO_OPERATION_MATCHED,
    REGRESSION_METHOD,
    EntryMode,
    ExactSolution,
    classify_polynomial,
    entry_text,
    evaluate_real_radical,
    is_prose,
    solve_exact,
    solve_one_equation,
    solve_over_points,
    solve_points_on_quadratic,
    solve_stated_linear_function,
    solve_vertex,
)

__all__ = [
    "EXACT_METHOD",
    "NO_OPERATION_MATCHED",
    "REGRESSION_METHOD",
    "EntryMode",
    "ExactCallResult",
    "ExactDebugInfo",
    "ExactSolution",
    "NotThisQuestion",
    "classify_polynomial",
    "entry_text",
    "evaluate_real_radical",
    "extract_final_math",
    "fit_quadratic",
    "is_prose",
    "rate_unit",
    "regression_optimum",
    "solve_exact",
    "solve_linear_function",
    "solve_one_equation",
    "solve_over_points",
    "solve_points_on_quadratic",
    "solve_stated_linear_function",
    "solve_vertex",
]
