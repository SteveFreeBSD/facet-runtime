"""A linear inequality in one variable, solved to the interval it describes.

`a < bx + c <= d` is two comparisons of one linear expression, and each is
settled by moving everything to one side and dividing by the coefficient --
turning the comparison round when that coefficient is negative. What is left
is an intersection of rays, which is an interval, the whole line, or nothing.
None of it is anything a model knows that exact rational arithmetic does not.

Live, on 2026-09-12, lesson 1.7: "solve the inequality and express your answer
in interval notation". The deterministic stage had no branch for an inequality
at all, declined with "no exact operation matched the instruction", and a
reasoning model on a GPU answered in nine seconds what this answers in one
millisecond with a proof behind it.

The answer is the solution set written in interval notation -- `(-8,7]`,
`[5/2,∞)`, `∅` -- because that is what the mathematics is and what the
question asked for. How a particular answer surface builds those brackets is
the consumer's business, and nothing here knows there is one.

Deliberately a closed family. One variable; every side linear with rational
coefficients; comparisons that are strict or not; any number of them joined by
"and", chained or written separately. Everything else is declined with a named
reason rather than approximated -- see `solve_linear_inequality`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

import sympy

from facet_runtime.exact.symbolic import _safe_sympy_expression

#: The name this solver answers to.
INEQUALITY_METHOD = "SymPy exact linear inequality"

#: A question that is about solving an inequality. Hawkes regenerates the
#: wording, so this reads the noun rather than a sentence: "solve the
#: inequality", "the following compound inequality", "solve the inequalities".
INEQUALITY_REQUEST = re.compile(r"\binequalit(?:y|ies)\b", re.IGNORECASE)

#: The notation the answer is asked in. Interval notation is the one this
#: family writes; set-builder and inequality notation are different answers to
#: the same question, and are declined rather than silently replaced.
_INTERVAL_NOTATION = re.compile(r"\binterval\s+notation\b", re.IGNORECASE)
_DECIMAL_FORM = re.compile(
    r"\bdecimal\s+(?:form|notation)\b|\bas\s+decimals?\b", re.IGNORECASE
)

#: Every spelling of a comparison that reaches this stage. MathML arrives as
#: the solver's LaTeX subset, so `≤` is already `\leq` by the time it is here;
#: the others are accepted because a question can also be typed.
_COMPARISONS = (
    (r"\leq", "<="),
    (r"\geq", ">="),
    (r"\le", "<="),
    (r"\ge", ">="),
    (r"\lt", "<"),
    (r"\gt", ">"),
    ("≤", "<="),
    ("≥", ">="),
    ("⩽", "<="),
    ("⩾", ">="),
    ("&lt;", "<"),
    ("&gt;", ">"),
)
_OPERATOR = re.compile(r"(<=|>=|<|>)")
#: A comparison this family does not answer: not-equal, or a bare equals sign
#: sitting in a chain of inequalities.
_FOREIGN_RELATION = re.compile(r"≠|\\neq|\\ne\b|!=|(?<![<>])=")
_CONJUNCTION = re.compile(r"\\text\{\s*and\s*\}|\band\b|\\land|∧", re.IGNORECASE)
_DISJUNCTION = re.compile(r"\\text\{\s*or\s*\}|\bor\b|\\lor|∨", re.IGNORECASE)

_RELATIONS = {
    "<": lambda left, right: left < right,
    "<=": lambda left, right: left <= right,
    ">": lambda left, right: left > right,
    ">=": lambda left, right: left >= right,
}
_REVERSED = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}


class InequalityDeclined(Exception):
    """Not a question this family answers exactly, and why."""


@dataclass(frozen=True, slots=True)
class Comparison:
    """One `left op right`, as it was written."""

    left: sympy.Expr
    operator: str
    right: sympy.Expr


@dataclass(frozen=True, slots=True)
class SolvedInequality:
    """The solution set, how it is written, and the working behind it."""

    written: str
    solution: sympy.Set
    evidence: dict[str, str]


def _normalized(text: str) -> str:
    value = text.strip().strip("$`")
    value = value.replace("−", "-").replace("–", "-")
    # Sizing commands first: `\left(` begins with `\le`.
    value = value.replace(r"\left", "").replace(r"\right", "")
    # Longest spelling first, so `\leq` is not read as `\le` followed by `q`.
    for spelling, operator in _COMPARISONS:
        value = value.replace(spelling, operator)
    return value


def states_an_inequality(expressions: list[str]) -> bool:
    """Whether any expression is written as a comparison at all."""
    return any(_OPERATOR.search(_normalized(item or "")) for item in expressions)


def read_comparisons(expressions: list[str]) -> list[Comparison]:
    """Every comparison the question states, or a named decline.

    A chain `a < bx + c <= d` is two comparisons sharing their middle. Several
    written apart -- in separate expressions, or joined by "and" -- are all
    required at once. "Or" asks for a union, which is not an interval and is
    declined.
    """
    comparisons: list[Comparison] = []
    for expression in expressions:
        text = _normalized(expression)
        if not _OPERATOR.search(text):
            continue
        if _DISJUNCTION.search(text):
            raise InequalityDeclined(
                "a disjunction of inequalities is not one interval"
            )
        if _FOREIGN_RELATION.search(text):
            raise InequalityDeclined(
                "only <, <=, > and >= comparisons are solved exactly"
            )
        for clause in _CONJUNCTION.split(text):
            if not clause.strip():
                continue
            pieces = _OPERATOR.split(clause)
            sides, operators = pieces[0::2], pieces[1::2]
            if not operators or any(not side.strip() for side in sides):
                raise InequalityDeclined("an inequality is missing one of its sides")
            if len(operators) > 2:
                raise InequalityDeclined("a chain of more than two comparisons")
            try:
                parsed = [_safe_sympy_expression(side) for side in sides]
            except (SyntaxError, TypeError, ValueError, ZeroDivisionError) as error:
                raise InequalityDeclined(
                    f"a side of the inequality could not be read exactly: {error}"
                ) from error
            comparisons.extend(
                Comparison(parsed[index], operator, parsed[index + 1])
                for index, operator in enumerate(operators)
            )
    if not comparisons:
        raise InequalityDeclined("no inequality was written")
    return comparisons


def _variable(comparisons: list[Comparison]) -> sympy.Symbol:
    names = set().union(
        *(item.left.free_symbols | item.right.free_symbols for item in comparisons)
    )
    if len(names) != 1:
        raise InequalityDeclined(
            "an inequality in exactly one variable is solved exactly"
            if names
            else "an inequality with no variable in it"
        )
    return names.pop()


def _ray(comparison: Comparison, variable: sympy.Symbol) -> tuple[sympy.Set, str]:
    """One comparison, solved by hand: the set it describes and how.

    `left op right` is `a*x + b op 0`. Dividing by `a` keeps the comparison
    when `a` is positive and turns it round when `a` is negative; when `a` is
    zero there is no variable left and the comparison is simply true or false.
    """
    residual = sympy.expand(comparison.left - comparison.right)
    try:
        polynomial = sympy.Poly(residual, variable)
    except sympy.PolynomialError as error:
        raise InequalityDeclined("an inequality that is not polynomial") from error
    if polynomial.degree() > 1:
        raise InequalityDeclined("an inequality that is not linear")
    slope = sympy.nsimplify(polynomial.coeff_monomial(variable))
    constant = sympy.nsimplify(polynomial.coeff_monomial(1))
    if not (slope.is_Rational and constant.is_Rational):
        raise InequalityDeclined(
            "an inequality with a coefficient that is not rational"
        )

    operator = comparison.operator
    if slope == 0:
        holds = _RELATIONS[operator](constant, 0)
        return (sympy.S.Reals if holds else sympy.S.EmptySet), (
            f"{constant} {operator} 0 is {'always' if holds else 'never'} true"
        )
    bound = -constant / slope
    if slope < 0:
        operator = _REVERSED[operator]
    strict = operator in ("<", ">")
    if operator in ("<", "<="):
        found = sympy.Interval(-sympy.oo, bound, right_open=strict)
    else:
        found = sympy.Interval(bound, sympy.oo, left_open=strict)
    turned = (
        f" (divided by {slope}, so the comparison turns round)" if slope < 0 else ""
    )
    return found, f"{variable} {operator} {bound}{turned}"


def _endpoint(value: sympy.Expr, decimal: bool) -> str:
    if value == sympy.oo:
        return "∞"
    if value == -sympy.oo:
        return "-∞"
    exact = Fraction(int(value.p), int(value.q))
    if exact.denominator == 1:
        return str(exact.numerator)
    if not decimal:
        return f"{exact.numerator}/{exact.denominator}"
    denominator = exact.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        raise InequalityDeclined(
            f"the endpoint {exact} has no exact decimal form, and decimals were asked for"
        )
    with localcontext() as context:
        context.prec = 64
        return format(Decimal(exact.numerator) / Decimal(exact.denominator), "f")


def write_interval(solution: sympy.Set, decimal: bool = False) -> str:
    """A solution set in interval notation, or a named decline."""
    if solution == sympy.S.EmptySet:
        return "∅"
    if solution == sympy.S.Reals:
        return "(-∞,∞)"
    if not isinstance(solution, sympy.Interval):
        # A single point, where a <= and a >= meet. `[3,3]` is not how that
        # set is written, and `{3}` is not interval notation.
        raise InequalityDeclined("the solution set is not an interval")
    opening = "(" if solution.left_open or solution.start == -sympy.oo else "["
    closing = ")" if solution.right_open or solution.end == sympy.oo else "]"
    return (
        f"{opening}{_endpoint(solution.start, decimal)},"
        f"{_endpoint(solution.end, decimal)}{closing}"
    )


def solve_linear_inequality(
    instruction: str, expressions: list[str]
) -> tuple[SolvedInequality | None, str]:
    """The interval a system of linear inequalities describes, or why not.

    Solved twice, by different means, and answered only when they agree: once
    by hand above, comparison by comparison, and once by SymPy's own
    set reading of the conjunction as written.
    """
    if not _INTERVAL_NOTATION.search(instruction):
        return (
            None,
            "only an inequality answered in interval notation is solved exactly",
        )
    try:
        comparisons = read_comparisons(expressions)
        variable = _variable(comparisons)
        solution: sympy.Set = sympy.S.Reals
        steps = []
        for comparison in comparisons:
            ray, step = _ray(comparison, variable)
            solution = sympy.Intersection(solution, ray)
            steps.append(step)

        stated = sympy.And(
            *(_RELATIONS[item.operator](item.left, item.right) for item in comparisons)
        )
        if stated == sympy.true:
            checked = sympy.S.Reals
        elif stated == sympy.false:
            checked = sympy.S.EmptySet
        else:
            checked = stated.as_set()
        if checked != solution:
            raise InequalityDeclined(
                "the solution could not be confirmed independently"
            )
        written = write_interval(
            solution, decimal=bool(_DECIMAL_FORM.search(instruction))
        )
    except InequalityDeclined as decline:
        return None, str(decline)

    return (
        SolvedInequality(
            written=written,
            solution=solution,
            evidence={
                "variable": str(variable),
                "comparisons": "; ".join(
                    f"{item.left} {item.operator} {item.right}" for item in comparisons
                ),
                "steps": "; ".join(steps),
                "solution_set": str(solution),
            },
        ),
        "",
    )
