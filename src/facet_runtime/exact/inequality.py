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

An absolute value of a linear expression belongs to the same family, because it
is two of these comparisons. `|ax+b| <= c` is `-c <= ax+b <= c`, an interval;
`|ax+b| > c` is `ax+b < -c` or `ax+b > c`, which for a positive `c` is two rays
and is written as their union. Live, on 2026-09-12, `|z + 3| <= 1` reached the
parser as `Abs(z + 3)`, the linear route declined it as "not polynomial", and a
reasoning model answered `(-∞,∞)` -- repeatedly -- for a set that is `[-4,-2]`.
One absolute value per comparison, of an expression linear in the one variable,
with nothing else in that variable beside it: anything more is declined.
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
#: A question asking for the solution set to be graphed. The set is the same
#: mathematics whichever way it is shown -- lesson 1.7 asks for it in interval
#: notation in one step and on a number line in the next -- and drawing it is
#: the consumer's business, so this returns the set exactly as it does for
#: interval notation.
_GRAPH_THE_SOLUTION = re.compile(
    r"\b(?:graph|plot)\b[^.?!]*\bsolutions?\b", re.IGNORECASE
)
_OTHER_NOTATION = re.compile(
    r"\bset[- ]builder\b|\binequality\s+notation\b", re.IGNORECASE
)
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


#: A step that asks for one of the displayed inequalities by its position:
#: "solve the first inequality", "the second inequality", "the last
#: inequality". Singular on purpose -- "the inequalities" and "the compound
#: inequality" ask for all of them at once.
_ORDINAL = re.compile(
    r"\b(?P<ordinal>first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)"
    r"\s+inequality\b",
    re.IGNORECASE,
)
_ORDINAL_POSITION = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
    "fifth": 4,
    "5th": 4,
    "last": -1,
}


def read_comparisons(expressions: list[str]) -> list[Comparison]:
    """Every comparison the question states, or a named decline.

    A chain `a < bx + c <= d` is two comparisons sharing their middle. Several
    written apart -- in separate expressions, or joined by "and" -- are all
    required at once. "Or" asks for a union, which is not an interval and is
    declined.
    """
    return [item for statement in read_statements(expressions) for item in statement]


def read_statements(expressions: list[str]) -> list[list[Comparison]]:
    """The inequalities the question displays, each as its own comparisons.

    One statement per inequality as written: a separate expression, or one
    clause of an expression joined by "and". A chain `a < bx + c <= d` is one
    inequality, two comparisons long. Kept apart so that a step asking for one
    of them can be answered about that one, before anything is combined.
    """
    statements: list[list[Comparison]] = []
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
            statements.append(
                [
                    Comparison(parsed[index], operator, parsed[index + 1])
                    for index, operator in enumerate(operators)
                ]
            )
    if not statements:
        raise InequalityDeclined("no inequality was written")
    return statements


def in_scope(
    statements: list[list[Comparison]], instruction: str
) -> tuple[list[list[Comparison]], str]:
    """The inequalities this step asks about, and how it named them.

    Hawkes displays a problem once and walks through it in steps: "solve the
    first inequality", then "the second", then the compound inequality they
    make together. Live, on 2026-09-12, step 1 displayed `-4(w-1) <= 28 and
    1+w < 9` and asked for the first; every displayed comparison was
    intersected regardless, and `[-6,8)` went into a box that wanted
    `[-6,∞)`. The step decides the scope, and it is decided here -- before
    anything is combined.

    A step that names no single inequality asks about all of them, which is
    what a compound inequality has always meant. A step naming more than one
    position, or a position past the last displayed inequality, is declined
    rather than guessed at.
    """
    named = {match.group("ordinal").lower() for match in _ORDINAL.finditer(instruction)}
    if not named:
        return statements, ""
    count = len(statements)
    positions = set()
    for name in named:
        position = _ORDINAL_POSITION[name]
        position = count - 1 if position < 0 else position
        if position >= count:
            raise InequalityDeclined(
                f"the step names inequality {position + 1} of {count} written"
            )
        positions.add(position)
    # "The last" of two is the second, and says nothing new. Two different
    # positions is a step about more than one of them, which is not this.
    if len(positions) > 1:
        raise InequalityDeclined("the step names more than one of the inequalities")
    (position,) = positions
    return [statements[position]], f"{position + 1} of {len(statements)}"


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


def _linear(expression: sympy.Expr, variable: sympy.Symbol, what: str):
    """`(slope, constant)` of a linear expression in the variable, or a decline."""
    try:
        polynomial = sympy.Poly(sympy.expand(expression), variable)
    except sympy.PolynomialError as error:
        raise InequalityDeclined(f"{what} that is not polynomial") from error
    if polynomial.degree() > 1:
        raise InequalityDeclined(f"{what} that is not linear")
    slope = sympy.nsimplify(polynomial.coeff_monomial(variable))
    constant = sympy.nsimplify(polynomial.coeff_monomial(1))
    if not (slope.is_Rational and constant.is_Rational):
        raise InequalityDeclined(f"{what} with a coefficient that is not rational")
    return slope, constant


#: The comparison an absolute value's lower branch makes: `|u| > c` holds
#: where `u < -c`, and `|u| >= c` where `u <= -c`.
_MIRRORED = {">": "<", ">=": "<="}


def _solve(comparison: Comparison, variable: sympy.Symbol) -> tuple[sympy.Set, str]:
    """One comparison, with or without an absolute value in it."""
    residual = sympy.expand(comparison.left - comparison.right)
    absolutes = residual.atoms(sympy.Abs)
    if not absolutes:
        return _ray(comparison, variable)
    if len(absolutes) != 1:
        raise InequalityDeclined("an inequality with more than one absolute value")
    (absolute,) = absolutes
    inner = absolute.args[0]
    if variable not in inner.free_symbols:
        raise InequalityDeclined("an absolute value with no variable in it")

    # The comparison as `scale*|u| + offset op 0`, with the variable nowhere
    # but inside the bars. `|x - 1| < x` is a real question, and not this one.
    marker = sympy.Dummy("absolute")
    outside = sympy.expand(residual.subs(absolute, marker))
    if variable in outside.free_symbols:
        raise InequalityDeclined(
            "an absolute-value inequality with the variable outside the bars"
        )
    scale, offset = _linear(outside, marker, "an absolute-value inequality")
    if scale == 0:
        raise InequalityDeclined("an absolute value that cancels out")
    _linear(inner, variable, "an absolute value")

    # Isolate the absolute value. Dividing by a negative scale turns the
    # comparison round, exactly as it does for the variable itself.
    operator = comparison.operator
    if scale < 0:
        operator = _REVERSED[operator]
    bound = -offset / scale
    isolated = f"|{inner}| {operator} {bound}"

    if operator in ("<", "<="):
        # Never true below zero, and `|u| < 0` never at all. `|u| <= 0` is the
        # single point `u = 0`, which the two branches below produce and which
        # is then refused as not an interval rather than written as one.
        if bound < 0 or (bound == 0 and operator == "<"):
            return sympy.S.EmptySet, f"{isolated} is never true"
        lower, lower_step = _ray(Comparison(-bound, operator, inner), variable)
        upper, upper_step = _ray(Comparison(inner, operator, bound), variable)
        return sympy.Intersection(lower, upper), (
            f"{isolated} means {-bound} {operator} {inner} {operator} {bound}: "
            f"{lower_step}; {upper_step}"
        )
    # `|u| > c` and `|u| >= c`: always true below zero, and `|u| >= 0` always.
    if bound < 0 or (bound == 0 and operator == ">="):
        return sympy.S.Reals, f"{isolated} is always true"
    lower, lower_step = _ray(Comparison(inner, _MIRRORED[operator], -bound), variable)
    upper, upper_step = _ray(Comparison(inner, operator, bound), variable)
    return sympy.Union(lower, upper), (
        f"{isolated} means {inner} {_MIRRORED[operator]} {-bound} or "
        f"{inner} {operator} {bound}: {lower_step}; {upper_step}"
    )


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
    """A solution set in interval notation, or a named decline.

    A union of disjoint intervals is written left to right joined by `∪`,
    which is how interval notation writes the solution of `|u| > c`.
    """
    if solution == sympy.S.EmptySet:
        return "∅"
    if solution == sympy.S.Reals:
        return "(-∞,∞)"
    if isinstance(solution, sympy.Union):
        pieces = sorted(solution.args, key=lambda piece: piece.inf)
        if not all(isinstance(piece, sympy.Interval) for piece in pieces):
            raise InequalityDeclined("the solution set is not a union of intervals")
        return "∪".join(write_interval(piece, decimal) for piece in pieces)
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
    if _OTHER_NOTATION.search(instruction) or not (
        _INTERVAL_NOTATION.search(instruction)
        or _GRAPH_THE_SOLUTION.search(instruction)
    ):
        return (
            None,
            (
                "only an inequality answered in interval notation, or graphed as "
                "its solution set, is solved exactly"
            ),
        )
    try:
        statements, selected = in_scope(read_statements(expressions), instruction)
        comparisons = [item for statement in statements for item in statement]
        variable = _variable(comparisons)
        solution: sympy.Set = sympy.S.Reals
        steps = []
        for comparison in comparisons:
            found, step = _solve(comparison, variable)
            solution = sympy.Intersection(solution, found)
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
                **({"selected": selected} if selected else {}),
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
