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
from dataclasses import dataclass, field
from typing import Literal

from facet_runtime.exact.answer import extract_final_math
from facet_runtime.exact.consecutive import (
    CONSECUTIVE_METHOD,
    CONSECUTIVE_REQUEST,
    solve_consecutive_sum,
)
from facet_runtime.exact.coordinate import (
    COORDINATE_METHOD,
    COORDINATE_REQUEST,
    solve_missing_coordinate,
)
from facet_runtime.exact.discount import (
    DISCOUNT_METHOD,
    DISCOUNT_REQUEST,
    solve_discount,
)
from facet_runtime.exact.distance import (
    DISTANCE_METHOD,
    DISTANCE_REQUEST,
    solve_point_distance,
)
from facet_runtime.exact.inequality import (
    INEQUALITY_METHOD,
    INEQUALITY_REQUEST,
    solve_linear_inequality,
    states_an_inequality,
)
from facet_runtime.exact.intercepts import (
    INTERCEPT_METHOD,
    INTERCEPT_REQUEST,
    AxisIntercepts,
    solve_axis_intercepts,
)
from facet_runtime.exact.linear import (
    LINEAR_REQUEST,
    render,
    solve_linear_function,
    subject,
)
from facet_runtime.exact.linearity import (
    LINEARITY_METHOD,
    LINEARITY_REQUEST,
    classify_equation_linearity,
)
from facet_runtime.exact.midpoint import (
    MIDPOINT_METHOD,
    MIDPOINT_REQUEST,
    solve_midpoint,
)
from facet_runtime.exact.polynomial import answer_polynomial_product
from facet_runtime.exact.quadrant import (
    QUADRANT_METHOD,
    QUADRANT_REQUEST,
    solve_quadrant,
)
from facet_runtime.exact.regression import NotThisQuestion, regression_optimum
from facet_runtime.exact.solution_kind import (
    SolutionKindRefused,
)
from facet_runtime.exact.solution_kind import (
    select as select_solution_kind,
)
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
from facet_runtime.exact.table import (
    TABLE_METHOD,
    AnswerTable,
    Representation,
    TableRefused,
    complete_table,
)


class ExactlyRefused(Exception):
    """A question the deterministic stage claimed, and cannot answer.

    Different from a decline. A decline says "not mine", and sending the
    question on to a model is exactly the right thing to do with it. This says
    "mine, and unanswerable as asked" -- and falling through on it would hand a
    model a question whose answer it cannot know, because the answer is one of
    a set of alternatives the model was never shown.
    """


#: The name the deterministic stage answers to. It is a solver, and it is named
#: as one: a reader must never have to work out whether this was a model.
EXACT_METHOD = "SymPy exact symbolic"

#: The same solver, named for what it actually did when it fitted a curve to
#: measured data. A reader looking at "9 and 36" should be able to see that a
#: regression produced them, not a symbolic manipulation of something written.
REGRESSION_METHOD = "SymPy exact least-squares regression"

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
_POINTS_ON_QUADRATIC = re.compile(
    r"""
    \s*(?:find|give|identify|determine|provide)\s+
    (?P<count>one|two|three|four|[1-4])\s+(?P<noun>points?)\s+
    on\s+(?:the\s+)?(?:graph|parabola)\s+
    (?:other\s+than|excluding)\s+
    (?:
      (?:the\s+)?vertex\s+and\s+(?:the\s+)?x[- ]intercepts?
      |
      (?:the\s+)?x[- ]intercepts?\s+and\s+(?:the\s+)?vertex
    )
    [.!?]?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)
_POINT_COUNTS = {"one": 1, "two": 2, "three": 3, "four": 4}
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


#: What *kind of thing* an answer is, beside what it says.
#:
#: The wire has always carried an answer as a string, so a consumer that has to
#: type it into a real answer surface has had to rediscover its structure by
#: parsing it -- and every new exact family has been found to be uninsertable
#: only after it was already answering live questions. A midpoint returning
#: `(17/2,-1/2)` is a pair *and* a rational at once, and nothing said so.
#:
#: This is deliberately the family and not a description of the value. What
#: notation the value is written in stays readable from the value; what a
#: consumer cannot recover cheaply, and needs before it can say whether it has
#: a path at all, is which of these it is holding.
#:
#: A closed set. Growing it is a protocol change, and the point of it is that
#: `docs/ANSWER_CAPABILITIES.md` on the consumer side must have a row for every
#: member -- a path, or an explicit statement that there is not one.
SCALAR = "scalar"
ORDERED_PAIR = "ordered-pair"
PARTS = "parts"
CHOICE = "choice"
CONDITIONAL_CHOICE = "conditional-choice"
#: An answer that is an *equation* rather than a value: `y = -2x + 5`, which is
#: what "the equation of the line in slope-intercept form" asks for. It is its
#: own family because a consumer has two ways to enter it and has to choose:
#: an answer surface that already states the subject takes the value alone, and
#: one that states nothing takes the whole equation. Neither is recoverable
#: from a value, which is why this was for a long time answered with the value
#: and no way for anyone to tell.
RELATION = "relation"
AXIS_INTERCEPTS = "axis-intercepts"
ANSWER_FORMS: tuple[str, ...] = (
    SCALAR,
    ORDERED_PAIR,
    PARTS,
    CHOICE,
    CONDITIONAL_CHOICE,
    RELATION,
    AXIS_INTERCEPTS,
)


@dataclass(frozen=True, slots=True)
class Relation:
    """An answer that states one quantity equals another, kept as both sides.

    `y` and `-2x+5`, never the string `y=-2x+5` alone. A consumer decides which
    of the two its own answer surface takes, and recovering the boundary by
    splitting the written form later is guessing at mathematics after the fact
    -- the same argument that keeps `parts` separate from `display`. It matters
    here for a reason a split would get wrong: `2x+y=5` is the same line in
    standard form, and its left side is not something any page supplies.
    """

    #: What the equation is about, written as the question names it: `y` for a
    #: line in slope-intercept form, `f(x)` where the question named a function.
    subject: str
    #: What the subject is equal to.
    value: str

    @property
    def written(self) -> str:
        """The whole equation, for a surface that states neither side itself."""
        return f"{self.subject}={self.value}"


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
    #: Both sides of the answer, when the answer is an equation. `entry` is
    #: then the whole equation and this is what it is made of, so a consumer
    #: whose answer surface states the subject itself can enter the value
    #: alone without splitting the written form to find it.
    relation: Relation | None = None
    #: The published alternative that enables `relation`, when answering is a
    #: choice followed conditionally by an equation entry.
    choice: str = ""
    #: Both named axis intercepts.  ``None`` for one axis means the line has no
    #: point on that axis; it is not a phrase or a scalar value.
    intercepts: AxisIntercepts | None = None
    #: Which family this answer belongs to. `scalar` is the default because it
    #: is what most exact answers are: one written value.
    form: str = SCALAR
    method: str = EXACT_METHOD
    #: The working, when a solver computed one worth showing: the fitted
    #: coefficients, the turning point, the value there. Carried out as
    #: evidence rather than folded into the answer, so a consumer can redo the
    #: whole computation from the same inputs and compare instead of trusting.
    evidence: dict[str, str] = field(default_factory=dict)


def solve_table_completion(
    expressions: list[str],
    table: AnswerTable,
    answer_parts: int,
    representation: Representation | None,
) -> tuple[ExactSolution | None, str]:
    """Fill a table of values from the relation it is a table of.

    The strongest signal a question can carry. A grid of stated values with
    numbered blanks, beside the relation those values satisfy, determines every
    blank by substitution -- so this is tried before any reading of the
    instruction's verbs, which are at best a description of what the grid
    already says exactly.
    """
    try:
        done = complete_table(
            expressions,
            table,
            answer_parts=answer_parts,
            representation=representation,
        )
    except TableRefused as error:
        return None, f"the table was not completed exactly: {error}"
    single = len(done.parts) == 1
    return (
        ExactSolution(
            display=", ".join(done.parts),
            entry=done.parts[0] if single else "",
            parts=() if single else done.parts,
            entry_mode="math",
            form=SCALAR if single else PARTS,
            method=TABLE_METHOD,
            evidence=done.evidence,
        ),
        "",
    )


def solve_exact(
    instruction: str,
    expressions: list[str],
    *,
    points: list[tuple[str, str]] | None = None,
    answer_parts: int = 1,
    table: AnswerTable | None = None,
    representation: Representation | None = None,
    choices: list[str] | None = None,
) -> tuple[ExactSolution | None, str]:
    """Answer the question exactly, or decline it and say why.

    Returns the solution and an empty reason, or None and the reason.

    A question is about written expressions or about measured points, never
    both. Points arrive when nobody wrote the function down -- it exists only
    as the fit to the data -- and are answered by their own solver. A table of
    values is a third shape: the question and its data are the same grid.

    `choices` is a fourth: a question answered by *choosing*, whose answer is
    one of the alternatives it published rather than a value anyone computes a
    written form for. Where they are given they are the contract, and a solver
    that claims such a question must return one of them exactly.
    """
    if table is not None:
        return solve_table_completion(expressions, table, answer_parts, representation)
    if points:
        return solve_over_points(instruction, points, answer_parts)
    # A price and a percentage off it is one division, and which of the three
    # quantities is wanted is stated in the question. Claimed here because
    # nothing below reads a word problem at all: live, this fell through to a
    # reasoning model, which did the arithmetic correctly on the wrong unknown
    # and returned the discount where the original price was asked for.
    if DISCOUNT_REQUEST.search(instruction):
        amount, refusal = solve_discount(instruction, expressions)
        if amount:
            return (
                ExactSolution(
                    display=amount,
                    entry=amount,
                    entry_mode="math",
                    method=DISCOUNT_METHOD,
                ),
                "",
            )
        # Named and never fallen through, for `solve_quadrant`'s reason: a
        # model asked this question has to guess which unknown was wanted.
        raise ExactlyRefused(refusal)

    # A stated sum fixes an ordinary consecutive-integer sequence exactly.  The
    # total is often inline MathJax and therefore arrives in `expressions` while
    # the prose carries the sequence length and what the page asks for.
    if CONSECUTIVE_REQUEST.search(instruction):
        consecutive, refusal = solve_consecutive_sum(
            instruction, expressions, answer_parts
        )
        if consecutive is None:
            raise ExactlyRefused(refusal)
        values = consecutive.values
        single = len(values) == 1
        return (
            ExactSolution(
                display=", ".join(values),
                entry=values[0] if single else "",
                parts=() if single else values,
                entry_mode="math",
                form=SCALAR if single else PARTS,
                method=CONSECUTIVE_METHOD,
                evidence={
                    "first": str(consecutive.first),
                    "count": str(consecutive.count),
                    "total": str(consecutive.total),
                },
            ),
            "",
        )

    if not expressions:
        return None, "no exact expression was supplied"

    # Whether an equation is linear is a property of the relation after both
    # sides are reduced, not of the tokens visible before cancellation.  This
    # is a choice answer, so the exact result uses the page's published label
    # and never implies that Facet or its consumer should click that choice.
    if LINEARITY_REQUEST.search(instruction):
        try:
            classification = classify_equation_linearity(
                instruction, expressions, choices or []
            )
        except ValueError as error:
            raise ExactlyRefused(str(error)) from error
        if classification is None:  # pragma: no cover - guarded by the regex above
            raise ExactlyRefused("the linearity request could not be classified")
        relation = (
            None
            if classification.standard_form is None
            else Relation(*classification.standard_form)
        )
        return (
            ExactSolution(
                display=(
                    classification.choice
                    if relation is None
                    else (
                        f"{classification.choice}; "
                        f"{relation.subject.replace('+', ' + ').replace('-', ' - ').strip()}"
                        f" = {relation.value}"
                    )
                ),
                entry=classification.choice if relation is None else relation.written,
                entry_mode="verbatim" if relation is None else "math",
                form=CHOICE if relation is None else CONDITIONAL_CHOICE,
                relation=relation,
                choice=classification.choice if relation is not None else "",
                method=LINEARITY_METHOD,
                evidence={
                    "simplified_relation": classification.simplified_relation,
                    "total_degree": str(classification.total_degree),
                },
            ),
            "",
        )

    # Both axis intercepts are one structured answer: a named point or an
    # explicit absence for each axis.  Claimed before the one-variable equation
    # solver, which otherwise sees ``4y=8`` and answers only ``2`` while losing
    # the x-axis question entirely.
    if INTERCEPT_REQUEST.search(instruction):
        intercepts, refusal = solve_axis_intercepts(instruction, expressions)
        if intercepts is None:
            raise ExactlyRefused(refusal)
        return (
            ExactSolution(
                display=intercepts.display,
                entry_mode="math",
                form=AXIS_INTERCEPTS,
                method=INTERCEPT_METHOD,
                intercepts=intercepts,
                evidence={
                    "x_intercept": intercepts.x.written if intercepts.x else "absent",
                    "y_intercept": intercepts.y.written if intercepts.y else "absent",
                },
            ),
            "",
        )

    # Which quadrant a point is in is two sign comparisons, and it is claimed
    # before every solver below because nothing below it can answer a choice
    # question at all. Live, this whole family reached a reasoning model on a
    # GPU -- for a question decided by comparing two numbers to zero -- because
    # the deterministic stage had no branch for it.
    if QUADRANT_REQUEST.search(instruction):
        choice, refusal = solve_quadrant(instruction, expressions, choices or [])
        if choice:
            return (
                ExactSolution(
                    display=choice,
                    # Already exactly what belongs in the answer: it is the
                    # page's own words for one of its own choices, and
                    # rewriting it into entry syntax would damage it.
                    entry=choice,
                    entry_mode="verbatim",
                    form=CHOICE,
                    method=QUADRANT_METHOD,
                ),
                "",
            )
        # Named, and never fallen through. A quadrant question that cannot be
        # answered from its own choices is not one a model should be asked --
        # it would be guessing at which alternatives the page offered.
        raise ExactlyRefused(refusal)

    # An inequality is a comparison, and nothing below reads one: the equation
    # solver and the symbolic operations all want an expression or an equals
    # sign. Live, lesson 1.7's compound inequalities fell straight through to a
    # reasoning model for an interval exact arithmetic settles. Claimed only
    # when the question names an inequality *and* one is written; a decline
    # names its gap and still reaches the reasoning route, because a quadratic
    # or a union is a real question this family simply does not answer.
    if INEQUALITY_REQUEST.search(instruction) and states_an_inequality(expressions):
        solved, refusal = solve_linear_inequality(instruction, expressions)
        if solved is None:
            return None, refusal
        return (
            ExactSolution(
                display=solved.written,
                entry=solved.written,
                # The solution set, written in interval notation. It is
                # mathematics, and how its brackets are built is the
                # consumer's decision.
                entry_mode="math",
                form=SCALAR,
                method=INEQUALITY_METHOD,
                evidence=solved.evidence,
            ),
            "",
        )

    point_count = requested_quadratic_point_count(instruction)
    if point_count is not None:
        return solve_points_on_quadratic(expressions, point_count, answer_parts)

    if _VERTEX.search(instruction):
        return solve_vertex(expressions)

    # The distance between two written points. Its answer is a square root, and
    # on the live page the box that takes it publishes `0123456789-` with a
    # Radical template -- so the exact radical is the only enterable form and a
    # decimal is not a rounder answer but an unusable one. Claimed only when
    # the instruction asks for a distance and two points can be read exactly.
    if DISTANCE_REQUEST.search(instruction):
        distance, refusal = solve_point_distance(instruction, expressions)
        if distance is not None:
            return (
                ExactSolution(
                    display=distance,
                    entry=distance,
                    entry_mode="math",
                    method=DISTANCE_METHOD,
                ),
                "",
            )
        if refusal:
            return None, refusal

    # The midpoint of the same two points, and the same argument as the
    # distance above it: halving the sum of two rationals is exact, and a model
    # asked to do it renders the half as a decimal about as often as not.
    # `(8.5,-0.5)` is a correct midpoint written in a notation the answer box
    # does not take, which the page refuses -- so it is computed here.
    if MIDPOINT_REQUEST.search(instruction):
        midpoint, refusal = solve_midpoint(instruction, expressions)
        if midpoint is not None:
            return (
                ExactSolution(
                    display=midpoint,
                    entry=midpoint,
                    # An ordered pair is already exactly what belongs in the
                    # answer: its parentheses are part of it, and rewriting it
                    # into entry syntax would damage it.
                    entry_mode="verbatim",
                    form=ORDERED_PAIR,
                    method=MIDPOINT_METHOD,
                ),
                "",
            )
        if refusal:
            return None, refusal

    # A line stated as its properties rather than written down. Two facts fix
    # it and one does not, so this either computes both and proves every stated
    # property against the result, or declines and names what stopped it. An
    # instruction that names a linear function but states no property is left
    # to the solvers below rather than claimed here.
    if LINEAR_REQUEST.search(instruction):
        line, refusal = solve_stated_linear_function(
            instruction, expressions, answer_parts
        )
        if line is not None or refusal:
            return line, refusal

    # The other half of an ordered pair: the equation is written, one
    # coordinate is given, and the box takes the value that makes the pair a
    # solution. One row of a table of values with one blank in it, and answered
    # by the same machinery -- bind what was stated, solve for what was left
    # out, and prove it by substitution back into the stated relation.
    if COORDINATE_REQUEST.search(instruction):
        coordinate, refusal = solve_missing_coordinate(instruction, expressions)
        if coordinate is not None:
            return (
                ExactSolution(
                    display=coordinate,
                    entry=coordinate,
                    entry_mode="math",
                    method=COORDINATE_METHOD,
                ),
                "",
            )
        if refusal:
            return None, refusal

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

    equation = solve_one_equation(instruction, expressions, choices or [])
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


def solve_over_points(
    instruction: str, points: list[tuple[str, str]], answer_parts: int
) -> tuple[ExactSolution | None, str]:
    """Answer a question about measured points, or decline it and say why.

    The only family answered here is a curve fitted to data and then read at
    its turning point. Its decline is worded like every other decline: the
    reason travels out to whoever asked, so a question that fell through says
    which gap it fell through.
    """
    try:
        values, working = regression_optimum(instruction, points, answer_parts)
    except NotThisQuestion as refusal:
        return None, str(refusal)
    if len(values) == 1:
        return (
            ExactSolution(
                display=values[0],
                entry=values[0],
                entry_mode="math",
                method=REGRESSION_METHOD,
                evidence=working,
            ),
            "",
        )
    # Two answers stay two values. The display is only for reading; what is
    # typed comes from the parts, and nothing downstream has to recover the
    # boundary between them by splitting prose.
    return (
        ExactSolution(
            display=", ".join(values),
            parts=tuple(values),
            entry_mode="math",
            form=PARTS,
            method=REGRESSION_METHOD,
            evidence=working,
        ),
        "",
    )


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
    return ExactSolution(display=pair, entry=pair, form=ORDERED_PAIR), ""


def solve_stated_linear_function(
    instruction: str, expressions: list[str], answer_parts: int
) -> tuple[ExactSolution | None, str]:
    """Shape a derived line as the equation a linear-function question asks for.

    The mathematics and its proof are in `facet_runtime.exact.linear`. What
    happens here is only the shaping, and for a long time it shaped the answer
    wrongly: it returned `render(...)` -- the right-hand side of `y = mx + b`
    -- as the whole answer. That is not the equation of a line, and it is only
    an answer at all on a page that prints the left side beside its answer box.
    Whether a page does that is not something Facet can know, and it does not
    ask: `MathProblem` has no way to describe a page and gains none here.

    So the answer is the whole equation, and both of its sides travel with it.
    A consumer whose answer surface states the subject enters the value alone;
    one whose surface states nothing enters the equation. That decision belongs
    to whoever owns the page, and this side stops making it silently.
    """
    line, decline = solve_linear_function(instruction, expressions, answer_parts)
    if line is None:
        return None, decline
    equation = Relation(
        subject=subject(instruction, expressions),
        value=render(line.slope, line.intercept),
    )
    return (
        ExactSolution(
            display=equation.written,
            entry=equation.written,
            entry_mode="math",
            form=RELATION,
            relation=equation,
            evidence=line.evidence,
        ),
        "",
    )


def requested_quadratic_point_count(instruction: str) -> int | None:
    """Read the bounded live point request, or decline ambiguous wording."""
    match = _POINTS_ON_QUADRATIC.fullmatch(instruction)
    if match is None:
        return None
    count_text = match.group("count").lower()
    count = _POINT_COUNTS.get(
        count_text, int(count_text) if count_text.isdigit() else 0
    )
    if (count == 1) != (match.group("noun").lower() == "point"):
        return None
    return count


def solve_points_on_quadratic(
    expressions: list[str], requested: int, answer_parts: int
) -> tuple[ExactSolution | None, str]:
    """Choose and prove simple points away from a quadratic's named landmarks."""
    import sympy

    if requested != answer_parts:
        return None, (
            f"point request asks for {requested} answers but the answer shape "
            f"requires {answer_parts}"
        )
    if len(expressions) != 1:
        return None, "points on a quadratic require one exact function"
    try:
        a, b, c = quadratic_coefficients(expressions[0])
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError):
        return None, "points on a quadratic require a rational quadratic"

    x = sympy.Symbol("x", real=True)
    polynomial = a * x**2 + b * x + c
    vertex_x = -b / (2 * a)
    vertex_y = sympy.factor(polynomial.subs(x, vertex_x))
    discriminant = sympy.factor(b**2 - 4 * a * c)
    if discriminant.is_negative:
        intercepts: tuple[sympy.Expr, ...] = ()
    elif discriminant.is_nonnegative:
        root = sympy.sqrt(discriminant)
        left = sympy.simplify((-b - root) / (2 * a))
        right = sympy.simplify((-b + root) / (2 * a))
        intercepts = (left,) if sympy.simplify(left - right) == 0 else (left, right)
    else:
        return None, "the real x-intercepts could not be determined exactly"

    excluded = (vertex_x, *intercepts)

    def is_excluded(candidate: sympy.Expr) -> bool:
        return any(sympy.simplify(candidate - value) == 0 for value in excluded)

    candidates = [sympy.Integer(0)]
    for distance in range(1, requested + 4):
        candidates.extend((sympy.Integer(distance), sympy.Integer(-distance)))

    points: list[tuple[sympy.Expr, sympy.Expr]] = []
    for candidate in candidates:
        if is_excluded(candidate):
            continue
        value = sympy.factor(polynomial.subs(x, candidate))
        # Incidence and exclusion are proved over exact SymPy expressions.
        # A failure is a refusal; an unchecked coordinate is never returned.
        if value == 0 or sympy.simplify(polynomial.subs(x, candidate) - value) != 0:
            continue
        points.append((candidate, value))
        if len(points) == requested:
            break
    if len(points) != requested:
        return None, "not enough simple non-landmark points could be proved"

    pairs = tuple(f"({point_x},{point_y})" for point_x, point_y in points)
    evidence = {
        "coefficients": ",".join(str(value) for value in (a, b, c)),
        "vertex": f"({vertex_x},{vertex_y})",
        "x_intercepts": ",".join(str(value) for value in intercepts) or "none",
        "points": ",".join(pairs),
        "incidence": "; ".join(
            f"f({point_x})={point_y}" for point_x, point_y in points
        ),
    }
    if requested == 1:
        return (
            ExactSolution(
                display=pairs[0],
                entry=pairs[0],
                form=ORDERED_PAIR,
                evidence=evidence,
            ),
            "",
        )
    # Several ordered pairs. `parts` is the top-level family; that each part is
    # itself a pair is a composition, and `ANSWER_CAPABILITIES.md` records it
    # as one rather than pretending the two are the same shape.
    return (
        ExactSolution(
            display=", ".join(pairs), parts=pairs, form=PARTS, evidence=evidence
        ),
        "",
    )


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
            return ExactSolution(
                display="Non-Polynomial", entry="Non-Polynomial", form=CHOICE
            )
        return ExactSolution(display="Polynomial", entry="Polynomial", form=CHOICE)
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
        return ExactSolution(
            display="Not a Real Number", entry="Not a Real Number", form=CHOICE
        )
    value = sympy.real_root(radicand, index)
    exact = sympy.nsimplify(value, rational=True)
    if not exact.is_rational:
        return None  # an irrational value is not what this question asks for
    return ExactSolution(display=str(exact), entry=str(exact))


def _as_published_choice(classification: str, choices: list[str]) -> ExactSolution:
    """One solution-set classification, as the choice the page published.

    Only ever reached with choices in hand. The algebra above stays in its own
    vocabulary -- "Infinite Solutions" is what reducing both sides established,
    and it is what a caller with no choice surface still receives -- and this
    is where that fact is said in the words of the page that asked for it.

    Refused rather than approximated, and never fallen through: a choice
    question whose alternatives cannot state this answer is not one a model
    should be asked, because it would have to guess which alternatives the page
    offered before it could name one of them. That is the same argument
    `solve_quadrant` makes and the same disposal it gets.
    """
    try:
        published = select_solution_kind(classification, choices)
    except SolutionKindRefused as refusal:
        raise ExactlyRefused(str(refusal)) from refusal
    return ExactSolution(
        display=published,
        # Already exactly what belongs in the answer: the page's own words for
        # one of its own choices, which is what selects the control.
        entry=published,
        entry_mode="verbatim",
        form=CHOICE,
    )


def solve_one_equation(
    instruction: str, expressions: list[str], choices: list[str] | None = None
) -> ExactSolution | None:
    """Map one exact equation result onto the structured answer model.

    `choices` are the alternatives a page published, when the question is asked
    by choosing between them. Where they are given, a classification is
    returned as one of them; where they are not, nothing about this function
    changes -- which is what keeps the revealed-textbox lifecycle below intact,
    since a revealed box is a field and publishes no choices at all.
    """
    if _requested_operation(instruction) != "solve" or len(expressions) != 1:
        return None
    target = _requested_variable(instruction)
    result = solve_equation(expressions[0], variable=target)
    if result is None:
        return None
    published = list(choices or [])
    # A question asked by choosing is answered by choosing, whatever the algebra
    # underneath it. `entry` on every branch below is a value to type, and this
    # surface has no box to type one into: the page is showing its alternatives,
    # and the answer is whichever of them states what the algebra established.
    #
    # "One Solution" keeps its value; it just is not entered here. Hawkes
    # reveals a textbox once that choice is made, and that read is a field --
    # one drawn box, no group, no choices published -- so it arrives with
    # `choices` empty and takes the branches below exactly as it always has.
    if published:
        classification = getattr(result, "classification", None)
        if classification is None:
            raise ExactlyRefused(
                "this question published alternatives to choose between and its "
                "answer is a value rather than one of them"
            )
        return _as_published_choice(classification, published)
    if isinstance(result, AbsoluteValueEquationResult):
        # A solved value is mathematics and a classification is a phrase, so
        # they cannot share a mode. `verbatim` means "type exactly this", and
        # `_display` writes a rational as `\frac{3}{2}` -- which nobody types.
        if len(result.solutions) == 1:
            return ExactSolution(
                display=result.display_text,
                entry=result.solutions[0],
                entry_mode="math",
            )
        if result.solutions:
            # Two roots are two answers, as they are for a polynomial below.
            # "Two Solutions" was once published here, and a page taking the
            # roots could be given neither (audit F08); a count of solutions is
            # an answer only where it is a published choice, answered above.
            return ExactSolution(
                display=result.display_text,
                parts=tuple(result.solutions),
                entry_mode="math",
                form=PARTS,
            )
        return ExactSolution(display=result.display_text, entry=result.classification)
    if isinstance(result, LinearEquationResult):
        if result.solution is None:
            return ExactSolution(
                display=result.classification,
                entry=result.classification,
                form=CHOICE,
            )
        if target is not None:
            return ExactSolution(
                display=f"{result.variable} = {result.solution}",
                entry=result.solution,
                entry_mode="math",
            )
        # Same value, same solver: the branch above already says `math`, and
        # this one differs only in whether the question named its variable.
        return ExactSolution(
            display=(
                f"{result.classification} ({result.variable} = {result.solution})"
            ),
            entry=result.solution,
            entry_mode="math",
        )
    if not result.solutions:
        return ExactSolution(
            display=result.classification, entry=result.classification, form=CHOICE
        )
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
        form=PARTS,
    )
