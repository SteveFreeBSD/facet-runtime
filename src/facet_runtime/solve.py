"""Facet's solver router: exact mathematics first, reasoning only for the rest.

A consumer that has a *question* rather than a prompt asks for `solve_math`. It
hands over the instruction in words, the exact expressions the question is
about, and how many separate values its answer takes -- and nothing else. Facet
then decides how the question gets answered.

That decision is the point of this module, and it is Facet's alone to make. The
deterministic solvers run first, because anything they settle is settled in a
millisecond, with no model, no accelerator and no network, and their answer is
checkable rather than merely plausible. Only what genuinely falls past them
reaches a reasoning model, and the result says which route ran.

A constraint like `accelerator_required` is a statement about where *model*
execution may land. An exact solve engages no backend at all, so it satisfies
any such constraint by never needing one, and it reports `actual_backend` as
null rather than claiming a processor it never used. Nothing is substituted
quietly: the route is named in every result.

What crosses back is structured. Separate answers stay separate values, and a
value carries how literally to take it. Presenting them -- keystrokes, fields,
options, a graph -- belongs to the consumer that owns the surface, and Facet
owns no surface.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import sympy

from facet_runtime.exact import (
    CHOICE,
    PARTS,
    SCALAR,
    AnswerTable,
    EntryMode,
    ExactlyRefused,
    ExactSolution,
    Relation,
    Representation,
    TableRefused,
    TableUnverifiable,
    extract_final_math,
    parse_answer_table,
    parse_representation,
    solve_exact,
    verify_completion,
)
from facet_runtime.graph import (
    LINEAR_GRAPH_PLAN,
    LINEAR_INEQUALITY_GRAPH_PLAN,
    LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN,
    MIN_REGRESSION_POINTS,
    PARABOLA_PLAN,
    POINT_PLOT_PLAN,
    QUADRATIC_REGRESSION,
    GraphContext,
    PlanRefused,
    Point,
    build_linear_graph_plan,
    build_linear_inequality_graph_plan,
    build_linear_inequality_system_graph_plan,
    build_point_plot_plan,
    parabola_prompt,
    parse_graph_context,
    parse_parabola_plan,
    parse_points,
    parse_regression_plan,
    regression_prompt,
)
from facet_runtime.result import RunResult

MAX_EXPRESSIONS = 8
MAX_EXPRESSION_CHARS = 2000
MAX_INSTRUCTION_CHARS = 4000
MAX_LABEL_CHARS = 200
MAX_ANSWER_PARTS = 5

#: The alternatives a choice question publishes, bounded. Two is the fewest
#: that is a choice at all; the ceiling is generous because these are short
#: strings and the page decides how many there are, not Facet.
MAX_ANSWER_CHOICES = 12
MAX_CHOICE_CHARS = 120

Route = Literal["exact", "reasoning"]

#: What a consumer may ask to get back. `value` is an answer to write down;
#: the other two are *plans* -- a proposal a consumer will prove for itself
#: before it draws anything. Growing this set is a protocol change.
VALUE = "value"
RESULT_KINDS: tuple[str, ...] = (
    VALUE,
    PARABOLA_PLAN,
    QUADRATIC_REGRESSION,
    POINT_PLOT_PLAN,
    LINEAR_GRAPH_PLAN,
    LINEAR_INEQUALITY_GRAPH_PLAN,
    LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN,
)

#: Which problem fields belong to which requested result. A field that means
#: nothing to the kind being asked for is refused rather than ignored: it is a
#: question about something else, not a question Facet half-understands.
PROBLEM_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # A value question is about written expressions or about measured points.
    # Which of the two is required is decided in `parse_problem`, because it is
    # a rule about the pair rather than about either field.
    VALUE: (
        ("instruction",),
        (
            "expressions",
            "points",
            "answer_parts",
            "label",
            # A question that is a grid: the values it states, the blanks it
            # asks for, and the form every answer must take. Structured, and
            # deliberately not prose -- the alternative is Facet recovering a
            # table out of a sentence it was handed, which is the reparsing
            # every seam in this file exists to avoid.
            "answer_table",
            "answer_representation",
            # The alternatives a question answered by *choosing* published.
            # Structure rather than prose, for the same reason the grid is:
            # an answer that must be one of these can be checked against them,
            # and a list flattened into the instruction can be read by a model
            # and by nothing else.
            "answer_choices",
        ),
    ),
    PARABOLA_PLAN: (("instruction", "expressions", "graph"), ("label",)),
    QUADRATIC_REGRESSION: (("instruction", "points"), ("label",)),
    # No `graph` context: a plotting plan is read from the question's own
    # words, and the live graph is what the browser proves it against.
    POINT_PLOT_PLAN: (("instruction",), ("expressions", "label")),
    LINEAR_GRAPH_PLAN: (("instruction", "expressions", "graph"), ("label",)),
    LINEAR_INEQUALITY_GRAPH_PLAN: (("instruction", "expressions", "graph"), ("label",)),
    LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN: (
        ("instruction", "expressions", "graph"),
        ("label",),
    ),
}

#: `PART 1: ...` from a multi-part reasoning reply. Structured on purpose: the
#: alternative is recovering mathematical boundaries out of display prose,
#: which is exactly the reparsing this exists to avoid.
PART_LINE = re.compile(r"(?im)^\s*PART\s+(\d+)\s*:\s*(.+?)\s*$")

#: The variable a formula question isolates, which whoever asked has usually
#: already written down as `r =`. Read out of the question, so the reasoning
#: route is told to answer with the value alone rather than repeating it.
ANSWER_PREFIX = re.compile(r"\bsolve\s+for\s+([A-Za-z])\b", re.IGNORECASE)

#: A word of English: two or more letters, and nothing else in it.
#:
#: An answer is written in mathematics. It may name itself in a few words --
#: "Not a Real Number", "All Real Numbers" -- and those named forms are short,
#: because they are names. Past a handful of words the thing being read is a
#: sentence, and a sentence arriving where an answer belongs is a reply about
#: the question rather than to it: most often this prompt's own contract,
#: echoed back by a model that pattern-completed the line it was shown instead
#: of answering.
PROSE_WORD = re.compile(r"^[A-Za-z]{2,}$")

#: How many such words an answer may contain before it is prose.
#:
#: Three, which is the longest named answer there is: "Not a Real Number" --
#: "a" is one letter and is not one of them. "Real Number" and "Not Factorable"
#: are two. Set against the things that must be refused rather than against a
#: round number: this prompt's own "answer number 1 by itself" is four, and its
#: "all answers as they would ordinarily be written" is eight.
MAX_ANSWER_WORDS = 3

#: How long an answer may be, in characters. Generous, because display notation
#: can be long -- a domain in interval notation runs to a couple of dozen -- and
#: because length is the weaker of the two tests here.
MAX_ANSWER_DISPLAY = 200


def answer_shaped(text: str) -> bool:
    """Whether this reads as an answer rather than as a sentence about one.

    A shape test, deliberately, and not a list of sentences to refuse. What
    makes prompt text recognisable is not which words it uses; it is that it is
    made of words at all, and no list of forbidden phrases survives the next
    rewording of the prompt.
    """
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_ANSWER_DISPLAY:
        return False
    words = [token for token in stripped.split() if PROSE_WORD.match(token)]
    return len(words) <= MAX_ANSWER_WORDS


class SolveRefused(Exception):
    """Facet will not answer this, and says which kind of refusal it is."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True, slots=True)
class MathProblem:
    """One question, in the only terms Facet accepts it.

    There is no page here, and there is deliberately no way to describe one.
    `answer_parts` is a requirement on the *reply* -- how many separate values
    the answer has -- which is a property of the question. Where those values
    are then typed is the consumer's business and never crosses.
    """

    instruction: str
    expressions: tuple[str, ...] = ()
    answer_parts: int = 1
    label: str = ""
    #: What the consumer asked to get back. `value` unless a specialist is
    #: wanted, and a specialist decides what the other fields mean.
    result_kind: str = VALUE
    #: Normalised geometry for a parabola plan: a grid, never a page.
    graph: GraphContext | None = None
    #: Normalised coordinates a regression is fitted to.
    points: tuple[Point, ...] = ()
    #: The grid a completion question is answered in, when it is one: stated
    #: cells and numbered blanks, in the columns the question names. Held as
    #: structure so the exact route can compute it and the verifier can prove
    #: an answer against it; neither can be done to a sentence.
    answer_table: AnswerTable | None = None
    #: The form every separate answer must take, when the question publishes
    #: one. A requirement on the reply, like `answer_parts`.
    answer_representation: Representation | None = None
    #: What this question may be answered with, when it is answered by
    #: choosing. The page's own words for its own alternatives, and the whole
    #: contract: an answer to a choice question that is not one of the choices
    #: is not an answer to it, whichever route produced it.
    answer_choices: tuple[str, ...] = ()


def _checked_kind(payload: dict[str, Any]) -> str:
    kind = payload.get("result_kind", VALUE)
    if kind not in RESULT_KINDS:
        raise SolveRefused(
            "invalid_request",
            f"result_kind must be one of {', '.join(RESULT_KINDS)}",
        )
    required, optional = PROBLEM_FIELDS[kind]
    missing = [name for name in required if name not in payload]
    unknown = sorted(set(payload) - {"result_kind", *required, *optional})
    if missing or unknown:
        detail = ", ".join(
            part
            for part in (
                f"{kind} needs {', '.join(missing)}" if missing else "",
                f"{kind} takes no {', '.join(unknown)}" if unknown else "",
            )
            if part
        )
        raise SolveRefused("invalid_request", f"problem fields are wrong: {detail}")
    return kind


def parse_problem(payload: Any) -> MathProblem:
    """Validate one problem completely, or refuse it with a reason."""
    if not isinstance(payload, dict):
        raise SolveRefused("invalid_request", "problem must be an object")
    kind = _checked_kind(payload)
    instruction = payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise SolveRefused("invalid_request", "instruction must be a non-empty string")
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        raise SolveRefused("invalid_request", "instruction exceeds the size limit")
    expressions = payload.get("expressions", [])
    # A question is about one thing. Expressions are mathematics somebody wrote
    # down; points are measurements nobody wrote a function for. Both at once
    # is two questions, and neither is none -- so a value question must carry
    # exactly one of them, and says which is missing rather than half-running.
    if kind == VALUE and ("points" in payload) == ("expressions" in payload):
        raise SolveRefused(
            "invalid_request",
            "a value question is about expressions or about points, not both "
            "and not neither",
        )
    written = kind in {
        PARABOLA_PLAN,
        LINEAR_GRAPH_PLAN,
        LINEAR_INEQUALITY_GRAPH_PLAN,
        LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN,
    } or (kind == VALUE and "points" not in payload)
    if written and (
        not isinstance(expressions, list)
        or not 1 <= len(expressions) <= MAX_EXPRESSIONS
    ):
        raise SolveRefused(
            "invalid_request",
            f"expressions must be a list of 1 to {MAX_EXPRESSIONS} strings",
        )
    for expression in expressions:
        if not isinstance(expression, str) or not expression.strip():
            raise SolveRefused("invalid_request", "every expression must be a string")
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise SolveRefused(
                "invalid_request", "an expression exceeds the size limit"
            )
    parts = payload.get("answer_parts", 1)
    # `True == 1` in Python, so the type is checked before the value.
    if (
        not isinstance(parts, int)
        or isinstance(parts, bool)
        or not 1 <= parts <= MAX_ANSWER_PARTS
    ):
        raise SolveRefused(
            "invalid_request", f"answer_parts must be 1 to {MAX_ANSWER_PARTS}"
        )
    label = payload.get("label", "")
    if not isinstance(label, str) or len(label) > MAX_LABEL_CHARS:
        raise SolveRefused("invalid_request", "label must be a short string")
    try:
        graph = (
            parse_graph_context(payload["graph"])
            if kind
            in {
                PARABOLA_PLAN,
                LINEAR_GRAPH_PLAN,
                LINEAR_INEQUALITY_GRAPH_PLAN,
                LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN,
            }
            else None
        )
        points = (
            parse_points(
                payload["points"],
                minimum=MIN_REGRESSION_POINTS if kind == QUADRATIC_REGRESSION else 1,
            )
            if kind == QUADRATIC_REGRESSION or "points" in payload
            else ()
        )
    except PlanRefused as error:
        raise SolveRefused("invalid_request", str(error)) from error
    try:
        table = (
            parse_answer_table(payload["answer_table"])
            if "answer_table" in payload
            else None
        )
        representation = (
            parse_representation(payload["answer_representation"])
            if "answer_representation" in payload
            else None
        )
    except TableRefused as error:
        raise SolveRefused("invalid_request", str(error)) from error
    # A grid with a different number of blanks than the answer has parts is two
    # descriptions of one question that do not agree, and there is no reading
    # of it that is not a guess about which one to believe.
    if table is not None and table.blanks != parts:
        raise SolveRefused(
            "invalid_request",
            f"answer_table has {table.blanks} blanks and answer_parts is {parts}",
        )
    choices = parse_choices(payload.get("answer_choices", []))
    # A question answered by choosing has one answer: the choice. Several
    # alternatives are not several answers, and reading them as one is exactly
    # the fault this field exists to end -- a five-option radio group crossed
    # as a five-part question, and a model was asked for five values to a
    # question with one.
    if choices and parts != 1:
        raise SolveRefused(
            "invalid_request",
            f"a question answered by choosing has one answer, not {parts}",
        )
    return MathProblem(
        instruction=instruction,
        expressions=tuple(expressions),
        answer_parts=parts,
        label=label,
        result_kind=kind,
        graph=graph,
        points=points,
        answer_table=table,
        answer_representation=representation,
        answer_choices=choices,
    )


def parse_choices(payload: Any) -> tuple[str, ...]:
    """The alternatives a choice question published, or refuse them.

    Distinct, because two identical choices make "the answer is this one"
    unanswerable, and non-empty because a choice with no words cannot be an
    answer anybody could check.
    """
    if not isinstance(payload, list):
        raise SolveRefused("invalid_request", "answer_choices must be a list")
    if not payload:
        return ()
    if not 2 <= len(payload) <= MAX_ANSWER_CHOICES:
        raise SolveRefused(
            "invalid_request",
            f"answer_choices must hold 2 to {MAX_ANSWER_CHOICES} alternatives",
        )
    for choice in payload:
        if not isinstance(choice, str) or not choice.strip():
            raise SolveRefused("invalid_request", "every choice must be a string")
        if len(choice) > MAX_CHOICE_CHARS:
            raise SolveRefused("invalid_request", "a choice exceeds the size limit")
    choices = tuple(choice.strip() for choice in payload)
    if len(set(choices)) != len(choices):
        raise SolveRefused("invalid_request", "answer_choices must be distinct")
    return choices


def answer_prefix(instruction: str) -> str:
    """The variable a formula question isolates, or an empty string."""
    match = ANSWER_PREFIX.search(instruction)
    return match.group(1) if match else ""


def reasoning_prompt(problem: MathProblem) -> str:
    """State the question in the exact terms Facet was handed.

    Every expression here arrived exactly, so nothing is a transcription and
    none of it needed a picture. The question's own label is context rather
    than instruction: it is sometimes the only thing that distinguishes one
    step of a problem from the next.

    The answer shape is stated as a requirement on the reply, never as a
    description of a page. The model is told how many values to produce and
    what a value may contain; it is told nothing about fields, editors, or
    where an answer is going, because none of that is its to reason about.
    """
    rendered = "\n".join(
        f"- {expression}" for expression in problem.expressions
    ) or "\n".join(f"- ({point.x}, {point.y})" for point in problem.points)
    # The grid, written out for a reader that has only words. It arrives as
    # structure and stays that way for the exact route and for the verifier;
    # this is the one place it becomes prose, and it becomes prose here rather
    # than before it crossed, so nothing downstream has to read it back.
    #
    # The requirement on the *form* of each answer is not repeated here. It
    # already reaches this prompt inside the instruction, stated by whoever
    # owns the answer surface, and saying it twice in two wordings is how a
    # model ends up arbitrating between them.
    grid = ""
    if problem.answer_table is not None:
        table = problem.answer_table
        lines = [" | ".join(table.columns)]
        for row in table.rows:
            lines.append(
                " | ".join(
                    f"(part {cell.blank})" if cell.blank is not None else cell.value
                    for cell in row
                )
            )
        grid = (
            "This question states the table below and is answered by completing "
            "it. Each blank is written as the numbered answer part that belongs "
            "in it.\n" + "\n".join(lines) + "\n"
        )
    # The alternatives, when the question is answered by choosing one. Listed
    # as they were published and required back verbatim, because that is what
    # the consumer will select by. A model that writes its own wording for the
    # right choice has answered the mathematics and not the question.
    alternatives = ""
    if problem.answer_choices:
        alternatives = (
            "This question is answered by choosing one of these, and nothing "
            "else:\n"
            + "".join(f"- {choice}\n" for choice in problem.answer_choices)
            + "Reply with exactly one of them, copied word for word.\n"
        )
    heading = f"Question: {problem.label.strip()}\n" if problem.label.strip() else ""
    prefix = answer_prefix(problem.instruction)
    # Whoever asked has already written the variable and the equals sign, so
    # answering with them again would repeat what is there. Deliberately one
    # sentence for every answer count: a question can name the variable *and*
    # take several answers -- solving for x with two roots does both -- and a
    # singular "the value" beside a contract asking for two of them is the kind
    # of self-contradiction this model spends its whole budget arbitrating.
    labelled = (
        f"`{prefix} =` is already written for you, so give only what follows "
        "it in each answer.\n"
        if prefix
        else ""
    )
    parts = problem.answer_parts
    if parts > 1:
        # The labels are listed bare, and what belongs after each of them is
        # said afterwards, in prose that is plainly about the reply rather than
        # in it.
        #
        # They used to be listed as filled-in examples -- "FINAL ANSWER: all
        # answers as they would ordinarily be written" -- and a model that
        # pattern-completes the shape it is shown copies the line whole. That
        # exact sentence reached a live answer card as the answer. Nothing here
        # is the defence against it (`answer_shaped` is), but a contract that
        # cannot be mistaken for a worked example is one fewer thing to echo,
        # and a label that ends at its colon carries no text worth copying.
        contract = (
            f"This question takes {parts} separate answers.\n"
            f"Reply with exactly {parts + 1} lines and nothing else. Each line "
            "begins with one of these labels, in this order, spelled exactly "
            "as shown:\n"
            "FINAL ANSWER:\n"
            + "".join(f"PART {index}:\n" for index in range(1, parts + 1))
            + "After FINAL ANSWER: write the answers as they would ordinarily "
            "be written together. After each PART label write that one answer "
            "and nothing else: no label repeated inside it, no variable name, "
            'no equals sign, no "or", no explanation. Write answers, never a '
            "description of what to write."
        )
    else:
        contract = (
            "Your entire response must be one line beginning with the exact words "
            "FINAL ANSWER: followed by only the answer itself. "
            "Do not repeat the input expression or output an equals sign. Never output "
            "angle brackets or a trailing period. Do not explain."
        )
    return (
        "Solve this precalculus question.\n"
        f"{heading}"
        f"Instruction: {problem.instruction}\n"
        f"Expression(s):\n{rendered}\n"
        f"{grid}"
        f"{alternatives}"
        f"{labelled}"
        f"{contract}"
    )


def labelled_parts(text: str, expected: int) -> list[str] | None:
    """Read the `PART n:` lines of a multi-part reply, or refuse.

    Returns None unless the reply carries exactly the parts that were asked
    for, numbered from one and in order. A reply that produced a different
    count did not answer the question that was asked -- it answered a
    differently shaped one -- and an answer of the wrong shape is worse than no
    answer, because a consumer would place it into real answer boxes.
    """
    found = PART_LINE.findall(text)
    if len(found) != expected:
        return None
    if [index for index, _ in found] != [str(n) for n in range(1, expected + 1)]:
        return None
    values = [value.strip() for _, value in found]
    return values if all(values) else None


def _answer(
    display: str,
    entry: str,
    parts: tuple[str, ...],
    entry_mode: EntryMode,
    form: str = SCALAR,
    relation: Relation | None = None,
    intercepts=None,
    choice: str = "",
    inequality_pair=None,
) -> dict[str, Any]:
    """One answer to write down, tagged with the kind it is.

    `form` says which *family* the answer belongs to -- a scalar, an ordered
    pair, several separate values, a chosen alternative, an equation -- beside
    `entry_mode`, which says how literally to take it. A consumer that has to
    enter the answer into a real surface needs the family before it can say
    whether it has a path for it at all, and until this existed it had to
    recover it by parsing the string. Additive and optional: a reader that does
    not know the field reads the same answer it always did.

    `relation` is the one family whose answer is not a single value: an
    equation has two sides, and which of them a consumer types depends on what
    its own answer surface already states. Both are carried so that neither has
    to be recovered by splitting `entry` -- the same rule `parts` follows, and
    for the same reason.
    """
    answer = {
        "kind": VALUE,
        "display": display,
        "entry": entry,
        "parts": list(parts),
        "entry_mode": entry_mode,
        "form": form,
        "relation": (
            None
            if relation is None
            else {"subject": relation.subject, "value": relation.value}
        ),
        "intercepts": (
            None
            if intercepts is None
            else {
                "x": None
                if intercepts.x is None
                else {"x": intercepts.x.x, "y": intercepts.x.y},
                "y": None
                if intercepts.y is None
                else {"x": intercepts.y.x, "y": intercepts.y.y},
            }
        ),
    }
    if choice:
        answer["choice"] = choice
    if inequality_pair is not None:
        answer["inequality_pair"] = {
            "left": {
                "left": inequality_pair.left.left,
                "relation": inequality_pair.left.relation,
                "right": inequality_pair.left.right,
            },
            "connector": inequality_pair.connector,
            "right": {
                "left": inequality_pair.right.left,
                "relation": inequality_pair.right.relation,
                "right": inequality_pair.right.right,
            },
        }
    return answer


def _plan_answer(kind: str, plan: dict[str, Any]) -> dict[str, Any]:
    """One proposed plan, carrying no value at all.

    A plan result deliberately has no `display`, `entry` or `parts`. There is
    nothing here for a consumer to write into an answer box, and leaving room
    for one would let a reasoned proposal arrive shaped like a settled answer.
    """
    return {"kind": kind, "plan": plan}


def _exact_result(solution: ExactSolution, elapsed_ms: float) -> dict[str, Any]:
    return {
        "route": "exact",
        "answer": _answer(
            solution.display,
            solution.entry,
            solution.parts,
            solution.entry_mode,
            solution.form,
            solution.relation,
            solution.intercepts,
            solution.choice,
            solution.inequality_pair,
        ),
        "provenance": {
            # The identity Facet answers to when it computed the answer itself.
            "source": "Facet Exact",
            "method": solution.method,
            "router": "solved",
            "router_detail": "",
            "runtime": f"SymPy {sympy.__version__}",
            "model": None,
            "device": None,
            "requested_backend": None,
            "actual_backend": None,
            "elapsed_ms": elapsed_ms,
            "fallback": False,
            "metrics": {},
            # The proof that no model took part, in the same place a model run
            # proves where it ran.
            "evidence": {
                "source": "facet exact solver",
                "model_calls": 0,
                # The working, when the solver produced one. Nested so it can
                # never shadow the two claims above it, which are Facet's
                # account of itself rather than of the mathematics.
                **({"computation": solution.evidence} if solution.evidence else {}),
            },
        },
    }


def _held_to_the_choices(problem: MathProblem, answer: str) -> None:
    """Refuse an answer to a choice question that is not one of the choices.

    Exactly one of them, character for character after trimming. Not "close
    to", not "contains": the consumer selects a control by the words the page
    published, and an answer that merely resembles one of them selects nothing.
    """
    if not problem.answer_choices:
        return
    if answer.strip() not in problem.answer_choices:
        raise SolveRefused(
            "unusable_result",
            "this question is answered by choosing one of the alternatives it "
            "published, and the answer is not one of them",
        )


def _renders(display: str, values: tuple[str, ...] | list[str]) -> bool:
    """Whether one line could be these separate answers, written together.

    Not an equality: how several answers are ordinarily written together is the
    model's to decide -- "x = -3 or x = 3", "(1, 2) and (3, 4)" -- and pinning a
    format here would refuse correct renderings. What a rendering cannot do is
    fail to contain the answers it renders.
    """
    return all(value.strip() and value.strip() in display for value in values)


_FINITE_DECIMAL = re.compile(r"[+-]?(?:\d+\.\d+|\.\d+)")


def _fraction_requested(instruction: str) -> bool:
    """Whether the question explicitly asks for a reduced fraction form."""
    lowered = instruction.lower()
    return (
        "fraction" in lowered
        and ("lowest terms" in lowered or "reduced fraction" in lowered)
        and "round" not in lowered
    )


def _reduced_fraction_if_decimal(value: str) -> str:
    """Preserve a reasoner's stated finite decimal value in requested form."""
    if _FINITE_DECIMAL.fullmatch(value.strip()) is None:
        return value
    rational = Fraction(value.strip())
    return str(rational)


def _reasoning_result(
    problem: MathProblem, run: RunResult, decline: str, elapsed_ms: float
) -> dict[str, Any]:
    final_math = extract_final_math(run.text)
    if not final_math:
        raise SolveRefused(
            "unusable_result", "the reasoning route returned no FINAL ANSWER"
        )
    entry, parts = final_math, ()
    if problem.answer_parts > 1:
        values = labelled_parts(run.text, problem.answer_parts)
        if values is None:
            # Fail closed. The question needs a known number of values and this
            # reply does not carry them, so nothing here may become an answer.
            raise SolveRefused(
                "unusable_result",
                f"the reasoning route did not return the {problem.answer_parts} "
                "separate answers this question needs",
            )
        # A multi-part answer has no single string that could be typed into
        # several separate boxes, so there is no single entry.
        entry, parts = "", tuple(values)
        # The FINAL ANSWER line of a multi-part reply is only ever a *rendering*
        # of the parts: the same answers written the way they would ordinarily
        # be written together. The parts are the answer, and they have been
        # checked; the rendering has not, and it is the one field on this route
        # that a model can fill with anything at all.
        #
        # It reached a live answer card as "all answers as they would ordinarily
        # be written" -- this prompt's own description of the line, pattern-
        # completed instead of answered. So it is not taken on trust: a line
        # that neither reads as an answer nor accounts for the parts it claims
        # to render is replaced by the parts themselves, which is what it was
        # supposed to say. Rebuilt rather than refused, because the answer is
        # known and only its presentation was lost.
        # A PART line can be echoed exactly as the FINAL ANSWER line can, and
        # nothing above this reads what a part says -- only that there are the
        # right number of them, numbered in order and not empty. These are the
        # values that would be typed into real answer boxes.
        unshaped = [value for value in values if not answer_shaped(value)]
        if unshaped:
            raise SolveRefused(
                "unusable_result",
                f"the reasoning route answered part {values.index(unshaped[0]) + 1} "
                "with prose rather than with an answer",
            )
        if not answer_shaped(final_math) or not _renders(final_math, values):
            final_math = ", ".join(values)
    # A question answered by choosing is answered with one of its own choices,
    # whichever route produced it. A model handed free text here would be
    # offering an answer the page has no way to accept.
    _held_to_the_choices(problem, final_math)
    # Whatever the count, what leaves here is an answer or it is nothing. On
    # the single-value route `final_math` is both the answer and its rendering
    # and there is no checked part list to rebuild it from, so a sentence is
    # refused outright rather than published as a value.
    if not answer_shaped(final_math):
        raise SolveRefused(
            "unusable_result",
            "the reasoning route answered with prose rather than with an answer",
        )
    if _fraction_requested(problem.instruction) and not problem.answer_choices:
        if parts:
            rendered = tuple(_reduced_fraction_if_decimal(value) for value in parts)
            if rendered != parts:
                parts = rendered
                final_math = ", ".join(parts)
        else:
            entry = final_math = _reduced_fraction_if_decimal(final_math)
    # A reasoned answer to a grid is proved against that grid before it becomes
    # an answer. The reasoning route is stochastic and the shape of its reply is
    # not evidence about the mathematics in it: five parts arriving is not five
    # parts being right, and a live run returned five wrong ones. Where the
    # question carries enough structure to check, checking is not optional.
    #
    # Fail closed, and no retry. Asking again until a reply passes would turn a
    # verifier into a filter on repeated guessing, and the run would report an
    # answer with no account of how many were thrown away to get it.
    checked = "not-applicable"
    if problem.answer_table is not None:
        try:
            verify_completion(
                parts or (entry,),
                problem.expressions,
                problem.answer_table,
                representation=problem.answer_representation,
            )
            checked = "verified"
        except TableUnverifiable as error:
            # The grid could not be read as mathematics, so nothing about this
            # answer was checked -- which is a different claim from the answer
            # being wrong, and refusing on it would be refusing on the strength
            # of a check that never ran. Said out loud rather than assumed.
            checked = f"not checkable: {error}"
        except TableRefused as error:
            raise SolveRefused(
                "unusable_result",
                f"the reasoning route's answer does not complete the table: {error}",
            ) from error
    return {
        "route": "reasoning",
        # The reasoning route's family is read off the shape it produced --
        # several checked parts, one chosen alternative, or one written value.
        # It never claims `ordered-pair`: no reasoned answer is held to that
        # structure, and claiming a family nothing verified would be worse than
        # claiming none.
        "answer": _answer(
            final_math,
            entry,
            parts,
            "auto",
            CHOICE if problem.answer_choices else (PARTS if parts else SCALAR),
        ),
        "provenance": {
            # Where the work ran is reported, not assumed.
            "source": f"Facet Reasoning · {run.actual_backend.upper()}",
            "method": run.model,
            "router": "declined",
            # Which gap the deterministic stage fell through, in its own words.
            "router_detail": decline,
            "runtime": run.runtime,
            "model": run.model,
            "device": run.device,
            "requested_backend": run.requested_backend,
            "actual_backend": run.actual_backend,
            "elapsed_ms": elapsed_ms,
            "fallback": run.fallback,
            "metrics": run.to_dict()["metrics"],
            # Whether the grid this answer claims to complete was checked, and
            # when it was not, why not. A reader must never have to work out
            # whether a reasoned answer was proved or merely counted.
            "evidence": {**dict(run.evidence), "answer_table": checked},
        },
    }


#: Each specialist: how it asks, how it reads a reply, what to call it, and
#: why the deterministic stage was not the one that ran.
SPECIALISTS: dict[str, tuple[Any, Any, str, str]] = {
    PARABOLA_PLAN: (
        lambda problem: parabola_prompt(
            problem.instruction, problem.expressions, problem.graph
        ),
        parse_parabola_plan,
        "Parabola Plan",
        "a graph plan has no deterministic route",
    ),
    QUADRATIC_REGRESSION: (
        lambda problem: regression_prompt(problem.instruction, problem.points),
        parse_regression_plan,
        "Quadratic Regression",
        "a quadratic regression has no deterministic route",
    ),
}


def _specialist_result(
    problem: MathProblem, run: RunResult, elapsed_ms: float
) -> dict[str, Any]:
    _, read_reply, identity, detail = SPECIALISTS[problem.result_kind]
    try:
        plan = read_reply(run.text)
    except PlanRefused as error:
        # Fail closed. A plan that does not match its schema exactly is not a
        # plan a consumer may go on to prove; it is a reply about something
        # else, and repairing it here would be inventing geometry.
        raise SolveRefused("unusable_result", str(error)) from error
    return {
        "route": "reasoning",
        "answer": _plan_answer(problem.result_kind, plan),
        "provenance": {
            # Identifiable as the specialist it was, on the processor that ran
            # it. Which specialist answered is not a detail a reader can infer
            # from a model name.
            "source": f"Facet {identity} · {run.actual_backend.upper()}",
            "method": run.model,
            # The deterministic solvers answer expressions, not geometry, so
            # they were not asked. Saying they declined would be a claim they
            # were tried, and saying they solved it would be a lie.
            "router": "not-run",
            "router_detail": detail,
            "runtime": run.runtime,
            "model": run.model,
            "device": run.device,
            "requested_backend": run.requested_backend,
            "actual_backend": run.actual_backend,
            "elapsed_ms": elapsed_ms,
            "fallback": run.fallback,
            "metrics": run.to_dict()["metrics"],
            "evidence": dict(run.evidence),
        },
    }


def _solve_exactly(problem: MathProblem):
    """The deterministic stage, given exactly what it is allowed to know."""
    return solve_exact(
        problem.instruction,
        list(problem.expressions),
        points=[(point.x, point.y) for point in problem.points],
        answer_parts=problem.answer_parts,
        table=problem.answer_table,
        representation=problem.answer_representation,
        choices=list(problem.answer_choices),
    )


def solve_math(problem: MathProblem, *, reason) -> dict[str, Any]:
    """Route one question, run it, and return the structured result.

    `reason(prompt)` executes the reasoning route under whatever constraints
    the caller established, and is only called when the deterministic stage
    declines -- or, for a specialist, when there was no deterministic stage to
    decline in the first place.
    """
    started = time.perf_counter()
    # A plotting question states its own answer: the pairs are written down, so
    # there is nothing for a model to work out and asking one would be
    # inventing uncertainty. Deterministic, and no backend is engaged at all.
    if problem.result_kind == POINT_PLOT_PLAN:
        try:
            plan = build_point_plot_plan(problem.instruction, list(problem.expressions))
        except PlanRefused as error:
            raise SolveRefused("unusable_result", str(error)) from error
        return {
            "route": "exact",
            "answer": _plan_answer(problem.result_kind, plan),
            "provenance": {
                "source": "Facet Exact",
                "method": "stated points read from the question",
                "router": "solved",
                "router_detail": "",
                "runtime": f"SymPy {sympy.__version__}",
                "model": None,
                "device": None,
                "requested_backend": None,
                "actual_backend": None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "fallback": False,
                "metrics": {},
                "evidence": {"source": "facet exact solver", "model_calls": 0},
            },
        }
    if problem.result_kind == LINEAR_GRAPH_PLAN:
        try:
            assert problem.graph is not None
            plan = build_linear_graph_plan(
                problem.instruction, list(problem.expressions), problem.graph
            )
        except PlanRefused as error:
            raise SolveRefused("unusable_result", str(error)) from error
        return {
            "route": "exact",
            "answer": _plan_answer(problem.result_kind, plan),
            "provenance": {
                "source": "Facet Exact",
                "method": "SymPy exact linear graph",
                "router": "solved",
                "router_detail": "",
                "runtime": f"SymPy {sympy.__version__}",
                "model": None,
                "device": None,
                "requested_backend": None,
                "actual_backend": None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "fallback": False,
                "metrics": {},
                "evidence": {"source": "facet exact solver", "model_calls": 0},
            },
        }
    if problem.result_kind == LINEAR_INEQUALITY_GRAPH_PLAN:
        try:
            assert problem.graph is not None
            plan = build_linear_inequality_graph_plan(
                problem.instruction, list(problem.expressions), problem.graph
            )
        except PlanRefused as error:
            raise SolveRefused("unusable_result", str(error)) from error
        return {
            "route": "exact",
            "answer": _plan_answer(problem.result_kind, plan),
            "provenance": {
                "source": "Facet Exact",
                "method": "SymPy exact linear inequality graph",
                "router": "solved",
                "router_detail": "",
                "runtime": f"SymPy {sympy.__version__}",
                "model": None,
                "device": None,
                "requested_backend": None,
                "actual_backend": None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "fallback": False,
                "metrics": {},
                "evidence": {"source": "facet exact solver", "model_calls": 0},
            },
        }
    if problem.result_kind == LINEAR_INEQUALITY_SYSTEM_GRAPH_PLAN:
        try:
            assert problem.graph is not None
            plan = build_linear_inequality_system_graph_plan(
                problem.instruction, list(problem.expressions), problem.graph
            )
        except PlanRefused as error:
            raise SolveRefused("unusable_result", str(error)) from error
        return {
            "route": "exact",
            "answer": _plan_answer(problem.result_kind, plan),
            "provenance": {
                "source": "Facet Exact",
                "method": "SymPy exact linear inequality system",
                "router": "solved",
                "router_detail": "",
                "runtime": f"SymPy {sympy.__version__}",
                "model": None,
                "device": None,
                "requested_backend": None,
                "actual_backend": None,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
                "fallback": False,
                "metrics": {},
                "evidence": {"source": "facet exact solver", "model_calls": 0},
            },
        }
    if problem.result_kind != VALUE:
        ask, *_ = SPECIALISTS[problem.result_kind]
        run = reason(ask(problem))
        return _specialist_result(
            problem, run, round((time.perf_counter() - started) * 1000, 3)
        )
    try:
        solution, decline = _solve_exactly(problem)
    except ExactlyRefused as refusal:
        # The deterministic stage claimed this question and could not answer it
        # as asked. That is a refusal, not a decline: there is no reasoning
        # route for a question whose answer is one of a set of alternatives a
        # model was never shown.
        raise SolveRefused("unusable_result", str(refusal)) from refusal
    if solution is not None:
        # An exact answer to a choice question is one of its choices by
        # construction, and is checked anyway: this is the one claim about such
        # an answer a consumer can verify, so it is verified before it leaves.
        _held_to_the_choices(problem, solution.choice or solution.display)
        return _exact_result(solution, round((time.perf_counter() - started) * 1000, 3))

    run = reason(reasoning_prompt(problem))
    return _reasoning_result(
        problem, run, decline, round((time.perf_counter() - started) * 1000, 3)
    )
