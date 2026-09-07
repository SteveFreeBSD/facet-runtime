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
from typing import Any, Literal

import sympy

from facet_runtime.exact import (
    EntryMode,
    ExactSolution,
    extract_final_math,
    solve_exact,
)
from facet_runtime.graph import (
    PARABOLA_PLAN,
    QUADRATIC_REGRESSION,
    GraphContext,
    PlanRefused,
    Point,
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

Route = Literal["exact", "reasoning"]

#: What a consumer may ask to get back. `value` is an answer to write down;
#: the other two are *plans* -- a proposal a consumer will prove for itself
#: before it draws anything. Growing this set is a protocol change.
VALUE = "value"
RESULT_KINDS: tuple[str, ...] = (VALUE, PARABOLA_PLAN, QUADRATIC_REGRESSION)

#: Which problem fields belong to which requested result. A field that means
#: nothing to the kind being asked for is refused rather than ignored: it is a
#: question about something else, not a question Facet half-understands.
PROBLEM_FIELDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # A value question is about written expressions or about measured points.
    # Which of the two is required is decided in `parse_problem`, because it is
    # a rule about the pair rather than about either field.
    VALUE: (("instruction",), ("expressions", "points", "answer_parts", "label")),
    PARABOLA_PLAN: (("instruction", "expressions", "graph"), ("label",)),
    QUADRATIC_REGRESSION: (("instruction", "points"), ("label",)),
}

#: `PART 1: ...` from a multi-part reasoning reply. Structured on purpose: the
#: alternative is recovering mathematical boundaries out of display prose,
#: which is exactly the reparsing this exists to avoid.
PART_LINE = re.compile(r"(?im)^\s*PART\s+(\d+)\s*:\s*(.+?)\s*$")

#: The variable a formula question isolates, which whoever asked has usually
#: already written down as `r =`. Read out of the question, so the reasoning
#: route is told to answer with the value alone rather than repeating it.
ANSWER_PREFIX = re.compile(r"\bsolve\s+for\s+([A-Za-z])\b", re.IGNORECASE)


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
    written = kind == PARABOLA_PLAN or (kind == VALUE and "points" not in payload)
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
        graph = parse_graph_context(payload["graph"]) if kind == PARABOLA_PLAN else None
        points = (
            parse_points(payload["points"])
            if kind == QUADRATIC_REGRESSION or "points" in payload
            else ()
        )
    except PlanRefused as error:
        raise SolveRefused("invalid_request", str(error)) from error
    return MathProblem(
        instruction=instruction,
        expressions=tuple(expressions),
        answer_parts=parts,
        label=label,
        result_kind=kind,
        graph=graph,
        points=points,
    )


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
        contract = (
            f"This question takes {parts} separate answers.\n"
            f"Reply with exactly {parts + 1} labelled lines and nothing else, "
            "and keep every label exactly as written here:\n"
            "FINAL ANSWER: all answers as they would ordinarily be written\n"
            + "".join(
                f"PART {index}: answer number {index} by itself\n"
                for index in range(1, parts + 1)
            )
            + "Every line must begin with its own label, including each PART "
            "line. A PART line holds only one of those answers: no label "
            "repeated inside it, no variable name, no equals sign, no "
            '"or", no explanation.'
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
    display: str, entry: str, parts: tuple[str, ...], entry_mode: EntryMode
) -> dict[str, Any]:
    """One answer to write down, tagged with the kind it is."""
    return {
        "kind": VALUE,
        "display": display,
        "entry": entry,
        "parts": list(parts),
        "entry_mode": entry_mode,
    }


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
            solution.display, solution.entry, solution.parts, solution.entry_mode
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
    return {
        "route": "reasoning",
        "answer": _answer(final_math, entry, parts, "auto"),
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
            "evidence": dict(run.evidence),
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


def solve_math(problem: MathProblem, *, reason) -> dict[str, Any]:
    """Route one question, run it, and return the structured result.

    `reason(prompt)` executes the reasoning route under whatever constraints
    the caller established, and is only called when the deterministic stage
    declines -- or, for a specialist, when there was no deterministic stage to
    decline in the first place.
    """
    started = time.perf_counter()
    if problem.result_kind != VALUE:
        ask, *_ = SPECIALISTS[problem.result_kind]
        run = reason(ask(problem))
        return _specialist_result(
            problem, run, round((time.perf_counter() - started) * 1000, 3)
        )
    solution, decline = solve_exact(
        problem.instruction,
        list(problem.expressions),
        points=[(point.x, point.y) for point in problem.points],
        answer_parts=problem.answer_parts,
    )
    if solution is not None:
        return _exact_result(solution, round((time.perf_counter() - started) * 1000, 3))

    run = reason(reasoning_prompt(problem))
    return _reasoning_result(
        problem, run, decline, round((time.perf_counter() - started) * 1000, 3)
    )
