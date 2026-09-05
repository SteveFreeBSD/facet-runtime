"""A live check that Facet's own prompts still answer, against the real model.

Every prompt a model ever sees is built in this repository, by three
constructors reached from `solve_math`. Nothing exercised them end to end: the
tests assert substrings against fake adapters, and the throughput benchmark
sends prompts of its own invention. A prompt could therefore stop working
against its assigned model without a single test noticing, which is precisely
what happened to the quadratic regression specialist -- a question this
repository documents answering correctly began returning nothing, and no gate
here could see it.

This is the smallest thing that would have caught it. Each case is a real
`MathProblem` answered by the real `solve_math`: the same routing, the same
constructors, the same strict parsers, the same adapters. Nothing here writes a
prompt of its own. A case that reaches a model reaches it with the exact bytes
production would have sent, because the recorder is simply the `reason`
callable that `solve_math` already takes.

Two things it deliberately does not do. It never repairs a reply, and it never
relaxes a parser to make a case pass -- a case that fails is reported as
failing, with the runtime's own reason. And it pins an expected answer only
where the answer is genuinely determined; a question with many correct
spellings reports what it got instead, because a harness that fails on a right
answer teaches nobody anything.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from facet_runtime.adapters.base import BackendAdapter
from facet_runtime.errors import FacetRuntimeError
from facet_runtime.graph import PARABOLA_PLAN, QUADRATIC_REGRESSION, GraphContext, Point
from facet_runtime.result import BackendName, RunResult
from facet_runtime.runtime import default_adapters, run_prompt
from facet_runtime.solve import MathProblem, SolveRefused, solve_math

#: Vocabulary that belongs to a consumer rather than to a question. A model
#: told which product is asking, what its reply will be drawn on, or who will
#: check it afterwards has been handed something it cannot act on -- and it may
#: reason about it instead. That is not a style rule here: one such phrase told
#: the regression specialist its exact coefficients would be "rounded for
#: display" while the schema demanded exact ones, and the model spent its whole
#: output budget deciding which to obey and answered nothing.
#:
#: Matched on word boundaries, so `domain` is a domain and not a DOM.
CONSUMER_VOCABULARY: tuple[str, ...] = (
    "hawkes",
    "ethnos",
    "svg",
    "browser",
    "page",
    "box",
    "boxes",
    "field",
    "editor",
    "selector",
    "screenshot",
    "dom",
    "url",
    "frame",
    "tab",
    "click",
    "press",
    "submit",
    "button",
)

_CONSUMER_WORDS = re.compile(
    r"\b(?:" + "|".join(CONSUMER_VOCABULARY) + r")\b", re.IGNORECASE
)


def leaks(prompt: str) -> tuple[str, ...]:
    """Any consumer's vocabulary in this prompt, which should always be none."""
    return tuple(sorted({found.lower() for found in _CONSUMER_WORDS.findall(prompt)}))


@dataclass(frozen=True, slots=True)
class PromptCase:
    """One question, and what a right answer to it must carry."""

    name: str
    problem: MathProblem
    #: A subset every correct answer agrees with, or None where the answer has
    #: many correct spellings and only the contract can be checked.
    expected: dict[str, Any] | None
    note: str


#: The routes worth holding open, one case each. Between them they reach every
#: prompt constructor in this repository, and one route that reaches none.
CASES: tuple[PromptCase, ...] = (
    PromptCase(
        name="exact",
        problem=MathProblem(instruction="Simplify.", expressions=("sqrt(-324)",)),
        expected={"route": "exact", "answer": {"display": "18i"}},
        note="answered by mathematics; must reach no model at all",
    ),
    PromptCase(
        name="reasoning",
        problem=MathProblem(
            instruction="Find the domain of the following function.",
            expressions=(r"\frac{x+1}{x^2-9}",),
            label="Question 4 of 12",
        ),
        expected=None,
        note=(
            "the exact solvers decline this, so it exercises reasoning_prompt; "
            "a domain has too many correct spellings to pin one"
        ),
    ),
    PromptCase(
        name="parabola",
        problem=MathProblem(
            instruction="Graph the parabola.",
            expressions=("f(x)=(x-3)^2-1",),
            result_kind=PARABOLA_PLAN,
            graph=GraphContext(
                family="parabola",
                orientation="vertical",
                bounds=(-10.0, 10.0, -10.0, 10.0),
                snap=(0.5, 0.5),
                controls="vertex-and-symmetric-points",
            ),
        ),
        # The vertex and the opening are determined. Which symmetric pair the
        # model picks is not, so the pair is left unpinned.
        expected={
            "answer": {
                "plan": {
                    "kind": "parabola",
                    "orientation": "vertical",
                    "opening": "up",
                    "vertex": {"x": "3", "y": "-1"},
                }
            }
        },
        note="exercises parabola_prompt and parse_parabola_plan",
    ),
    PromptCase(
        name="regression",
        problem=MathProblem(
            instruction="Use quadratic regression. Round to three decimal places.",
            result_kind=QUADRATIC_REGRESSION,
            points=(Point("-5", "5"), Point("-2", "-4"), Point("-1", "5")),
        ),
        # All three points lie exactly on 3x^2+18x+20, so the least-squares fit
        # is that parabola and there is exactly one right answer.
        expected={
            "answer": {
                "plan": {
                    "kind": "quadratic-regression",
                    "coefficients": ["3", "18", "20"],
                }
            }
        },
        note="exercises regression_prompt; the case that failed live",
    ),
)


class _Rendered(Exception):
    """Carries the prompt back out of a run that was never going to happen."""

    def __init__(self, prompt: str) -> None:
        super().__init__("prompt rendered")
        self.prompt = prompt


def case(name: str) -> PromptCase:
    for known in CASES:
        if known.name == name:
            return known
    raise ValueError(f"unknown prompt case: {name}")


def render(subject: PromptCase) -> str | None:
    """The exact prompt this case would send, or None if it reaches no model.

    Routing is not reimplemented here. `solve_math` decides, exactly as it does
    in production, and the recorder simply refuses to be a model -- so a case
    that returns None is a case the deterministic stage answered, which is a
    fact about the routing rather than about this function.
    """

    def capture(prompt: str) -> RunResult:
        raise _Rendered(prompt)

    try:
        solve_math(subject.problem, reason=capture)
    except _Rendered as rendered:
        return rendered.prompt
    return None


def contains(actual: Any, expected: Any) -> bool:
    """Whether `actual` carries everything `expected` names, and agrees on it."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and contains(actual[key], value)
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(contains(a, e) for a, e in zip(actual, expected, strict=True))
        )
    return bool(actual == expected)


def check(
    subject: PromptCase,
    backend: BackendName = "auto",
    *,
    adapters: dict[str, BackendAdapter] | None = None,
    run: Callable[..., RunResult] | None = None,
    keep_prompt: bool = False,
) -> dict[str, Any]:
    """Answer one case for real and report what happened, without repairing it."""
    adapter_map = dict(adapters or default_adapters())
    execute = run if run is not None else run_prompt
    sent: list[str] = []

    def reason(prompt: str) -> RunResult:
        sent.append(prompt)
        return execute(prompt, backend, adapters=adapter_map)

    row: dict[str, Any] = {"case": subject.name, "note": subject.note}
    started = time.perf_counter()
    try:
        result = solve_math(subject.problem, reason=reason)
    except SolveRefused as refusal:
        # The run itself may well have succeeded; what came back could not be
        # an answer. That distinction is the whole point of reporting it.
        row.update(status=refusal.kind, error=str(refusal))
    except FacetRuntimeError as failure:
        row.update(status="execution_failed", error=str(failure))
    else:
        row.update(
            status="ok",
            route=result["route"],
            source=result["provenance"]["source"],
            model=result["provenance"]["model"],
            device=result["provenance"]["device"],
            metrics=result["provenance"]["metrics"],
            answer=result["answer"],
        )
    row["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    row["model_calls"] = len(sent)
    row["prompt_chars"] = len(sent[0]) if sent else 0
    row["leaks"] = leaks(sent[0]) if sent else ()
    if keep_prompt and sent:
        row["prompt"] = sent[0]
    row["expected_met"] = (
        None
        if subject.expected is None
        else (row["status"] == "ok" and contains(row, subject.expected))
    )
    # A case that pins nothing still has a contract: it ran, and something
    # survived the strict parser.
    row["passed"] = row["status"] == "ok" and row["expected_met"] is not False
    return row


def check_all(
    cases: tuple[PromptCase, ...] = CASES,
    backend: BackendName = "auto",
    *,
    adapters: dict[str, BackendAdapter] | None = None,
    run: Callable[..., RunResult] | None = None,
    keep_prompt: bool = False,
) -> dict[str, Any]:
    """Answer every case in turn and report the sweep."""
    rows = [
        check(
            subject,
            backend,
            adapters=adapters,
            run=run,
            keep_prompt=keep_prompt,
        )
        for subject in cases
    ]
    return {
        "backend": backend,
        "cases": rows,
        "passed": sum(1 for row in rows if row["passed"]),
        "total": len(rows),
        "all_passed": all(row["passed"] for row in rows),
    }


def render_all(cases: tuple[PromptCase, ...] = CASES) -> dict[str, Any]:
    """Show every prompt, with no model and no network, for reading and diffing."""
    return {
        "cases": [
            {
                "case": subject.name,
                "note": subject.note,
                "result_kind": subject.problem.result_kind,
                "reaches_a_model": render(subject) is not None,
                "leaks": leaks(render(subject) or ""),
                "prompt": render(subject),
            }
            for subject in cases
        ]
    }


def model_facing_prompts() -> tuple[str, ...]:
    """Every prompt any model can be sent, for a check over all of them at once."""
    rendered = (render(subject) for subject in CASES)
    return tuple(prompt for prompt in rendered if prompt is not None)


__all__ = [
    "CASES",
    "CONSUMER_VOCABULARY",
    "PromptCase",
    "case",
    "check",
    "check_all",
    "contains",
    "leaks",
    "model_facing_prompts",
    "render",
    "render_all",
]
