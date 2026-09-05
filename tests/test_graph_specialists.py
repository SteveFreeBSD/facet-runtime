"""Facet routing a question whose answer is a plan rather than a value.

Two families ask for geometry: a vertical parabola somebody will draw, and the
coefficients of a quadratic regression over points somebody measured. Neither
has a deterministic route -- the exact solvers answer expressions -- so both are
reasoned, and what Facet returns is a *proposal*.

That word carries the whole design. Facet parses a reply strictly, which is a
check on the model, and then hands the plan on. It does not prove the geometry,
because the authority that matters is whoever owns the surface the plan will be
drawn on. What is checked here is that Facet asks the right specialist, refuses
a reply that is not a plan, and never returns a plan wearing a value's clothes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from facet_runtime.adapters.base import AdapterOutput, ExecutionMetrics
from facet_runtime.graph import (
    PlanRefused,
    parse_graph_context,
    parse_parabola_plan,
    parse_points,
    parse_regression_plan,
)
from facet_runtime.remote import PROTOCOL_VERSION, handle
from facet_runtime.solve import MathProblem, SolveRefused, parse_problem, solve_math

PLAN = {
    "kind": "parabola",
    "orientation": "vertical",
    "opening": "up",
    "vertex": {"x": "3", "y": "-1"},
    "points": [{"x": "4", "y": "0"}, {"x": "2", "y": "0"}],
}
REGRESSION = {"kind": "quadratic-regression", "coefficients": ["3", "18", "20"]}
CONTEXT = {
    "family": "parabola",
    "orientation": "vertical",
    "bounds": [-10.0, 10.0, -10.0, 10.0],
    "snap": [0.5, 0.5],
    "controls": "vertex-and-symmetric-points",
}
POINTS = [{"x": "-5", "y": "5"}, {"x": "-2", "y": "-4"}, {"x": "-1", "y": "5"}]
GRAPH_INSTRUCTION = "Graph the parabola."
REGRESSION_INSTRUCTION = "Use quadratic regression. Round to three decimal places."


@dataclass
class Reasoner:
    text: str = json.dumps(PLAN)
    calls: list[str] = field(default_factory=list)

    def __call__(self, prompt: str):
        from facet_runtime.result import RunResult

        self.calls.append(prompt)
        return RunResult(
            text=self.text,
            requested_backend="auto",
            actual_backend="gpu",
            runtime="Ollama 0.33.2",
            model="gpt-oss:20b",
            device="AMD Radeon 890M Graphics (RADV STRIX1)",
            elapsed_ms=910.0,
            fallback=False,
            metrics=ExecutionMetrics(prompt_tokens=44, generated_tokens=61),
            evidence={"source": "ollama /api/ps"},
        )


def parabola_problem(**changes) -> MathProblem:
    payload = {
        "result_kind": "parabola_plan",
        "instruction": GRAPH_INSTRUCTION,
        "expressions": ["f(x)=(x-3)^2-1"],
        "graph": CONTEXT,
    }
    payload.update(changes)
    return parse_problem(payload)


def regression_problem(**changes) -> MathProblem:
    payload = {
        "result_kind": "quadratic_regression",
        "instruction": REGRESSION_INSTRUCTION,
        "points": POINTS,
    }
    payload.update(changes)
    return parse_problem(payload)


# --- the specialists Facet routes to ---------------------------------------


def test_a_parabola_question_reaches_the_parabola_specialist() -> None:
    reasoner = Reasoner()

    result = solve_math(parabola_problem(), reason=reasoner)

    prompt = reasoner.calls[0]
    assert "Produce a graph plan for the exact function." in prompt
    assert "f(x)=(x-3)^2-1" in prompt
    assert "vertex-and-symmetric-points" in prompt
    assert result["route"] == "reasoning"
    assert result["answer"] == {"kind": "parabola_plan", "plan": PLAN}
    assert result["provenance"]["source"] == "Facet Parabola Plan · GPU"
    # The exact solvers answer expressions, not geometry. They were not asked,
    # which is a different claim from having tried and declined.
    assert result["provenance"]["router"] == "not-run"
    assert result["provenance"]["router_detail"] == (
        "a graph plan has no deterministic route"
    )
    assert result["provenance"]["model"] == "gpt-oss:20b"
    assert result["provenance"]["actual_backend"] == "gpu"
    assert result["provenance"]["device"].startswith("AMD Radeon")
    assert result["provenance"]["fallback"] is False
    assert result["provenance"]["elapsed_ms"] >= 0


def test_a_regression_question_reaches_the_regression_specialist() -> None:
    reasoner = Reasoner(text=json.dumps(REGRESSION))

    result = solve_math(regression_problem(), reason=reasoner)

    prompt = reasoner.calls[0]
    assert "quadratic least-squares regression" in prompt
    assert '{"x": "-5", "y": "5"}' in prompt
    assert result["answer"] == {"kind": "quadratic_regression", "plan": REGRESSION}
    assert result["provenance"]["source"] == "Facet Quadratic Regression · GPU"
    assert result["provenance"]["router"] == "not-run"


def test_a_plan_carries_no_value_to_write_anywhere() -> None:
    """A proposal must not arrive shaped like a settled answer."""
    result = solve_math(parabola_problem(), reason=Reasoner())

    assert set(result["answer"]) == {"kind", "plan"}
    for value in ("display", "entry", "parts", "entry_mode"):
        assert value not in result["answer"]


def test_a_value_question_is_untouched_by_the_specialists() -> None:
    """The ordinary route still answers exactly, and asks no model."""

    def refuses(prompt: str):
        raise AssertionError("a model was asked an exactly solvable question")

    result = solve_math(
        parse_problem(
            {
                "instruction": "Simplify. Express your answer using rational exponents.",
                "expressions": [r"y^{3/4} \cdot y^{2/5}"],
            }
        ),
        reason=refuses,
    )

    assert result["route"] == "exact"
    assert result["answer"]["kind"] == "value"
    assert result["answer"]["display"] == "y^(23/20)"


# --- a reply that is not a plan is not repaired -----------------------------


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("```json\n" + json.dumps(PLAN) + "\n```", "a fenced code block"),
        (json.dumps(PLAN) + " prose", "trailing prose"),
        (json.dumps({**PLAN, "click": "Submit"}), "an extra key"),
        (json.dumps(PLAN).replace('"x": "3"', '"x": 3'), "a bare number"),
        (json.dumps(PLAN).replace('"x": "3"', '"x": "3", "x": "4"'), "a repeated key"),
        (json.dumps(PLAN).replace('"3"', '"NaN"'), "a coordinate that is not one"),
        (json.dumps(PLAN).replace('"3"', '"3.0"'), "a decimal approximation"),
        (json.dumps({**PLAN, "opening": "sideways"}), "an opening that is neither"),
        (json.dumps({**PLAN, "orientation": "horizontal"}), "another orientation"),
        (json.dumps({**PLAN, "points": [PLAN["points"][0]]}), "one defining point"),
        (json.dumps({**PLAN, "vertex": {"x": "3"}}), "half a vertex"),
        ("", "no reply at all"),
    ],
)
def test_a_reply_that_is_not_a_parabola_plan_fails_closed(text: str, why: str) -> None:
    with pytest.raises(SolveRefused) as refusal:
        solve_math(parabola_problem(), reason=Reasoner(text=text))

    assert refusal.value.kind == "unusable_result", why


@pytest.mark.parametrize(
    ("text", "why"),
    [
        (json.dumps(REGRESSION) + " prose", "trailing prose"),
        (json.dumps({**REGRESSION, "kind": "linear-regression"}), "another family"),
        (json.dumps({**REGRESSION, "coefficients": ["3", "18"]}), "two coefficients"),
        (json.dumps({**REGRESSION, "coefficients": ["3.0", "18", "20"]}), "a decimal"),
        (json.dumps({**REGRESSION, "coefficients": [3, 18, 20]}), "bare numbers"),
        (json.dumps({**REGRESSION, "note": "check it"}), "an extra key"),
    ],
)
def test_a_reply_that_is_not_a_regression_fails_closed(text: str, why: str) -> None:
    with pytest.raises(SolveRefused) as refusal:
        solve_math(regression_problem(), reason=Reasoner(text=text))

    assert refusal.value.kind == "unusable_result", why


def test_an_exact_rational_coefficient_is_still_a_coefficient() -> None:
    """The refusals above must be refusing decimals, not fractions."""
    exact = {"kind": "quadratic-regression", "coefficients": ["5/4", "-9/20", "1/20"]}

    result = solve_math(regression_problem(), reason=Reasoner(text=json.dumps(exact)))

    assert result["answer"]["plan"] == exact


# --- what a consumer may ask for -------------------------------------------


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({"result_kind": "cubic_plan"}, "a specialist that does not exist"),
        ({"graph": None}, "a parabola plan with no geometry"),
        ({"answer_parts": 2}, "a plan asked for in separate values"),
        ({"points": POINTS}, "regression points on a parabola request"),
        ({"graph": {**CONTEXT, "family": "hyperbola"}}, "an unsupported family"),
        ({"graph": {**CONTEXT, "orientation": "horizontal"}}, "another orientation"),
        ({"graph": {**CONTEXT, "controls": "drag"}}, "unsupported controls"),
        (
            {"graph": {**CONTEXT, "bounds": [10.0, -10.0, -10.0, 10.0]}},
            "reversed bounds",
        ),
        ({"graph": {**CONTEXT, "snap": [0.0, 0.5]}}, "a zero snap"),
        ({"graph": {**CONTEXT, "bounds": [-10.0, 10.0, -10.0]}}, "three bounds"),
        ({"graph": {k: v for k, v in CONTEXT.items() if k != "snap"}}, "no snap"),
        ({"graph": {**CONTEXT, "element": "#QGraph1"}}, "an element on the geometry"),
        ({"expressions": []}, "no expression to plan from"),
    ],
)
def test_a_parabola_request_that_is_not_one_is_refused(payload, why) -> None:
    with pytest.raises(SolveRefused) as refusal:
        parabola_problem(**payload)

    assert refusal.value.kind == "invalid_request", why


@pytest.mark.parametrize(
    ("payload", "why"),
    [
        ({"points": POINTS[:2]}, "too few points to fit a quadratic"),
        ({"points": POINTS * 12}, "more points than the protocol carries"),
        ({"points": [{"x": "0.5", "y": "1"}]}, "a decimal coordinate"),
        ({"points": [{"x": "1", "y": "1", "id": "p0"}] * 3}, "an identity on a point"),
        ({"points": [{"x": "1"}] * 3}, "half a point"),
        ({"expressions": ["x^2"]}, "an expression a regression does not take"),
        ({"graph": CONTEXT}, "geometry a regression does not take"),
    ],
)
def test_a_regression_request_that_is_not_one_is_refused(payload, why) -> None:
    with pytest.raises(SolveRefused) as refusal:
        regression_problem(**payload)

    assert refusal.value.kind == "invalid_request", why


@pytest.mark.parametrize(
    "field", ["field_id", "selector", "tab_id", "frame", "screenshot", "svg", "node"]
)
def test_a_graph_problem_still_cannot_describe_a_browser(field: str) -> None:
    with pytest.raises(SolveRefused) as refusal:
        parabola_problem(**{field: "anything"})

    assert refusal.value.kind == "invalid_request"
    assert field in str(refusal.value)


# --- across the wire --------------------------------------------------------


@dataclass
class FakeAdapter:
    backend: str
    text: str = json.dumps(PLAN)

    def is_available(self) -> bool:
        return True

    def run(self, prompt: str) -> AdapterOutput:
        return AdapterOutput(
            text=self.text,
            runtime="Ollama 0.33.2",
            model="gpt-oss:20b",
            device=f"fake {self.backend} device",
            metrics=ExecutionMetrics(),
            evidence={},
        )


def test_a_plan_survives_the_wire_exactly() -> None:
    request = json.dumps(
        {
            "facet_protocol_version": PROTOCOL_VERSION,
            "operation": "solve_math",
            "request_id": "graph-1",
            "problem": {
                "result_kind": "parabola_plan",
                "instruction": GRAPH_INSTRUCTION,
                "expressions": ["f(x)=(x-3)^2-1"],
                "graph": CONTEXT,
            },
        }
    ).encode()

    envelope, code = handle(
        request, adapters={name: FakeAdapter(name) for name in ("cpu", "gpu", "npu")}
    )

    assert code == 0
    # Serialised and parsed again, because that is what actually happens.
    assert json.loads(json.dumps(envelope))["result"]["answer"]["plan"] == PLAN


# --- the parsers, on their own ----------------------------------------------


def test_the_parsers_accept_what_they_are_meant_to() -> None:
    assert parse_parabola_plan(json.dumps(PLAN)) == PLAN
    assert parse_regression_plan(json.dumps(REGRESSION)) == REGRESSION
    assert parse_graph_context(CONTEXT).as_json() == CONTEXT
    assert [point.as_json() for point in parse_points(POINTS)] == POINTS


def test_a_repeated_key_is_refused_rather_than_resolved() -> None:
    """`json.loads` keeps the last silently; a plan may not be chosen that way."""
    with pytest.raises(PlanRefused, match="more than once"):
        parse_parabola_plan(
            json.dumps(PLAN).replace('"opening"', '"opening": "down", "opening"')
        )
