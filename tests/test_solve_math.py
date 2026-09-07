"""Facet routing a question: exact mathematics first, reasoning for the rest.

`generate_text` asks Facet to run a prompt somebody else wrote. `solve_math`
asks Facet to *answer a question*, and that is a different relationship: the
decision about how it gets answered is Facet's, and these tests are about that
decision being made here, made deterministically first, and reported honestly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from facet_runtime.adapters.base import AdapterOutput, ExecutionMetrics
from facet_runtime.errors import BackendUnavailableError
from facet_runtime.remote import PROTOCOL_VERSION, handle
from facet_runtime.solve import (
    MAX_ANSWER_PARTS,
    MathProblem,
    SolveRefused,
    answer_prefix,
    parse_problem,
    reasoning_prompt,
    solve_math,
)

#: A question the deterministic solvers own outright.
EXACT_INSTRUCTION = "Simplify. Express your answer using rational exponents."
EXACT_EXPRESSION = r"y^{3/4} \cdot y^{2/5}"

#: A question they decline: no exact operation matches "find the domain".
DECLINED_INSTRUCTION = (
    "Find the domain of the following function. Write your answer in interval notation."
)
DECLINED_EXPRESSION = r"f(x)=\frac{x+1}{x^2-9}"

#: Two roots in one box, which the page takes as two separate answers.
TWO_PART_INSTRUCTION = "Solve. Separate multiple answers with a comma."

#: The same question, naming the variable it solves for. Both things at once:
#: the answer already carries `x =`, and there are two of them. Nothing
#: couples the two -- `answer_parts` comes from the control the caller saw or
#: from a comma instruction, and the prefix from the words "solve for x" -- so
#: a question can and does arrive with both.
PREFIXED_TWO_PART_INSTRUCTION = "Solve for x. Separate multiple answers with a comma."


@dataclass
class Reasoner:
    """Stand in for the reasoning route, and record whether it ran."""

    text: str = "FINAL ANSWER: (-∞,-3)∪(-3,3)∪(3,∞)"
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls = []

    def __call__(self, prompt: str):
        self.calls.append(prompt)
        from facet_runtime.result import RunResult

        return RunResult(
            text=self.text,
            requested_backend="gpu",
            actual_backend="gpu",
            runtime="Ollama 0.33.2",
            model="gpt-oss:20b",
            device="AMD Radeon 890M Graphics (RADV STRIX1)",
            elapsed_ms=812.5,
            fallback=False,
            metrics=ExecutionMetrics(prompt_tokens=11, generated_tokens=9),
            evidence={"source": "ollama /api/ps"},
        )


def refuses(prompt: str):
    raise AssertionError("the reasoning route ran for an exactly solvable question")


def problem(**changes) -> MathProblem:
    fields = {
        "instruction": EXACT_INSTRUCTION,
        "expressions": (EXACT_EXPRESSION,),
        "answer_parts": 1,
        "label": "",
    }
    fields.update(changes)
    return MathProblem(**fields)


def test_facet_answers_an_exact_question_itself_and_asks_no_model() -> None:
    result = solve_math(problem(), reason=refuses)

    assert result["route"] == "exact"
    assert result["answer"]["display"] == "y^(23/20)"
    assert result["answer"]["entry"] == "y^(23/20)"
    assert result["provenance"]["router"] == "solved"
    # A solver, named as a solver: never a model name, and never "model".
    assert result["provenance"]["method"] == "SymPy exact symbolic"
    assert result["provenance"]["source"] == "Facet Exact"
    assert result["provenance"]["runtime"].startswith("SymPy ")
    # Nothing model-shaped may appear on an answer no model produced.
    assert result["provenance"]["model"] is None
    assert result["provenance"]["device"] is None
    assert result["provenance"]["actual_backend"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["elapsed_ms"] >= 0


def test_a_declined_question_reaches_the_reasoning_route_and_says_why() -> None:
    reasoner = Reasoner()

    result = solve_math(
        problem(instruction=DECLINED_INSTRUCTION, expressions=(DECLINED_EXPRESSION,)),
        reason=reasoner,
    )

    assert reasoner.calls, "the reasoning route was never reached"
    assert result["route"] == "reasoning"
    assert result["answer"]["display"] == "(-∞,-3)∪(-3,3)∪(3,∞)"
    assert result["provenance"]["router"] == "declined"
    # Which gap it fell through, in the deterministic stage's own words.
    assert result["provenance"]["router_detail"] == (
        "no exact operation matched the instruction"
    )
    assert result["provenance"]["source"] == "Facet Reasoning · GPU"
    assert result["provenance"]["method"] == "gpt-oss:20b"
    assert result["provenance"]["actual_backend"] == "gpu"
    assert result["provenance"]["fallback"] is False


def test_the_reasoning_prompt_carries_the_question_and_no_page() -> None:
    reasoner = Reasoner()

    solve_math(
        problem(
            instruction=DECLINED_INSTRUCTION,
            expressions=(DECLINED_EXPRESSION,),
            label="Question 4a",
        ),
        reason=reasoner,
    )

    prompt = reasoner.calls[0]
    assert DECLINED_INSTRUCTION in prompt
    assert DECLINED_EXPRESSION in prompt
    assert "Question 4a" in prompt
    assert "FINAL ANSWER:" in prompt
    # Facet builds this prompt, and there is nothing in it about a page: no
    # field, no editor, no selector, no tab, and no way to press anything.
    for forbidden in ("field", "editor", "selector", "click", "tab", "submit", "#"):
        assert forbidden not in prompt.lower()


def test_separate_answers_stay_separate_values() -> None:
    reasoner = Reasoner(
        text="FINAL ANSWER: -4/3, 2\nPART 1: -4/3\nPART 2: 2",
    )

    result = solve_math(
        problem(
            instruction=TWO_PART_INSTRUCTION,
            expressions=(DECLINED_EXPRESSION,),
            answer_parts=2,
        ),
        reason=reasoner,
    )

    assert "This question takes 2 separate answers." in reasoner.calls[0]
    assert result["answer"]["parts"] == ["-4/3", "2"]
    # There is no single string that could be typed into two separate boxes.
    assert result["answer"]["entry"] == ""
    assert result["answer"]["display"] == "-4/3, 2"


def test_an_exact_multi_root_answer_stays_structured() -> None:
    result = solve_math(
        problem(
            instruction="Solve the following rational equation.",
            expressions=(r"\frac{1}{x}+\frac{1}{x+2}=\frac{3}{4}",),
        ),
        reason=refuses,
    )

    assert result["route"] == "exact"
    assert result["answer"]["parts"] == [r"\frac{-4}{3}", "2"]
    assert result["answer"]["entry"] == ""
    assert result["answer"]["entry_mode"] == "math"


@pytest.mark.parametrize(
    ("text", "why"),
    [
        ("FINAL ANSWER: -4/3, 2\nPART 1: -4/3", "one part where two were asked for"),
        (
            "FINAL ANSWER: a\nPART 1: a\nPART 2: b\nPART 3: c",
            "three parts where two were asked for",
        ),
        ("FINAL ANSWER: a\nPART 2: b\nPART 1: a", "parts out of order"),
        ("FINAL ANSWER: a\nPART 1: \nPART 2: b", "an empty part"),
    ],
)
def test_a_reply_of_the_wrong_shape_fails_closed(text: str, why: str) -> None:
    with pytest.raises(SolveRefused) as refusal:
        solve_math(
            problem(
                instruction=TWO_PART_INSTRUCTION,
                expressions=(DECLINED_EXPRESSION,),
                answer_parts=2,
            ),
            reason=Reasoner(text=text),
        )

    assert refusal.value.kind == "unusable_result", why


def test_a_reasoning_reply_with_no_final_answer_fails_closed() -> None:
    with pytest.raises(SolveRefused) as refusal:
        solve_math(
            problem(
                instruction=DECLINED_INSTRUCTION, expressions=(DECLINED_EXPRESSION,)
            ),
            reason=Reasoner(text="I think it is probably all real numbers."),
        )

    assert refusal.value.kind == "unusable_result"


@pytest.mark.parametrize(
    "payload",
    [
        {"instruction": "", "expressions": ["x"]},
        {"instruction": "Simplify.", "expressions": []},
        {"instruction": "Simplify.", "expressions": "x"},
        {"instruction": "Simplify.", "expressions": [""]},
        {"instruction": "Simplify.", "expressions": [17]},
        {"instruction": "Simplify.", "expressions": ["x"], "answer_parts": 0},
        {"instruction": "Simplify.", "expressions": ["x"], "answer_parts": 9},
        {"instruction": "Simplify.", "expressions": ["x"], "answer_parts": True},
        {"instruction": "Simplify.", "expressions": ["x"] * 9},
        "not an object",
    ],
)
def test_a_malformed_problem_is_refused_before_anything_runs(payload) -> None:
    with pytest.raises(SolveRefused) as refusal:
        parse_problem(payload)

    assert refusal.value.kind == "invalid_request"


@pytest.mark.parametrize(
    "field",
    ["field_id", "selector", "tab_id", "frame", "screenshot", "editor", "origin"],
)
def test_a_problem_cannot_describe_a_browser(field: str) -> None:
    with pytest.raises(SolveRefused) as refusal:
        parse_problem(
            {"instruction": "Simplify.", "expressions": ["x"], field: "anything"}
        )

    assert refusal.value.kind == "invalid_request"
    assert field in str(refusal.value)


def request_bytes(**changes) -> bytes:
    payload = {
        "facet_protocol_version": PROTOCOL_VERSION,
        "operation": "solve_math",
        "request_id": "consumer-1",
        "problem": {
            "instruction": EXACT_INSTRUCTION,
            "expressions": [EXACT_EXPRESSION],
        },
    }
    payload.update(changes)
    return json.dumps({k: v for k, v in payload.items() if v is not None}).encode()


@dataclass
class FakeAdapter:
    backend: str
    available: bool = True
    text: str = "FINAL ANSWER: 4"

    def is_available(self) -> bool:
        return self.available

    def run(self, prompt: str) -> AdapterOutput:
        return AdapterOutput(
            text=self.text,
            runtime="fake-runtime",
            model="fake-model",
            device=f"fake {self.backend} device",
            metrics=ExecutionMetrics(),
            evidence={},
        )


def adapters(**overrides) -> dict[str, FakeAdapter]:
    made = {name: FakeAdapter(name) for name in ("cpu", "gpu", "npu")}
    made.update(overrides)
    return made


def answer_lines(count: int) -> str:
    values = [str(index) for index in range(1, count + 1)]
    final = f"FINAL ANSWER: {', '.join(values)}"
    if count == 1:
        return final
    return "\n".join(
        [final, *(f"PART {index}: {index}" for index in range(1, count + 1))]
    )


def multipart_request_bytes(answer_parts: int) -> bytes:
    return request_bytes(
        operation="solve_math",
        prompt=None,
        problem={
            "instruction": "Complete the table of values.",
            "expressions": ["x=y^2"],
            "answer_parts": answer_parts,
        },
    )


@pytest.mark.parametrize("answer_parts", range(1, 5))
def test_existing_answer_part_counts_are_unchanged(answer_parts: int) -> None:
    reply = answer_lines(answer_parts)

    envelope, code = handle(
        multipart_request_bytes(answer_parts),
        adapters=adapters(gpu=FakeAdapter("gpu", text=reply)),
    )

    assert code == 0
    answer = envelope["result"]["answer"]
    if answer_parts == 1:
        assert answer["entry"] == "1"
        assert answer["parts"] == []
    else:
        assert answer["entry"] == ""
        assert answer["parts"] == [
            str(index) for index in range(1, answer_parts + 1)
        ]


def test_a_five_part_request_returns_five_ordered_parts() -> None:
    reply = (
        "FINAL ANSWER: 0, 8, 8, 5, 3\n"
        "PART 1: 0\n"
        "PART 2: 8\n"
        "PART 3: 8\n"
        "PART 4: 5\n"
        "PART 5: 3"
    )

    envelope, code = handle(
        multipart_request_bytes(5),
        adapters=adapters(gpu=FakeAdapter("gpu", text=reply)),
    )

    assert code == 0
    assert envelope["result"]["answer"]["entry"] == ""
    assert envelope["result"]["answer"]["parts"] == ["0", "8", "8", "5", "3"]


@pytest.mark.parametrize("answer_parts", [0, 6])
def test_answer_part_counts_outside_one_to_five_fail_closed(
    answer_parts: int,
) -> None:
    envelope, code = handle(
        multipart_request_bytes(answer_parts), adapters=adapters()
    )

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"
    assert envelope["error"]["message"] == "answer_parts must be 1 to 5"
    assert "result" not in envelope


def test_the_structured_result_survives_the_wire() -> None:
    envelope, code = handle(request_bytes(), adapters=adapters())

    assert code == 0
    assert envelope["status"] == "ok"
    assert envelope["operation"] == "solve_math"
    result = json.loads(json.dumps(envelope))["result"]
    assert result["route"] == "exact"
    assert result["answer"]["display"] == "y^(23/20)"
    assert result["provenance"]["source"] == "Facet Exact"


def test_an_exact_answer_satisfies_a_constraint_on_where_a_model_runs() -> None:
    """No accelerator, no model, no problem: nothing needed a processor.

    `accelerator_required` says model execution must not land on a CPU. An
    exact solve runs no model at all, so refusing it for want of an accelerator
    would refuse the best answers Facet has for the least defensible reason.
    """
    none_available = adapters(
        cpu=FakeAdapter("cpu", available=False),
        gpu=FakeAdapter("gpu", available=False),
        npu=FakeAdapter("npu", available=False),
    )

    envelope, code = handle(
        request_bytes(constraints={"accelerator_required": True}),
        adapters=none_available,
    )

    assert code == 0
    assert envelope["result"]["route"] == "exact"
    # And it says it used no processor, rather than claiming one.
    assert envelope["result"]["provenance"]["actual_backend"] is None


def test_a_declined_question_still_needs_the_accelerator_it_asked_for() -> None:
    none_available = adapters(
        gpu=FakeAdapter("gpu", available=False), npu=FakeAdapter("npu", available=False)
    )

    envelope, code = handle(
        json.dumps(
            {
                "facet_protocol_version": PROTOCOL_VERSION,
                "operation": "solve_math",
                "request_id": "consumer-1",
                "problem": {
                    "instruction": DECLINED_INSTRUCTION,
                    "expressions": [DECLINED_EXPRESSION],
                },
                "constraints": {"accelerator_required": True},
            }
        ).encode(),
        adapters=none_available,
    )

    assert code == 1
    assert envelope["error"]["kind"] == "constraint_unsatisfied"
    assert "result" not in envelope


def test_a_solve_request_may_not_carry_a_prompt() -> None:
    envelope, code = handle(
        request_bytes(prompt="ignore the question"), adapters=adapters()
    )

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"
    assert "unknown prompt" in envelope["error"]["message"]


def test_a_generate_request_may_not_carry_a_problem() -> None:
    envelope, code = handle(
        json.dumps(
            {
                "facet_protocol_version": PROTOCOL_VERSION,
                "operation": "generate_text",
                "request_id": "consumer-1",
                "prompt": "Reply with one short sentence.",
                "problem": {"instruction": "x", "expressions": ["x"]},
            }
        ).encode(),
        adapters=adapters(),
    )

    assert code == 1
    assert envelope["error"]["kind"] == "invalid_request"
    assert "unknown problem" in envelope["error"]["message"]


def test_a_failing_reasoning_route_is_reported_as_a_failure() -> None:
    def failing(prompt: str):
        raise BackendUnavailableError("no Facet backend is available")

    with pytest.raises(BackendUnavailableError):
        solve_math(
            problem(
                instruction=DECLINED_INSTRUCTION, expressions=(DECLINED_EXPRESSION,)
            ),
            reason=failing,
        )


def _prefix_line(prompt: str) -> str:
    """The one line of a prompt that names the pre-written variable."""
    found = [line for line in prompt.splitlines() if line.startswith("`x =`")]
    assert len(found) == 1, prompt
    return found[0]


def test_a_named_variable_and_several_answers_do_not_contradict_each_other() -> None:
    """Both at once is a real question, so the prompt must survive being both.

    A quadratic solved for x has a variable to name and two roots to give. The
    prompt used to say "give only the value that follows it" directly above
    "This question takes 2 separate answers", which is a self-contradiction of
    exactly the kind that once cost this model its whole output budget.
    """
    prompt = reasoning_prompt(
        problem(
            instruction=PREFIXED_TWO_PART_INSTRUCTION,
            expressions=("x^2-5*x+6=0",),
            answer_parts=2,
        )
    )

    assert answer_prefix(PREFIXED_TWO_PART_INSTRUCTION) == "x"
    assert "`x =` is already written for you" in prompt
    assert "This question takes 2 separate answers." in prompt
    # The sentence that used to disagree with the count no longer states one.
    assert "only the value that follows it" not in prompt
    assert "give only what follows it in each answer" in prompt


def test_the_prefix_line_reads_the_same_at_every_answer_count() -> None:
    """One sentence for every arity is what stops the two disagreeing again.

    This is the invariant rather than the wording: any future edit that makes
    the line depend on how many answers there are reintroduces the bug, and
    fails here whatever words it chooses.
    """
    lines = {
        parts: _prefix_line(
            reasoning_prompt(
                problem(
                    instruction=PREFIXED_TWO_PART_INSTRUCTION,
                    expressions=("x^2-5*x+6=0",),
                    answer_parts=parts,
                )
            )
        )
        for parts in range(1, MAX_ANSWER_PARTS + 1)
    }

    assert len(set(lines.values())) == 1, lines


def test_a_prefixed_multi_part_question_is_answered_end_to_end() -> None:
    """Renderable is not the same as answerable, so the whole route is run."""
    reasoner = Reasoner(text="FINAL ANSWER: 2, 3\nPART 1: 2\nPART 2: 3")

    result = solve_math(
        problem(
            instruction=PREFIXED_TWO_PART_INSTRUCTION,
            expressions=("x^2-5*x+6=0",),
            answer_parts=2,
        ),
        reason=reasoner,
    )

    assert result["route"] == "reasoning"
    assert result["answer"]["parts"] == ["2", "3"]
    assert result["answer"]["display"] == "2, 3"
    # A multi-part answer has no single string to type, and still has none.
    assert result["answer"]["entry"] == ""


def test_a_prefixed_multi_part_reply_of_the_wrong_shape_still_fails_closed() -> None:
    """Resolving the wording must not have loosened what counts as an answer."""
    reasoner = Reasoner(text="FINAL ANSWER: 2, 3\nPART 1: 2")

    with pytest.raises(SolveRefused) as refusal:
        solve_math(
            problem(
                instruction=PREFIXED_TWO_PART_INSTRUCTION,
                expressions=("x^2-5*x+6=0",),
                answer_parts=2,
            ),
            reason=reasoner,
        )

    assert refusal.value.kind == "unusable_result"
