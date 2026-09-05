"""The harness must exercise the real prompts, and must not flatter them."""

from __future__ import annotations

import json

import pytest

from facet_runtime import prompts
from facet_runtime.errors import FacetRuntimeError
from facet_runtime.result import RunResult
from facet_runtime.solve import MathProblem, reasoning_prompt

CORRECT = {
    "reasoning": r"FINAL ANSWER: (-\infty,-3)\cup(-3,3)\cup(3,\infty)",
    "parabola": (
        '{"kind":"parabola","orientation":"vertical","opening":"up",'
        '"vertex":{"x":"3","y":"-1"},'
        '"points":[{"x":"4","y":"0"},{"x":"2","y":"0"}]}'
    ),
    "regression": ('{"kind":"quadratic-regression","coefficients":["3","18","20"]}'),
}


def answering(replies: dict[str, str] | str, *, raises: Exception | None = None):
    """A stand-in model that replies by which prompt it was actually sent."""

    def run(prompt: str, backend="auto", *, adapters=None) -> RunResult:
        if raises is not None:
            raise raises
        if isinstance(replies, str):
            text = replies
        elif "quadratic-regression" in prompt:
            text = replies["regression"]
        elif "parabola" in prompt:
            text = replies["parabola"]
        else:
            text = replies["reasoning"]
        return RunResult(
            text=text,
            requested_backend=backend,
            actual_backend="gpu",
            runtime="Ollama test",
            model="gpt-oss:20b",
            device="AMD Radeon 890M Graphics (RADV STRIX1)",
            elapsed_ms=1.0,
            fallback=False,
        )

    return run


def test_every_case_renders_through_the_real_constructors() -> None:
    """The harness writes no prompt of its own; solve_math builds all of them."""
    from facet_runtime.graph import parabola_prompt, regression_prompt
    from facet_runtime.solve import reasoning_prompt

    reasoning = prompts.case("reasoning")
    parabola = prompts.case("parabola")
    regression = prompts.case("regression")

    assert prompts.render(reasoning) == reasoning_prompt(reasoning.problem)
    assert prompts.render(parabola) == parabola_prompt(
        parabola.problem.instruction,
        parabola.problem.expressions,
        parabola.problem.graph,
    )
    assert prompts.render(regression) == regression_prompt(
        regression.problem.instruction, regression.problem.points
    )


def test_an_exactly_answered_case_reaches_no_model() -> None:
    """Rendering nothing is the routing's answer, not a gap in the harness."""
    assert prompts.render(prompts.case("exact")) is None

    row = prompts.check(prompts.case("exact"), run=answering("unused"))

    assert row["model_calls"] == 0
    assert row["route"] == "exact"
    assert row["answer"]["display"] == "18i"
    assert row["passed"] is True


def test_a_sweep_of_correct_replies_passes_every_case() -> None:
    report = prompts.check_all(run=answering(CORRECT))

    assert report["all_passed"] is True
    assert report["passed"] == report["total"] == len(prompts.CASES)
    assert [row["case"] for row in report["cases"]] == [
        subject.name for subject in prompts.CASES
    ]
    assert {row["model_calls"] for row in report["cases"]} == {0, 1}


def test_a_case_that_pins_no_answer_still_has_to_answer() -> None:
    """The domain has many spellings, so only the contract is checked."""
    row = prompts.check(prompts.case("reasoning"), run=answering(CORRECT))

    assert row["expected_met"] is None
    assert row["passed"] is True
    assert row["answer"]["display"].startswith("(-\\infty")


def test_a_wrong_but_well_formed_answer_does_not_pass() -> None:
    """A schema-valid regression that is wrong is still wrong."""
    wrong = dict(CORRECT)
    wrong["regression"] = (
        '{"kind":"quadratic-regression","coefficients":["3","18","21"]}'
    )

    row = prompts.check(prompts.case("regression"), run=answering(wrong))

    assert row["status"] == "ok"
    assert row["expected_met"] is False
    assert row["passed"] is False


def test_a_reply_of_the_wrong_shape_is_reported_not_repaired() -> None:
    """The harness must never soften a parser to make a case go green."""
    row = prompts.check(
        prompts.case("regression"),
        run=answering({"regression": '{"coefficients":["3","18","20"]}'}),
    )

    assert row["status"] == "unusable_result"
    assert row["passed"] is False
    assert "exactly kind, coefficients" in row["error"]


def test_an_empty_completion_is_reported_with_the_runtimes_own_reason() -> None:
    """The live failure, as this harness would have shown it."""
    row = prompts.check(
        prompts.case("regression"),
        run=answering(
            CORRECT,
            raises=FacetRuntimeError(
                "Ollama stopped gpt-oss:20b on its output cap after 1024 of "
                "1024 tokens, all of it internal reasoning"
            ),
        ),
    )

    assert row["status"] == "execution_failed"
    assert row["passed"] is False
    assert "output cap" in row["error"]
    assert row["model_calls"] == 1


def test_a_live_row_carries_the_evidence_a_reader_needs() -> None:
    row = prompts.check(
        prompts.case("regression"), run=answering(CORRECT), keep_prompt=True
    )

    assert row["model_calls"] == 1
    assert row["prompt_chars"] == len(row["prompt"])
    assert row["prompt"] == prompts.render(prompts.case("regression"))
    assert row["device"].startswith("AMD Radeon 890M")
    assert row["elapsed_ms"] >= 0


@pytest.mark.parametrize(
    ("actual", "expected", "held"),
    [
        ({"a": 1, "b": 2}, {"a": 1}, True),
        ({"a": 1}, {"a": 1, "b": 2}, False),
        ({"a": {"b": 1, "c": 2}}, {"a": {"b": 1}}, True),
        ({"a": {"b": 1}}, {"a": {"b": 2}}, False),
        ({"a": ["1", "2"]}, {"a": ["1", "2"]}, True),
        ({"a": ["1", "2", "3"]}, {"a": ["1", "2"]}, False),
        ({"a": 1}, {"b": 1}, False),
    ],
)
def test_expected_is_a_subset_that_must_agree(actual, expected, held) -> None:
    assert prompts.contains(actual, expected) is held


def test_an_unknown_case_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown prompt case"):
        prompts.case("nonexistent")


def test_the_command_renders_without_reaching_a_model(capsys) -> None:
    from facet_runtime.cli import run_cli

    assert run_cli(["prompts", "--case", "exact,regression"]) == 0
    rendered = json.loads(capsys.readouterr().out)

    assert [row["case"] for row in rendered["cases"]] == ["exact", "regression"]
    assert rendered["cases"][0]["reaches_a_model"] is False
    assert rendered["cases"][1]["prompt"] == prompts.render(prompts.case("regression"))


def test_the_command_leaves_nonzero_when_a_prompt_stops_answering(
    monkeypatch, capsys
) -> None:
    """It is a gate, not only a report."""
    from facet_runtime.cli import run_cli

    monkeypatch.setattr(
        prompts,
        "run_prompt",
        answering(
            CORRECT, raises=FacetRuntimeError("Ollama stopped it on its output cap")
        ),
    )

    assert run_cli(["prompts", "--live", "--case", "regression"]) == 1
    report = json.loads(capsys.readouterr().out)

    assert report["all_passed"] is False
    assert report["cases"][0]["status"] == "execution_failed"


def test_the_command_leaves_zero_when_every_prompt_answers(monkeypatch, capsys) -> None:
    from facet_runtime.cli import run_cli

    monkeypatch.setattr(prompts, "run_prompt", answering(CORRECT))

    assert run_cli(["prompts", "--live", "--case", "regression"]) == 0
    assert json.loads(capsys.readouterr().out)["all_passed"] is True


#: Every shape `reasoning_prompt` can take. Its label, its prefix line and its
#: multi-part contract are each built by a separate branch, and the live cases
#: exercise only one of them, so the vocabulary gate would otherwise never see
#: the other three.
PROMPT_VARIANTS = (
    MathProblem(instruction="Find the domain.", expressions=("1/(x-2)",)),
    MathProblem(
        instruction="Find the domain.",
        expressions=("1/(x-2)",),
        label="Question 4 of 12",
    ),
    MathProblem(
        instruction="Solve the following formula for the indicated variable. "
        "Solve for r.",
        expressions=("C=2*pi*r",),
    ),
    MathProblem(
        instruction="Find the x-intercepts.",
        expressions=("3*x^2+2*x-8",),
        answer_parts=2,
    ),
)


def test_no_prompt_speaks_any_consumers_vocabulary() -> None:
    """The prompts are Facet's, and name nobody they are being asked on behalf of.

    This checks the wording Facet writes. The instruction inside a prompt is
    the caller's own sentence and stays as it was sent; the fixtures here use
    neutral ones so that what is left is exactly Facet's contribution.
    """
    for prompt in prompts.model_facing_prompts():
        assert prompts.leaks(prompt) == (), prompt


def test_every_branch_of_the_value_prompt_is_checked_too() -> None:
    for variant in PROMPT_VARIANTS:
        prompt = reasoning_prompt(variant)
        assert prompts.leaks(prompt) == (), prompt


def test_the_value_prompt_still_says_everything_it_has_to() -> None:
    """Generalising the wording must not drop a requirement from the contract."""
    plain, labelled, prefixed, multi = (
        reasoning_prompt(variant) for variant in PROMPT_VARIANTS
    )

    assert plain.startswith("Solve this precalculus question.")
    assert "FINAL ANSWER:" in plain and "Do not explain." in plain
    assert "Question: Question 4 of 12" in labelled
    assert "`r =` is already written for you" in prefixed
    assert "This question takes 2 separate answers." in multi
    assert "PART 1: answer number 1 by itself" in multi
    assert "PART 2: answer number 2 by itself" in multi


def test_the_regression_prompt_settles_the_rounding_it_used_to_argue_with() -> None:
    """The contradiction that cost a whole output budget is decided in the prompt."""
    prompt = prompts.render(prompts.case("regression"))

    # The caller's own instruction still crosses untouched...
    assert "Round to three decimal places." in prompt
    # ...and the prompt says which of the two wins, so the model need not.
    assert "NOT decimal approximations" in prompt
    assert "even where the instruction asks for rounded ones" in prompt


def test_the_word_check_reads_words_and_not_fragments() -> None:
    assert prompts.leaks("Find the domain of the function.") == ()
    assert prompts.leaks("A box on the page") == ("box", "page")
    assert prompts.leaks("Ethnos verifies this") == ("ethnos",)
