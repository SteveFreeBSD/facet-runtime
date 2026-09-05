"""Golden snapshots of every prompt Facet sends, so a rewording cannot be silent.

Until now every assertion about prompt wording was a substring: `"FINAL ANSWER:"
in prompt`, a handful of forbidden words, and one test that compared a prompt to
itself. All of them pass no matter how the rest of the sentence is written, which
is how nine consumer-specific lines survived in the prompts for as long as they
did, and how the clause that eventually cost a whole output budget went unread.

A golden file is the missing half. It holds the exact text of one prompt, so any
edit to the wording -- deliberate or accidental -- shows up as a diff in a file a
reviewer reads, rather than as nothing at all. Deliberate is easy:

    FACET_UPDATE_GOLDEN=1 uv run --frozen pytest -q tests/test_prompt_snapshots.py

then read `git diff` and commit it as the intended change.

Nothing here builds a prompt. The value prompts are rendered by
`reasoning_prompt` and the two plan prompts by `solve_math` through the harness,
which are the production paths; a golden is what production would send. The value
prompts are rendered by calling the constructor rather than by routing, because
several of these questions -- `Solve for r` among them -- are answered exactly and
would never reach a model at all. That the constructor is what routing actually
calls is held separately, by
`test_prompts.py::test_every_case_renders_through_the_real_constructors`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from facet_runtime import prompts
from facet_runtime.solve import (
    MAX_ANSWER_PARTS,
    MathProblem,
    answer_prefix,
    reasoning_prompt,
)

GOLDEN = Path(__file__).parent / "fixtures" / "prompts"

#: Rewrite the goldens instead of comparing to them. Deliberately an
#: environment variable: regenerating is a decision, and the diff it leaves is
#: the thing a reviewer is meant to look at.
UPDATING = os.environ.get("FACET_UPDATE_GOLDEN") == "1"

REGENERATE = (
    "the prompt no longer matches its golden file. If the change was intended, "
    "regenerate with FACET_UPDATE_GOLDEN=1 pytest tests/test_prompt_snapshots.py "
    "and commit the diff."
)

#: Every shape the value prompt takes. `reasoning_prompt` assembles four
#: optional pieces, and each of them changes the text that reaches a model:
#: the `Question:` heading, the `x =` prefix line, the expression list, and one
#: of two closing contracts whose multi-part form also counts its own lines. A
#: case here for each, and one with all of them at once, because the pieces are
#: concatenated and an interaction between them would otherwise be unread.
VALUE_PROMPTS: tuple[tuple[str, MathProblem], ...] = (
    (
        "reasoning_plain",
        MathProblem(
            instruction="Find the domain of the following function.",
            expressions=(r"\frac{x+1}{x^2-9}",),
        ),
    ),
    (
        "reasoning_labelled",
        MathProblem(
            instruction="Find the domain of the following function.",
            expressions=(r"\frac{x+1}{x^2-9}",),
            label="Question 4 of 12",
        ),
    ),
    (
        "reasoning_prefixed",
        MathProblem(
            instruction=(
                "Solve the following formula for the indicated variable. Solve for r."
            ),
            expressions=("C=2*pi*r",),
        ),
    ),
    (
        "reasoning_several_expressions",
        MathProblem(
            instruction="Find the domain of each function.",
            expressions=(r"\frac{x+1}{x^2-9}", r"\sqrt{x-4}", r"\log(x+2)"),
        ),
    ),
    (
        "reasoning_two_parts",
        MathProblem(
            instruction="Find the x-intercepts.",
            expressions=("3*x^2+2*x-8",),
            answer_parts=2,
        ),
    ),
    (
        "reasoning_most_parts",
        MathProblem(
            instruction="Find all of the roots.",
            expressions=("x^4-16",),
            answer_parts=MAX_ANSWER_PARTS,
        ),
    ),
    # A named variable and several answers at once, isolated from the noise of
    # the every-branch case below. Reachable two ways in production -- an
    # observed multi control, or an instruction asking for comma-separated
    # answers -- and this is the pairing whose wording used to disagree with
    # itself, so it gets a snapshot of its own.
    (
        "reasoning_prefixed_multi",
        MathProblem(
            instruction="Solve for x. Separate multiple answers with a comma.",
            expressions=("x^2-5*x+6=0",),
            answer_parts=2,
        ),
    ),
    (
        "reasoning_every_branch",
        MathProblem(
            instruction=(
                "Solve the following formula for the indicated variable. Solve for r."
            ),
            expressions=("C=2*pi*r", "A=pi*r^2"),
            answer_parts=2,
            label="Question 7 of 12",
        ),
    ),
)

#: The two specialists. Their wording does not branch, but the JSON payload
#: their prompt carries is built from the request, so the golden pins both. The
#: problems are the harness's own live cases, so a golden is byte for byte what
#: `facet prompts --live` sends.
PLAN_PROMPTS: tuple[tuple[str, str], ...] = (
    ("parabola_plan", "parabola"),
    ("quadratic_regression", "regression"),
)

GOLDEN_NAMES = tuple(name for name, _ in (*VALUE_PROMPTS, *PLAN_PROMPTS))


def golden_path(name: str) -> Path:
    return GOLDEN / f"{name}.txt"


def compare(name: str, rendered: str) -> None:
    """Hold one prompt against its golden, exactly, or rewrite it on request."""
    path = golden_path(name)
    if UPDATING:
        path.write_text(rendered, encoding="utf-8")
        return
    assert path.exists(), f"no golden file for {name}: {REGENERATE}"
    # Exact text, including every newline. The prompts end without a trailing
    # one, and so do these files.
    assert rendered == path.read_text(encoding="utf-8"), REGENERATE


def rendered(name: str) -> str:
    """One golden's prompt, from whichever production path builds it."""
    for known, problem in VALUE_PROMPTS:
        if known == name:
            return reasoning_prompt(problem)
    for known, live_case in PLAN_PROMPTS:
        if known == name:
            prompt = prompts.render(prompts.case(live_case))
            assert prompt is not None, f"{live_case} reached no model"
            return prompt
    raise AssertionError(f"unknown golden: {name}")


@pytest.mark.parametrize("name", [name for name, _ in VALUE_PROMPTS])
def test_every_shape_of_the_value_prompt_matches_its_golden(name: str) -> None:
    compare(name, reasoning_prompt(dict(VALUE_PROMPTS)[name]))


@pytest.mark.parametrize(("name", "live_case"), PLAN_PROMPTS)
def test_every_plan_prompt_matches_its_golden(name: str, live_case: str) -> None:
    prompt = prompts.render(prompts.case(live_case))

    assert prompt is not None, f"the {live_case} case reached no model"
    compare(name, prompt)


def test_the_goldens_still_cover_every_branch_that_changes_wording() -> None:
    """A snapshot suite that quietly loses a branch is worse than none.

    This checks the *inputs*, so it fails when a case is dropped or edited into
    duplicating another, rather than only when a prompt changes.
    """
    problems = [problem for _, problem in VALUE_PROMPTS]

    assert any(not problem.label.strip() for problem in problems), "no unlabelled case"
    assert any(problem.label.strip() for problem in problems), "no labelled case"
    assert any(not answer_prefix(problem.instruction) for problem in problems), (
        "no case without the prefix line"
    )
    assert any(answer_prefix(problem.instruction) for problem in problems), (
        "no case with the prefix line"
    )
    assert any(problem.answer_parts == 1 for problem in problems), (
        "no single-value case"
    )
    assert any(problem.answer_parts > 1 for problem in problems), "no multi-part case"
    assert any(problem.answer_parts == MAX_ANSWER_PARTS for problem in problems), (
        "the largest answer_parts is never rendered"
    )
    assert any(len(problem.expressions) == 1 for problem in problems)
    assert any(len(problem.expressions) > 1 for problem in problems)
    # A named variable together with several answers: the pairing whose
    # wording used to disagree with itself, and a real question either way.
    assert any(
        answer_prefix(problem.instruction) and problem.answer_parts > 1
        for problem in problems
    ), "no case names a variable and asks for several answers at once"
    # And one case where every optional piece is present at once.
    assert any(
        problem.label.strip()
        and answer_prefix(problem.instruction)
        and problem.answer_parts > 1
        and len(problem.expressions) > 1
        for problem in problems
    ), "no case exercises all four pieces together"


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_no_golden_speaks_a_consumers_vocabulary(name: str) -> None:
    """The prohibition applies to the exact bytes, not only to a live render."""
    text = golden_path(name).read_text(encoding="utf-8")

    assert prompts.leaks(text) == (), text


def test_every_golden_file_on_disk_is_one_a_test_reads() -> None:
    """A renamed case must not leave its old snapshot behind to rot."""
    on_disk = {path.stem for path in GOLDEN.glob("*.txt")}

    assert on_disk == set(GOLDEN_NAMES)


@pytest.mark.parametrize("name", GOLDEN_NAMES)
def test_a_golden_is_a_whole_prompt_and_not_an_excerpt(name: str) -> None:
    """Cheap insurance against a truncated or hand-edited snapshot."""
    text = golden_path(name).read_text(encoding="utf-8")

    assert text == rendered(name)
    assert text.strip() == text.rstrip(), "a golden must not start with whitespace"
    assert not text.endswith("\n"), "the prompts carry no trailing newline"
