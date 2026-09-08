"""What a model may hand back as an answer, and what is a sentence about one.

Live, on a quadrant question, the answer card read:

    all answers as they would ordinarily be written

That is not an answer. It is this file's own multi-part contract, describing
the line the model was asked to write, pattern-completed instead of answered --
and it was published because the FINAL ANSWER line of a multi-part reply was
the one field on the reasoning route nothing read. The parts were checked: the
right number of them, numbered in order, non-empty. Their *rendering* was
taken on trust and became the answer a person was shown.

The tests here are about shape, not about that sentence. A reply is refused for
being made of words, whichever words those are, so the next rewording of the
prompt is covered by the same rule that covers this one.
"""

from __future__ import annotations

import pytest
from test_solve_math import (
    DECLINED_EXPRESSION,
    DECLINED_INSTRUCTION,
    TWO_PART_INSTRUCTION,
    Reasoner,
)

from facet_runtime.solve import (
    MathProblem,
    SolveRefused,
    answer_shaped,
    reasoning_prompt,
    solve_math,
)

#: The sentence that reached a live card, and the rest of the contract it came
#: from. Used as *data* here -- what the rule refuses is their shape.
ECHOED = "all answers as they would ordinarily be written"


def declined(**changes) -> MathProblem:
    fields = {
        "instruction": DECLINED_INSTRUCTION,
        "expressions": (DECLINED_EXPRESSION,),
        "answer_parts": 1,
        "label": "",
    }
    fields.update(changes)
    return MathProblem(**fields)


def two_part(**changes) -> MathProblem:
    return declined(
        instruction=TWO_PART_INSTRUCTION,
        expressions=("x^2-9=0",),
        answer_parts=2,
        **changes,
    )


# --- the shape rule itself -------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "(-∞,-3)∪(-3,3)∪(3,∞)",
        "x = -3 or x = 3",
        "-3, 3",
        "y^(23/20)",
        "√101",
        "42",
        "(1,2), (3,4)",
        # The named answers Hawkes really does ask for. They are names, and
        # names are short; this is the longest of them.
        "Not a Real Number",
        "All Real Numbers",
        "Not Factorable",
        "No Solution",
    ],
)
def test_an_answer_is_answer_shaped(answer: str) -> None:
    assert answer_shaped(answer)


@pytest.mark.parametrize(
    "prose",
    [
        ECHOED,
        "answer number 1 by itself",
        "Reply with exactly 3 lines and nothing else.",
        "After each PART label write that one answer and nothing else",
        "Write answers, never a description of what to write.",
        "The answer is the set of all real numbers except three",
        "I cannot determine the answer from the information given",
        "",
        "   ",
        "x" * 201,
    ],
)
def test_a_sentence_is_not_answer_shaped(prose: str) -> None:
    assert not answer_shaped(prose)


def test_the_rule_is_about_words_and_not_about_digits() -> None:
    """A digit anywhere used to be enough to pass, which most instructions have."""
    assert not answer_shaped("Reply with exactly 3 labelled lines and nothing else")


# --- the single-value route ------------------------------------------------


def test_prose_on_the_single_value_route_is_refused_not_published() -> None:
    reasoner = Reasoner(text=f"FINAL ANSWER: {ECHOED}")

    with pytest.raises(SolveRefused) as raised:
        solve_math(declined(), reason=reasoner)

    assert raised.value.kind == "unusable_result"
    assert "prose" in str(raised.value)


def test_a_real_single_answer_still_comes_through() -> None:
    result = solve_math(declined(), reason=Reasoner())

    assert result["route"] == "reasoning"
    assert result["answer"]["display"] == "(-∞,-3)∪(-3,3)∪(3,∞)"


# --- the multi-part route --------------------------------------------------


def test_the_rendering_is_rebuilt_from_the_parts_it_failed_to_render() -> None:
    """The live leak, exactly: checked parts behind an echoed FINAL ANSWER.

    Rebuilt rather than refused. The answer is known -- it is the parts, and
    they were checked -- and only the sentence that was supposed to present
    them was lost.
    """
    reasoner = Reasoner(text=f"FINAL ANSWER: {ECHOED}\nPART 1: -3\nPART 2: 3")

    result = solve_math(two_part(), reason=reasoner)

    assert result["answer"]["parts"] == ["-3", "3"]
    assert result["answer"]["display"] == "-3, 3"
    assert ECHOED not in result["answer"]["display"]


def test_a_rendering_that_does_not_contain_its_parts_is_rebuilt() -> None:
    """Answer-shaped is not enough: it must be these answers, written together."""
    reasoner = Reasoner(text="FINAL ANSWER: x = 7 or x = 11\nPART 1: -3\nPART 2: 3")

    result = solve_math(two_part(), reason=reasoner)

    assert result["answer"]["display"] == "-3, 3"


def test_a_genuine_rendering_of_the_parts_is_kept_as_written() -> None:
    """How several answers are written together is the model's to decide."""
    reasoner = Reasoner(text="FINAL ANSWER: x = -3 or x = 3\nPART 1: -3\nPART 2: 3")

    result = solve_math(two_part(), reason=reasoner)

    assert result["answer"]["display"] == "x = -3 or x = 3"
    assert result["answer"]["parts"] == ["-3", "3"]


def test_an_echoed_part_is_refused_rather_than_typed_into_a_box() -> None:
    """A PART line can be echoed exactly as the FINAL ANSWER line can, and a
    part is what would be typed into a real answer field."""
    reasoner = Reasoner(
        text="FINAL ANSWER: -3, 3\nPART 1: answer number 1 by itself\nPART 2: 3"
    )

    with pytest.raises(SolveRefused) as raised:
        solve_math(two_part(), reason=reasoner)

    assert raised.value.kind == "unusable_result"
    assert "part 1" in str(raised.value)


# --- the prompt that was echoed --------------------------------------------


def test_the_contract_no_longer_shows_a_line_worth_copying() -> None:
    """Not the defence -- `answer_shaped` is -- but one fewer thing to echo."""
    prompt = reasoning_prompt(two_part())

    assert "\nFINAL ANSWER:\n" in prompt
    assert ECHOED not in prompt
    # Every label ends at its colon, so echoing one yields no value at all and
    # the reply is refused for carrying no FINAL ANSWER rather than for what it
    # carried.
    for line in prompt.splitlines():
        if line.startswith(("FINAL ANSWER:", "PART ")):
            assert line.rstrip().endswith(":"), line
