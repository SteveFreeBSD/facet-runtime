"""Completing a table of values exactly, and refusing an answer that does not.

Lesson 2.1 states `x = y²` and a grid with five blank cells. Every blank is
determined by the row it sits in, so the question has a deterministic answer
and never needed a model at all. What it got was one: the reasoning route was
asked for five values, returned five, and a live run returned five wrong ones --
because five parts arriving is not five parts being right.

These tests cover both halves. The route that computes the answer, and the
proof any answer has to survive before it counts as one.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact import (
    TableRefused,
    complete_table,
    parse_answer_table,
    parse_representation,
    solve_exact,
    verify_completion,
)
from facet_runtime.solve import SolveRefused, parse_problem, solve_math

RELATION = ["x=y^2"]

#: The live grid, exactly as the page states it: y for x=0, x for y=2√2, y for
#: x=64, y for x=25, and x for y=-√3.
LIVE_ROWS = [
    [{"value": "0"}, {"blank": 1}],
    [{"blank": 2}, {"value": r"2\sqrt{2}"}],
    [{"value": "64"}, {"blank": 3}],
    [{"value": "25"}, {"blank": 4}],
    [{"blank": 5}, {"value": r"-\sqrt{3}"}],
]

INTEGER_4 = {"kind": "signed-integer", "max_length": 4}


def table(rows=None, columns=("x", "y")):
    return parse_answer_table(
        {"columns": list(columns), "rows": LIVE_ROWS if rows is None else rows}
    )


def representation(payload=None):
    return parse_representation(INTEGER_4 if payload is None else payload)


class Refused:
    """A model that fails the test if anything asks it to reason."""

    def __call__(self, prompt):  # pragma: no cover - only runs on failure
        raise AssertionError("the deterministic route should have answered")


# --- the exact route --------------------------------------------------------


def test_the_live_grid_is_completed_exactly() -> None:
    done = complete_table(
        RELATION, table(), answer_parts=5, representation=representation()
    )

    assert done.parts == ("0", "8", "8", "5", "3")


def test_every_row_states_its_own_working() -> None:
    """The candidates a row admitted, and which of them was taken."""
    done = complete_table(
        RELATION, table(), answer_parts=5, representation=representation()
    )

    assert done.evidence["relation"] == "x - y**2 = 0"
    assert done.evidence["row 1"] == "x=0 -> y in {0} -> 0"
    assert done.evidence["row 3"] == "x=64 -> y in {8, -8} -> 8"


def test_a_row_with_two_roots_is_answered_by_the_canonical_one() -> None:
    """Smallest in magnitude, and the non-negative one where two share it.

    `x = 64` is satisfied by 8 and by -8, and Hawkes takes either. Choosing is
    unavoidable and being consistent about it is the whole requirement, so the
    rule is stated once and applied everywhere: the simplest representative,
    and the non-negative one among equals.
    """
    both = complete_table(
        RELATION,
        table([[{"value": "64"}, {"blank": 1}], [{"value": "25"}, {"blank": 2}]]),
        answer_parts=2,
        representation=representation(),
    )

    assert both.parts == ("8", "5")


def test_the_smaller_magnitude_wins_before_the_sign_does() -> None:
    """`(y-2)(y+3) = 0` admits 2 and -3; 2 is the smaller in magnitude."""
    done = complete_table(
        ["x=(y-2)*(y+3)"],
        table([[{"value": "0"}, {"blank": 1}], [{"value": "-6"}, {"value": "0"}]]),
        answer_parts=1,
    )

    assert done.parts == ("2",)


def test_radicals_and_rationals_stay_exact() -> None:
    """Nothing is rounded on the way in or on the way out."""
    done = complete_table(
        ["y=x^2"],
        table(
            [
                [{"value": r"\sqrt{2}"}, {"blank": 1}],
                [{"value": r"\frac{1}{3}"}, {"blank": 2}],
            ]
        ),
        answer_parts=2,
    )

    assert done.parts == ("2", "1/9")


def test_a_solution_that_cannot_be_written_that_way_is_refused_not_rounded() -> None:
    """`y = 1/3` is exact and is not an integer, so the table is declined."""
    with pytest.raises(TableRefused, match="no solution the answer may be written"):
        complete_table(
            ["y=x"],
            table([[{"value": r"\frac{1}{3}"}, {"blank": 1}],
                   [{"value": "2"}, {"value": "2"}]]),
            answer_parts=1,
            representation=representation(),
        )


def test_a_solution_too_long_for_the_answer_is_refused() -> None:
    with pytest.raises(TableRefused, match="no solution the answer may be written"):
        complete_table(
            ["y=x"],
            table([[{"value": "123456"}, {"blank": 1}],
                   [{"value": "2"}, {"value": "2"}]]),
            answer_parts=1,
            representation=representation(),
        )


# --- failing closed ---------------------------------------------------------


@pytest.mark.parametrize(
    "expressions,rows,columns,why",
    [
        (["x=y^2"], LIVE_ROWS, ("x", "revenue per photo"),
         "column heading is not a variable"),
        (["x=y^2", "y=x"], LIVE_ROWS, ("x", "y"), "one stated relation"),
        (["x+y"], LIVE_ROWS, ("x", "y"), "not one equation"),
        (["x=z^2"], LIVE_ROWS, ("x", "y"), "names something the table does not"),
        (["x=y^2"], [[{"blank": 1}, {"blank": 2}],
                     [{"value": "1"}, {"value": "1"}]], ("x", "y"),
         "more than one blank"),
        (["x=y^2"], [[{"value": "9"}, {"value": "2"}],
                     [{"value": "4"}, {"blank": 1}]], ("x", "y"),
         "contradicts the stated relation"),
        (["x=y^2"], [[{"value": "-1"}, {"blank": 1}],
                     [{"value": "4"}, {"value": "2"}]], ("x", "y"),
         "no exact solution"),
        (["x*y=0"], [[{"value": "0"}, {"blank": 1}],
                     [{"value": "1"}, {"value": "0"}]], ("x", "y"),
         "does not constrain its blank"),
    ],
    ids=[
        "phrase-heading", "two-relations", "not-an-equation", "foreign-symbol",
        "two-blanks-in-a-row", "inconsistent-row", "no-real-solution",
        "blank-unconstrained",
    ],
)
def test_an_ambiguous_or_unsupported_table_declines(
    expressions, rows, columns, why
) -> None:
    with pytest.raises(TableRefused, match=why):
        complete_table(
            expressions,
            table(rows, columns),
            answer_parts=sum(
                1 for row in rows for cell in row if "blank" in cell
            ),
        )


def test_a_grid_and_a_part_count_that_disagree_decline() -> None:
    with pytest.raises(TableRefused, match="blanks and the answer's parts disagree"):
        complete_table(RELATION, table(), answer_parts=4)


@pytest.mark.parametrize(
    "payload,why",
    [
        ({"columns": ["x"], "rows": LIVE_ROWS}, "columns must be"),
        ({"columns": ["x", "x"], "rows": LIVE_ROWS}, "share a name"),
        ({"columns": ["x", "y"], "rows": [[{"value": "1"}, {"value": "2"}]] * 2},
         "has a blank in it"),
        ({"columns": ["x", "y"], "rows": [
            [{"value": "1"}, {"blank": 2}], [{"value": "2"}, {"blank": 3}]]},
         "numbered from one"),
        ({"columns": ["x", "y"], "rows": [
            [{"value": "1", "blank": 1}, {"value": "2"}],
            [{"value": "2"}, {"blank": 2}]]},
         "one blank or one value"),
    ],
    ids=["one-column", "repeated-column", "no-blank", "misnumbered", "cell-is-both"],
)
def test_a_malformed_grid_is_refused_at_the_door(payload, why) -> None:
    with pytest.raises(TableRefused, match=why):
        parse_answer_table(payload)


# --- verifying an answer that came from somewhere else ----------------------


def test_the_computed_answer_verifies() -> None:
    verify_completion(
        ("0", "8", "8", "5", "3"), RELATION, table(), representation=representation()
    )


def test_the_other_root_verifies_too() -> None:
    """`-8` and `-5` are as true as `8` and `5`, and the verifier says so.

    The canonical rule decides what this route *produces*. It has no business
    deciding what is correct, and a reasoned answer that took the other root
    must not be rejected for disagreeing with a tie-break.
    """
    verify_completion(
        ("0", "8", "-8", "-5", "3"), RELATION, table(), representation=representation()
    )


@pytest.mark.parametrize(
    "parts,why",
    [
        (("0", "8", "8", "5", "9"), "does not satisfy row 5"),
        (("0", "8", "5", "8", "3"), "does not satisfy row 3"),
        (("1", "8", "8", "5", "3"), "does not satisfy row 1"),
        (("0", "8", "8", "5"), "a part for every blank"),
        (("0", "8", "8", "5", "3", "1"), "a part for every blank"),
        (("0", "8", "8", "5", "x"), "not an exact real number"),
        (("0", "8", "8", "5", "<b>3</b>"), "not exact mathematics"),
    ],
    ids=["wrong-value", "swapped", "first-wrong", "too-few", "too-many",
         "symbolic", "not-mathematics"],
)
def test_an_answer_that_does_not_complete_the_table_is_refused(parts, why) -> None:
    with pytest.raises(TableRefused, match=why):
        verify_completion(parts, RELATION, table(), representation=representation())


def test_an_exact_but_unwritable_part_is_refused() -> None:
    """`2√2` satisfies no row here, and could not be typed if it did."""
    with pytest.raises(TableRefused, match="not written the way"):
        verify_completion(
            ("0", "8", "8", "5", r"2\sqrt{2}"),
            RELATION,
            table(),
            representation=representation(),
        )


def test_a_part_may_be_a_radical_when_the_question_permits_one() -> None:
    """Without a representation, exactness is the only requirement."""
    verify_completion(
        (r"\sqrt{2}",),
        ["y=x^2"],
        table([[{"blank": 1}, {"value": "2"}], [{"value": "3"}, {"value": "9"}]]),
    )


# --- the route Facet takes --------------------------------------------------


def test_the_router_takes_the_table_route_and_names_it() -> None:
    solution, decline = solve_exact(
        "Complete the table of values below.",
        RELATION,
        answer_parts=5,
        table=table(),
        representation=representation(),
    )

    assert decline == ""
    assert solution.parts == ("0", "8", "8", "5", "3")
    assert solution.method == "SymPy exact table completion"
    assert solution.entry_mode == "math"


def test_a_refused_table_declines_with_its_reason() -> None:
    solution, decline = solve_exact(
        "Complete the table.", ["x=y^2"], answer_parts=4, table=table()
    )

    assert solution is None
    assert "the table was not completed exactly" in decline


def test_no_table_leaves_every_other_exact_route_alone() -> None:
    """The one guarantee this change owes everything that came before it."""
    solution, _ = solve_exact("Simplify the expression.", ["(x+1)*(x+1)"])

    assert solution is not None
    assert solution.method == "SymPy exact symbolic"


def problem(parts: int = 5, **changes):
    payload = {
        "instruction": "Complete the table of values below for the given equation.",
        "expressions": RELATION,
        "answer_parts": parts,
        "answer_table": {"columns": ["x", "y"], "rows": LIVE_ROWS[:parts]},
        "answer_representation": INTEGER_4,
    }
    payload.update(changes)
    return parse_problem(payload)


def test_the_live_question_reaches_no_model_at_all() -> None:
    result = solve_math(problem(), reason=Refused())

    assert result["route"] == "exact"
    assert result["answer"]["parts"] == ["0", "8", "8", "5", "3"]
    assert result["provenance"]["evidence"]["model_calls"] == 0
    assert result["provenance"]["method"] == "SymPy exact table completion"
    assert result["provenance"]["actual_backend"] is None


def test_a_grid_that_disagrees_with_the_part_count_is_refused_as_a_request() -> None:
    with pytest.raises(SolveRefused, match="blanks and answer_parts is"):
        problem(parts=4, answer_table={"columns": ["x", "y"], "rows": LIVE_ROWS})


def test_the_grid_is_stated_to_a_model_only_when_one_is_asked() -> None:
    """It cannot be computed, so the reasoning route is told what the grid is."""
    from facet_runtime.solve import reasoning_prompt

    asked = reasoning_prompt(problem(answer_table={
        "columns": ["x", "y"],
        "rows": [[{"value": "0"}, {"blank": 1}], [{"blank": 2}, {"value": "9"}]],
    }, parts=2))

    assert "0 | (part 1)" in asked
    assert "(part 2) | 9" in asked
    assert "This question takes 2 separate answers." in asked


# --- the backstop, where a reasoned answer becomes an answer -----------------


def rounding_problem(parts: int = 1):
    """A table the exact route declines, because a third is not an integer."""
    return parse_problem({
        "instruction": "Complete the table of values below.",
        "expressions": ["y=x"],
        "answer_parts": parts,
        "answer_table": {
            "columns": ["x", "y"],
            "rows": [
                [{"value": r"\frac{1}{3}"}, {"blank": 1}],
                [{"value": "2"}, {"value": "2"}],
            ],
        },
        "answer_representation": INTEGER_4,
    })


def answering(text: str):
    from facet_runtime.result import RunResult

    def reason(prompt):
        return RunResult(
            text=text,
            requested_backend="gpu",
            actual_backend="gpu",
            runtime="fake",
            model="fake-model",
            device="fake device",
            elapsed_ms=1.0,
            fallback=False,
            evidence={"source": "test"},
        )

    return reason


def test_a_rounded_reasoned_answer_is_rejected_rather_than_presented() -> None:
    """The exact route declined because no integer completes the row.

    A model will happily produce one anyway. Substituting it back is what says
    so, and there is no arithmetic anywhere in this path that could turn a
    third into a zero and call it an answer.
    """
    with pytest.raises(SolveRefused, match="does not complete the table"):
        solve_math(rounding_problem(), reason=answering("FINAL ANSWER: 0"))


def test_a_reasoned_answer_that_completes_the_table_is_kept() -> None:
    """The backstop rejects what fails; it does not reject on principle.

    A quintic has no solution set this route will enumerate, so it declines and
    a model is asked. `y = 1` still satisfies the row exactly, and proving that
    costs one substitution -- which is the whole asymmetry this backstop rests
    on: finding an answer can be hard where checking one is easy.
    """
    quintic = parse_problem({
        "instruction": "Complete the table of values below.",
        "expressions": ["x=y^5+y+1"],
        "answer_parts": 1,
        "answer_table": {
            "columns": ["x", "y"],
            "rows": [[{"value": "3"}, {"blank": 1}], [{"value": "1"}, {"value": "0"}]],
        },
    })

    result = solve_math(quintic, reason=answering("FINAL ANSWER: 1"))

    assert result["route"] == "reasoning"
    assert result["provenance"]["evidence"]["answer_table"] == "verified"


def test_a_reasoned_answer_to_the_same_row_that_is_wrong_is_rejected() -> None:
    """The same question, one off. Checking is what tells the two apart."""
    quintic = parse_problem({
        "instruction": "Complete the table of values below.",
        "expressions": ["x=y^5+y+1"],
        "answer_parts": 1,
        "answer_table": {
            "columns": ["x", "y"],
            "rows": [[{"value": "3"}, {"blank": 1}], [{"value": "1"}, {"value": "0"}]],
        },
    })

    with pytest.raises(SolveRefused, match="does not complete the table"):
        solve_math(quintic, reason=answering("FINAL ANSWER: 2"))


def test_a_grid_that_cannot_be_read_leaves_the_reasoned_answer_alone() -> None:
    """Nothing was checked, and the result says so rather than implying more.

    Refusing here would refuse on the strength of a check that never ran, and
    would make every completion question with an unreadable relation
    unanswerable rather than merely unproved.
    """
    unreadable = parse_problem({
        "instruction": "Complete the table of values below.",
        # Two relations: this module will not choose between them.
        "expressions": ["y=x", "y=2*x"],
        "answer_parts": 1,
        "answer_table": {
            "columns": ["x", "y"],
            "rows": [[{"value": "3"}, {"blank": 1}], [{"value": "2"}, {"value": "2"}]],
        },
    })

    result = solve_math(unreadable, reason=answering("FINAL ANSWER: 3"))

    assert result["route"] == "reasoning"
    assert result["provenance"]["evidence"]["answer_table"].startswith("not checkable")


def test_a_question_without_a_grid_reports_no_verification_at_all() -> None:
    plain = parse_problem({
        "instruction": "Complete the sentence.",
        "expressions": ["x+1"],
    })

    result = solve_math(plain, reason=answering("FINAL ANSWER: x+1"))

    assert result["provenance"]["evidence"]["answer_table"] == "not-applicable"


def test_the_backstop_asks_the_model_exactly_once() -> None:
    """A verifier is not a filter on repeated guessing.

    Retrying until a reply passes would report an answer with no account of how
    many were thrown away to reach it, and would make a stochastic route look
    deterministic.
    """
    asked = []

    def counting(prompt):
        asked.append(prompt)
        return answering("FINAL ANSWER: 0")(prompt)

    with pytest.raises(SolveRefused):
        solve_math(rounding_problem(), reason=counting)

    assert len(asked) == 1
