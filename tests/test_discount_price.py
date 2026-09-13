"""A price after a percentage off is one division, not a reasoning question.

Live, on 2026-09-10: a Hawkes problem stating a price *after* a stated discount
and asking for the original. Nothing exact claimed it, so it went to a reasoning
model on a GPU, and what came back was the discount amount -- correct
arithmetic on the wrong unknown.

The same two numbers answer three different questions, and all three answers
are plausible prices. So these hold the reading as tightly as the arithmetic.
"""

from __future__ import annotations

import pytest

from facet_runtime.exact.discount import asked_for, solve_discount
from facet_runtime.solve import MathProblem, solve_math


def refuses(prompt: str):
    raise AssertionError("a discount question reached the reasoning route")


def answer(instruction: str, expressions: list[str] | None = None) -> dict:
    return solve_math(
        MathProblem(instruction=instruction, expressions=expressions or []),
        reason=refuses,
    )


#: The live shape: the price given is what is paid, the question wants what it
#: was before. 63.20 / 0.80 = 79.00
LIVE = (
    "A jacket is on sale for $63.20 after a 20% discount. "
    "What was the original price of the jacket?"
)


def test_the_live_question_returns_the_original_price_exactly():
    result = answer(LIVE)

    assert result["answer"]["entry"] == "79.00"
    assert result["route"] == "exact"
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


@pytest.mark.parametrize("price_expression", ["$18.90", "$*18.90", "18.90"])
def test_mathjax_holes_supply_quantities_without_duplicating_prose(price_expression):
    result = answer(
        "Edith buys last year's best-selling novel in hardcover for . "
        "This is with a discount from the original price. "
        "What was the original price of the novel?",
        [price_expression, "10%"],
    )

    assert result["answer"]["entry"] == "21.00"
    assert result["route"] == "exact"
    assert result["provenance"]["model"] is None
    assert result["provenance"]["evidence"]["model_calls"] == 0


def test_the_original_price_is_not_the_discount_amount():
    """The failure that was returned live. 79.00 - 63.20 = 15.80, which is a
    perfectly plausible price and the answer to a different question."""
    assert solve_discount(LIVE, [])[0] == "79.00"
    assert solve_discount(LIVE, [])[0] != "15.80"


@pytest.mark.parametrize(
    "instruction",
    [
        (
            "The regular price of a coat is unknown. After a 20% discount it sells "
            "for $63.20. Find the regular price."
        ),
        (
            "A television is marked down 20%. The sale price is $63.20. "
            "What was the list price?"
        ),
        (
            "After taking 20 percent off, a bicycle costs $63.20. "
            "Determine the price before the discount."
        ),
        "A shirt costs 63.20 dollars after a 20% markdown. What was its original cost?",
    ],
)
def test_the_wording_may_be_regenerated(instruction):
    """Hawkes rewrites these. The arithmetic does not move."""
    assert solve_discount(instruction, [])[0] == "79.00"


def test_the_three_questions_are_told_apart():
    """Same two numbers, three answers. Which one is read, never assumed."""
    stated = "A coat originally priced at $79.00 is discounted 20%."

    assert solve_discount(f"{stated} What is the sale price?", [])[0] == "63.20"
    assert solve_discount(f"{stated} How much is the discount?", [])[0] == "15.80"
    assert (
        solve_discount(
            "A coat is on sale for $63.20 after a 20% discount. "
            "What was the original price?",
            [],
        )[0]
        == "79.00"
    )


#: Audit F03, 2026-09-12: "The original price of a coat is . It is discounted
#: 20%. What is the sale price?" answered 80.00. The original price was looked
#: for anywhere in the question, and the sentence *stating* the price named it.
#: Each is stated in prose and as MathJax leaves it -- the numbers moved into
#: expressions, a hole left in the sentence.
STATED_ORIGINAL_80 = [
    ("The original price of a coat is $80.00. It is discounted 20%.", []),
    ("The original price of a coat is . It is discounted 20%.", ["80"]),
    ("The original price of a coat is . It is discounted .", ["$80.00", "20%"]),
]
STATED_SALE_64 = [
    (
        "A coat is on sale for $64.00. It was discounted 20% from its original price.",
        [],
    ),
    (
        "A coat is on sale for . This is with a discount from the original price.",
        ["$*64.00", "20%"],
    ),
]


@pytest.mark.parametrize(("stated", "expressions"), STATED_ORIGINAL_80 + STATED_SALE_64)
@pytest.mark.parametrize(
    ("asked", "expected"),
    [
        ("What is the sale price?", "64.00"),
        ("What is the sale price after the discount?", "64.00"),
        ("How much is the discount?", "16.00"),
        ("What was the original price?", "80.00"),
    ],
)
def test_what_is_wanted_is_read_from_what_the_question_asks(
    stated, expressions, asked, expected
):
    """Same coat, both prices, all three questions. A price a question states
    is never the price it asks for unless that is what it asks."""
    result = answer(f"{stated} {asked}", expressions)

    assert result["answer"]["entry"] == expected
    assert result["route"] == "exact"


def test_a_request_asks_for_the_first_quantity_it_names():
    """ "The sale price after a 20% discount" names the discount as what the
    sale price comes after. It was answered as the discount, 16.00."""
    assert (
        solve_discount(
            "A coat has a regular price of $80.00. "
            "What is the sale price after a 20% discount?",
            [],
        )[0]
        == "64.00"
    )


@pytest.mark.parametrize(
    ("instruction", "expressions"),
    [
        ("If a coat costs $80.00, what is the sale price after a 20% discount?", []),
        ("A coat costs . What is the sale price after a 20% discount?", ["80"]),
    ],
)
def test_what_a_question_asks_never_says_which_price_it_stated(
    instruction, expressions
):
    """The same defect from the other side. "...the sale price after a 20%
    discount" describes the price asked for, and was read as describing the
    price stated -- so $80.00 became a sale price, and the answer 20.00. Nothing
    left in what these state says which side of the discount it is on."""
    answerable, refusal = solve_discount(instruction, expressions)

    assert answerable == ""
    assert "before or after the discount" in refusal


def test_a_price_stated_before_the_request_in_one_sentence_is_still_read():
    """Separating the request must not lose the statement it shares a sentence
    with."""
    assert (
        solve_discount(
            "If a shirt originally costs $40.00, what is the sale price after 25% off?",
            [],
        )[0]
        == "30.00"
    )


def test_a_question_asking_for_two_of_them_is_declined():
    """One answer cannot be both, and taking either is a pick nobody asked for."""
    answerable, refusal = solve_discount(
        "A coat originally priced at $80.00 is discounted 20%. "
        "Find the sale price. How much is the discount?",
        [],
    )

    assert answerable == ""
    assert "does not say whether it wants" in refusal


def test_a_question_that_does_not_say_which_is_declined():
    """Guessing here is how the discount is offered as the original price."""
    answerable, refusal = solve_discount(
        "A coat is on sale for $63.20 after a 20% discount.", []
    )

    assert answerable == ""
    assert "does not say whether it wants" in refusal
    assert asked_for("A coat is on sale for $63.20 after a 20% discount.") is None


def test_two_percentages_are_a_different_question():
    """Successive discounts are not this equation."""
    answerable, refusal = solve_discount(
        "A coat is discounted 20% and then a further 10%. It sells for $63.20. "
        "What was the original price?",
        [],
    )

    assert answerable == ""
    assert "one percentage" in refusal


def test_the_arithmetic_stays_exact():
    """A third off is not 0.3333. 66.00 / (2/3) = 99.00 exactly."""
    assert (
        solve_discount(
            "A lamp sells for $66.00 after a 33% discount. "
            "What was the original price?",
            [],
        )[0]
        != ""
    )
    assert (
        solve_discount(
            "A lamp sells for $75.00 after a 25% discount. "
            "What was the original price?",
            [],
        )[0]
        == "100.00"
    )
