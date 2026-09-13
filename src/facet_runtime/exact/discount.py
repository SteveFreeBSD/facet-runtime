"""Original price, sale price, or the discount itself -- one division apart.

A price after a stated percentage off is one equation:

    sale = original * (1 - rate)

so `original = sale / (1 - rate)` and the discount is `original - sale`. All
three are exact rational arithmetic on two numbers a person can read off the
question, and none of them is anything a model knows that dividing does not.

Live, on 2026-09-10: a price given *after* a stated discount, asked for the
original. Nothing here claimed it, it fell through to the reasoning route on a
GPU, and what came back was the discount amount -- the right arithmetic on the
wrong unknown, which is the failure this family exists to make impossible.

Which of the three is wanted is read from the question rather than assumed,
because the same two numbers answer all three and the wrong one is a plausible
number. Where the wording does not say, this declines instead of picking.
"""

from __future__ import annotations

import re
from fractions import Fraction

#: The name this solver answers to.
DISCOUNT_METHOD = "exact percentage discount"

#: A question about a price and a percentage off it. Broad enough to survive
#: rewording -- Hawkes regenerates these -- and narrow enough that it is about
#: money and a percentage, which is what the arithmetic below needs.
DISCOUNT_REQUEST = re.compile(
    r"(?=.*(?:\bdiscount(?:ed)?\b|\bmarked\s+down\b|\bmark-?down\b|\bon\s+sale\b"
    r"|\breduced\b|\b(?:percent|%)\s*off\b|\boff\s+the\b))"
    r"(?=.*(?:\bprice\b|\bcost\b|\$))",
    re.IGNORECASE | re.DOTALL,
)

#: Where a question asks rather than states: from an asking word to the end of
#: its sentence, and a decimal point inside a price does not end one.
#:
#: The two readings below are held apart by this. Audit F03, 2026-09-12: "The
#: original price of a coat is . It is discounted 20%. What is the sale price?"
#: answered 80.00, because the sentence stating the price named the original
#: price and the whole question was read for what it wanted.
_REQUEST = re.compile(
    r"\b(?:what|how\s+much|find|determine|calculate|compute)\b.*?(?:[.?!](?=\s|$)|$)",
    re.IGNORECASE | re.DOTALL,
)

#: What each quantity is called when it is asked for. A request wants the first
#: one it names: in "the sale price after the discount" the discount is what
#: the sale price comes after, not what is wanted. The order only breaks a tie.
_DISCOUNT_AMOUNT = re.compile(
    r"\b(?:amount\s+of\s+(?:the\s+)?discount|discount\s+amount|discount)\b"
    r"|\b(?:saved?|savings?|reduced\s+by|taken\s+off)\b",
    re.IGNORECASE,
)
_ORIGINAL_PRICE = re.compile(
    r"\b(?:original|list|regular|pre-?sale|before\s+the\s+discount|full)\s+"
    r"(?:price|cost|amount)\b"
    r"|\b(?:price|cost)\s+(?:before|prior)\b",
    re.IGNORECASE,
)
_SALE_PRICE = re.compile(
    r"\b(?:sale|discounted|final|new|reduced)\s+(?:price|cost|amount)\b"
    r"|\b(?:price|cost)\s+after\b|\b(?:pays?|paid)\b",
    re.IGNORECASE,
)
_WANTED = (
    (_DISCOUNT_AMOUNT, "discount"),
    (_ORIGINAL_PRICE, "original"),
    (_SALE_PRICE, "sale"),
)

#: Whether the price a sentence states is the price before the discount or the
#: price after it. Both are ordinary wordings, they lead to different
#: arithmetic on the same two numbers, and which one it is is never inferred
#: from the size of the number.
#:
#: Read from the sentence holding the price and no wider. "The regular price of
#: a coat is unknown. After a 20% discount it sells for $63.20." names both
#: sides of the discount in one question, and only the second sentence is about
#: the number: a reader taking the whole question sees both cues and can only
#: refuse a question that says exactly what it means.
_STATED_IS_ORIGINAL = re.compile(
    r"\b(?:original|regular|list|full)(?:ly)?\s+(?:price[ds]?|cost[s]?)\b"
    r"|\b(?:originally|regularly)\s+(?:priced|cost[s]?|sells?)\b"
    r"|\bprice[ds]?\s+at\b[^.?!]*\b(?:then\s+)?discounted\b"
    r"|\bbefore\s+(?:the\s+)?discount\b",
    re.IGNORECASE,
)
_STATED_IS_SALE = re.compile(
    r"\bon\s+sale\s+for\b"
    r"|\bsale\s+price\b"
    r"|\bsells?\s+for\b"
    r"|\b(?:buys?|pays?|purchased?)\b[^.?!]*\bfor\b"
    r"|\bafter\b[^.?!]*\b(?:discount(?:ed)?|mark(?:ed)?[\s-]?down|off)\b",
    re.IGNORECASE,
)

#: A MathJax value removed from prose leaves the surrounding sentence intact.
#: This locates a price-sized hole without putting the value back into the
#: prompt. It is used only when the price came from `expressions`.
_PRICE_HOLE = re.compile(r"\b(?:for|at|is|was)\s*(?=[.?!])", re.IGNORECASE)

#: A percentage: `20%`, `20 percent`, `12.5%`.
_RATE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", re.IGNORECASE)
#: An amount of money, with or without its sign, and with optional thousands
#: separators. Anchored on `$` or on being followed by a money word, so the
#: `20` in "20% off" is never read as a price.
_MONEY = re.compile(
    r"\$\s*(\d[\d,]*(?:\.\d+)?)"
    r"|(\d[\d,]*(?:\.\d+)?)\s*dollars\b",
    re.IGNORECASE,
)


class DiscountRefused(Exception):
    """This is not a question this module can answer, and says why."""


def _rational(text: str) -> Fraction:
    """One written number, exactly. Decimals stay rational; nothing floats."""
    return Fraction(text.replace(",", ""))


def _money(value: Fraction) -> str:
    """A price, written the way a price is written.

    Two decimal places when the value is exact there, because that is what a
    price is; anything that is not exact in cents is left as the fraction it
    is rather than silently rounded into a wrong answer.
    """
    cents = value * 100
    if cents.denominator != 1:
        return str(value)
    whole, remainder = divmod(int(cents), 100)
    return f"{whole}.{remainder:02d}"


def _sentence_around(question: str, match: re.Match[str] | None) -> str:
    """The one sentence a price was written in.

    The side of the discount belongs to the number, not to the question: a
    question is free to name the original price in one sentence and state the
    sale price in the next, and reading both together says only that it named
    both.
    """
    if match is None:
        return question
    starts = [0, *(found.end() for found in re.finditer(r"[.?!]\s+", question))]
    ends = [
        *(found.start() for found in re.finditer(r"[.?!]", question)),
        len(question),
    ]
    start = max((at for at in starts if at <= match.start()), default=0)
    end = min((at for at in ends if at >= match.end()), default=len(question))
    return question[start:end]


def _statements(question: str) -> str:
    """The question with what it asks blanked out, leaving what it states.

    Blanked rather than cut, so a sentence ends where it ended and nothing is
    joined to whatever followed the request.
    """
    return _REQUEST.sub(lambda request: " " * len(request.group()), question)


def asked_for(question: str) -> str | None:
    """Which of the three quantities the question wants, or None.

    Read from what the question asks and from nothing it states. None means the
    wording did not say, or asked for two of them. The same two numbers answer
    all three and every answer is a plausible price, so guessing here is how
    the discount amount gets offered as the original price.
    """
    wanted: set[str] = set()
    for request in _REQUEST.finditer(question):
        named = [
            (found.start(), name)
            for pattern, name in _WANTED
            if (found := pattern.search(request.group()))
        ]
        if named:
            wanted.add(min(named, key=lambda item: item[0])[1])
    return wanted.pop() if len(wanted) == 1 else None


def _expression_quantities(expressions: list[str]) -> tuple[list[str], list[str]]:
    """Percentages and prices stated as standalone MathJax expressions.

    Hawkes may keep the currency symbol outside the MathML, so after MathML
    conversion a price can be a bare decimal. That is unambiguous here only
    because each expression is classified independently: a percentage is
    removed first, and the remaining standalone numeric expression is money.
    """
    rates: list[str] = []
    amounts: list[str] = []
    for expression in expressions:
        normalized = expression.replace(r"\%", "%").replace(r"\$", "$").strip()
        # MathJax places an invisible multiplication operator between a
        # currency symbol and its number. The general MathML converter
        # faithfully renders that operator as `*`, producing `$*17.85` at this
        # boundary. It is currency punctuation here, not multiplication.
        normalized = re.sub(r"^\$\s*\*\s*(?=\d)", "$", normalized)
        found_rates = _RATE.findall(normalized)
        rates.extend(found_rates)
        found_amounts = [match[0] or match[1] for match in _MONEY.findall(normalized)]
        amounts.extend(found_amounts)
        if found_rates or found_amounts:
            continue
        standalone = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)", normalized)
        if standalone:
            amounts.append(standalone.group(1))
    return rates, amounts


def read_discount(
    question: str, expressions: list[str] | None = None
) -> tuple[Fraction, Fraction]:
    """The stated percentage and the one price, or raise.

    Exactly one of each. Two percentages is a question about successive
    discounts, and two prices is one that already states its own answer;
    neither is this equation, and answering either from the first number found
    would be arithmetic on inputs nobody chose.
    """
    expression_rates, expression_amounts = _expression_quantities(expressions or [])
    rates = [*_RATE.findall(question), *expression_rates]
    if len(rates) != 1:
        raise DiscountRefused(
            f"a discount question states one percentage; this states {len(rates)}"
        )
    amounts = [
        *(match[0] or match[1] for match in _MONEY.findall(question)),
        *expression_amounts,
    ]
    if len(amounts) != 1:
        raise DiscountRefused(
            f"a discount question states one price; this states {len(amounts)}"
        )
    rate = _rational(rates[0]) / 100
    if not 0 < rate < 1:
        raise DiscountRefused(f"a discount of {rates[0]}% is not a discount")
    return rate, _rational(amounts[0])


def solve_discount(instruction: str, expressions: list[str]) -> tuple[str, str]:
    """Answer a percentage-discount question exactly, or say why not.

    Returns `(answer, "")` or `("", refusal)`. The refusal is final for its
    caller in the same way `solve_quadrant`'s is: this arithmetic is two
    numbers and a division, and a model asked the same question would be
    guessing at which unknown was wanted -- which is exactly what it did.
    """
    wanted = asked_for(instruction)
    if wanted is None:
        return "", (
            "this discount question does not say whether it wants the original "
            "price, the sale price or the discount"
        )
    try:
        rate, stated = read_discount(instruction, expressions)
    except DiscountRefused as refusal:
        return "", str(refusal)

    # Which price was stated is a separate reading from which is wanted, and
    # both are needed: "originally priced at $79, discounted 20%, how much is
    # the discount" and "on sale for $63.20 after 20% off, how much was the
    # discount" state one number each and mean different ones. It is read from
    # what the question states: "...what is the sale price after a 20%
    # discount?" describes the price asked for, never the one given.
    statements = _statements(instruction)
    price_in_prose = _MONEY.search(statements)
    price_context = price_in_prose or _PRICE_HOLE.search(statements)
    clause = _sentence_around(statements, price_context)
    before = _STATED_IS_ORIGINAL.search(clause) is not None
    after = _STATED_IS_SALE.search(clause) is not None
    if before == after:
        # Neither wording, or both. A price whose side of the discount is not
        # stated is two different questions sharing a number.
        return "", (
            "this discount question does not say whether the price it states "
            "is before or after the discount"
        )
    original = stated if before else stated / (1 - rate)
    if wanted == "original":
        return _money(original), ""
    if wanted == "sale":
        return _money(original * (1 - rate)), ""
    return _money(original * rate), ""
