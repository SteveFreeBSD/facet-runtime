"""How many solutions an equation has, said in the page's own words.

Reducing both sides of a linear equation answers three questions at once: is it
true for every value, for exactly one, or for none. That is algebra, and
`exact/symbolic.py` states it as algebra -- "No Solution", "One Solution",
"Infinite Solutions" -- deliberately in no page's vocabulary.

A page that asks the question by publishing radio buttons has its own words for
the same three facts, and they are not those. Live, on 2026-09-10, lesson 1.6
question 1: `4t + 5 = 4(t + 3) - 7`, which reduces to `0 = 0`. Facet classified
it exactly and correctly as "Infinite Solutions"; the page had published

    No Solution (∅)   One Solution   Infinite Solutions (ℝ)

and the contract that an answer to a choice question must *be* one of the
choices refused it, because "Infinite Solutions" is not "Infinite Solutions
(ℝ)". The mathematics was right and unusable. All three classifications refused
the same way, "One Solution" included.

The fix is not to teach the algebra Hawkes' notation -- a classification that
carried `(ℝ)` would be wrong everywhere else, and wrong in a way that spreads.
It is to translate, once, at the only boundary that has both: the classification
on one side and the published alternatives on the other.

`kind_of` reads a published choice back into one of the three facts, and
`select` returns the single choice that states the fact this equation has. Both
are narrow on purpose. A recognised choice is one of the three phrases, written
in one of the few ways English writes them, optionally carrying a notation for
the same fact -- `(∅)`, `(ℝ)`. Everything else is unreadable, which is not the
same as unmatched: a choice this cannot read contributes nothing rather than
being guessed at. Nothing here matches on resemblance, on substrings, or on
distance between strings; a page whose wording it cannot read declines and says
so, and the answer stays refused rather than becoming the nearest label.
"""

from __future__ import annotations

import re
import unicodedata

#: The three facts about a solution set that this vocabulary covers, named as
#: the exact solvers name them. Closed: an equation reduces to one of these
#: three cases and to nothing else, so a classification outside this set is a
#: defect rather than a case to widen for.
NO_SOLUTION = "No Solution"
ONE_SOLUTION = "One Solution"
INFINITE_SOLUTIONS = "Infinite Solutions"

SOLUTION_KINDS: frozenset[str] = frozenset(
    {NO_SOLUTION, ONE_SOLUTION, INFINITE_SOLUTIONS}
)


class SolutionKindRefused(Exception):
    """These choices cannot state this fact, and this says why."""


#: How each fact may be written in words. Anchored whole: `_ONE` must not read
#: "One Solution" out of "More Than One Solution", which states a different
#: fact and would otherwise match on the tail of this pattern.
_NO = re.compile(r"^no\s+solutions?$|^the\s+empty\s+set$|^empty\s+set$")
_ONE = re.compile(
    r"^(?:exactly\s+|precisely\s+)?one\s+solution$"
    r"|^a?\s*unique\s+solution$"
    r"|^one\s+unique\s+solution$"
)
_INFINITE = re.compile(
    r"^infinite(?:ly)?\s+(?:many\s+)?solutions?$"
    r"|^an?\s+infinite\s+number\s+of\s+solutions$"
    r"|^all\s+real\s+numbers$"
    r"|^the\s+set\s+of\s+all\s+real\s+numbers$"
)

#: How each fact may be written as notation, in a parenthetical beside the
#: words or standing alone as the whole choice. `R` is admitted only as the
#: blackboard-bold letter: a bare capital R is a variable far more often than
#: it is the reals, and reading one as the other is exactly the kind of guess
#: this module refuses to make.
_NOTATIONS: dict[str, str] = {
    "∅": NO_SOLUTION,
    "{}": NO_SOLUTION,
    "empty set": NO_SOLUTION,
    "the empty set": NO_SOLUTION,
    "ℝ": INFINITE_SOLUTIONS,
    "all reals": INFINITE_SOLUTIONS,
    "all real numbers": INFINITE_SOLUTIONS,
}

#: A notation carried beside the words, as Hawkes writes it: `(∅)`, `[ℝ]`.
_PARENTHETICAL = re.compile(r"[\(\[\{]\s*(?P<notation>[^\)\]\}]+?)\s*[\)\]\}]\s*$")


def _normalized(text: str) -> str:
    """One choice, reduced to what can be compared without losing meaning.

    Case and whitespace are presentation. So are the several dashes and the
    several apostrophes a page may be authored with, and the compatibility
    forms of the set symbols -- `ℝ` has a plain `R` as its compatibility
    decomposition, so normalizing to NFKC here would silently turn the reals
    into a letter this module refuses to read. NFC keeps them distinct.
    """
    collapsed = unicodedata.normalize("NFC", text).replace("–", "-")
    return re.sub(r"\s+", " ", collapsed).strip().strip(".,;").casefold()


def kind_of(choice: str) -> str | None:
    """Which of the three facts this published choice states, or None.

    None means unreadable, not "no". A page is free to offer a fourth
    alternative -- "Cannot be determined" -- and reading it as one of these
    would be an answer to a question nobody asked.
    """
    text = _normalized(choice)
    if not text:
        return None

    # A notation standing alone as the whole choice: a page is free to offer
    # `∅` and `ℝ` as the alternatives themselves rather than as glosses. Read
    # before the parenthetical below, so that `{}` is the empty set rather than
    # a bracket around nothing.
    standalone = _NOTATIONS.get(text)
    if standalone is not None:
        return standalone

    notation: str | None = None
    match = _PARENTHETICAL.search(text)
    if match is not None:
        # The words and the notation must state the *same* fact. "No Solution
        # (ℝ)" is a page saying two contradictory things, and a reader that
        # took either half of it would be choosing which half to believe.
        notation = _NOTATIONS.get(_normalized(match.group("notation")))
        if notation is None:
            return None
        text = text[: match.start()].strip()

    if not text:
        # The notation stood alone and is the whole choice: `∅`, `ℝ`.
        return notation

    for pattern, kind in (
        (_NO, NO_SOLUTION),
        (_ONE, ONE_SOLUTION),
        (_INFINITE, INFINITE_SOLUTIONS),
    ):
        if pattern.match(text):
            return kind if notation in (None, kind) else None
    return None


def select(kind: str, choices: list[str]) -> str:
    """The one published choice stating this fact, or raise.

    Fail closed in both directions, for the same reason `exact/quadrant.py`
    does. No choice states it: this is not the question it looked like, and the
    nearest alternative is not the right one. Two choices state it: the page
    said something this cannot read, and picking either would be picking for a
    reason nobody could check afterwards.
    """
    if kind not in SOLUTION_KINDS:
        raise SolutionKindRefused(f"{kind!r} is not a solution-set classification")
    matched = [choice for choice in choices if kind_of(choice) == kind]
    if len(matched) == 1:
        return matched[0]
    if matched:
        raise SolutionKindRefused(
            f"{len(matched)} published choices state {kind.lower()}"
        )
    raise SolutionKindRefused(
        f"this equation has {kind.lower()} and no published choice states it"
    )
