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


#: How each fact may be written, as the phrase a page leads with. Anchored at
#: the start rather than over the whole string, because what follows the words
#: is a gloss and is checked separately below -- and anchored at the start all
#: the same, so "More Than One Solution" is not read as "One Solution".
_PHRASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^no\s+solutions?\b"), NO_SOLUTION),
    (re.compile(r"^(?:the\s+)?empty\s+set\b"), NO_SOLUTION),
    (re.compile(r"^(?:exactly\s+|precisely\s+)?one\s+solution\b"), ONE_SOLUTION),
    (re.compile(r"^(?:a\s+|one\s+)?unique\s+solution\b"), ONE_SOLUTION),
    (re.compile(r"^infinite(?:ly)?\s+(?:many\s+)?solutions?\b"), INFINITE_SOLUTIONS),
    (
        re.compile(r"^an?\s+infinite\s+(?:number|amount)\s+of\s+solutions?\b"),
        INFINITE_SOLUTIONS,
    ),
    (
        re.compile(r"^(?:the\s+set\s+of\s+)?all\s+real\s+numbers\b"),
        INFINITE_SOLUTIONS,
    ),
)

#: How each fact may be written as notation, in the gloss beside the words or
#: standing alone as the whole choice.
#:
#: Hawkes sets these with MathJax, so what the words are followed by is not
#: reliably the character on screen: the same `ℝ` reaches a reader as `ℝ`, as
#: the `R` its MathML carries, or twice over when an assistive copy is in the
#: element too. Which of those it is does not change the answer -- the words
#: have already said which fact this is -- so the gloss is not what decides.
#: It only has to not contradict them.
_GLOSSES: dict[str, str] = {
    "∅": NO_SOLUTION,
    "⌀": NO_SOLUTION,
    "ø": NO_SOLUTION,
    "{}": NO_SOLUTION,
    "empty": NO_SOLUTION,
    "empty set": NO_SOLUTION,
    "the empty set": NO_SOLUTION,
    "null set": NO_SOLUTION,
    "ℝ": INFINITE_SOLUTIONS,
    "ℛ": INFINITE_SOLUTIONS,
    "r": INFINITE_SOLUTIONS,
    "reals": INFINITE_SOLUTIONS,
    "all reals": INFINITE_SOLUTIONS,
    "real numbers": INFINITE_SOLUTIONS,
    "all real numbers": INFINITE_SOLUTIONS,
    "the set of all real numbers": INFINITE_SOLUTIONS,
}

#: A gloss that may stand alone as the whole choice. Narrower than `_GLOSSES`:
#: a page is free to offer `∅` itself as an alternative, but a lone `R` is a
#: variable far more often than it is the reals, and reading one as the other
#: with no words to go on is exactly the guess this module refuses to make.
_STANDALONE = frozenset({"∅", "⌀", "{}", "ℝ", "empty set", "the empty set"})

#: What separates one gloss from the next: the brackets a page wraps them in
#: and the punctuation it puts between them. Not whitespace -- "all real
#: numbers" is one gloss and splitting it into words would leave three that
#: mean nothing on their own.
_GLOSS_SEPARATORS = re.compile(r"[()\[\]{}|,;:/]+")


def _normalized(text: str) -> str:
    """One choice, reduced to what can be compared without losing meaning.

    Case and whitespace are presentation. So are the several dashes a page may
    be authored with. The compatibility forms are not: `ℝ` decomposes to a
    plain `R` under NFKC, which would silently turn the reals into a letter
    this module reads only under the protection of the words beside it. NFC
    keeps them distinct.
    """
    collapsed = unicodedata.normalize("NFC", text).replace("–", "-")
    return re.sub(r"\s+", " ", collapsed).strip().strip(".,;").casefold()


def kind_of(choice: str) -> str | None:
    """Which of the three facts this published choice states, or None.

    None means unreadable, not "no". A page is free to offer a fourth
    alternative -- "Cannot be determined" -- and reading it as one of these
    would be answering a question nobody asked.

    The words decide. What follows them is a gloss on the same fact, and is
    held only to not stating a different one: `Infinite Solutions (ℝ)` and
    `Infinite Solutions (R)` are one page's notation and another's, and both
    say what their first two words already said. `Infinite Solutions (∅)` is a
    page saying two contradictory things, and a reader that believed either
    half of it would be choosing which half to believe.
    """
    text = _normalized(choice)
    if not text:
        return None

    # A gloss standing alone as the whole choice: `∅`, `ℝ`. Read before the
    # words below, so `{}` is the empty set rather than an empty bracket.
    if text in _STANDALONE:
        return _GLOSSES[text]

    stated = next(
        ((match, kind) for pattern, kind in _PHRASES if (match := pattern.match(text))),
        None,
    )
    if stated is None:
        return None
    match, kind = stated
    phrase = match.group(0)

    # Everything after the words, with the brackets and separators taken out.
    # What is left must gloss the fact the words stated -- including those
    # words a second time, which is how an element carrying an assistive copy
    # of its own label reads.
    remainder = text[match.end() :].replace(phrase, " ")
    for segment in _GLOSS_SEPARATORS.split(remainder):
        segment = segment.strip()
        if not segment or _GLOSSES.get(segment) == kind:
            continue
        # Not one gloss whole, so it must be several written side by side --
        # `(ℝ)ℝ` is the glyph and an assistive copy of it. Each has to be a
        # gloss on this same fact in its own right.
        if all(_GLOSSES.get(word) == kind for word in segment.split()):
            continue
        return None
    return kind


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
