"""Reading one final answer out of a block of solver or model text.

Both of Facet's routes produce prose around their answer: the exact solvers
write their working and their verification, and a reasoning model writes
whatever it writes. Exactly one line of either is the answer, and this is the
single place that says which.
"""

from __future__ import annotations

import re

_ESCAPED_UNICODE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _decode_escaped_unicode(text: str) -> str:
    r"""Turn an escape a model wrote out as text back into the character.

    Live, `√-27` came back as `3i×sqrt(3)`: six literal characters where a
    multiplication sign belongs. A model asked for one line of mathematics will
    sometimes emit JSON's escape form, and nothing downstream expects it --
    `keyboard_entry_for_math` mangled it further into `3*i\u*221*a*5`, and the
    add-on's answer pattern refuses a backslash outright, so the whole answer
    was dropped and the panel showed the backslash-stripped remains.

    Safe against LaTeX, which has no `\uXXXX` control sequence: `\underline`
    and `\upsilon` both fail the four-hex-digit test on their second character.
    """
    return _ESCAPED_UNICODE.sub(lambda match: chr(int(match.group(1), 16)), text)


def extract_final_math(answer_text: str) -> str | None:
    """Extract the solver's final displayed expression for a compact answer card."""
    answer_text = _decode_escaped_unicode(answer_text)
    boxed_start = answer_text.rfind(r"\boxed{")
    if boxed_start >= 0:
        content_start = boxed_start + len(r"\boxed{")
        depth = 1
        for index in range(content_start, len(answer_text)):
            character = answer_text[index]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return answer_text[content_start:index].strip()

    matches = list(re.finditer(r"(?im)^\s*FINAL ANSWER:\s*(.+?)\s*$", answer_text))
    if matches:
        return matches[-1].group(1).strip().strip("$`")
    return None
