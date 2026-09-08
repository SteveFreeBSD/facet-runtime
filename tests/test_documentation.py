"""The documentation points where it says it points.

Facet's README is the authority for everything current, and `docs/README.md`
routes to it. Both link across the repository boundary to the consumer, whose
paths move independently -- so a link that rots here rots silently unless
something checks it.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")
MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*$", flags=re.MULTILINE)
FENCED_BLOCK = re.compile(r"^```.*?^```", flags=re.MULTILINE | re.DOTALL)


def _prose(text: str) -> str:
    """Markdown with fenced code removed.

    A shell comment at the start of a line inside a code block is not a
    heading, and counting it as one turns a correct document into a failure.
    """
    return FENCED_BLOCK.sub("", text)


def _markdown_files() -> list[Path]:
    return [
        PROJECT_ROOT / "README.md",
        *sorted((PROJECT_ROOT / "docs").rglob("*.md")),
    ]


def _heading_anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    counts: dict[str, int] = {}
    for heading in MARKDOWN_HEADING.findall(_prose(path.read_text(encoding="utf-8"))):
        slug = re.sub(r"[^\w\- ]", "", heading.lower())
        slug = re.sub(r"\s+", "-", slug.strip())
        seen = counts.get(slug, 0)
        counts[slug] = seen + 1
        anchors.add(slug if seen == 0 else f"{slug}-{seen}")
    return anchors


def test_local_markdown_links_resolve():
    """A link into the sibling repository is skipped when it is not checked out.

    Its absence is a fact about someone's working copy, not a broken link.
    """
    broken = []
    for source in _markdown_files():
        for raw in MARKDOWN_LINK.findall(source.read_text(encoding="utf-8")):
            if "://" in raw or raw.startswith("mailto:"):
                continue
            target, _, anchor = raw.partition("#")
            resolved = (source if not target else source.parent / target).resolve()
            if (
                "facet-hawkes" in raw
                and not (PROJECT_ROOT.parent / "facet-hawkes").exists()
            ):
                continue
            if not resolved.exists() or (
                anchor
                and resolved.suffix == ".md"
                and anchor not in _heading_anchors(resolved)
            ):
                broken.append(f"{source.relative_to(PROJECT_ROOT)} -> {raw}")

    assert broken == []


def test_markdown_files_have_one_top_level_heading():
    failures = []
    for source in _markdown_files():
        headings = re.findall(
            r"^#\s+(.+?)\s*$",
            _prose(source.read_text(encoding="utf-8")),
            flags=re.MULTILINE,
        )
        if len(headings) != 1:
            failures.append(f"{source.relative_to(PROJECT_ROOT)}: {len(headings)} H1s")

    assert failures == []


def test_the_active_readme_does_not_describe_facet_as_a_first_pass():
    """It said so for a while, in the same file that documents solver routing.

    An opening that contradicts the middle of its own document is worse than
    either, because a reader stops at the opening.
    """
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for retired in (
        "This first pass deliberately contains no agent architecture",
        "the only way another machine reaches Facet",
        "routing policy, tools, memory, and agent behavior belong to later",
    ):
        assert retired not in readme, retired
