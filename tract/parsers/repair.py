"""Named, counted repairs for damaged source text.

A parser opts into a repair by name and declares how many times it may fire.
BaseParser refuses to write when a repair exceeds its declared ceiling, so a
transform that starts eating good text is caught by its own counter rather
than by someone reading the corpus months later.

A count is not a diff. Repairs that move text across control boundaries also
emit before/after pairs into an audit file, because a fragment attributed to
the wrong control id is a wrong compliance assertion with a plausible-looking
provenance record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final, Iterable


@dataclass(frozen=True)
class RepairResult:
    """Repaired text and how many times the repair fired."""

    text: str
    applied: int


# PDF column wrapping splits a word and leaves the hyphen surrounded by
# spaces: "secu - rity". Requires a lowercase letter on both sides, which
# distinguishes it from an aside ("the organization - The owner shall") and
# from a real compound, whose hyphen carries no spaces ("topic-specific").
_HYPHEN_BREAK: Final[re.Pattern[str]] = re.compile(r"([a-z])\s+-\s+([a-z])")


def fix_hyphen_breaks(text: str) -> RepairResult:
    """Rejoin words split across a PDF line break.

    Unrepaired these tokenize to fragments the encoder cannot match against
    anything, which is worse than the title the row would otherwise carry.
    """
    repaired, count = _HYPHEN_BREAK.subn(r"\1\2", text)
    return RepairResult(text=repaired, applied=count)


def strip_page_furniture(
    lines: Iterable[str], patterns: tuple[str, ...],
) -> tuple[list[str], int]:
    """Drop running headers and footers before row extraction.

    Returns (kept_lines, dropped_count).
    """
    compiled = [re.compile(p) for p in patterns]
    kept: list[str] = []
    dropped = 0
    for line in lines:
        if any(p.search(line) for p in compiled):
            dropped += 1
            continue
        kept.append(line)
    return kept, dropped
