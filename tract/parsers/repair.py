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


_WORD: Final[re.Pattern[str]] = re.compile(r"[A-Za-z]+")


def build_vocabulary(
    texts: Iterable[str], min_length: int = 3,
) -> frozenset[str]:
    """Collect a lowercase word set from corpus text.

    Built from this corpus rather than an imported wordlist so it reflects
    security boilerplate, and so the repair is reproducible from artifacts in
    the repository rather than from whatever /usr/share/dict holds.
    """
    return frozenset(
        word.lower()
        for text in texts
        for word in _WORD.findall(text)
        if len(word) >= min_length
    )


def _segment(token: str, vocabulary: frozenset[str]) -> list[str] | None:
    """Return the fewest-segment split of *token*, or None if none exists.

    Fails closed on purpose. A partial or greedy split produces plausible
    nonsense, and the encoder cannot tell that from real text.
    """
    lowered = token.lower()
    n = len(lowered)
    # best[i] = fewest segments covering lowered[:i], with the split point.
    best: list[tuple[int, int] | None] = [None] * (n + 1)
    best[0] = (0, 0)
    for end in range(1, n + 1):
        for start in range(end):
            prefix = best[start]
            if prefix is None:
                continue
            if lowered[start:end] in vocabulary:
                candidate = (prefix[0] + 1, start)
                current = best[end]
                if current is None or candidate[0] < current[0]:
                    best[end] = candidate
    if best[n] is None:
        return None
    parts: list[str] = []
    cursor = n
    while cursor > 0:
        entry = best[cursor]
        assert entry is not None  # invariant: reachable positions are set
        parts.append(token[entry[1]:cursor])
        cursor = entry[1]
    return list(reversed(parts))


def split_run_together(
    text: str, vocabulary: frozenset[str], min_token_length: int = 20,
) -> RepairResult:
    """Split concatenated words that lost their spaces in PDF conversion.

    Only tokens at or above *min_token_length* are considered, so ordinary
    long words are never touched: "responsibilities" is 16 characters and a
    naive length test would shred it.
    """
    applied = 0
    out: list[str] = []
    for token in text.split(" "):
        stripped = token.strip(".,;:()")
        if len(stripped) < min_token_length or not stripped.isalpha():
            out.append(token)
            continue
        parts = _segment(stripped, vocabulary)
        if parts is None or len(parts) < 2:
            out.append(token)
            continue
        out.append(token.replace(stripped, " ".join(parts)))
        applied += 1
    return RepairResult(text=" ".join(out), applied=applied)


def repair_cell_bleed(
    rows: list[tuple[str, str, str]], marker: str = "Control",
) -> tuple[list[tuple[str, str, str]], int]:
    """Move a spilled sentence fragment back to the row it belongs to.

    PDF table extraction can carry the tail of one cell into the next row, so
    the predecessor ends mid-sentence and the successor opens with a fragment.
    Every real row's text begins with the marker word, which is what makes the
    boundary recoverable.

    Returns (rows, applied). Only fires when the successor has a predecessor
    and the marker appears after position 0, so a genuinely leading fragment
    with nowhere to go is left visible rather than silently discarded.
    """
    repaired: list[tuple[str, str, str]] = []
    applied = 0
    for control_id, title, text in rows:
        stripped = text.strip()
        index = stripped.find(marker)
        if not repaired or index <= 0:
            repaired.append((control_id, title, stripped))
            continue
        fragment = stripped[:index].strip()
        remainder = stripped[index:].strip()
        prev_id, prev_title, prev_text = repaired[-1]
        repaired[-1] = (prev_id, prev_title, f"{prev_text} {fragment}".strip())
        repaired.append((control_id, title, remainder))
        applied += 1
    return repaired, applied
