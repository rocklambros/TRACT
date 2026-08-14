"""Prose-first text selection for control anchors.

Standing rule: consider all available prose, always. A control's full text is
preferred over its title everywhere text is chosen, for training anchors and
eval items alike, and falling back to the section title is a last resort that
gets counted rather than passing unnoticed.

This matters more than it looks. The pipeline previously took
``link["section_name"]`` everywhere, which is a three-word title, while
production hands the model paragraph-length control text. Training on titles
and serving prose is a distribution shift built into the data path, and it was
invisible because both sides of the eval agreed with each other.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from tract.config import (
    FRAMEWORK_NAME_ALIASES,
    PROCESSED_DIR,
    PROSE_MIN_EXTRA_CHARS,
)
from tract.io import load_json

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

TextSource = Literal["full_text", "description", "title"]


def canonical_framework(name: str) -> str:
    """Normalise a framework name to its control-side spelling."""
    key = _WHITESPACE.sub(" ", (name or "").strip()).lower()
    return FRAMEWORK_NAME_ALIASES.get(key, name)


def _is_prose(description: str, title: str) -> bool:
    """True when a description carries more than a restatement of the title."""
    return len(description.strip()) > len(title.strip()) + PROSE_MIN_EXTRA_CHARS


@dataclass
class TextSelection:
    """The chosen anchor text and where it came from."""

    text: str
    source: TextSource

    @property
    def is_prose(self) -> bool:
        return self.source != "title"


@dataclass
class SelectionStats:
    """Counts of what the selector actually found, per framework.

    The standing rule says fallbacks are logged and counted. A run that
    silently fell back to titles for most of its corpus looks identical in the
    metrics to one that used prose throughout, which is exactly how the
    title-only evaluation went unnoticed.
    """

    by_source: Counter[str] = field(default_factory=Counter)
    fallback_by_framework: Counter[str] = field(default_factory=Counter)
    total_by_framework: Counter[str] = field(default_factory=Counter)

    def record(self, framework: str, selection: TextSelection) -> None:
        self.by_source[selection.source] += 1
        self.total_by_framework[framework] += 1
        if not selection.is_prose:
            self.fallback_by_framework[framework] += 1

    @property
    def total(self) -> int:
        return sum(self.by_source.values())

    @property
    def prose_fraction(self) -> float:
        return 0.0 if not self.total else 1.0 - self.by_source["title"] / self.total

    def log_summary(self, label: str) -> None:
        logger.info(
            "%s text selection: %d items, %.1f%% prose (%s)",
            label, self.total, 100 * self.prose_fraction,
            dict(self.by_source),
        )
        for framework, n_fallback in self.fallback_by_framework.most_common():
            total = self.total_by_framework[framework]
            logger.info(
                "  title fallback: %-40s %d/%d (%.0f%%)",
                framework, n_fallback, total, 100 * n_fallback / total,
            )


class ProseIndex:
    """Lookup from a link's (framework, section) to that control's full text.

    Indexed on both control_id and lowercased title, because OpenCRE's
    section_id matches the control id for some frameworks and the section_name
    matches the title for others.
    """

    def __init__(self, controls: list[dict[str, Any]]) -> None:
        self._by_id: dict[tuple[str, str], TextSelection] = {}
        self._by_title: dict[tuple[str, str], TextSelection] = {}

        for record in controls:
            framework = canonical_framework(record.get("framework_name", ""))
            for control in record.get("controls") or []:
                title = str(control.get("title") or "")
                description = str(control.get("description") or "")
                full_text = str(control.get("full_text") or "")

                if full_text.strip():
                    selection = TextSelection(full_text.strip(), "full_text")
                elif _is_prose(description, title):
                    selection = TextSelection(description.strip(), "description")
                else:
                    continue  # title restated; nothing to gain over the link

                control_id = str(control.get("control_id") or "").strip()
                if control_id:
                    self._by_id[(framework, control_id)] = selection

                # A section heading and the name OpenCRE links it under are
                # often the same concept spelled differently: "Evasion Attacks
                # and Mitigations" against "Evasion Attacks". Parsers declare
                # those variants rather than the index guessing at them, since
                # a fuzzy match here would silently attach the wrong prose.
                metadata = control.get("metadata") or {}
                alternates = metadata.get("alt_titles") or []
                if isinstance(alternates, str):
                    alternates = [alternates]
                for name in [title, *alternates]:
                    key = str(name).strip().lower()
                    # First writer wins: the real title is offered first, so an
                    # alternate can add a name but never displace one.
                    if key and (framework, key) not in self._by_title:
                        self._by_title[(framework, key)] = selection

    @classmethod
    def load(cls, path: Path | None = None) -> ProseIndex:
        data = load_json(path or PROCESSED_DIR / "all_controls.json")
        records = data if isinstance(data, list) else next(
            (v for v in data.values() if isinstance(v, list)), []
        )
        index = cls(records)
        logger.info(
            "Prose index: %d controls by id, %d by title",
            len(index._by_id), len(index._by_title),
        )
        return index

    def __len__(self) -> int:
        return len(self._by_id)

    def lookup(
        self, framework: str, section_id: str | None, section_name: str | None,
    ) -> TextSelection | None:
        """Resolve a link to its control's prose. Title first, then id.

        The order matters and is not arbitrary. A link's section_id is
        sometimes coarser than the thing it links: NIST AI 100-2 links
        "Adversarial training", "Formal verification" and "Randomized
        smoothing" all carry the section id of the Mitigations subsection that
        contains them. Resolving by id first handed all three the same
        paragraph, and because the eval corpus de-duplicates on control text,
        three distinct eval items silently collapsed into one.

        The section name identifies the specific item, so it is tried first. An
        id lookup remains the fallback for frameworks whose links carry no
        usable name.
        """
        canonical = canonical_framework(framework)
        if section_name:
            hit = self._by_title.get((canonical, str(section_name).strip().lower()))
            if hit:
                return hit
        if section_id:
            hit = self._by_id.get((canonical, str(section_id).strip()))
            if hit:
                return hit
        return None


def select_control_text(
    index: ProseIndex | None,
    framework: str,
    section_id: str | None,
    section_name: str | None,
    stats: SelectionStats | None = None,
    stopwords: frozenset[str] | None = None,
) -> TextSelection:
    """Choose the richest available text for one control.

    Order: the control's full_text, then its description when that is more than
    a restatement of the title, then the section title. Raises when there is no
    text at all, rather than returning an empty anchor that would train on
    nothing.

    Args:
        stopwords: When given, low-information words are filtered from the
            chosen text. Callers must apply the same set to hub texts: the
            firewall compares control text against hub text by exact substring,
            so filtering one side and not the other would make a real leak
            unmatchable.
    """
    selection = index.lookup(framework, section_id, section_name) if index else None
    if selection is None:
        fallback = (section_name or section_id or "").strip()
        if not fallback:
            raise ValueError(
                f"No text of any kind for {framework!r} section "
                f"{section_id!r}/{section_name!r}."
            )
        selection = TextSelection(fallback, "title")

    if stopwords:
        from tract.stopwords import filter_stopwords

        selection = TextSelection(
            filter_stopwords(selection.text, stopwords), selection.source,
        )

    if stats is not None:
        stats.record(canonical_framework(framework), selection)
    return selection


def build_parsed_controls(
    links: Iterable[Any],
    index: ProseIndex | None = None,
    stopwords: frozenset[str] | None = None,
) -> dict[tuple[str, str], str]:
    """Map (framework, section_id) to prose, in the shape the eval corpus wants.

    scripts/phase0/common.build_evaluation_corpus already accepts exactly this
    dict and prefers it over the section title. Every Phase 1B caller was
    passing an empty one, which is why the evaluation measured three-word
    titles while production is handed paragraphs. Supplying it is the whole
    change; the corpus builder needed no edit.

    Only entries with real prose are returned. A title would round-trip to the
    same fallback the corpus builder already applies, and including it would
    make the "full-text" track it reports meaningless.
    """
    resolved: dict[tuple[str, str], str] = {}
    if index is None:
        return resolved

    # A section_id is not always unique per linked item. NIST AI 100-2 links
    # "Adversarial training", "Formal verification" and "Randomized smoothing"
    # under the id of the Mitigations subsection that contains all three, so
    # keying on section_id alone would give all three whichever text won.
    #
    # Prefer apply_prose_to_corpus for evaluation. This function feeds
    # build_evaluation_corpus, which de-duplicates on control text and so can
    # change the item count when the anchor changes.
    candidates: dict[tuple[str, str], set[str]] = {}
    for link in links:
        framework = getattr(link, "standard_name", None) or ""
        section_id = getattr(link, "section_id", None)
        section_name = getattr(link, "section_name", None)
        if not framework or section_id is None:
            continue

        selection = index.lookup(framework, section_id, section_name)
        if selection is None:
            continue
        text = selection.text
        if stopwords:
            from tract.stopwords import filter_stopwords

            text = filter_stopwords(text, stopwords)
        candidates.setdefault((framework, str(section_id)), set()).add(text)

    ambiguous = 0
    for key, texts in candidates.items():
        if len(texts) == 1:
            resolved[key] = next(iter(texts))
        else:
            ambiguous += 1

    logger.info(
        "Prose available for %d link sections (%d dropped: one section id, "
        "several distinct controls)", len(resolved), ambiguous,
    )
    return resolved


def apply_prose_to_corpus(
    corpus: list[Any],
    index: ProseIndex | None,
    stopwords: frozenset[str] | None = None,
    stats: SelectionStats | None = None,
) -> list[Any]:
    """Swap each eval item's anchor for its control's prose, in place of nothing else.

    The item set is decided before this runs and is not touched: same items,
    same order, same ground truth, same valid hub sets. Only control_text
    changes.

    That property is the whole point. build_evaluation_corpus de-duplicates on
    control text, so building the corpus separately per arm lets the anchor
    decide how many items exist: substituting prose collapsed 147 items to 144
    because several NIST sections share wording once expanded. Comparing arms
    over different item sets is not a paired comparison, and paired_bootstrap_delta
    requires equal per-fold lengths to run at all.

    So the corpus is built once from titles, which fixes identity, and the text
    is swapped afterwards.
    """
    if index is None:
        return corpus

    from dataclasses import replace

    updated: list[Any] = []
    for item in corpus:
        selection = index.lookup(
            item.framework_name, item.section_id, item.control_text,
        )
        if selection is None:
            if stats is not None:
                stats.record(
                    canonical_framework(item.framework_name),
                    TextSelection(item.control_text, "title"),
                )
            updated.append(item)
            continue

        text = selection.text
        if stopwords:
            from tract.stopwords import filter_stopwords

            text = filter_stopwords(text, stopwords)
        if stats is not None:
            stats.record(canonical_framework(item.framework_name),
                         TextSelection(text, selection.source))
        # track is part of the record the evaluation reports, so keep it honest.
        updated.append(replace(item, control_text=text, track="full-text"))

    return updated
