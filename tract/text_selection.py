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
from typing import Any, Final, Literal

from tract.config import (
    FRAMEWORK_NAME_ALIASES,
    MAX_ANCHOR_CHARS,
    PROCESSED_DIR,
    PROSE_MIN_EXTRA_CHARS,
    REMEDIATION_HEADINGS,
)
from tract.io import load_json

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")

TextSource = Literal["full_text", "description", "title"]


_SECTION_ID_PREFIX = re.compile(r"^(?:sec\.?|section|clause)\s+", re.IGNORECASE)


def normalize_section_id(section_id: str | None) -> str:
    """Strip the prose prefix OpenCRE puts on some section ids.

    NIST AI 100-2 links carry "Sec. 2.2" and "Sec 2.4.2" while the parser emits
    the bare section number, so the id fallback could never fire for that
    framework and every miss fell through to a title match.
    """
    text = _WHITESPACE.sub(" ", str(section_id or "").strip())
    return _SECTION_ID_PREFIX.sub("", text).strip()


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
    truncated: bool = False

    @property
    def is_prose(self) -> bool:
        return self.source != "title"


# Markdown and site furniture that survives parsing and reaches the encoder as
# tokens. OWASP source carries "#### **Example Attack Scenarios**" and AI
# Exchange carries ">Category: ..." and ">Permalink: https://owaspai.org/...".
# None of it describes a control, all of it is framework-branded, and a
# framework-identifying token in the anchor is a shortcut a bi-encoder can
# learn instead of the mapping.
_MARKDOWN_NOISE: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    # Only the label. Consuming to end-of-line was wrong: the text is already
    # flattened to a single line, so ">Permalink: ..." swallowed the entire
    # control body. The URL rule below removes the value; a Category value is
    # ordinary words and worth keeping.
    (re.compile(r">\s*(?:Category|Permalink|Purpose|Discussion)\s*:\s*"), " "),
    (re.compile(r"https?://\S+"), " "),
    (re.compile(r"!?\[([^\]]*)\]\([^)]*\)"), r"\1"),      # links and images
    (re.compile(r"^#{1,6}\s*", re.M), " "),                   # ATX headings
    (re.compile(r"#{2,6}\s*(?=\*|[A-Z])"), " "),              # inline, post-flattening
    (re.compile(r"\*{1,3}|_{2,3}|`{1,3}"), " "),               # emphasis and code ticks
    (re.compile(r"^\s*[-*+]\s+", re.M), " "),                 # bullets
)


def strip_markup(text: str) -> str:
    """Remove markdown and site furniture, keeping the prose."""
    for pattern, replacement in _MARKDOWN_NOISE:
        text = pattern.sub(replacement, text)
    return _WHITESPACE.sub(" ", text).strip()


# A heading only counts as a section boundary when it starts one: at the string
# start, or after sentence-ending punctuation, and followed by a capital. Run
# after strip_markup, so "#### **Example Attack Scenarios** **Scenario #1"
# has already become "Example Attack Scenarios Scenario 1".
_REMEDIATION_RE = re.compile(
    r"(?:(?<=[.!?:])\s+|^|(?<=\s))(?:"
    + "|".join(re.escape(h) for h in REMEDIATION_HEADINGS)
    + r")\b\s*:?\s+(?=[A-Z0-9])",
)
# Below this, cutting produced a stub rather than a description, so the whole
# body is kept instead.
MIN_DESCRIPTION_CHARS: Final[int] = 120


def strip_remediation(text: str) -> tuple[str, bool]:
    """Cut a control's body at the first remediation heading.

    Returns (text, was_cut). The encoder's 512-token budget is fixed by the
    architecture, so this spends it on the part that says what the control is
    rather than the part that says how to satisfy it.
    """
    text = strip_markup(text or "")
    match = _REMEDIATION_RE.search(text)
    if not match:
        return text, False
    head = text[: match.start()].strip()
    if len(head) < MIN_DESCRIPTION_CHARS:
        # Cutting here would leave a stub. A short description plus its
        # remediation beats a fragment.
        return text, False
    return head, True


def prepare_anchor(
    text: str, max_chars: int | None = None
) -> tuple[str, bool]:
    """Normalise an anchor and cut it to the budget the encoder can read.

    Returns (text, was_truncated).

    Two things the eval path was skipping. The corpus builder sanitises every
    title it produces, so substituting raw index text left the arms running
    different input contracts, and CLAUDE.md requires NFC normalisation and a
    length bound on stored text. And the encoder silently discards anything
    past its sequence limit, so an anchor of 13,007 characters was ~96%
    invisible to the model while looking complete in the artifact.

    max_chars defaults to the 512-token budget. A long-context encoder must
    pass its own, or this cut binds first and the extra context buys nothing:
    at the old fixed 2000 only 7 of 147 eval anchors exceeded 512 tokens,
    while 51 exceeded it before the cut.
    """
    budget = MAX_ANCHOR_CHARS if max_chars is None else max_chars
    import unicodedata

    cleaned = strip_markup(
        unicodedata.normalize("NFC", (text or "").replace("\x00", ""))
    )
    if len(cleaned) <= budget:
        return cleaned, False
    return cleaned[:budget].rstrip(), True


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
    truncated_by_framework: Counter[str] = field(default_factory=Counter)

    def record(self, framework: str, selection: TextSelection) -> None:
        self.by_source[selection.source] += 1
        self.total_by_framework[framework] += 1
        if not selection.is_prose:
            self.fallback_by_framework[framework] += 1
        if selection.truncated:
            self.truncated_by_framework[framework] += 1

    @property
    def n_truncated(self) -> int:
        return sum(self.truncated_by_framework.values())

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
        for framework, n_cut in self.truncated_by_framework.most_common():
            logger.info(
                "  truncated at the encoder budget: %-30s %d/%d",
                framework, n_cut, self.total_by_framework[framework],
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
        pending_alternates: list[tuple[tuple[str, str], TextSelection]] = []

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

                control_id = normalize_section_id(control.get("control_id"))
                if control_id:
                    self._by_id[(framework, control_id)] = selection

                # Real titles are indexed in this pass. Alternates are held
                # back and applied afterwards, because "first writer wins"
                # within one control does NOT stop one control's generated
                # alternate from taking the slot belonging to another
                # control's real title. That already happened: NIST AI 100-2
                # section 2.3's alternate "Poisoning Attacks" claimed the key
                # before section 3.2.2, whose actual title is "Poisoning
                # Attacks", so the Generative-AI eval item resolved to the
                # Predictive-AI chapter's text. That is a wrong anchor, not a
                # fallback, and nothing downstream could see it.
                metadata = control.get("metadata") or {}
                alternates = metadata.get("alt_titles") or []
                if isinstance(alternates, str):
                    alternates = [alternates]

                key = title.strip().lower()
                if key and (framework, key) not in self._by_title:
                    self._by_title[(framework, key)] = selection
                for name in alternates:
                    alt_key = str(name).strip().lower()
                    if alt_key:
                        pending_alternates.append(((framework, alt_key), selection))

        # Second pass: an alternate may add a name, never displace a real one.
        for key_pair, selection in pending_alternates:
            if key_pair not in self._by_title:
                self._by_title[key_pair] = selection

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
        normalized = normalize_section_id(section_id)
        if normalized:
            hit = self._by_id.get((canonical, normalized))
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
    description_only: bool = False,
    max_chars: int | None = None,
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

    # Fixed order, and it matters. Stop word filtering rebuilds text from
    # alphabetic tokens, so running it before markup removal shredded URLs into
    # word fragments ("https owaspai org go ratelimit") that survived as
    # tokens. The stopword arm then differed from the prose arm by markdown
    # handling as well as by stop words, which is a confound rather than an
    # ablation.
    text = strip_markup(selection.text)
    if description_only:
        text, _ = strip_remediation(text)
    if stopwords:
        from tract.stopwords import filter_stopwords

        text = filter_stopwords(text, stopwords)
    selection = TextSelection(text, selection.source)

    prepared, truncated = prepare_anchor(selection.text, max_chars)
    selection = TextSelection(prepared, selection.source, truncated)

    if stats is not None:
        stats.record(canonical_framework(framework), selection)
    return selection


def apply_prose_to_corpus(
    corpus: list[Any],
    index: ProseIndex | None,
    stopwords: frozenset[str] | None = None,
    stats: SelectionStats | None = None,
    description_only: bool = False,
    max_chars: int | None = None,
) -> list[Any]:
    """Swap each eval item's anchor for its control's prose, in place of nothing else.

    The item set is decided before this runs and is not touched: same items,
    same order, same ground truth, same valid hub sets. Only control_text
    changes.

    That property is the whole point. build_evaluation_corpus de-duplicates on
    control text, so building the corpus separately per arm lets the anchor
    decide how many items exist: substituting prose collapsed 147 items to 146
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

        # Same fixed order as select_control_text: markup, section cut, stop
        # words, budget. See the note there on why markup must come first.
        text = strip_markup(selection.text)
        if description_only:
            text, _ = strip_remediation(text)
        if stopwords:
            from tract.stopwords import filter_stopwords

            text = filter_stopwords(text, stopwords)
        text, truncated = prepare_anchor(text, max_chars)
        if stats is not None:
            stats.record(canonical_framework(item.framework_name),
                         TextSelection(text, selection.source, truncated))
        # track is part of the record the evaluation reports, so keep it honest.
        updated.append(replace(item, control_text=text, track="full-text"))

    return _keep_items_resolvable(corpus, updated, stats)


def _keep_items_resolvable(
    original: list[Any],
    updated: list[Any],
    stats: SelectionStats | None = None,
) -> list[Any]:
    """Revert any substitution that made two items indistinguishable.

    Identity preservation is not sufficient on its own. Several NIST AI 100-2
    sections expand to the same paragraph, so after substitution ten eval items
    formed groups that share an anchor while pointing at DIFFERENT ground-truth
    hubs. No model can score better than one item per group, which puts a
    ceiling on the prose arms that the title arm does not have -- the comparison
    would then be measuring a data collision rather than the anchor.

    Items whose prose collides with a different answer keep their title, which
    is unique by construction. Colliding items that share the SAME answer are
    left alone: they are genuinely the same question asked twice, and the
    grader treats them identically either way.
    """
    from collections import defaultdict

    answers_by_text: dict[str, set[str]] = defaultdict(set)
    for item in updated:
        answers_by_text[item.control_text].add(item.ground_truth_hub_id)

    conflicted = {t for t, hubs in answers_by_text.items() if len(hubs) > 1}
    if not conflicted:
        return updated

    resolved: list[Any] = []
    reverted = 0
    for before, after in zip(original, updated):
        if after.control_text in conflicted:
            resolved.append(before)
            reverted += 1
            if stats is not None:
                stats.record(
                    canonical_framework(before.framework_name),
                    TextSelection(before.control_text, "title"),
                )
        else:
            resolved.append(after)

    logger.warning(
        "Reverted %d eval item(s) to their title: their prose was shared with "
        "an item having a different ground-truth hub, which would have capped "
        "accuracy on those items by construction.", reverted,
    )
    return resolved
