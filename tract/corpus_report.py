"""The corpus join report: the one instrument the parser plan is gated on.

A count of links resolved cannot distinguish 615 links unstacked onto 615
distinct anchors from 615 links collapsed onto 40 coarse ones. Both make the
same number rise, and the second is a regression dressed as progress. So this
module reports the anchor side as well as the link side, and reports both
through the same lookup order the training and evaluation paths use, rather
than through a set intersection that would accept a join the consumer cannot
perform.

Two anchor columns exist because the first version of this instrument reported
a three-times-overstated gain. `distinct_anchors` counts the anchors a resolved
link reaches. `fallback_anchors` counts the distinct section names the trainer
already gets for links that resolve to nothing, because `select_control_text`
falls back to `section_name` rather than failing. A framework with 734
unresolved links is not a framework with zero anchors, and reporting it that
way turned a +152 gain into a +452 headline.

Columns, and the failure each one answers:

    by_title / by_id / unresolved   which channel carried the join
    fallback_anchors                what the trainer gets without a join
    distinct_anchors                the number every downstream metric rests on
    distinct_anchors_pre_truncation two anchors merging into one after the cut
    links_per_anchor                collapse, visible
    truncated                       anchors the encoder budget cuts
    nested_anchors                  an anchor contained in another anchor
    contained_anchors               the stricter prefix-only form, for continuity
    dropped_by_prose_rule           controls ProseIndex never indexed, corpus-wide
    wrong_anchor_risk               three detectors, two of them id-side
    anchor_source_*                 what kind of text the anchor is
    distinct_hubs / links_per_hub   hub-side concentration, for a later
                                    agreement study
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from tract.config import (
    MAX_ANCHOR_CHARS,
    PROJECT_ROOT,
    RESTRICTED_FRAMEWORK_IDS,
    TRAINING_DIR,
)
from tract.io import atomic_write_text
from tract.text_selection import (
    ProseIndex,
    TextSelection,
    _is_prose,
    canonical_framework,
    merged_corpus_path,
    normalize_section_id,
    prepare_anchor,
    strip_markup,
)

logger = logging.getLogger(__name__)

CURATED_LINKS_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"

# Evidence, not results. Tracked by design through the .gitignore negations
# this task adds, and anchored to PROJECT_ROOT so a reader never depends on the
# directory pytest happened to start in.
CORPUS_EVIDENCE_DIR: Final[Path] = PROJECT_ROOT / "results" / "corpus"

# The tracked corpus carries 29 frameworks, the licensed overlay 31. [measured]
# A report built from fewer than the full set cannot assert the restricted
# rows, and gating on file existence never skips because the tracked file
# always exists. Task 15 owns updating this if the rebuild changes the census.
FULL_CORPUS_FRAMEWORK_COUNT: Final[int] = 31

# A parser that assembles an anchor out of several source fragments marks it,
# so the report can separate parser-written text from publisher-written text.
# Absent means the publisher wrote it.
TEXT_ORIGIN_METADATA_KEY: Final[str] = "text_origin"
SYNTHETIC_TEXT_ORIGIN: Final[str] = "synthetic"

_WHITESPACE = re.compile(r"\s+")

# Separators that make one identifier a parent of another: "5.2" of "5.2.2",
# "WSTG-INFO" of "WSTG-INFO-01", "IPY" of "IPY-01".
_ID_SEPARATORS: Final[tuple[str, ...]] = (".", "-", "_", ":", " ")


def _fold(text: str | None) -> str:
    """Whitespace-collapsed, case-folded form, for comparing a name to a title."""
    return _WHITESPACE.sub(" ", (text or "").strip()).casefold()


@dataclass(frozen=True)
class ControlFacts:
    """What the report needs about a control that TextSelection does not carry."""

    title: str
    origin: str


@dataclass
class LinkResolution:
    """One curated link, resolved, carrying no anchor text.

    Digest and length only. The file this serialises to is tracked for every
    framework including the licence-restricted ones, so it must hold nothing a
    publisher reserves. `section_id` and `section_name` come from
    hub_links_curated.jsonl, which is already tracked.
    """

    framework_id: str
    section_id: str
    section_name: str
    cre_id: str
    link_type: str
    channel: str
    anchor_source: str
    anchor_sha256: str
    anchor_chars: int
    truncated: bool
    wrong_anchor: bool
    wrong_anchor_checked: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameworkJoin:
    """One framework's join, on both the link side and the anchor side."""

    framework_id: str
    standard_name: str
    links: int = 0
    by_title: int = 0
    by_id: int = 0
    unresolved: int = 0
    fallback_anchors: int = 0
    distinct_anchors: int = 0
    distinct_anchors_pre_truncation: int = 0
    links_per_anchor: float = 0.0
    truncated: int = 0
    nested_anchors: int = 0
    contained_anchors: int = 0
    dropped_by_prose_rule: int = 0
    wrong_anchor_risk: int = 0
    anchor_source_full_text: int = 0
    anchor_source_description: int = 0
    anchor_source_title: int = 0
    anchor_source_synthetic: int = 0
    distinct_hubs: int = 0
    links_per_hub: float = 0.0
    resolution_rate: float = 0.0

    def finalise(self) -> None:
        resolved = self.by_title + self.by_id
        self.resolution_rate = 0.0 if not self.links else resolved / self.links
        self.links_per_anchor = (
            0.0 if not self.distinct_anchors else resolved / self.distinct_anchors
        )
        # All links, not only the resolved ones: hub concentration is a
        # property of the curated link file and must not move when a parser
        # lands.
        self.links_per_hub = (
            0.0 if not self.distinct_hubs else self.links / self.distinct_hubs
        )


@dataclass
class CorpusReport:
    """Every framework's join plus the identity of what produced it."""

    per_framework: list[FrameworkJoin]
    totals: FrameworkJoin
    corpus_path: str
    corpus_sha256: str
    links_path: str
    links_sha256: str
    corpus_framework_count: int = 0
    max_anchor_chars: int = MAX_ANCHOR_CHARS
    # Serialised to the JSONL rather than into to_json(): 4,405 rows would
    # dominate the summary artifact a reader opens first.
    resolution_rows: list[LinkResolution] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "corpus_framework_count": self.corpus_framework_count,
            "links_path": self.links_path,
            "links_sha256": self.links_sha256,
            "max_anchor_chars": self.max_anchor_chars,
            "totals": asdict(self.totals),
            "per_framework": [asdict(row) for row in self.per_framework],
        }

    def by_id(self, framework_id: str) -> FrameworkJoin:
        """One framework's row.

        Raises:
            KeyError: If the framework contributed no curated links.
        """
        for row in self.per_framework:
            if row.framework_id == framework_id:
                return row
        raise KeyError(
            f"{framework_id!r} has no curated links, so it has no join to "
            f"report. Check the framework_id spelling against "
            f"data/training/hub_links_curated.jsonl."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_links(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            grouped.setdefault(row["framework_id"], []).append(row)
    return grouped


def _load_records(path: Path) -> list[dict[str, Any]]:
    """The framework records, from either corpus shape.

    Both real corpus files are mappings keyed
    [framework_count, frameworks, generated_date, total_controls]. [measured]
    Preferring the named key rather than "the first list value" means a new
    top-level list cannot silently take over.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = data.get("frameworks")
        if isinstance(records, list):
            return records
        for value in data.values():
            if isinstance(value, list):
                return value
    raise ValueError(
        f"{path} holds no list of framework records. The merged corpus is "
        f"either a list or a mapping carrying one under 'frameworks'."
    )


def _control_facts(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, str], ControlFacts], dict[str, int]]:
    """Title and text origin per indexed anchor, plus the prose-rule census.

    Keyed on the selection text rather than on the control id, because
    TextSelection carries no back-reference and duplicating ProseIndex's key
    logic here would be a second implementation to keep in step. Two controls
    with byte-identical text collapse to one entry, and the first in corpus
    order wins, and their anchors are indistinguishable to the encoder anyway.

    The census counts every framework in the corpus, including those with no
    curated links. Summing only the link-bearing subset read 522 where the
    corpus holds 558.
    """
    facts: dict[tuple[str, str], ControlFacts] = {}
    dropped: dict[str, int] = {}
    for record in records:
        framework = canonical_framework(str(record.get("framework_name") or ""))
        for control in record.get("controls") or []:
            title = str(control.get("title") or "")
            description = str(control.get("description") or "")
            full_text = str(control.get("full_text") or "")
            if full_text.strip():
                text = full_text.strip()
            elif _is_prose(description, title):
                text = description.strip()
            else:
                dropped[framework] = dropped.get(framework, 0) + 1
                continue
            metadata = control.get("metadata") or {}
            facts.setdefault(
                (framework, text),
                ControlFacts(
                    title=title,
                    origin=str(
                        metadata.get(TEXT_ORIGIN_METADATA_KEY) or "source"
                    ),
                ),
            )
    return facts, dropped


def _lookup_with_channel(
    index: ProseIndex,
    canonical: str,
    section_id: str | None,
    section_name: str | None,
) -> tuple[TextSelection | None, str]:
    """ProseIndex.lookup, plus which channel answered.

    Deliberately reimplements lookup's branch order rather than calling it: the
    report has to say *how* a link resolved, and lookup returns only the text.
    The order here must stay identical to lookup's, title then id, and
    tests/test_corpus_report.py::TestChannelParity asserts the two agree on
    every curated link in the real corpus.
    """
    if section_name:
        hit = index.by_title(canonical, str(section_name))
        if hit is not None:
            return hit, "title"
    normalized = normalize_section_id(section_id)
    if normalized:
        hit = index.by_id(canonical, normalized)
        if hit is not None:
            return hit, "id"
    return None, "unresolved"


def _fallback_anchor(section_id: str | None, section_name: str | None) -> str:
    """What select_control_text hands the trainer when the index misses.

    Same expression and same normalisation as the fallback branch of
    select_control_text, so the count here is the count the trainer sees.
    """
    fallback = (section_name or section_id or "").strip()
    if not fallback:
        return ""
    text, _ = prepare_anchor(strip_markup(fallback))
    return text


def _classify_anchor(
    selection: TextSelection, anchor: str, facts: ControlFacts | None,
) -> str:
    """Which of the four kinds of text this anchor is.

    Parser-assembled text wins over everything, because its provenance is the
    parser rather than the publisher. A stored anchor that only restates the
    control's own title is reported as `title` even though it arrived through
    full_text or description, since that is what the encoder reads and the
    prose rule cannot see it once a parser writes full_text.
    """
    if facts is not None and facts.origin == SYNTHETIC_TEXT_ORIGIN:
        return "synthetic"
    if facts is not None and facts.title and _fold(anchor) == _fold(facts.title):
        return "title"
    return selection.source


def _count_nested(anchors: set[str]) -> int:
    """Anchors contained anywhere inside a longer anchor of the same framework.

    Containment, not a strict prefix. An ETSI clause 5.2 that rolls up 5.2.2
    opens with its own lead paragraph, so the child sits in the middle of the
    parent and a prefix test reads 0. Measured on the current corpus this
    column reads 0 for every framework. [measured]
    """
    ordered = sorted(anchors, key=lambda item: (len(item), item))
    return sum(
        1
        for position, short in enumerate(ordered)
        if any(short in longer for longer in ordered[position + 1:])
    )


def _count_contained(anchors: set[str]) -> int:
    """The strict-prefix count the first version of this module reported.

    Kept so a reader comparing two runs across this change can separate the
    definition change from a corpus change.
    """
    ordered = sorted(anchors, key=lambda item: (len(item), item))
    return sum(
        1
        for position, short in enumerate(ordered)
        if any(longer.startswith(short) for longer in ordered[position + 1:])
    )


def _is_ancestor_id(parent: str, child: str) -> bool:
    """Whether one normalised section id is a parent of another."""
    if not parent or not child or len(child) <= len(parent):
        return False
    if not child.startswith(parent):
        return False
    return child[len(parent)] in _ID_SEPARATORS


@dataclass
class _Resolved:
    """One link that reached an anchor, with everything a detector needs."""

    link: Mapping[str, Any]
    normalized_id: str
    channel: str
    anchor: str
    raw_text: str
    truncated: bool
    anchor_source: str
    control_title: str


def _wrong_anchor(
    index: ProseIndex,
    canonical: str,
    entry: _Resolved,
    by_id_anchor: Mapping[str, str],
) -> tuple[bool, bool]:
    """Whether this link's anchor is suspect, and whether anything checked it.

    Three detectors, because the first version had one and it lived entirely
    inside the title branch. Nine of the eleven frameworks this plan adds are
    engineered to resolve through the id channel, so a title-only detector made
    `wrong_anchor_risk == 0` unfailable for them.

    A: the title channel answered and the id channel would have answered
       differently. The curator wrote both, and they disagree.
    B: the id channel answered, the link also carried a name that says
       something the id does not, and the control the id reached does not carry
       that name anywhere in its title.
    C: a coarser id and a finer id in the same framework reached the same
       paragraph. This is the NIST AI 100-2 failure that put title first in the
       lookup order, and neither A nor B can see it.

    The second return value is the denominator. A framework whose links carry
    `section_name == section_id` and no ancestor relations has zero applicable
    checks, so a zero in this column proves nothing about it, and
    wrong_anchor_applicable() makes that legible instead of leaving it implied.
    """
    name = str(entry.link.get("section_name") or "")
    checked = False
    flagged = False

    if entry.channel == "title":
        if entry.normalized_id:
            checked = True
            other = index.by_id(canonical, entry.normalized_id)
            if other is not None and prepare_anchor(other.text)[0] != entry.anchor:
                flagged = True
        return flagged, checked

    if name and _fold(name) != _fold(entry.normalized_id):
        checked = True
        title = _fold(entry.control_title)
        if title and _fold(name) not in title and title not in _fold(name):
            flagged = True

    for other_id, other_anchor in by_id_anchor.items():
        if other_id == entry.normalized_id:
            continue
        if _is_ancestor_id(other_id, entry.normalized_id) or _is_ancestor_id(
            entry.normalized_id, other_id
        ):
            checked = True
            if other_anchor == entry.anchor:
                flagged = True
    return flagged, checked


def build_corpus_report(
    links_path: Path | None = None, corpus_path: Path | None = None,
) -> CorpusReport:
    """Resolve every curated link through ProseIndex and report the join."""
    links_file = links_path or CURATED_LINKS_PATH
    corpus_file = corpus_path or merged_corpus_path()

    records = _load_records(corpus_file)
    index = ProseIndex(records)
    facts, dropped = _control_facts(records)
    grouped = _load_links(links_file)

    rows: list[FrameworkJoin] = []
    resolution_rows: list[LinkResolution] = []
    totals = FrameworkJoin(framework_id="TOTAL", standard_name="")
    all_anchors: set[str] = set()
    all_pre_truncation: set[str] = set()
    all_fallbacks: set[str] = set()
    all_hubs: set[str] = set()

    for framework_id in sorted(grouped):
        links = grouped[framework_id]
        standard = str(links[0].get("standard_name") or "")
        canonical = canonical_framework(standard)
        row = FrameworkJoin(
            framework_id=framework_id,
            standard_name=standard,
            links=len(links),
            dropped_by_prose_rule=dropped.get(canonical, 0),
        )

        anchors: set[str] = set()
        pre_truncation: set[str] = set()
        fallbacks: set[str] = set()
        hubs: set[str] = set()
        resolved: list[_Resolved] = []
        unresolved_rows: list[tuple[Mapping[str, Any], str]] = []

        for link in links:
            hubs.add(str(link.get("cre_id") or ""))
            selection, channel = _lookup_with_channel(
                index, canonical, link.get("section_id"), link.get("section_name"),
            )
            normalized = normalize_section_id(link.get("section_id"))
            if selection is None:
                row.unresolved += 1
                fallback = _fallback_anchor(
                    link.get("section_id"), link.get("section_name"),
                )
                if fallback:
                    fallbacks.add(fallback)
                unresolved_rows.append((link, fallback))
                continue

            anchor, was_cut = prepare_anchor(selection.text)
            control = facts.get((canonical, selection.text))
            resolved.append(
                _Resolved(
                    link=link,
                    normalized_id=normalized,
                    channel=channel,
                    anchor=anchor,
                    raw_text=selection.text,
                    truncated=was_cut,
                    anchor_source=_classify_anchor(selection, anchor, control),
                    control_title=control.title if control is not None else "",
                )
            )
            anchors.add(anchor)
            pre_truncation.add(selection.text)
            row.truncated += int(was_cut)
            if channel == "title":
                row.by_title += 1
            else:
                row.by_id += 1

        by_id_anchor: dict[str, str] = {}
        for entry in resolved:
            if entry.channel == "id" and entry.normalized_id:
                by_id_anchor.setdefault(entry.normalized_id, entry.anchor)

        for entry in resolved:
            flagged, checked = _wrong_anchor(index, canonical, entry, by_id_anchor)
            row.wrong_anchor_risk += int(flagged)
            if entry.anchor_source == "full_text":
                row.anchor_source_full_text += 1
            elif entry.anchor_source == "description":
                row.anchor_source_description += 1
            elif entry.anchor_source == SYNTHETIC_TEXT_ORIGIN:
                row.anchor_source_synthetic += 1
            else:
                row.anchor_source_title += 1
            resolution_rows.append(
                LinkResolution(
                    framework_id=framework_id,
                    section_id=str(entry.link.get("section_id") or ""),
                    section_name=str(entry.link.get("section_name") or ""),
                    cre_id=str(entry.link.get("cre_id") or ""),
                    link_type=str(entry.link.get("link_type") or ""),
                    channel=entry.channel,
                    anchor_source=entry.anchor_source,
                    anchor_sha256=hashlib.sha256(
                        entry.anchor.encode("utf-8")
                    ).hexdigest(),
                    anchor_chars=len(entry.anchor),
                    truncated=entry.truncated,
                    wrong_anchor=flagged,
                    wrong_anchor_checked=checked,
                )
            )

        # Unresolved links carry the fallback anchor the trainer receives, so
        # the BEFORE file holds the text-quality baseline the AFTER is read
        # against. Without these rows the JSONL would describe only the links
        # that already work.
        for missed, fallback in unresolved_rows:
            resolution_rows.append(
                LinkResolution(
                    framework_id=framework_id,
                    section_id=str(missed.get("section_id") or ""),
                    section_name=str(missed.get("section_name") or ""),
                    cre_id=str(missed.get("cre_id") or ""),
                    link_type=str(missed.get("link_type") or ""),
                    channel="unresolved",
                    anchor_source="title",
                    anchor_sha256=hashlib.sha256(
                        fallback.encode("utf-8")
                    ).hexdigest(),
                    anchor_chars=len(fallback),
                    truncated=False,
                    wrong_anchor=False,
                    wrong_anchor_checked=False,
                )
            )

        row.distinct_anchors = len(anchors)
        row.distinct_anchors_pre_truncation = len(pre_truncation)
        row.fallback_anchors = len(fallbacks)
        row.distinct_hubs = len(hubs)
        row.nested_anchors = _count_nested(anchors)
        row.contained_anchors = _count_contained(anchors)
        row.finalise()
        rows.append(row)

        all_anchors |= anchors
        all_pre_truncation |= pre_truncation
        all_fallbacks |= fallbacks
        all_hubs |= hubs
        totals.links += row.links
        totals.by_title += row.by_title
        totals.by_id += row.by_id
        totals.unresolved += row.unresolved
        totals.truncated += row.truncated
        totals.nested_anchors += row.nested_anchors
        totals.contained_anchors += row.contained_anchors
        totals.wrong_anchor_risk += row.wrong_anchor_risk
        totals.anchor_source_full_text += row.anchor_source_full_text
        totals.anchor_source_description += row.anchor_source_description
        totals.anchor_source_title += row.anchor_source_title
        totals.anchor_source_synthetic += row.anchor_source_synthetic

    # The census covers every framework in the corpus, including the ones with
    # no curated links and therefore no row above.
    totals.dropped_by_prose_rule = sum(dropped.values())
    totals.distinct_anchors = len(all_anchors)
    totals.distinct_anchors_pre_truncation = len(all_pre_truncation)
    totals.fallback_anchors = len(all_fallbacks)
    totals.distinct_hubs = len(all_hubs)
    totals.finalise()

    logger.info(
        "Corpus join: %d links, %d resolved, %d distinct anchors, %d fallback "
        "anchors, %d controls outside the prose index, over %d frameworks",
        totals.links, totals.by_title + totals.by_id, totals.distinct_anchors,
        totals.fallback_anchors, totals.dropped_by_prose_rule, len(records),
    )

    return CorpusReport(
        per_framework=rows,
        totals=totals,
        corpus_path=str(corpus_file),
        corpus_sha256=_sha256(corpus_file),
        corpus_framework_count=len(records),
        links_path=str(links_file),
        links_sha256=_sha256(links_file),
        resolution_rows=resolution_rows,
    )


def write_link_resolution(report: CorpusReport, path: Path) -> None:
    """One row per curated link, digests only, safe to track for any framework.

    This is what a later label-agreement study needs to sample the frameworks
    this plan re-weights: which channel carried each link, what kind of text it
    reached, how long that text was, and whether a detector questioned it. The
    premortem's answer to "does the plan create that artifact" was no.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        "".join(
            json.dumps(entry.to_json(), sort_keys=True) + "\n"
            for entry in report.resolution_rows
        ),
        path,
    )
    logger.info("wrote %d link resolutions to %s", len(report.resolution_rows), path)


def wrong_anchor_applicable(report: CorpusReport) -> dict[str, int]:
    """Per framework, how many links a wrong-anchor detector could fire on.

    `wrong_anchor_risk == 0` over a denominator of 0 is a fact about the link
    file, not about the parser. Reporting the denominator is what stops that
    zero from being read as a pass.
    """
    counts: dict[str, int] = {row.framework_id: 0 for row in report.per_framework}
    for entry in report.resolution_rows:
        if entry.wrong_anchor_checked:
            counts[entry.framework_id] = counts.get(entry.framework_id, 0) + 1
    return counts


def check_join_floors(
    report: CorpusReport, floors: Mapping[str, float],
) -> list[str]:
    """One message per framework whose resolution rate is under its floor.

    A floor is derived from the link file and the source before the parser is
    written, never pasted from the run being gated. See JOIN_CEILINGS for the
    arithmetic that produced each one.

    Raises:
        KeyError: If a floor names a framework with no curated links. Silently
            skipping it would retire the gate.
    """
    failures: list[str] = []
    for framework_id, floor in sorted(floors.items()):
        row = report.by_id(framework_id)
        if row.resolution_rate + 1e-9 < floor:
            failures.append(
                f"{framework_id}: resolved {row.by_title + row.by_id} of "
                f"{row.links} links ({row.resolution_rate:.4f}) against a "
                f"derived floor of {floor:.4f}. The floor is the arithmetic "
                f"ceiling of this framework's link data rounded down, so a "
                f"miss means the parser lost anchors the source supplies."
            )
    return failures


def floors_for_report(
    report: CorpusReport,
    floors: Mapping[str, float],
    restricted: frozenset[str] = RESTRICTED_FRAMEWORK_IDS,
) -> tuple[dict[str, float], list[str]]:
    """The floors this corpus can carry, and the named group it cannot.

    CI has no licensed overlay, so its corpus holds 29 frameworks against the
    overlay's 31, and every restricted row would read 0.0000 and hard-fail.
    Gating on file existence never skips, because the tracked corpus always
    exists. This gates on content instead, and returns the skipped group by
    name so the reason is stated rather than implied. Never delete a floor to
    make CI green: that retires the only gate on a parser nobody can inspect.

    The Rule 3 author widens the default to OVERLAY_FRAMEWORK_IDS when text
    routing moves the conditional frameworks into the overlay.
    """
    if report.corpus_framework_count >= FULL_CORPUS_FRAMEWORK_COUNT:
        return dict(floors), []
    applicable = {k: v for k, v in floors.items() if k not in restricted}
    skipped = sorted(k for k in floors if k in restricted)
    if skipped:
        logger.warning(
            "corpus has %d frameworks against %d in the full set, so the "
            "licensed overlay is absent from this checkout and these floors "
            "cannot be asserted: %s",
            report.corpus_framework_count, FULL_CORPUS_FRAMEWORK_COUNT,
            ", ".join(skipped),
        )
    return applicable, skipped


def format_table(report: CorpusReport) -> str:
    """The report as a fixed-width table, for logs and for the run ledger."""
    header = (
        f"{'framework':26s} {'links':>5s} {'ttl':>5s} {'id':>4s} {'unres':>5s} "
        f"{'fb':>4s} {'anch':>5s} {'pre':>5s} {'l/a':>5s} {'trunc':>5s} "
        f"{'nest':>4s} {'cont':>4s} {'noidx':>5s} {'wrong':>5s} "
        f"{'ftxt':>5s} {'desc':>5s} {'titl':>4s} {'synt':>4s} "
        f"{'hubs':>5s} {'l/h':>5s} {'rate':>6s}"
    )
    lines = [header, "-" * len(header)]
    for row in [*report.per_framework, report.totals]:
        lines.append(
            f"{row.framework_id:26s} {row.links:5d} {row.by_title:5d} "
            f"{row.by_id:4d} {row.unresolved:5d} {row.fallback_anchors:4d} "
            f"{row.distinct_anchors:5d} {row.distinct_anchors_pre_truncation:5d} "
            f"{row.links_per_anchor:5.2f} {row.truncated:5d} "
            f"{row.nested_anchors:4d} {row.contained_anchors:4d} "
            f"{row.dropped_by_prose_rule:5d} {row.wrong_anchor_risk:5d} "
            f"{row.anchor_source_full_text:5d} {row.anchor_source_description:5d} "
            f"{row.anchor_source_title:4d} {row.anchor_source_synthetic:4d} "
            f"{row.distinct_hubs:5d} {row.links_per_hub:5.2f} "
            f"{row.resolution_rate:6.4f}"
        )
    return "\n".join(lines)


# Per-framework join ceilings, each derived from the curated link file and the
# pinned source in that framework's own plan task, BEFORE its parser existed.
# Written as the fraction so a transcription error is visible. The eleven
# pending frameworks resolve 0 of 734 links today, so none of these was read
# off the run it gates.
#
#   dsomm        213/214  one activity's statement is 11 characters
#   wstg         109/118  nine links name ids absent from the archive
#   nist_800_63   78/79   one section_id is the fragment "are g"
#   biml          21/21   with the two declared alternates
#   enisa         68/68   with Table 3, Annex C and name repair
#   etsi          36/36   every technique declared to its own clause
#   csa_ccm       29/29   seven renamed ids resolve by title
#   nist_ssdf     46/46   with the two declared alt_ids
#   samm          30/30
#   owasp_top10_2021          17/17
#   owasp_proactive_controls  76/76
#
# The eleven below the fold already resolve today, and their ceilings are the
# rates measured on the BEFORE corpus. They are here as a regression gate: the
# rebuild in Task 15 must not cost them. Each miss is known and named.
#   cwe          612/613  CWE-937 was withdrawn upstream
#   nist_800_53  298/300  SC-23(1) and SC-23(3) were withdrawn
#   iso_27001     92/94   A.7.8 and A.7.9 are shorter than their own titles
#                         plus PROSE_MIN_EXTRA_CHARS, so ProseIndex excludes
#                         them on purpose
JOIN_CEILINGS: Final[Mapping[str, float]] = {
    "asvs": 277 / 277,
    "biml": 21 / 21,
    "capec": 1799 / 1799,
    "csa_ccm": 29 / 29,
    "cwe": 612 / 613,
    "dsomm": 213 / 214,
    "enisa": 68 / 68,
    "etsi": 36 / 36,
    "iso_27001": 92 / 94,
    "mitre_atlas": 65 / 65,
    "nist_800_53": 298 / 300,
    "nist_800_63": 78 / 79,
    "nist_ai_100_2": 45 / 45,
    "nist_ssdf": 46 / 46,
    "owasp_ai_exchange": 64 / 64,
    "owasp_cheat_sheets": 391 / 391,
    "owasp_llm_top10": 13 / 13,
    "owasp_ml_top10": 10 / 10,
    "owasp_proactive_controls": 76 / 76,
    "owasp_top10_2021": 17 / 17,
    "samm": 30 / 30,
    "wstg": 109 / 118,
}

# Each ceiling rounded down to two decimals, which
# tests/test_corpus_report.py::TestDerivedFloors asserts rather than trusts.
# A floor of 1.00 where the ceiling is 1.00 is the right number, not an
# oversight: below it, a link the source can answer stopped resolving.
JOIN_FLOORS: Final[Mapping[str, float]] = {
    "asvs": 1.00,
    "biml": 1.00,
    "capec": 1.00,
    "csa_ccm": 1.00,
    "cwe": 0.99,
    "dsomm": 0.99,
    "enisa": 1.00,
    "etsi": 1.00,
    "iso_27001": 0.97,
    "mitre_atlas": 1.00,
    "nist_800_53": 0.99,
    "nist_800_63": 0.98,
    "nist_ai_100_2": 1.00,
    "nist_ssdf": 1.00,
    "owasp_ai_exchange": 1.00,
    "owasp_cheat_sheets": 1.00,
    "owasp_llm_top10": 1.00,
    "owasp_ml_top10": 1.00,
    "owasp_proactive_controls": 1.00,
    "owasp_top10_2021": 1.00,
    "samm": 1.00,
    "wstg": 0.92,
}


# Pre-registered wrong-anchor counts, one entry per framework where the title
# channel can answer at all. Task 16 gates on these instead of on `== 0`.
#
# `== 0` is unfailable for nine of the eleven, because their links resolve
# entirely through the id channel and `wrong_anchor_risk` increments only inside
# the title branch. A gate whose maximum attainable value is zero certifies
# nothing. These two are the frameworks where the title channel is live and the
# count is genuinely non-zero, so the assertion can fail in both directions.
#
#   csa_ccm  1  link `IPY` carries `section_name` "Interoperability and
#               portability policy and procedures", which is control IPY-01's
#               title, not the IPY domain's name ("Interoperability &
#               Portability"). Title-first therefore answers with IPY-01.
#               Task 8 rules that IPY-01 is the correct target. [measured,
#               ML Engineer; link text confirmed by the orchestrator]
#   etsi     1  link `6.3.1` carries the name "Mitigating model stealing",
#               which resolves to clause 6.3. [measured, ML Engineer]
#
# A framework absent from this mapping must report zero, and Task 16 asserts
# `by_title == 0` for it rather than asserting an unfailable risk count.
JOIN_WRONG_ANCHOR_BUDGET: Final[Mapping[str, int]] = {
    "csa_ccm": 1,
    "etsi": 1,
}
