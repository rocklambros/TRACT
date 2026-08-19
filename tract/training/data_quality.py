"""Training data quality pipeline.

Filters curated hub links by quality, assigns tier metadata,
and computes data hash chain for provenance tracking.

Quality tiers:
  T1     — Human LinkedTo with a resolved anchor (traditional frameworks)
  T1-AI  — Human-curated AI framework links
  T3     — AutomaticallyLinkedTo with a resolved anchor
  DROPPED — No resolved anchor, or an anchor under the length floor

Every gate here tests the anchor the encoder is handed, never the link's
section_name. The two gates this replaced both tested the title.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from tract.config import (
    CONTESTED_RECOVERY_DEFAULT,
    CONTESTED_RECOVERY_FRAMEWORK_IDS,
    PHASE1B_MIN_ANCHOR_TEXT_LENGTH,
    TRAINING_DIR,
)
from tract.io import atomic_write_json
from tract.text_selection import ProseIndex, merged_corpus_path, merged_corpus_sha256

logger = logging.getLogger(__name__)

AI_FRAMEWORK_NAMES: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS",
    "NIST AI 100-2",
    "OWASP AI Exchange",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
})

CURATED_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"
TRAINING_OUTPUT_PATH: Final[Path] = TRAINING_DIR / "hub_links_training.jsonl"
TRAINING_META_PATH: Final[Path] = TRAINING_DIR / "hub_links_training.meta.json"


class QualityTier(enum.Enum):
    T1 = "T1"
    T1_AI = "T1-AI"
    T3 = "T3"
    AL = "AL"
    DROPPED = "DROPPED"


@dataclass(frozen=True)
class TieredLink:
    link: dict[str, str]
    tier: QualityTier


def link_key(link: dict[str, str]) -> str:
    """Stable identity for one curated link, so a drop can be named.

    A count tells an operator that something moved. Only a name tells them
    whether the thing that moved is the thing they expected.
    """
    return "|".join((
        link.get("framework_id", ""),
        link.get("section_id", ""),
        link.get("section_name", ""),
        link.get("cre_id", ""),
    ))


@dataclass(frozen=True)
class FilterReport:
    """What the anchor gate kept, and why it dropped everything else.

    Three drop reasons, reported apart, because they call for three different
    responses: an unresolved link means a parser or a join is missing, a thin
    anchor means the source is that terse, and a contested drop is a
    deliberate exclusion this run chose.
    """

    kept: list[TieredLink]
    dropped_unresolved: list[str]
    dropped_thin_anchor: list[str]
    dropped_contested: list[str]
    # The corpus the INDEX read, not merged_corpus_path() asked a second time.
    # Both are None when a caller built the index from literals, because an
    # in-memory index has no file to name and inventing one would be the same
    # false-provenance defect this field exists to close.
    corpus_path: str | None
    corpus_sha256: str | None

    @property
    def n_dropped(self) -> int:
        return (
            len(self.dropped_unresolved)
            + len(self.dropped_thin_anchor)
            + len(self.dropped_contested)
        )


def is_contested_recovery(link: dict[str, str]) -> bool:
    """True for a link this change newly admits from capec or cwe.

    Exactly the links the retired title floor dropped: 44 capec and 17 cwe, of
    which 60 resolve to a parsed control and one (cwe 937) does not.
    [measured]

    This is the one place a section_name length is still read, and it is not a
    gate. It reconstructs which links the RETIRED title floor excluded, so the
    lever's scope is the set a reviewer can check against the old behaviour
    rather than a fresh judgement call about which links are contested.
    """
    return (
        link.get("framework_id", "") in CONTESTED_RECOVERY_FRAMEWORK_IDS
        and len(link.get("section_name", "").strip()) < PHASE1B_MIN_ANCHOR_TEXT_LENGTH
    )


def assign_quality_tier(
    link: dict[str, str], resolved_text: str | None,
) -> QualityTier:
    """Assign a quality tier to a single hub link.

    `resolved_text` is the anchor the encoder will be handed for this link,
    from ProseIndex, or None when the link resolves to no parsed control. It
    has no default on purpose. tract/ceiling_study.py calls this function under
    a docstring promising it mirrors training, and a defaulted parameter would
    let that call keep compiling while the two pools silently diverged.

    A link with no resolved anchor is dropped rather than falling back to
    link["section_name"]. That fallback is the field this change exists to stop
    training on, and twelve links clear the ten-character floor on it: nine
    wstg ids absent from the archive, two iso_27001 titles, and one dsomm
    activity whose statement _is_prose refuses to index. Training
    "WSTG-BUSL-$$" against a real CRE hub is worse than training nothing.
    [measured]
    """
    if resolved_text is None:
        return QualityTier.DROPPED

    if len(resolved_text.strip()) < PHASE1B_MIN_ANCHOR_TEXT_LENGTH:
        return QualityTier.DROPPED

    if link.get("standard_name", "") in AI_FRAMEWORK_NAMES:
        return QualityTier.T1_AI

    if link.get("link_type", "") == "AutomaticallyLinkedTo":
        return QualityTier.T3

    return QualityTier.T1


def filter_training_links(
    links: list[dict[str, str]],
    index: ProseIndex,
    *,
    recover_contested: bool = CONTESTED_RECOVERY_DEFAULT,
) -> FilterReport:
    """Filter links by the resolved anchor and assign tier metadata."""
    kept: list[TieredLink] = []
    unresolved: list[str] = []
    thin: list[str] = []
    contested: list[str] = []
    tier_counts: dict[QualityTier, int] = {t: 0 for t in QualityTier}

    for link in links:
        if not recover_contested and is_contested_recovery(link):
            contested.append(link_key(link))
            continue

        selection = index.lookup(
            link.get("standard_name", ""),
            link.get("section_id"),
            link.get("section_name"),
        )
        text = selection.text if selection else None
        tier = assign_quality_tier(link, text)
        tier_counts[tier] += 1
        if tier is not QualityTier.DROPPED:
            kept.append(TieredLink(link=link, tier=tier))
        elif text is None:
            unresolved.append(link_key(link))
        else:
            thin.append(link_key(link))

    for tier, count in tier_counts.items():
        logger.info("Quality tier %s: %d links", tier.value, count)
    logger.info(
        "Dropped %d unresolved, %d thin anchors, %d contested",
        len(unresolved), len(thin), len(contested),
    )

    source = index.source_path
    return FilterReport(
        kept=kept,
        dropped_unresolved=sorted(unresolved),
        dropped_thin_anchor=sorted(thin),
        dropped_contested=sorted(contested),
        corpus_path=None if source is None else str(source),
        corpus_sha256=None if source is None else merged_corpus_sha256(source),
    )


def compute_data_hash(data: list[dict[str, Any]]) -> str:
    """Compute deterministic SHA-256 hash of structured data."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _artifact_sha256(path: Path) -> str | None:
    """Hash an input artifact, or None if the arm did not read it."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fold_input_digests(
    *, with_prose: bool, with_stopwords: bool,
) -> dict[str, str | None]:
    """The digests that pin the data one fold read.

    The git SHA pins the code; these pin the data the code was pointed at,
    which is the half that changes when a parser is re-run. Hashing the files
    rather than the parsed objects means anyone with the repo can re-derive
    them with sha256sum.

    all_controls_sha256 hashes merged_corpus_path(), the file ProseIndex.load
    opens. run_single_fold used to hash PROCESSED_DIR / "all_controls.json"
    unconditionally while the run read the licensed overlay, so the field named
    a file the run had not opened and two runs hundreds of links apart recorded
    the same digest for two different corpora.

    It lives here rather than in tract.training.orchestrate because that module
    imports the training stack, `datasets` is not in requirements.txt, and a
    provenance rule no test can import is a rule that regresses unobserved.
    """
    from tract.stopwords import STOPWORDS_PATH

    return {
        "curated_links_sha256": _artifact_sha256(CURATED_PATH),
        "all_controls_sha256": merged_corpus_sha256() if with_prose else None,
        "stopwords_sha256": (
            _artifact_sha256(STOPWORDS_PATH) if with_stopwords else None
        ),
    }


def curated_link_filter_report(
    path: Path | None = None,
    index: ProseIndex | None = None,
    *,
    recover_contested: bool = CONTESTED_RECOVERY_DEFAULT,
) -> tuple[FilterReport, str]:
    """Load the curated links and run the anchor gate over them.

    The single implementation of the gate. tract/ceiling_study.py calls this
    rather than repeating the tier call beside its own loop, which is how the
    two pools stopped agreeing.

    Returns:
        (report, sha256 of the raw curated records).
    """
    p = path or CURATED_PATH
    raw_links: list[dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_links.append(json.loads(line))

    raw_hash = compute_data_hash(raw_links)
    logger.info("Loaded %d curated links (hash=%s)", len(raw_links), raw_hash[:16])

    report = filter_training_links(
        raw_links, index or ProseIndex.load(), recover_contested=recover_contested,
    )
    logger.info(
        "After the anchor gate: %d usable links (dropped %d) against %s",
        len(report.kept), report.n_dropped, report.corpus_path,
    )
    return report, raw_hash


def load_and_filter_curated_links(
    path: Path | None = None,
) -> tuple[list[TieredLink], str]:
    """Load curated links, filter by the resolved anchor, return with data hash.

    Kept at its original arity so the six existing callers in
    tract/training/orchestrate.py, scripts/phase1b/run_fold.py and
    scripts/phase1c/ do not change. Callers that need the drop reasons call
    curated_link_filter_report directly.

    Returns:
        Tuple of (filtered links with tiers, SHA-256 hash of raw data).
    """
    report, raw_hash = curated_link_filter_report(path)
    return report.kept, raw_hash


def save_training_links(
    links: list[TieredLink],
    raw_hash: str,
    corpus_sha256: str,
    path: Path | None = None,
) -> str:
    """Save filtered training links to JSONL, and record what produced them.

    corpus_sha256 has no default. After the anchor gate this file is a function
    of the corpus as well as of the curated links, and recording only raw_hash
    made two runs over different corpora indistinguishable in their own
    provenance.

    Writes a sidecar beside the JSONL rather than a header line inside it, so
    the JSONL stays one-record-per-line for every reader.

    Returns SHA-256 hash of the output data.
    """
    p = path or TRAINING_OUTPUT_PATH
    meta_path = (
        TRAINING_META_PATH if path is None
        else p.with_suffix(".meta.json")
    )
    output_records: list[dict[str, Any]] = []
    for tiered in links:
        record = dict(tiered.link)
        record["quality_tier"] = tiered.tier.value
        output_records.append(record)

    output_hash = compute_data_hash(output_records)

    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    atomic_write_json(
        {
            "corpus_path": str(merged_corpus_path()),
            "corpus_sha256": corpus_sha256,
            "curated_links_sha256": raw_hash,
            "n_links": len(output_records),
            "output_sha256": output_hash,
        },
        meta_path,
    )

    logger.info(
        "Saved %d training links to %s (hash=%s, raw_hash=%s, corpus=%s)",
        len(output_records),
        p.name,
        output_hash[:16],
        raw_hash[:16],
        corpus_sha256[:16],
    )
    return output_hash
