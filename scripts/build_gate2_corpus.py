"""Merge a Phase 2C annotation round into the single corpus file training reads.

THIS STEP DID NOT EXIST, and its absence was a Critical premortem finding. The
Gate 2 draft plan named two arms -- "round 1 (80 links)" and "round 2 (173
links)" -- and neither was a file. `load_bridge_links` requires `path.is_file()`;
a round on disk is a pair of per-annotator `.jsonl`s plus their sidecars. The
merge was going to be a manual `cat` with no schema, no order and no test, and
merging the two volumes in the two possible orders gives two different digests.

Worse, the confidence floor that defines "80" lives only in
`scripts/analysis/gate1_report.py`. `bridge_training_records` drops the
confidence field entirely, so a naive concatenation trains on 86 rows including
six the pre-registration calls "data, not evidence" -- while every artifact says
80. The number in the document and the number in the weights were going to
differ, and the sha256 would have faithfully pinned the wrong one.

So this is where a round becomes a corpus:

  * the Q3 confidence floor is applied ON THE TRAINING PATH, not just in the
    gate reporter, and what it removed is counted;
  * identical (section, hub) edges from two annotators collapse to one, keeping
    the higher-confidence record -- matching what `build_training_pairs` does
    downstream anyway, so the corpus count and the trained count agree;
  * a control mapped to DIFFERENT hubs by two annotators keeps both, because the
    CRE graph is multi-hop and Q2 permits up to six hubs per control -- but the
    disagreement is counted, because nobody had decided this and it lands in the
    treatment arm only;
  * the output is sorted on content, so two machines produce identical bytes and
    the recorded digest means something;
  * a manifest is written for commit, carrying counts and digests but no
    annotator free text, because the corpus itself is gitignored and a digest
    with no public referent is not provenance.

Read-only over the round directories. Loads no model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Final

from tract.bridge.links import BridgeLink, load_bridge_links
from tract.config import (
    BRIDGE_CORPUS_DIR,
    BRIDGE_CORPUS_DIR_R2,
    PHASE2C_Q3_CONFIDENCE_FLOOR,
    TRAINING_DIR,
)
from tract.io import atomic_write_json, repo_relative

logger = logging.getLogger(__name__)

# Filenames that mean "you have pointed this at the gold links, not at a round".
# `gate1_report` learned this after being pointed at data/training/, where
# globbing *.jsonl loads 4,405 Tier-1 curated links as though two volunteers had
# written them. Exact names, not a prefix: `hub_links_bridge.r2.jsonl` is this
# script's own output and must not be caught.
GOLD_LINK_FILENAMES: Final[frozenset[str]] = frozenset({
    "hub_links.jsonl",
    "hub_links_curated.jsonl",
    "hub_links_training.jsonl",
})

DEFAULT_MANIFEST: Final[Path] = (
    Path("results") / "phase2c" / "gate2_corpus_manifest.json"
)


@dataclass(frozen=True)
class CorpusStats:
    """What the merge did, in the terms the write-up has to report."""

    round_label: str
    min_confidence: int
    n_raw: int
    n_below_floor: int
    n_kept: int
    n_agreed_edges: int
    n_conflicting_controls: int
    n_distinct_controls: int
    n_distinct_hubs: int
    per_annotator_raw: dict[str, int]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_sha() -> str:
    """Best-effort commit id. Never fatal: a manifest is better than no manifest."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("Could not read git SHA: %s", exc)
        return "unknown"
    return out.stdout.strip()


def _sort_key(link: BridgeLink) -> tuple[str, str, str, str]:
    return (link.framework_id, link.section_id, link.cre_id, link.annotator_id)


def merge_round(
    round_dir: Path, *, min_confidence: int = PHASE2C_Q3_CONFIDENCE_FLOOR
) -> tuple[list[BridgeLink], CorpusStats]:
    """One round directory -> a deterministic, floored, deduplicated link list.

    Raises rather than returning a partial or empty corpus. Every failure mode
    here -- an empty directory, a gold file, a malformed record -- produces a
    smaller corpus that trains successfully and reports success, which is the
    shape of a finding nobody catches for six months.
    """
    if not round_dir.is_dir():
        raise ValueError(f"{round_dir} is not a directory")

    present = {p.name for p in round_dir.glob("*.jsonl")}
    if gold := present & GOLD_LINK_FILENAMES:
        raise ValueError(
            f"{round_dir} contains gold link file(s) {sorted(gold)}. This is "
            "the curated Tier-1 corpus, not an annotation round; merging it "
            "would relabel 4,405 OpenCRE links as volunteer-authored Tier 2."
        )
    sources = sorted(round_dir.glob("*.jsonl"))
    if not sources:
        raise ValueError(
            f"{round_dir} holds no *.jsonl files, so there is no round to "
            "merge. An empty corpus trains with no bridge supervision and "
            "reports success."
        )

    raw: list[BridgeLink] = []
    per_annotator: dict[str, int] = {}
    for source in sources:
        links = load_bridge_links(source)
        per_annotator[source.stem] = len(links)
        raw.extend(links)

    kept_floor = [b for b in raw if b.confidence >= min_confidence]

    # Collapse identical edges; keep the higher-confidence record, breaking ties
    # on annotator id so the survivor does not depend on glob order.
    best: dict[tuple[str, str], BridgeLink] = {}
    n_agreed = 0
    for link in sorted(kept_floor, key=_sort_key):
        key = (link.section_id, link.cre_id)
        incumbent = best.get(key)
        if incumbent is None:
            best[key] = link
            continue
        n_agreed += 1
        if link.confidence > incumbent.confidence:
            best[key] = link

    merged = sorted(best.values(), key=_sort_key)

    # A control two annotators sent to DIFFERENT hubs. Both survive -- the CRE
    # graph is multi-hop and Q2 allows six hubs per control -- but this is a
    # disagreement landing in the treatment arm and nowhere in the comparator,
    # so it is counted rather than left implicit.
    hubs_by_control: dict[str, set[str]] = {}
    annotators_by_control: dict[str, set[str]] = {}
    for link in kept_floor:
        hubs_by_control.setdefault(link.section_id, set()).add(link.cre_id)
        annotators_by_control.setdefault(link.section_id, set()).add(
            link.annotator_id
        )
    n_conflicting = sum(
        1
        for control, hubs in hubs_by_control.items()
        if len(hubs) > 1 and len(annotators_by_control[control]) > 1
    )

    stats = CorpusStats(
        round_label=round_dir.name,
        min_confidence=min_confidence,
        n_raw=len(raw),
        n_below_floor=len(raw) - len(kept_floor),
        n_kept=len(merged),
        n_agreed_edges=n_agreed,
        n_conflicting_controls=n_conflicting,
        n_distinct_controls=len({b.section_id for b in merged}),
        n_distinct_hubs=len({b.cre_id for b in merged}),
        per_annotator_raw=per_annotator,
    )
    return merged, stats


def write_corpus(links: list[BridgeLink], out: Path) -> None:
    """Write the merged corpus atomically, with sorted keys, one record a line.

    Sorted keys and a fixed record order are what make the sha256 a stable
    identifier rather than a property of whichever machine ran the merge.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(
        json.dumps(asdict(link), sort_keys=True) + "\n" for link in links
    )
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(body, encoding="utf-8")
    tmp.replace(out)


def build_manifest(
    *, round_label: str, round_dir: Path, corpus_path: Path, stats: CorpusStats
) -> dict[str, object]:
    """The committed record. Counts and digests; no annotator free text.

    The corpus is gitignored because it carries pseudonyms and verbatim
    rationales, and `.gitignore` says publishing those "is not a decision an
    import command should make". That leaves `bridge_links_sha256` in a fold
    record pointing at content no third party can obtain. This manifest is the
    digest's public referent.
    """
    # repo-RELATIVE, both of them. The default --out is derived from
    # TRAINING_DIR, which is absolute, so the first manifest this script wrote
    # carried "/home/rock/github_projects/TRACT/..." into a committed artifact
    # -- a path that resolves on exactly one machine, and the reason
    # tests/test_io.py guards every tracked JSON file for this.
    manifest: dict[str, object] = {
        "round_label": round_label,
        "round_dir": repo_relative(round_dir),
        "corpus_path": repo_relative(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "git_sha": _git_sha(),
        "source_files": {
            p.name: _sha256(p) for p in sorted(round_dir.glob("*.jsonl"))
        },
    }
    counts = asdict(stats)
    counts.pop("round_label", None)
    manifest.update(counts)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round-dir", type=Path, default=BRIDGE_CORPUS_DIR_R2,
        help="Annotation round to merge. Defaults to round 2, the corpus of "
             "record under Amendment 1.",
    )
    parser.add_argument("--round-label", default="r2")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Merged corpus. Defaults to "
             "data/training/hub_links_bridge.<round-label>.jsonl",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--min-confidence", type=int, default=PHASE2C_Q3_CONFIDENCE_FLOOR,
        help="Q3 floor. Applied HERE, on the training path -- it previously "
             "existed only in the gate reporter.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out = args.out or (
        TRAINING_DIR / f"hub_links_bridge.{args.round_label}.jsonl"
    )
    links, stats = merge_round(args.round_dir, min_confidence=args.min_confidence)
    write_corpus(links, out)
    manifest = build_manifest(
        round_label=args.round_label, round_dir=args.round_dir,
        corpus_path=out, stats=stats,
    )
    atomic_write_json(manifest, args.manifest)

    logger.info("=" * 68)
    logger.info("Gate 2 corpus: %s", args.round_label)
    logger.info("  rows read              : %d", stats.n_raw)
    logger.info("  below confidence %d     : %d", stats.min_confidence,
                stats.n_below_floor)
    logger.info("  distinct edges kept    : %d", stats.n_kept)
    logger.info("  edges both annotators agreed on: %d", stats.n_agreed_edges)
    logger.info("  controls sent to different hubs: %d",
                stats.n_conflicting_controls)
    logger.info("  distinct controls / hubs: %d / %d",
                stats.n_distinct_controls, stats.n_distinct_hubs)
    logger.info("  sha256                 : %s", manifest["corpus_sha256"])
    logger.info("")
    logger.info("  corpus   -> %s  (gitignored: carries pseudonyms and prose)", out)
    logger.info("  manifest -> %s  (COMMIT THIS)", args.manifest)
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
