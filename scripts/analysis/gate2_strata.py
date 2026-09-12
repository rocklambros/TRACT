"""The Gate 2 exposure partition, fixed and committed before any arm is trained.

An eval item is EXPOSED when at least one hub in its `valid_hub_ids` receives a
training positive from the bridge corpus. Everything else is UNEXPOSED: under
the strict all-AI firewall its gold hubs have no positives in either arm, so it
cannot respond to the treatment and can only contribute noise.

This number is why the Gate 2 design was rebuilt. The pre-registration's
original ENISA+BIML evaluation gave 50 items of which the corpus could reach 18,
all 18 of them ENISA -- BIML's 17 items, 34% of the denominator, had ZERO
exposure at the counting floor. A pooled contrast over that denominator dilutes
a real effect by a factor of nearly three and was the difference between a
design that can see the corpus and one that cannot.

WHY IT IS COMMITTED FIRST. The partition is a property of the corpus and the
gold links only -- nothing in it can move with the arm being measured, which is
what `CAMPAIGN3.md` section 3 requires of a binding partition. Computing it
after the arms have run would make it exactly the kind of post-hoc stratum this
project has withdrawn headlines over. There is nothing to gain by waiting: it
costs no GPU and needs no model.

Items are identified by `control_text_sha256`, which is what `predictions.json`
carries, so the partition joins to a scored arm without anyone re-deriving
anchors -- and without a restricted framework's prose appearing in a tracked
file.

Read-only. Loads no model.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from scripts.phase0.common import build_evaluation_corpus, load_curated_links
from tract.bridge.links import load_bridge_links
from tract.config import (
    PHASE1B_MAX_SEQ_LENGTH,
    PHASE2C_GATE2_EVAL_FRAMEWORKS,
    PHASE2C_GATE2_HELD_OUT,
    max_anchor_chars,
)
from tract.framework_identity import filter_set
from tract.io import atomic_write_json, repo_relative
from tract.text_selection import ProseIndex, SelectionStats, apply_prose_to_corpus

logger = logging.getLogger(__name__)

DEFAULT_OUT = Path("results") / "phase2c" / "gate2_strata.json"


@dataclass(frozen=True)
class Strata:
    """The partition, plus what a reader needs to check it was not moved."""

    exposed_item_keys: list[str]
    unexposed_item_keys: list[str]
    n_items: int
    n_scored_hubs: int
    n_exposed_by_framework: dict[str, int]
    n_items_by_framework: dict[str, int]
    negative_control_frameworks: list[str]
    supervised_hubs: list[str]
    n_bridge_hubs: int
    n_bridge_hubs_invisible_to_eval: int


def build_strata(corpus_path: Path) -> Strata:
    """Partition the Gate 2 eval corpus by exposure to a bridge corpus.

    The eval corpus is built exactly as `run_fold` builds it -- titles first to
    fix the item set and ordering, then prose swapped in -- because the item key
    is a digest of the final anchor text and must match what the fold writes.
    """
    links = load_curated_links()
    corpus = build_evaluation_corpus(
        links, set(PHASE2C_GATE2_EVAL_FRAMEWORKS), {}
    )
    corpus = apply_prose_to_corpus(
        corpus,
        ProseIndex.load(),
        filter_set(use_stopwords=False, use_framework_identity=False),
        stats=SelectionStats(),
        description_only=False,
        max_chars=max_anchor_chars(PHASE1B_MAX_SEQ_LENGTH),
    )

    bridge_hubs = {b.cre_id for b in load_bridge_links(corpus_path)}
    scored_hubs: set[str] = set()
    for item in corpus:
        scored_hubs |= set(item.valid_hub_ids)

    exposed: list[str] = []
    unexposed: list[str] = []
    exposed_by_framework: collections.Counter[str] = collections.Counter()
    items_by_framework: collections.Counter[str] = collections.Counter()

    for item in corpus:
        key = hashlib.sha256(item.control_text.encode("utf-8")).hexdigest()
        items_by_framework[item.framework_name] += 1
        if set(item.valid_hub_ids) & bridge_hubs:
            exposed.append(key)
            exposed_by_framework[item.framework_name] += 1
        else:
            unexposed.append(key)

    # A framework no bridge link can reach is a NEGATIVE CONTROL, not signal:
    # if it moves between arms, the movement is training-draw drift, because
    # nothing in the treatment could have caused it.
    negative_controls = sorted(
        fw for fw in items_by_framework if exposed_by_framework[fw] == 0
    )

    return Strata(
        exposed_item_keys=sorted(exposed),
        unexposed_item_keys=sorted(unexposed),
        n_items=len(corpus),
        n_scored_hubs=len(scored_hubs),
        n_exposed_by_framework=dict(sorted(exposed_by_framework.items())),
        n_items_by_framework=dict(sorted(items_by_framework.items())),
        negative_control_frameworks=negative_controls,
        supervised_hubs=sorted(bridge_hubs & scored_hubs),
        n_bridge_hubs=len(bridge_hubs),
        n_bridge_hubs_invisible_to_eval=len(bridge_hubs - scored_hubs),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, required=True,
        help="The merged bridge corpus from scripts.build_gate2_corpus.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    strata = build_strata(args.corpus)
    n_exposed = len(strata.exposed_item_keys)

    payload = {
        "corpus_path": repo_relative(args.corpus),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "eval_frameworks": sorted(PHASE2C_GATE2_EVAL_FRAMEWORKS),
        "held_out_frameworks": sorted(PHASE2C_GATE2_HELD_OUT),
        **{k: v for k, v in vars(strata).items()},
    }
    atomic_write_json(payload, args.out)

    logger.info("=" * 70)
    logger.info("PHASE 2C GATE 2 -- exposure strata")
    logger.info("  eval        : %d items over %d scored hubs (%s)",
                strata.n_items, strata.n_scored_hubs,
                ", ".join(sorted(PHASE2C_GATE2_EVAL_FRAMEWORKS)))
    logger.info("  EXPOSED     : %d  %s", n_exposed,
                dict(strata.n_exposed_by_framework))
    logger.info("  unexposed   : %d", len(strata.unexposed_item_keys))
    logger.info("  items/fw    : %s", dict(strata.n_items_by_framework))
    logger.info("")
    logger.info("  bridge hubs : %d, of which %d are invisible to this eval",
                strata.n_bridge_hubs, strata.n_bridge_hubs_invisible_to_eval)
    if strata.negative_control_frameworks:
        logger.info(
            "  NEGATIVE CONTROL: %s -- no bridge link can reach these items, "
            "so movement there is drift, not treatment.",
            ", ".join(strata.negative_control_frameworks),
        )
    logger.info("")
    logger.info("  Wrote %s -- COMMIT THIS BEFORE THE FIRST POD.", args.out)
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
