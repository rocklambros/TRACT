"""A0R: a random bridge corpus, matched to the real one in everything but judgement.

WHY THIS ARM EXISTS. Under the strict all-AI firewall the bridge-free comparator
has ZERO training positives for all 62 scored hubs, because the AI and
traditional hub regions are disjoint. So `A1 - A0` does not ask "do these
human-curated links help"; it asks "does any supervision at all beat none", and
a PASS on it licenses almost nothing. That was a Critical premortem finding and
it is not fixable by analysis -- it needs a third arm.

WHAT IS HELD CONSTANT. The placebo keeps every real link's HUB and replaces its
CONTROL with a NIST 800-53 control the annotators did not map there. So it
matches the real corpus on:

  * count -- the same number of distinct edges, so training-set size and
    therefore batch composition and the LR schedule are unchanged;
  * source framework -- the same traditional standard supplies every anchor;
  * hub distribution -- EXACTLY the same hubs, at the same multiplicities.

That last one is deliberate and load-bearing. Holding the hub distribution fixed
means the placebo arm has the SAME exposure partition as the treatment arm, so
`results/phase2c/gate2_strata.json` applies unchanged to both contrasts and the
difference between them is the annotators' judgement about WHICH control belongs
to which hub -- which is the thing Phase 2C actually bought.

Drawing hubs uniformly instead would have changed exposure, and then A1 - A0R
would confound judgement with coverage.

Deterministic under a declared seed, so the placebo is reproducible and its
digest identifies it. Read-only over the real corpus. Loads no model.
"""

from __future__ import annotations

import argparse
import collections
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Final

from scripts.build_gate2_corpus import _git_sha, _sha256, write_corpus
from scripts.phase0.common import load_curated_links
from tract.bridge.links import BridgeLink, load_bridge_links
from tract.config import TRAINING_DIR
from tract.io import atomic_write_json, repo_relative

logger = logging.getLogger(__name__)

# Declared, not incidental. The placebo is an arm of a pre-registered gate and
# "we drew it randomly" is only reproducible if the draw is named.
PLACEBO_SEED: Final[int] = 20260911

# The traditional standard the real corpus draws from. Every placebo anchor must
# come from the same one, or the arms differ in their source distribution as
# well as in their judgement.
PLACEBO_SOURCE_STANDARD: Final[str] = "NIST 800-53 v5"

DEFAULT_OUT: Final[Path] = TRAINING_DIR / "hub_links_bridge.placebo.jsonl"
DEFAULT_MANIFEST: Final[Path] = (
    Path("results") / "phase2c" / "gate2_placebo_manifest.json"
)


def build_placebo(
    real: list[BridgeLink],
    candidate_sections: dict[str, str],
    *,
    seed: int = PLACEBO_SEED,
) -> list[BridgeLink]:
    """Same hubs, same count, same framework -- different controls.

    Raises rather than reusing a control the annotators actually chose for that
    hub: a placebo that accidentally reproduces part of the real corpus is a
    weaker comparator by exactly the overlap, and nothing downstream would show
    it.
    """
    if not real:
        raise ValueError("the real corpus is empty; there is nothing to match")
    if len(candidate_sections) <= max(
        collections.Counter(b.cre_id for b in real).values()
    ):
        raise ValueError(
            f"only {len(candidate_sections)} candidate controls for a hub "
            "needing more than that many distinct placebo anchors"
        )

    real_pairs = {(b.section_id, b.cre_id) for b in real}
    rng = random.Random(seed)
    # Sorted, so the draw depends on the seed and not on dict iteration order.
    pool = sorted(candidate_sections)

    placebo: list[BridgeLink] = []
    used: set[tuple[str, str]] = set()
    for link in sorted(
        real, key=lambda b: (b.framework_id, b.section_id, b.cre_id)
    ):
        for _ in range(len(pool) * 2):
            section = rng.choice(pool)
            pair = (section, link.cre_id)
            if pair not in real_pairs and pair not in used:
                used.add(pair)
                placebo.append(replace(
                    link,
                    section_id=section,
                    section_name=candidate_sections[section],
                    annotator_id="placebo",
                    rationale="Randomly assigned. Not a human judgement.",
                ))
                break
        else:
            raise ValueError(
                f"could not find an unused placebo control for hub "
                f"{link.cre_id} after exhausting the pool"
            )
    return placebo


def candidate_controls(standard: str = PLACEBO_SOURCE_STANDARD) -> dict[str, str]:
    """Every control of the source standard, as section_id -> section_name."""
    out = {
        link.section_id: link.section_name
        for link in load_curated_links()
        if link.standard_name == standard and link.section_id
    }
    if not out:
        raise ValueError(
            f"no curated links carry standard_name={standard!r}, so no placebo "
            "anchors can be drawn"
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real", type=Path,
        default=TRAINING_DIR / "hub_links_bridge.r2.jsonl",
        help="The treatment corpus this placebo is matched to.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=PLACEBO_SEED)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    real = load_bridge_links(args.real)
    placebo = build_placebo(
        real, candidate_controls(), seed=args.seed
    )
    write_corpus(placebo, args.out)

    real_hubs = collections.Counter(b.cre_id for b in real)
    placebo_hubs = collections.Counter(b.cre_id for b in placebo)
    if real_hubs != placebo_hubs:
        raise ValueError(
            "the placebo's hub distribution does not match the real corpus, "
            "so the two arms would differ in exposure as well as in judgement"
        )

    overlap = len(
        {(b.section_id, b.cre_id) for b in real}
        & {(b.section_id, b.cre_id) for b in placebo}
    )
    # Built here rather than through build_gate2_corpus.build_manifest, which
    # globs a ROUND DIRECTORY for its source digests. The placebo's source is a
    # single file in data/training/, and globbing that directory would hash the
    # curated gold links into a placebo manifest.
    manifest: dict[str, object] = {
        "round_label": "placebo",
        "seed": args.seed,
        "source_standard": PLACEBO_SOURCE_STANDARD,
        "real_corpus_path": repo_relative(args.real),
        "real_corpus_sha256": _sha256(args.real),
        "corpus_path": repo_relative(args.out),
        "corpus_sha256": _sha256(args.out),
        "git_sha": _git_sha(),
        "n_edges": len(placebo),
        "n_edges_real": len(real),
        "n_distinct_controls": len({b.section_id for b in placebo}),
        "n_distinct_hubs": len(placebo_hubs),
        "hub_distribution_matches_real": real_hubs == placebo_hubs,
        "overlap_with_real_corpus": overlap,
    }
    atomic_write_json(manifest, args.manifest)

    logger.info("=" * 68)
    logger.info("A0R placebo corpus (seed %d)", args.seed)
    logger.info("  edges           : %d (real: %d)", len(placebo), len(real))
    logger.info("  distinct hubs   : %d (identical to real: %s)",
                len(placebo_hubs), real_hubs == placebo_hubs)
    logger.info("  overlap with real: %d edges", overlap)
    logger.info("  sha256          : %s", manifest["corpus_sha256"])
    logger.info("  corpus   -> %s", args.out)
    logger.info("  manifest -> %s  (COMMIT THIS)", args.manifest)
    logger.info("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
