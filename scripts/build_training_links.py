"""Rebuild data/training/hub_links_training.jsonl and its metadata sidecar.

    python -m scripts.build_training_links --check
    python -m scripts.build_training_links

Both outputs are TRACKED, and until now there was no entry point that produced
them. `save_training_links` had exactly two kinds of caller: the test suite,
and whatever ad-hoc invocation originally wrote the committed files. That is
the same defect shape as a refusal nothing calls -- the artifact is in git and
the command that makes it is in somebody's shell history.

It matters because `assert_corpus_matches_training_links` compares the corpus
on disk against the digest recorded here, and that check now gates provision,
run_folds and run_fold.py. When the corpus legitimately changes, somebody has
to be able to regenerate this sidecar without reverse-engineering how.

RUN THIS WHENEVER THE MERGED CORPUS CHANGES, and read what it prints. A change
in `n_links` or `output_sha256` means the training set moved and every recorded
result is stale against it. A change in `corpus_sha256` alone means the corpus
moved somewhere the training links do not reach, which is the benign case.

`--check` reports drift and writes nothing, so a reviewer can tell a stale
sidecar from a current one.

Exit codes: 0 in sync, 1 drift under --check, 2 the inputs are unusable.

Owner: TRACT.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any, Final

from tract.text_selection import merged_corpus_path, merged_corpus_sha256
from tract.training.data_quality import (
    TRAINING_META_PATH,
    TRAINING_OUTPUT_PATH,
    load_and_filter_curated_links,
    save_training_links,
)
from tract.io import repo_relative

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXIT_IN_SYNC: Final[int] = 0
EXIT_DRIFT: Final[int] = 1
EXIT_UNUSABLE: Final[int] = 2

# The fields whose movement means the TRAINING SET changed rather than merely
# the corpus around it. Reported separately because the two have very
# different consequences for results already on disk.
TRAINING_SET_FIELDS: Final[tuple[str, ...]] = ("n_links", "output_sha256")


def refuse_reason(before: dict[str, Any], after: dict[str, Any]) -> str | None:
    """Why this regeneration must not be written, or None if it may be.

    There are two ways to satisfy the corpus refusal that gates provisioning.
    Stage the licensed overlay, which is the corpus the campaign is measured
    on. Or regenerate this sidecar against whatever the machine happens to
    hold, which makes the check pass and trains on 4,048 of 4,389 links while
    reporting the same shape of output.

    This function exists so the script cannot offer the second. The two
    signals that separate a legitimate regeneration from a short one are the
    corpus PATH the sidecar records, and whether the training set shrinks. A
    parser moving under the same corpus is legitimate and common; the corpus
    changing identity underneath is not.

    Returns:
        A sentence naming the fix, or None when the write is safe.
    """
    if not before:
        return None

    recorded_path = before.get("corpus_path")
    if recorded_path and recorded_path != after.get("corpus_path"):
        return (
            f"the sidecar records {recorded_path} and this machine is reading "
            f"{after.get('corpus_path')}. Those are different corpora, so "
            f"rewriting the sidecar would not fix a mismatch, it would record "
            f"the smaller corpus as correct and train on it silently. Stage "
            f"the licensed overlay instead: see docs/RUNNING_ELSEWHERE.md."
        )

    before_links = before.get("n_links")
    after_links = after.get("n_links")
    if isinstance(before_links, int) and isinstance(after_links, int):
        if after_links < before_links:
            return (
                f"the training set would shrink from {before_links} to "
                f"{after_links} links. Something upstream stopped resolving "
                f"anchors it used to resolve. Find out what before recording "
                f"the smaller set as the reference."
            )
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report drift against the committed sidecar, write nothing")
    args = parser.parse_args(argv)

    corpus = merged_corpus_path()
    if not corpus.exists():
        logger.error("%s does not exist; nothing to build against.",
                     repo_relative(corpus))
        return EXIT_UNUSABLE

    logger.info("Reading corpus %s", repo_relative(corpus))
    corpus_sha = merged_corpus_sha256()
    links, raw_hash = load_and_filter_curated_links()

    before: dict[str, Any] = (
        json.loads(TRAINING_META_PATH.read_text(encoding="utf-8"))
        if TRAINING_META_PATH.exists() else {}
    )
    after: dict[str, Any]

    # Always computed before anything is written, so the refusal below can run
    # against what WOULD be recorded. Writing first and judging afterwards
    # would mean the damage is on disk by the time the reason is printed.
    from tract.training.data_quality import compute_data_hash

    records: list[dict[str, Any]] = []
    for tiered in links:
        record = dict(tiered.link)
        record["quality_tier"] = tiered.tier.value
        records.append(record)
    after = {
        "corpus_path": repo_relative(corpus),
        "corpus_sha256": corpus_sha,
        "curated_links_sha256": raw_hash,
        "n_links": len(records),
        "output_sha256": compute_data_hash(records),
    }

    refusal = refuse_reason(before, after)
    if refusal is not None:
        logger.error("REFUSING to rewrite %s: %s",
                     repo_relative(TRAINING_META_PATH), refusal)
        logger.error(
            "There is no flag for this. If the project genuinely means to "
            "change which corpus is the reference, delete %s and run again -- "
            "that leaves a deleted tracked file in `git status` for a reviewer "
            "to see, which a flag would not.",
            repo_relative(TRAINING_META_PATH),
        )
        return EXIT_UNUSABLE

    if not args.check:
        save_training_links(links, raw_hash, corpus_sha)
        after = json.loads(TRAINING_META_PATH.read_text(encoding="utf-8"))

    moved = [k for k in after if before.get(k) != after[k]]
    if not moved:
        logger.info("In sync: %d links, corpus %s", after["n_links"],
                    after["corpus_sha256"][:12])
        return EXIT_IN_SYNC

    for key in moved:
        logger.warning("%-22s %s -> %s", key,
                       str(before.get(key))[:16], str(after[key])[:16])

    training_set_moved = [k for k in moved if k in TRAINING_SET_FIELDS]
    if training_set_moved:
        logger.warning(
            "THE TRAINING SET CHANGED (%s). Every result recorded against the "
            "old set is stale and must not be compared to a new one. Re-run "
            "the arms.", ", ".join(training_set_moved),
        )
    else:
        logger.info(
            "The training set is unchanged: same %d links, same output digest. "
            "Only the corpus digest moved, so the corpus changed somewhere the "
            "training links do not reach.", after["n_links"],
        )

    if args.check:
        logger.error("Committed sidecar %s is stale. Re-run without --check.",
                     repo_relative(TRAINING_META_PATH))
        return EXIT_DRIFT

    logger.info("Wrote %s and %s", repo_relative(TRAINING_OUTPUT_PATH),
                repo_relative(TRAINING_META_PATH))
    return EXIT_IN_SYNC


if __name__ == "__main__":
    sys.exit(main())
