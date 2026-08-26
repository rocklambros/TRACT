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

    if args.check:
        # Recompute what save_training_links would record, without writing.
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
    else:
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
