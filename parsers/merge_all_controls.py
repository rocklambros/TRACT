"""Merge validated framework JSONs into all_controls.json.

Two artifacts, not one.

`data/processed/all_controls.json` is git-tracked and excludes every framework
in `RESTRICTED_FRAMEWORK_IDS`. Those frameworks carry licensed control
statements, and this repository is CC0, which is an affirmative grant that the
publisher holds the rights and waives them. Gitignoring the per-framework file
alone left this wide channel open: the merge globbed the same directory and
inlined the untracked file verbatim into a tracked one.

`data/processed/licensed/all_controls.json` is gitignored and carries the full
corpus, restricted frameworks included, for local training and evaluation.

Read order for anything that trains or evaluates: prefer the licensed overlay
when it is present, else fall back to the tracked file. The overlay is written
only when a restricted framework is on disk, and a stale one is removed when
the restricted source disappears, so its presence always means "this checkout
holds licensed prose".

`generated_date` is derived from the frameworks actually inside each artifact,
never from the clock and never from files the artifact excludes. The merged
corpus's sha256 is recorded per fold and compared across folds, so the tracked
artifact's bytes must not depend on whether a gitignored file exists locally.
"""
from __future__ import annotations

import logging
from pathlib import Path

from tract.config import (
    HOLDOUT_FRAMEWORK_IDS,
    PROCESSED_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    PROCESSED_LICENSED_DIR,
    RESTRICTED_FRAMEWORK_IDS,
)
from tract.io import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MERGED_FILENAME: str = "all_controls.json"


def _control_count(framework: dict[str, object]) -> int:
    """Number of controls in one framework artifact.

    Raises rather than coercing. A framework file whose `controls` key is not
    a list is corrupt, and a merged corpus built from it would carry a wrong
    total that nothing downstream can tell from a right one.
    """
    controls = framework.get("controls", [])
    if not isinstance(controls, list):
        raise ValueError(
            f"framework {framework.get('framework_id', '?')!r} has a "
            f"'controls' key of type {type(controls).__name__}, expected list"
        )
    return len(controls)


def _drop_holdouts(
    frameworks: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Remove frameworks the model must never see, from both corpora.

    This is the other way a holdout's prose reaches a trainer. Every roster
    could name the right frameworks and the corpus builder would still glob a
    holdout in, because the prose index reads all_controls.json rather than any
    roster. The exclusion belongs here, at the one chokepoint both corpora pass
    through, not in each reader.

    A holdout is excluded for a different reason than a restricted framework:
    restricted is about what this repository may redistribute, holdout is about
    what the model may see. See tract.config.HOLDOUT_FRAMEWORK_IDS.

    Raises:
        ValueError: If nothing survives. An empty corpus is a different and
            much louder failure than one missing framework, and it would
            otherwise surface as a confusing schema error downstream.
    """
    kept = [
        f for f in frameworks
        if f.get("framework_id") not in HOLDOUT_FRAMEWORK_IDS
    ]
    dropped = [str(f.get("framework_id")) for f in frameworks
               if f.get("framework_id") in HOLDOUT_FRAMEWORK_IDS]
    for framework_id in sorted(dropped):
        logger.info(
            "Excluded %s from both merged corpora: it is a pretraining-"
            "contamination holdout", framework_id,
        )
    if not kept:
        raise ValueError(
            f"Every framework on disk is a holdout ({sorted(dropped)}), so "
            f"the merged corpus would be empty. Parse at least one framework "
            f"that is not in HOLDOUT_FRAMEWORK_IDS."
        )
    return kept


def _build(frameworks: list[dict[str, object]]) -> dict[str, object]:
    """Assemble one merged corpus from the frameworks it actually contains."""
    total_controls = sum(_control_count(f) for f in frameworks)
    generated_date = max(
        (str(f.get("fetched_date", "")) for f in frameworks), default=""
    )
    return {
        "generated_date": generated_date,
        "framework_count": len(frameworks),
        "total_controls": total_controls,
        "frameworks": frameworks,
    }


def main(
    frameworks_dir: Path | None = None,
    output_dir: Path | None = None,
    licensed_dir: Path | None = None,
) -> None:
    """Merge every per-framework JSON into the tracked corpus and the overlay."""
    frameworks_dir = frameworks_dir or PROCESSED_FRAMEWORKS_DIR
    output_dir = output_dir or PROCESSED_DIR
    licensed_dir = licensed_dir or PROCESSED_LICENSED_DIR

    files = sorted(frameworks_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No framework files in {frameworks_dir}")

    frameworks: list[dict[str, object]] = []
    for path in files:
        data = load_json(path)
        frameworks.append(data)
        logger.info("Loaded %s: %d controls", path.stem, _control_count(data))

    frameworks = _drop_holdouts(frameworks)

    restricted = [
        f for f in frameworks if f.get("framework_id") in RESTRICTED_FRAMEWORK_IDS
    ]
    public = [
        f for f in frameworks if f.get("framework_id") not in RESTRICTED_FRAMEWORK_IDS
    ]

    tracked = _build(public)
    tracked_path = output_dir / MERGED_FILENAME
    atomic_write_json(tracked, tracked_path)
    logger.info(
        "Wrote %s: %d frameworks, %d total controls",
        tracked_path, tracked["framework_count"], tracked["total_controls"],
    )

    overlay_path = licensed_dir / MERGED_FILENAME
    if restricted:
        overlay = _build(frameworks)
        atomic_write_json(overlay, overlay_path)
        logger.info(
            "Wrote gitignored overlay %s: %d frameworks (%d restricted), "
            "%d total controls",
            overlay_path, overlay["framework_count"], len(restricted),
            overlay["total_controls"],
        )
        return

    if overlay_path.exists():
        # A surviving overlay would shadow the tracked corpus for every reader
        # that prefers it, with no restricted source left to regenerate it.
        overlay_path.unlink()
        logger.info(
            "Removed stale overlay %s: no restricted framework is on disk",
            overlay_path,
        )


if __name__ == "__main__":
    main()
