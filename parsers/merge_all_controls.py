"""Merge validated framework JSONs into all_controls.json.

Two artifacts, not one.

`data/processed/all_controls.json` is git-tracked. It excludes every framework
in `RESTRICTED_FRAMEWORK_IDS` outright, and it carries every framework in
`OVERLAY_FRAMEWORK_IDS` reduced to identifiers and titles with no prose. This
repository is CC0, which is an affirmative grant that the publisher holds the
rights and waives them. Gitignoring the per-framework file alone left this wide
channel open: the merge globbed the same directory and inlined the untracked
file verbatim into a tracked one.

The filter used to read RESTRICTED_FRAMEWORK_IDS alone, which is the same
defect one tier down. The seven conditional frameworks carry GPL-3.0 and CC
BY-SA text, their per-framework files are gitignored, and the merge inlined
them into a tracked artifact anyway. Nothing fired, because they happen to
carry no prose today.

Reduced to titles rather than dropped, and the difference is deliberate. A
conditional framework's mapping stays tracked and published, because a mapping
is a fact about two documents rather than a reproduction of either, and its
titles are already published by OpenCRE. Measured on 2026-08-19: all 341
tracked controls across the seven already have `description == title` and no
`full_text`, so this reduction is byte-identical on current data and becomes a
real filter the first time a parser writes prose. Dropping the frameworks
outright would remove 341 controls and seven frameworks from the tracked
corpus, shifting every count downstream, which is an owner decision rather
than a leak fix.

`data/processed/licensed/all_controls.json` is gitignored and carries the full
corpus, every framework's prose included, for local training and evaluation.

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
    OVERLAY_FRAMEWORK_IDS,
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


def _redact_prose(framework: dict[str, object]) -> tuple[dict[str, object], int]:
    """Reduce one framework's controls to identifiers and titles.

    Returns (framework, redacted_control_count). A control that already carries
    no prose is returned as the same object, not a copy, so a corpus with
    nothing to redact serialises to identical bytes. The tracked artifact's
    sha256 is compared across LOFO folds, so a merge that rewrote equal values
    into new dicts would still be safe here, but proving byte-identity by
    construction is cheaper than arguing it from json.dumps's behaviour.

    Raises:
        ValueError: the framework's `controls` key is not a list. Same contract
            as _control_count: a corrupt artifact fails rather than passing
            through a filter that cannot inspect it.
    """
    controls = framework.get("controls", [])
    if not isinstance(controls, list):
        raise ValueError(
            f"framework {framework.get('framework_id', '?')!r} has a "
            f"'controls' key of type {type(controls).__name__}, expected list. "
            f"Prose cannot be filtered out of a structure that is not a list "
            f"of controls, and passing it through would publish it unchecked."
        )

    redacted = 0
    out_controls: list[object] = []
    for control in controls:
        if not isinstance(control, dict):
            raise ValueError(
                f"framework {framework.get('framework_id', '?')!r} has a "
                f"control of type {type(control).__name__}, expected dict"
            )
        title = str(control.get("title") or "")
        description = str(control.get("description") or "")
        carries_prose = description.strip() != title.strip()
        carries_full_text = bool(control.get("full_text"))
        if not (carries_prose or carries_full_text):
            out_controls.append(control)
            continue
        replacement = dict(control)
        replacement["description"] = title
        if carries_full_text:
            replacement["full_text"] = ""
        out_controls.append(replacement)
        redacted += 1

    if redacted == 0:
        return framework, 0
    return {**framework, "controls": out_controls}, redacted


def _tracked_view(
    frameworks: list[dict[str, object]],
) -> list[dict[str, object]]:
    """The corpus as it may be committed: no restricted source, no overlay prose.

    Two filters because the tiers mean two different things. A restricted
    source may not appear in git in any form. A conditional source may appear
    as identifiers and titles, which OpenCRE already publishes, and may not
    appear as the publisher's own control statements.
    """
    view: list[dict[str, object]] = []
    for framework in frameworks:
        framework_id = framework.get("framework_id")
        if framework_id in RESTRICTED_FRAMEWORK_IDS:
            continue
        if framework_id not in OVERLAY_FRAMEWORK_IDS:
            view.append(framework)
            continue
        reduced, redacted = _redact_prose(framework)
        if redacted:
            logger.info(
                "Reduced %s to titles in the tracked corpus: %d control "
                "statement(s) withheld. Its licence permits reproduction on "
                "terms a CC0 grant cannot carry; the full text is in the "
                "gitignored overlay.",
                framework_id, redacted,
            )
        view.append(reduced)
    return view


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

    # Presence of any overlay member on disk, not just a restricted one. The
    # overlay is the only artifact that may hold a conditional framework's
    # prose, so a checkout that parsed one needs the overlay written even when
    # no restricted source is present.
    overlay_members = [
        f for f in frameworks if f.get("framework_id") in OVERLAY_FRAMEWORK_IDS
    ]
    public = _tracked_view(frameworks)

    tracked = _build(public)
    tracked_path = output_dir / MERGED_FILENAME
    atomic_write_json(tracked, tracked_path)
    logger.info(
        "Wrote %s: %d frameworks, %d total controls",
        tracked_path, tracked["framework_count"], tracked["total_controls"],
    )

    overlay_path = licensed_dir / MERGED_FILENAME
    if overlay_members:
        overlay = _build(frameworks)
        atomic_write_json(overlay, overlay_path)
        logger.info(
            "Wrote gitignored overlay %s: %d frameworks (%d whose text may not "
            "be published), %d total controls",
            overlay_path, overlay["framework_count"], len(overlay_members),
            overlay["total_controls"],
        )
        return

    if overlay_path.exists():
        # A surviving overlay would shadow the tracked corpus for every reader
        # that prefers it, with no unpublishable source left to regenerate it.
        overlay_path.unlink()
        logger.info(
            "Removed stale overlay %s: no framework whose text may not be "
            "published is on disk",
            overlay_path,
        )


if __name__ == "__main__":
    main()
