"""Build a balanced re-run sample per annotator, blind to their prior answers.

The first round produced 87% and 85% NONE. That may be correct -- the regions
are disjoint by construction -- but the handbook nudged toward NONE four times,
once by asserting the answer distribution before the annotator had read
anything. This builds the sample that measures whether that mattered.

THE SAMPLE MUST CARRY NO SIGNAL. Sending only the controls someone marked NONE
would itself say "you said NONE too often" -- the request would deliver the very
nudge the re-run exists to remove. So each annotator gets every control they
linked, plus an equal number they marked NONE, shuffled together.

The sheet goes out BLANK. Their previous answers are not shown, not pre-filled,
and not implied by the ordering.

Deterministic: the shuffle is seeded per annotator, so the same round rebuilds
byte-identically and a coordinator can regenerate a lost packet.

Read-only over the round-1 corpora. Loads no model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import random
from pathlib import Path
from typing import Final

from scripts.build_bridge_packet import (
    ANNOTATION_FIELDS,
    ANNOTATION_SHEET_NAME,
    ANSWER_FIELDS,
    CONTROL_SHEET_NAME,
    HUB_SHEET_NAME,
    build_control_sheet,
    build_hub_sheet,
    write_manifest,
)
from tract.io import atomic_write_json
from tract.licensing import refuse_external_redistribution

logger = logging.getLogger(__name__)

# Where round-1 filled sheets were read from, and where round-2 packets go.
DEFAULT_ROUND1_DIR: Final[Path] = Path.home() / "tract-inbox" / "phase2c"
DEFAULT_OUT_DIR: Final[Path] = Path.home() / "tract-packets" / "phase2c-r2"

NO_HUB: Final[str] = "NONE"


def _seed_for(annotator: str) -> int:
    """A per-annotator seed derived from the id, so rebuilds are identical."""
    return int.from_bytes(
        hashlib.sha256(annotator.encode("utf-8")).digest()[:8], "big"
    )


def balanced_sample(
    filled: Path, annotator: str
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Every control they linked, plus an equal number they marked NONE.

    Returns (rows with answers stripped, counts). Raises rather than returning
    a lopsided sample: a sample whose composition reveals the hypothesis is
    worse than no re-run.
    """
    with filled.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    linked = [r for r in rows if (r["cre_id"] or "").strip().upper() != NO_HUB]
    none_rows = [r for r in rows if (r["cre_id"] or "").strip().upper() == NO_HUB]

    if not linked:
        raise ValueError(
            f"{filled} has no linked controls, so a balanced sample cannot be "
            "built and a NONE-only sheet would carry the hypothesis."
        )
    if len(none_rows) < len(linked):
        raise ValueError(
            f"{filled} has {len(none_rows)} NONE rows against {len(linked)} "
            "linked, so the sample cannot be balanced. Send the whole sheet "
            "rather than a sample that leans one way."
        )

    rng = random.Random(_seed_for(annotator))
    sampled = linked + rng.sample(none_rows, len(linked))
    rng.shuffle(sampled)

    # Blank every answer column. The sheet must arrive carrying no trace of what
    # they said last time -- not a value, not an ordering.
    blanked = [
        {**{k: r[k] for k in ANNOTATION_FIELDS if k not in ANSWER_FIELDS},
         **dict.fromkeys(ANSWER_FIELDS, "")}
        for r in sampled
    ]
    return blanked, {
        "n_sampled": len(blanked),
        "n_from_linked": len(linked),
        "n_from_none": len(linked),
        "n_original": len(rows),
    }


def build_rerun_packet(
    out_dir: Path, annotator: str, filled: Path, framework_id: str
) -> dict[str, int]:
    """Write one annotator's round-2 packet."""
    refuse_external_redistribution(framework_id)

    rows, counts = balanced_sample(filled, annotator)
    out_dir.mkdir(parents=True, exist_ok=True)

    build_hub_sheet(out_dir / HUB_SHEET_NAME)
    build_control_sheet(out_dir / CONTROL_SHEET_NAME, framework_id)
    with (out_dir / ANNOTATION_SHEET_NAME).open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ANNOTATION_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    n_hubs = sum(1 for _ in (out_dir / HUB_SHEET_NAME).open(encoding="utf-8")) - 1
    write_manifest(out_dir, framework_id, n_hubs, counts["n_sampled"])

    # Round-2 provenance, kept beside the packet and NOT sent: it records the
    # composition, which is the thing the annotator must not see.
    atomic_write_json(
        {"annotator": annotator, "round": 2, "framework_id": framework_id, **counts},
        out_dir / "sample_composition.json",
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotator", action="append", required=True,
        help="Pseudonym, repeatable. Must match a <pseudonym>.csv in --round1.",
    )
    parser.add_argument("--round1", type=Path, default=DEFAULT_ROUND1_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--framework-id", default="nist_800_53")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for annotator in args.annotator:
        filled = args.round1 / f"{annotator}.csv"
        if not filled.is_file():
            raise SystemExit(f"{filled} not found; cannot sample round 1.")
        counts = build_rerun_packet(
            args.out / annotator, annotator, filled, args.framework_id
        )
        logger.info(
            "%s: %d rows (%d they linked + %d they marked NONE) from %d, -> %s",
            annotator, counts["n_sampled"], counts["n_from_linked"],
            counts["n_from_none"], counts["n_original"], args.out / annotator,
        )
    logger.info("")
    logger.info(
        "Send each annotator their own directory, plus Part 2 of "
        "docs/phase2c-rerun-brief.md. Do NOT send sample_composition.json."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
