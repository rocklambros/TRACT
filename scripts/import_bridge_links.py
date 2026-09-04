"""Import a filled annotator sheet as Tier-2 bridge links.

This is the tier boundary. Everything it accepts becomes training supervision
and enters the Gate 1 orphan count, and the input has been through a
spreadsheet, an email client and a human before it arrives.

So it validates and refuses; it never coerces and never skips. A row naming an
unknown hub, an unknown control, a duplicate mapping, a confidence off the
scale or an empty rationale raises, naming the row number. Dropping such a row
instead would report a smaller corpus as though it were the whole one, and that
count is exactly what Gate 1 measures.

Provenance is stamped here rather than trusted from the sheet: tier, annotator
id and timestamp are supplied by the operator running the import, because a
spreadsheet column claiming `tier: 2` is a claim the spreadsheet cannot make.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Final

from tract.bridge.links import BridgeLink
from tract.config import PROCESSED_DIR, TRAINING_DIR
from tract.io import atomic_write_text

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH: Final[Path] = TRAINING_DIR / "hub_links_bridge.jsonl"

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "control_id", "cre_id", "confidence", "rationale",
)

# The annotator sheet's confidence scale, inclusive.
CONFIDENCE_MIN: Final[int] = 1
CONFIDENCE_MAX: Final[int] = 5


def _known_hub_ids() -> frozenset[str]:
    hierarchy = json.loads(
        (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
    )
    return frozenset(hierarchy["hubs"])


def _framework_controls(framework_id: str) -> dict[str, str]:
    """control_id -> title, for the framework the sheet covers."""
    payload = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    for framework in payload["frameworks"]:
        if framework["framework_id"] == framework_id:
            return {
                c["control_id"]: c.get("title", "") for c in framework["controls"]
            }
    raise ValueError(
        f"{framework_id!r} is not a parsed framework, so its control ids "
        "cannot be validated."
    )


def _standard_name(framework_id: str) -> str:
    """The standard_name the curated link set uses for this framework id."""
    payload: dict[str, list[dict[str, str]]] = json.loads(
        (TRAINING_DIR / "hub_links_by_framework_curated.json").read_text(
            encoding="utf-8"
        )
    )
    links = payload.get(framework_id)
    if not links:
        raise ValueError(
            f"{framework_id!r} has no links in the curated set, so its "
            "standard_name cannot be resolved. A bridge link whose "
            "standard_name disagrees with the curated set will not join to it."
        )
    return links[0]["standard_name"]


def import_bridge_links(
    source: Path,
    output: Path,
    *,
    framework_id: str,
    annotator_id: str,
    created_at: str,
) -> list[BridgeLink]:
    """Validate a filled sheet and write the Tier-2 corpus atomically.

    Returns the accepted links. Writes nothing at all unless every row passes,
    so a rejected sheet leaves no partial corpus and does not clobber an
    existing one.
    """
    if not annotator_id.strip():
        raise ValueError(
            "annotator_id is required: a Tier-2 link is a claim by a named "
            "person, and an unattributed one cannot be followed up."
        )
    if not created_at.strip():
        raise ValueError("created_at is required.")

    hubs = _known_hub_ids()
    controls = _framework_controls(framework_id)
    standard_name = _standard_name(framework_id)

    accepted: list[BridgeLink] = []
    seen: set[tuple[str, str]] = set()

    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"{source}: missing column(s) {sorted(missing_columns)}"
            )

        # start=2 so the number matches what a spreadsheet shows, header included.
        for row_number, row in enumerate(reader, start=2):
            control_id = (row["control_id"] or "").strip()
            cre_id = (row["cre_id"] or "").strip()
            if not control_id and not cre_id:
                continue

            if cre_id not in hubs:
                raise ValueError(
                    f"{source} row {row_number}: unknown hub id {cre_id!r}."
                )
            if control_id not in controls:
                raise ValueError(
                    f"{source} row {row_number}: unknown control id "
                    f"{control_id!r} for framework {framework_id!r}."
                )
            if (control_id, cre_id) in seen:
                raise ValueError(
                    f"{source} row {row_number}: duplicate mapping "
                    f"{control_id!r} -> {cre_id!r}."
                )
            seen.add((control_id, cre_id))

            raw_confidence = (row["confidence"] or "").strip()
            try:
                confidence = int(raw_confidence)
            except ValueError as exc:
                raise ValueError(
                    f"{source} row {row_number}: confidence "
                    f"{raw_confidence!r} is not an integer."
                ) from exc
            if not CONFIDENCE_MIN <= confidence <= CONFIDENCE_MAX:
                raise ValueError(
                    f"{source} row {row_number}: confidence {confidence} is "
                    f"outside {CONFIDENCE_MIN}-{CONFIDENCE_MAX}."
                )

            rationale = (row["rationale"] or "").strip()
            if not rationale:
                raise ValueError(
                    f"{source} row {row_number}: rationale is empty. It is "
                    "what makes a disputed link reviewable later."
                )

            accepted.append(
                BridgeLink(
                    framework_id=framework_id,
                    standard_name=standard_name,
                    section_id=control_id,
                    section_name=controls[control_id],
                    cre_id=cre_id,
                    tier=2,
                    annotator_id=annotator_id,
                    created_at=created_at,
                    confidence=confidence,
                    rationale=rationale,
                )
            )

    if not accepted:
        raise ValueError(
            f"{source} has no rows. An empty import would succeed and "
            "de-orphan nothing, which is indistinguishable from a round that "
            "was never run."
        )

    # Sorted so a re-import of the same sheet is byte-identical, and written in
    # one atomic step so a rejection above leaves the previous corpus intact.
    body = "".join(
        json.dumps(asdict(link), sort_keys=True) + "\n"
        for link in sorted(
            accepted, key=lambda link: (link.section_id, link.cre_id)
        )
    )
    atomic_write_text(body, output)
    logger.info("Imported %d Tier-2 bridge links to %s", len(accepted), output)
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Filled annotator CSV.")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_PATH,
        help="Where to write the Tier-2 bridge corpus.",
    )
    parser.add_argument(
        "--framework-id", default="nist_800_53",
        help="The framework the sheet covers.",
    )
    parser.add_argument(
        "--annotator-id", required=True,
        help="Who produced this sheet. Recorded on every link.",
    )
    parser.add_argument(
        "--created-at", required=True,
        help="ISO 8601 timestamp for the round, e.g. 2026-09-04T12:00:00Z.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    import_bridge_links(
        args.source,
        args.output,
        framework_id=args.framework_id,
        annotator_id=args.annotator_id,
        created_at=args.created_at,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
