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
from tract.io import atomic_write_json, atomic_write_text

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH: Final[Path] = TRAINING_DIR / "hub_links_bridge.jsonl"

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "control_id", "cre_id", "confidence", "rationale",
)

# The annotator sheet's confidence scale, inclusive. 1-3, per design decision
# D4, which also sets the Gate 1 counting floor at >= 2. Not 1-5: a wider scale
# here would silently admit values the gate's floor was never calibrated
# against.
CONFIDENCE_MIN: Final[int] = 1
CONFIDENCE_MAX: Final[int] = 3
# A link below this does not count toward Gate 1. Enforced by
# scripts/analysis/gate1_report.py, not here: a low-confidence link is data, it
# just is not evidence, so it is stored and then excluded from the count.
GATE1_CONFIDENCE_FLOOR: Final[int] = 2

# What an annotator writes in `cre_id` when no hub in the AI region fits the
# control. A first-class answer, not an error: the handbook calls it "a real,
# correct, expected answer", and the design names "too few links" as the round's
# most likely and most informative outcome. It used to be an unknown hub id, so
# a single NONE row rejected the entire sheet -- whose cheapest recovery is
# `grep -v NONE`, which deletes exactly the negative evidence and biases the
# round toward Gate 1 passing.
NO_HUB_SENTINEL: Final[str] = "NONE"

# The rationale is a human-channel field: it is read in spreadsheets and
# terminals by reviewers and adjudicators. It never reaches the model --
# bridge_training_records carries only ids and names.
RATIONALE_MAX_CHARS: Final[int] = 2_000

# A cell starting with one of these is executed by Excel, LibreOffice and
# Sheets on open. Refused rather than escaped: an annotator has no reason to
# begin a rationale this way, and silently rewriting their words is worse than
# asking them to resend.
FORMULA_PREFIXES: Final[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")

# Bidirectional overrides and isolates. These change what a human READS without
# changing what is stored, which is the whole attack.
BIDI_CONTROLS: Final[frozenset[str]] = frozenset(
    "\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
)


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


def _clean_rationale(raw: str, where: str) -> str:
    """Sanitise an annotator's free text, refusing what cannot be sanitised.

    CLAUDE.md requires null bytes stripped, unicode NFC-normalised and a length
    cap on every stored text field. tract/sanitize.py implements all three and
    eighteen modules call it; this boundary -- the only one ingesting text from
    outside the project -- did not.

    Formula prefixes and bidi controls are REFUSED rather than cleaned, because
    both are about what a human sees rather than what is stored, and quietly
    editing an annotator's words to make them safe is its own problem.
    """
    from tract.sanitize import sanitize_text

    if len(raw) > RATIONALE_MAX_CHARS:
        raise ValueError(
            f"{where}: rationale is too long -- {len(raw)} characters "
            f"against a {RATIONALE_MAX_CHARS} limit."
        )

    # sanitize_text strips null bytes and zero-width characters but leaves the
    # rest of C0 -- BEL, and ESC, which begins every ANSI sequence. A reviewer
    # reading these in a terminal is the channel that matters, so they go here
    # rather than in the shared helper, which many callers rely on to be a
    # text normaliser and not a control-character filter.
    stripped = "".join(ch for ch in raw if ord(ch) >= 0x20 or ch in "\t\n")
    try:
        cleaned = sanitize_text(stripped, max_length=RATIONALE_MAX_CHARS)
    except ValueError as exc:
        raise ValueError(f"{where}: rationale is empty after cleaning.") from exc

    # The prefix and bidi checks run on the CLEANED text, not the raw text.
    # Checking raw first was bypassable: sanitize_text strips zero-width
    # characters and HTML, so "\u200b=HYPERLINK(...)" and "<b></b>=cmd|..."
    # both failed the raw check and were then stored WITH a leading "=".
    # Whatever is stored is what a reviewer's spreadsheet opens, so that is
    # what has to be tested.
    if cleaned[:1] in FORMULA_PREFIXES:
        raise ValueError(
            f"{where}: rationale begins with {cleaned[:1]!r} once cleaned, "
            "which a spreadsheet executes as a formula when a reviewer opens "
            "the file."
        )
    if BIDI_CONTROLS & set(cleaned):
        raise ValueError(
            f"{where}: rationale contains a bidirectional override, which "
            "changes how it reads to a human without changing what is stored."
        )
    return cleaned


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
    replace: bool = False,
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

    # A second annotator's import used to overwrite the first's silently, with
    # an identical success message. Q4 requires a double-annotated overlap, the
    # importer takes one --annotator-id per call, and the natural operator
    # action is to run it once per returned sheet.
    if output.exists() and not replace:
        raise ValueError(
            f"{output} already exists. Importing over it would destroy the "
            f"previous annotator's corpus silently. Write one file per "
            f"annotator -- hub_links_bridge.<annotator_id>.jsonl -- or pass "
            f"replace=True if you genuinely mean to discard what is there."
        )

    hubs = _known_hub_ids()
    controls = _framework_controls(framework_id)
    standard_name = _standard_name(framework_id)

    accepted: list[BridgeLink] = []
    seen: set[tuple[str, str]] = set()
    reviewed: list[str] = []
    no_hub: list[str] = []

    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        present_columns = [c for c in (reader.fieldnames or []) if c is not None]
        missing_columns = set(REQUIRED_COLUMNS) - set(present_columns)
        if missing_columns:
            raise ValueError(
                f"{source}: missing column(s) {sorted(missing_columns)}"
            )
        # Refused, not ignored. The JSONL loader rejects unknown FIELDS for the
        # same reason, and this is the boundary facing the spreadsheet, so it is
        # where a second-hub column or a typo'd header actually shows up.
        unknown_columns = set(present_columns) - set(REQUIRED_COLUMNS)
        if unknown_columns:
            raise ValueError(
                f"{source}: unknown column(s) {sorted(unknown_columns)}. "
                "Dropping them silently would discard annotator judgements."
            )

        # start=2 so the number matches what a spreadsheet shows, header included.
        for row_number, row in enumerate(reader, start=2):
            control_id = (row["control_id"] or "").strip()
            cre_id = (row["cre_id"] or "").strip()
            if not control_id and not cre_id:
                continue

            if control_id not in controls:
                raise ValueError(
                    f"{source} row {row_number}: unknown control id "
                    f"{control_id!r} for framework {framework_id!r}."
                )
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

            raw_rationale = (row["rationale"] or "").strip()
            if not raw_rationale:
                raise ValueError(
                    f"{source} row {row_number}: rationale is empty. It is "
                    "what makes a disputed link reviewable later."
                )
            rationale = _clean_rationale(
                raw_rationale, f"{source} row {row_number}"
            )

            reviewed.append(control_id)
            if cre_id == NO_HUB_SENTINEL:
                # A judgement, not a link. Recorded so the round has a
                # denominator of controls actually worked.
                no_hub.append(control_id)
                continue

            if cre_id not in hubs:
                raise ValueError(
                    f"{source} row {row_number}: unknown hub id {cre_id!r}. "
                    f"Use {NO_HUB_SENTINEL!r} when no hub fits the control."
                )
            if (control_id, cre_id) in seen:
                raise ValueError(
                    f"{source} row {row_number}: duplicate mapping "
                    f"{control_id!r} -> {cre_id!r}."
                )
            seen.add((control_id, cre_id))

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

    if not reviewed:
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

    # The denominator. Without it, a volunteer who worked 300 controls and found
    # few links is indistinguishable in the record from one who worked 40, and
    # Q1 only counts controls that PRODUCED a link -- so it can detect neither a
    # shortfall in effort nor a genuinely hard task.
    reviewed_path = output.with_suffix(".reviewed.json")
    atomic_write_json(
        {
            "annotator_id": annotator_id,
            "created_at": created_at,
            "framework_id": framework_id,
            "source": str(source),
            "n_reviewed": len(reviewed),
            "n_linked": len(accepted),
            "n_no_hub": len(no_hub),
            "no_hub_controls": sorted(set(no_hub)),
        },
        reviewed_path,
    )

    logger.info(
        "Imported %d Tier-2 bridge links to %s (%d controls reviewed, "
        "%d judged to have no fitting hub) -> %s",
        len(accepted), output, len(reviewed), len(no_hub), reviewed_path,
    )
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
        "--replace", action="store_true",
        help=(
            "Overwrite an existing output file. Without it the import refuses, "
            "because a second annotator's sheet used to destroy the first's "
            "silently. Prefer one file per annotator."
        ),
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
        replace=args.replace,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
