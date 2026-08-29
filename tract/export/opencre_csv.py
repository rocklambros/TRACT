"""Generate OpenCRE-compatible CSV from filtered assignments (spec §3).

Output targets OpenCRE's parse_export_format() parser. Each row has:
- CRE 0: "hub_id|hub_name" (pipe-delimited)
- StandardName|name: control title
- StandardName|id: section_id
- StandardName|description: control description
- StandardName|hyperlink: URL to official source

CRE 0 ONLY — no hierarchy columns.

Licence filtering lives here as well as in .gitignore, and for the same reason
it does in tract/export/canonical.py. The description column carries the
publisher's own control statement, the default output directory is
./opencre_export at the repository root, and that directory holds tracked files
today, so `tract export --opencre && git add -A` staged control prose with
nothing checking whose it was. Gitignoring the directory closes the git channel
and does nothing about the one the command exists to open: the stated
destination is OpenCRE's importer, outside git entirely.

So a framework in OVERLAY_FRAMEWORK_IDS exports its section identifier, its
title, its hyperlink and its CRE mapping, and exports a standing sentence in
place of its control text. See tract.licensing.withheld_control_text for why
that shape and not omission.
"""
from __future__ import annotations

import csv
import logging
import os
import tempfile
from io import StringIO
from pathlib import Path

from tract.crosswalk.export import CSV_FORMULA_TRIGGERS, neutralize_csv_cell
from tract.export.filters import ExportableAssignment
from tract.export.opencre_names import build_hyperlink, get_opencre_name
from tract.licensing import exportable_description

logger = logging.getLogger(__name__)


def _is_formula_shaped(value: str) -> bool:
    """True when a spreadsheet would treat this cell as a formula.

    Shares CSV_FORMULA_TRIGGERS with the crosswalk exporter so the two cannot
    drift, and repeats its numeric carve-out: "-1" and "-0.5" are numbers in a
    cell, not formulas, and refusing them would reject legitimate identifiers.
    """
    if not value.startswith(CSV_FORMULA_TRIGGERS):
        return False
    try:
        float(value)
    except ValueError:
        return True
    return False


def _require_safe_identifier(value: str, column: str, section_id: str) -> str:
    """Refuse a formula-shaped value in a column OpenCRE parses as a key.

    Escaping is the right remedy for prose and the wrong one here.
    `parse_export_format()` splits `CRE 0` on the separator to recover a hub id
    and reads `|id` as a section identifier, so an apostrophe guard would be
    stored AS PART OF THE KEY -- `'342-641` resolves to no CRE, and the mapping
    silently vanishes from the import rather than arriving wrong-looking.

    Passing it through unguarded is the injection this function exists to stop,
    and escaping it is data corruption, so the only honest option left is to
    refuse. No identifier in any parsed framework is formula-shaped today; if
    one ever is, that is a parser bug worth surfacing loudly rather than
    papering over at the export boundary.
    """
    if _is_formula_shaped(value):
        raise ValueError(
            f"Row {section_id!r} carries a formula-shaped value in {column!r}: "
            f"{value!r}. OpenCRE parses this column as an identifier, so it can "
            "be neither escaped (the guard character would become part of the "
            "key and the mapping would not resolve) nor exported as-is (a "
            "spreadsheet would evaluate it). Fix the value at its parser."
        )
    return value


def generate_opencre_csv(rows: list[ExportableAssignment], framework_id: str) -> str:
    """Generate OpenCRE CSV string from filtered assignment rows.

    Prose columns are neutralised against CSV formula injection; identifier
    columns are refused if formula-shaped, because escaping a key OpenCRE
    parses would corrupt the mapping rather than protect anyone. See
    _require_safe_identifier.

    Raises:
        ValueError: a row carries no framework_id, so its licence tier cannot
            be resolved (from tract.licensing.exportable_description); or a row
            carries a formula-shaped hub id, hub name or section id.
        KeyError: framework_id has no OpenCRE name or hyperlink template.
    """
    opencre_name = get_opencre_name(framework_id)

    fieldnames = [
        "CRE 0",
        f"{opencre_name}|name",
        f"{opencre_name}|id",
        f"{opencre_name}|description",
        f"{opencre_name}|hyperlink",
    ]

    sorted_rows = sorted(rows, key=lambda r: (r["hub_id"], r["framework_id"], r["section_id"]))

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    withheld = 0
    for row in sorted_rows:
        # Identifier columns: refuse rather than escape. See
        # _require_safe_identifier for why those are the only two options.
        section_id = _require_safe_identifier(
            row["section_id"], "id", row["section_id"],
        )
        cre0 = "{}|{}".format(
            _require_safe_identifier(row["hub_id"], "CRE 0 (hub id)", section_id),
            _require_safe_identifier(row["hub_name"], "CRE 0 (hub name)", section_id),
        )
        hyperlink = build_hyperlink(framework_id, section_id)
        # Keyed on the ROW's framework, not on the argument. The two agree on
        # every call site today, and keying on the argument would let a row
        # belonging to a withheld framework ride out under a publishable one's
        # name the first time a caller passes a mixed list.
        description = exportable_description(
            row["framework_id"], row["description"],
        )
        if description != row["description"]:
            withheld += 1

        # Prose columns: neutralise. OpenCRE copies these verbatim into its
        # database and never evaluates them (parse_export_format has no formula
        # handling at all), so a leading apostrophe is cosmetic there and stops
        # a spreadsheet executing the cell when an analyst opens the exported
        # file -- which is where the real exposure is, since this CSV is the
        # RFC deliverable and gets read outside the project.
        writer.writerow({
            "CRE 0": cre0,
            f"{opencre_name}|name": neutralize_csv_cell(row["title"]),
            f"{opencre_name}|id": section_id,
            f"{opencre_name}|description": neutralize_csv_cell(description),
            f"{opencre_name}|hyperlink": neutralize_csv_cell(hyperlink),
        })

    if withheld:
        logger.info(
            "Withheld control text for %d of %s's %d exported rows: its "
            "licence does not permit TRACT to redistribute the publisher's "
            "wording. Identifier, title, hyperlink and CRE mapping are "
            "exported in full.",
            withheld, framework_id, len(sorted_rows),
        )

    return output.getvalue()


def write_opencre_csv(
    rows: list[ExportableAssignment],
    framework_id: str,
    output_dir: Path,
) -> Path:
    """Generate and atomically write OpenCRE CSV to output_dir."""
    csv_text = generate_opencre_csv(rows, framework_id)
    opencre_name = get_opencre_name(framework_id)
    safe_name = opencre_name.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"{safe_name}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return output_path
