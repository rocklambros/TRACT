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

from tract.export.filters import ExportableAssignment
from tract.export.opencre_names import build_hyperlink, get_opencre_name
from tract.licensing import exportable_description

logger = logging.getLogger(__name__)


def generate_opencre_csv(rows: list[ExportableAssignment], framework_id: str) -> str:
    """Generate OpenCRE CSV string from filtered assignment rows.

    Raises:
        ValueError: a row carries no framework_id, so its licence tier cannot
            be resolved. Raised by tract.licensing.exportable_description.
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
        cre0 = f"{row['hub_id']}|{row['hub_name']}"
        hyperlink = build_hyperlink(framework_id, row["section_id"])
        # Keyed on the ROW's framework, not on the argument. The two agree on
        # every call site today, and keying on the argument would let a row
        # belonging to a withheld framework ride out under a publishable one's
        # name the first time a caller passes a mixed list.
        description = exportable_description(
            row["framework_id"], row["description"],
        )
        if description != row["description"]:
            withheld += 1

        writer.writerow({
            "CRE 0": cre0,
            f"{opencre_name}|name": row["title"],
            f"{opencre_name}|id": row["section_id"],
            f"{opencre_name}|description": description,
            f"{opencre_name}|hyperlink": hyperlink,
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
