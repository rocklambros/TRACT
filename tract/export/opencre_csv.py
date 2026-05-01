"""Generate OpenCRE-compatible CSV from filtered assignments (spec §3).

Output targets OpenCRE's parse_export_format() parser. Each row has:
- CRE 0: "hub_id|hub_name" (pipe-delimited)
- StandardName|name: control title
- StandardName|id: section_id
- StandardName|description: control description
- StandardName|hyperlink: URL to official source

CRE 0 ONLY — no hierarchy columns.
"""
from __future__ import annotations

import csv
import logging
import os
import tempfile
from io import StringIO
from pathlib import Path

from tract.export.opencre_names import build_hyperlink, get_opencre_name

logger = logging.getLogger(__name__)


def generate_opencre_csv(rows: list[dict], framework_id: str) -> str:
    """Generate OpenCRE CSV string from filtered assignment rows."""
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

    for row in sorted_rows:
        cre0 = f"{row['hub_id']}|{row['hub_name']}"
        hyperlink = build_hyperlink(framework_id, row["section_id"])

        writer.writerow({
            "CRE 0": cre0,
            f"{opencre_name}|name": row["title"],
            f"{opencre_name}|id": row["section_id"],
            f"{opencre_name}|description": row["description"],
            f"{opencre_name}|hyperlink": hyperlink,
        })

    return output.getvalue()


def write_opencre_csv(
    rows: list[dict],
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
