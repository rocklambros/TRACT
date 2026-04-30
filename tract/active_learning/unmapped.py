"""Load unmapped AI controls from parsed framework JSON files."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_unmapped_controls(
    frameworks_dir: Path,
    framework_file_ids: list[str],
    framework_display_names: dict[str, str],
) -> list[dict]:
    """Load all controls from unmapped AI frameworks.

    Each control gets a composite control_id (framework_file_id:original_id)
    and a control_text built from title + full_text (or description as fallback).
    """
    all_controls: list[dict] = []

    for fid in sorted(framework_file_ids):
        fw_path = frameworks_dir / f"{fid}.json"
        with open(fw_path, encoding="utf-8") as f:
            data = json.load(f)

        display_name = framework_display_names.get(fid, data.get("framework_name", fid))

        for ctrl in data.get("controls", []):
            cid = ctrl.get("control_id", "")
            title = ctrl.get("title", "")
            full_text = ctrl.get("full_text", "")
            description = ctrl.get("description", "")

            body = full_text if full_text.strip() else description
            control_text = f"{cid}: {title}. {body}" if title else body

            all_controls.append({
                "control_id": f"{fid}:{cid}",
                "framework": display_name,
                "control_text": control_text,
                "title": title,
                "description": description,
            })

        logger.info("Loaded %d controls from %s", len(data.get("controls", [])), display_name)

    logger.info("Total unmapped controls: %d from %d frameworks", len(all_controls), len(framework_file_ids))
    return all_controls
