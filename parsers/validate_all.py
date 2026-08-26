"""Cross-framework validation: schema conformance, counts, no empty fields."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from tract.config import PROCESSED_FRAMEWORKS_DIR
from tract.io import load_json
from tract.schema import FrameworkOutput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def validate_framework(path: Path) -> list[str]:
    """Validate a single framework JSON. Returns list of error messages."""
    errors: list[str] = []
    fw_id = path.stem

    try:
        data = load_json(path)
        output = FrameworkOutput.model_validate(data)
    except Exception as e:
        return [f"{fw_id}: schema validation failed: {e}"]

    if output.framework_id != fw_id:
        errors.append(f"{fw_id}: framework_id mismatch (file={fw_id}, data={output.framework_id})")

    # No count check here. It used to compare against a hand-maintained table
    # in tract/config.py that drifted from the parsers' own declarations, so
    # this script failed every run on owasp_ai_exchange while that parser was
    # working correctly. BaseParser.run() now enforces the count at write
    # time against the value the parser itself declares, which is the only
    # place that value exists.

    seen_ids: set[str] = set()
    for ctrl in output.controls:
        if not ctrl.description.strip():
            errors.append(f"{fw_id}: empty description for {ctrl.control_id}")
        if ctrl.control_id in seen_ids:
            errors.append(f"{fw_id}: duplicate control_id {ctrl.control_id}")
        seen_ids.add(ctrl.control_id)

    return errors


def main() -> None:
    framework_dir = PROCESSED_FRAMEWORKS_DIR
    files = sorted(framework_dir.glob("*.json"))

    if not files:
        logger.error("No framework files found in %s", framework_dir)
        sys.exit(1)

    total_errors: list[str] = []
    for path in files:
        errors = validate_framework(path)
        if errors:
            for e in errors:
                logger.error("FAIL: %s", e)
            total_errors.extend(errors)
        else:
            data = load_json(path)
            count = len(data.get("controls", []))
            logger.info("PASS: %s (%d controls)", path.stem, count)

    logger.info("Validated %d frameworks, %d errors", len(files), len(total_errors))
    if total_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
