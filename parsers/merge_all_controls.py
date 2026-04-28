"""Merge all validated framework JSONs into a single all_controls.json."""
from __future__ import annotations

import datetime
import logging

from tract.config import PROCESSED_DIR, PROCESSED_FRAMEWORKS_DIR
from tract.io import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    files = sorted(PROCESSED_FRAMEWORKS_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No framework files in {PROCESSED_FRAMEWORKS_DIR}")

    frameworks: list[dict[str, object]] = []
    total_controls = 0

    for path in files:
        data = load_json(path)
        frameworks.append(data)
        count = len(data.get("controls", []))
        total_controls += count
        logger.info("Loaded %s: %d controls", path.stem, count)

    output = {
        "generated_date": datetime.date.today().isoformat(),
        "framework_count": len(frameworks),
        "total_controls": total_controls,
        "frameworks": frameworks,
    }

    output_path = PROCESSED_DIR / "all_controls.json"
    atomic_write_json(output, output_path)
    logger.info(
        "Wrote %s: %d frameworks, %d total controls",
        output_path, len(frameworks), total_controls,
    )


if __name__ == "__main__":
    main()
