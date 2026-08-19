"""Build the contamination probe's negative-control item set.

    python -m scripts.build_contamination_control

The closed-book probe asks each judge to recall OpenCRE's published hub id for
a control. On the 250 study items that question has a right answer, so
above-chance recall is evidence of memorisation. The question is what a model
does when it *cannot* know, and that needs a framework the judges could not
have seen mapped.

The 2026 edition of the OWASP LLM Top 10 is that framework. Its source
document is dated August 2026; the three panel members were released between
April and June 2026. OpenCRE has never mapped it, so every hub id a model
emits for these controls is confabulation by construction, which is exactly
the base rate needed to read the contaminable arm.

Writes the ten controls in ceiling_items.json's schema so the same
`--items` path and the same prompt cover both arms.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tract.config import CEILING_STUDY_DIR, EXIT_USER_ERROR, PROCESSED_DIR
from tract.io import atomic_write_json, load_json
from tract.sanitize import sanitize_text

SOURCE = PROCESSED_DIR / "frameworks" / "owasp_llm_top10_2026.json"
OUTPUT = CEILING_STUDY_DIR / "contamination_control_items.json"

# Negative item_index values so a control item can never be confused with, or
# silently merged into, a study item. The study is 1..250.
_INDEX_BASE = -1


def build_items(framework: dict[str, Any]) -> list[dict[str, Any]]:
    """Framework controls -> ceiling-study item rows, in source order."""
    items: list[dict[str, Any]] = []
    for offset, control in enumerate(framework["controls"]):
        items.append({
            "item_index": _INDEX_BASE - offset,
            "framework_id": str(framework["framework_id"]),
            "framework_name": str(framework["framework_name"]),
            "control_id": sanitize_text(str(control["control_id"])),
            "control_title": sanitize_text(str(control.get("title") or control["control_id"])),
            "control_text": sanitize_text(str(control["description"])),
            "stratum": "contamination_control",
            "text_source": "description",
        })
    return items


def main() -> int:
    if not SOURCE.exists():
        print(f"error: file not found: {SOURCE}", file=sys.stderr)
        return EXIT_USER_ERROR
    framework = load_json(SOURCE)
    items = build_items(framework)
    atomic_write_json(
        {
            "items": items,
            "n_items": len(items),
            "source": str(SOURCE.relative_to(Path.cwd())) if SOURCE.is_relative_to(Path.cwd()) else str(SOURCE),
            "source_version": str(framework["version"]),
            "fetched_date": str(framework["fetched_date"]),
            "purpose": (
                "Negative control for the closed-book contamination probe. "
                "OpenCRE has never mapped this framework and its source "
                "document postdates every panel member's release, so any hub "
                "id recalled for these controls is confabulation."
            ),
        },
        OUTPUT,
    )
    print(f"wrote {OUTPUT} ({len(items)} controls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
