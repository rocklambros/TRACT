"""TRACT reviewer guide and hub reference generation."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


def generate_reviewer_guide(output_dir: Path, metadata: dict) -> Path:
    """Generate reviewer_guide.md with instructions, decision criteria, and common pitfalls.

    Args:
        output_dir: Directory to write reviewer_guide.md into.
        metadata: Metadata dict from generate_review_export (used for dynamic values).

    Returns:
        Path to the written file.
    """
    total = metadata.get("total_predictions", 0)
    priority = metadata.get("priority_breakdown", {})
    routine = priority.get("routine", 0)
    careful = priority.get("careful", 0)
    critical = priority.get("critical", 0)

    guide = f"""\
# TRACT Crosswalk Review Guide

## Role

You are a cybersecurity domain expert reviewing AI-generated mappings between
security framework controls and the Common Requirement Enumeration (CRE) hub
taxonomy. Your review determines which mappings are published as a
peer-reviewed research dataset.

## Background

**CRE (Common Requirement Enumeration)** is a universal taxonomy that links
security requirements across standards. It organizes security topics into
**522 hubs** arranged in a hierarchy — each hub represents a distinct security
concept (e.g., "Cryptography > Key Management" or "Authentication > Multi-Factor").

**TRACT** (Transitive Reconciliation and Assignment of CRE Taxonomies) maps
controls from **31 security frameworks** to CRE hubs using a fine-tuned
bi-encoder model. Each mapping is called an **assignment** — it says "this
control belongs under this hub."

Your job is to review these assignments and decide whether each one is correct.

## Step-by-Step Process

1. Open `review_predictions.json` in your editor.
2. For each prediction, read `control_text` to understand the control's security intent.
3. Read `assigned_hub_name` and `assigned_hub_path` to understand the model's suggestion.
4. Check `confidence` — above 0.70 is fairly certain, below 0.30 needs careful attention.
5. Check `review_priority` — "critical" items need the most care.
6. **Accept:** Control belongs under this hub → set `"status": "accepted"`.
7. **Reassign:** Wrong hub, but a better one exists → set `"status": "reassigned"`, set `"reviewer_hub_id"` to the correct hub ID (find IDs in `hub_reference.json`).
8. **Reject:** No hub fits → set `"status": "rejected"`, explain in `"reviewer_notes"`.
9. Add `reviewer_notes` for any non-obvious decision.

## Decision Criteria

The control's security **PURPOSE** should align with the hub's security
**DOMAIN**. This is not keyword matching — a control about "encrypting AI
training data" maps to encryption, not AI training.

## Common Pitfalls

- **MITRE ATLAS hubs are the hardest** — many sound similar. Read the full path.
- **"Rejected" means no hub fits at all.** Check `alternative_hubs` and `hub_reference.json` before rejecting.
- **High `is_ood: true`** means the model thinks this control is outside its training distribution — review more carefully.
- **NIST AI RMF predictions** are based on short control descriptions and may be less reliable.
- **Items flagged `text_quality: "low"`** had sparse input text — predictions may be unreliable.

## Editor Requirement

Use a JSON-aware editor (VS Code, Notepad++, Sublime Text) that highlights
syntax errors. Do **NOT** edit in plain Notepad or a word processor.

Common mistakes: missing commas between fields, extra trailing comma after last
field, unclosed quotes. Run `tract review-validate` before submitting to catch
JSON errors early.

## Saving Progress

Work in batches. Leave unreviewed items as `"status": "pending"`. Partial files
can be imported — you do not need to finish everything in one session.

## Time Estimate

This file contains **{total} predictions** to review:
- **{routine} routine** items (~1 min each)
- **{careful} careful** items (~3-5 min each)
- **{critical} critical** items (~3-5 min each)

Estimated total: **25-40 hours** depending on complexity.
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "reviewer_guide.md"

    fd, tmp_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=".reviewer_guide.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(guide)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Reviewer guide written to %s", target)
    return target


def generate_hub_reference(db_path: Path, output_dir: Path) -> Path:
    """Generate hub_reference.json — all hubs with id, name, path, parent_id, is_leaf.

    Args:
        db_path: Path to crosswalk.db.
        output_dir: Directory to write hub_reference.json into.

    Returns:
        Path to the written file.
    """
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT id, name, path, parent_id FROM hubs ORDER BY path",
        ).fetchall()

        parent_ids: set[str | None] = {row["parent_id"] for row in rows}

        hubs: list[dict] = []
        for row in rows:
            hubs.append({
                "hub_id": row["id"],
                "name": row["name"],
                "path": row["path"] or "",
                "parent_id": row["parent_id"],
                "is_leaf": row["id"] not in parent_ids,
            })
    finally:
        conn.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "hub_reference.json"

    fd, tmp_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=".hub_reference.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(hubs, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Hub reference written to %s (%d hubs)", target, len(hubs))
    return target
