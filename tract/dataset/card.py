"""TRACT dataset card — HuggingFace Datasets README.md generation."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_YAML_FRONTMATTER = """\
---
language: en
license: cc-by-sa-4.0
task_categories:
  - text-classification
tags:
  - security
  - crosswalk
  - CRE
  - AI-security
  - framework-mapping
---"""


def generate_dataset_card(
    staging_dir: Path,
    framework_metadata: list[dict],
    review_metrics: dict,
    bundle_stats: dict,
) -> Path:
    """Generate HuggingFace Datasets card as README.md.

    Sections: What Is This, Quick Start, Dataset Structure,
    Framework Coverage, How It Was Made, Review Methodology,
    Limitations, License, Citation.

    Args:
        staging_dir: Directory to write README.md into.
        framework_metadata: Per-framework stats from _build_framework_metadata().
        review_metrics: Metrics dict from compute_review_metrics().
        bundle_stats: Stats dict from bundle_dataset().

    Returns:
        Path to the written README.md.
    """
    total_rows = bundle_stats.get("total_rows", 0)
    n_frameworks = bundle_stats.get("frameworks", len(framework_metadata))

    overall = review_metrics.get("overall", {})
    coverage = review_metrics.get("coverage", {})
    calibration = review_metrics.get("calibration", {})

    accepted_pct = overall.get("accepted_pct", 0)
    rejected_pct = overall.get("rejected_pct", 0)
    reassigned_pct = overall.get("reassigned_pct", 0)
    total_reviewed = coverage.get("reviewed", 0)
    quality_score = calibration.get("quality_score")

    framework_table_rows = _build_framework_table(framework_metadata)

    quality_line = ""
    if quality_score is not None:
        quality_line = (
            f"\n\n**Reviewer quality (calibration):** {quality_score:.0%} agreement "
            f"on {calibration.get('total_reviewed', 20)} hidden calibration items."
        )

    card = f"""{_YAML_FRONTMATTER}

# TRACT Crosswalk Dataset

## What Is This

A human-reviewed crosswalk dataset mapping security controls from {n_frameworks} frameworks to CRE (Common Requirement Enumeration) hubs. Contains {total_rows:,} unique control-to-hub assignments combining OpenCRE ground truth links, model predictions from a fine-tuned BGE-large-v1.5 encoder, and expert human review.

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("rockCO78/tract-crosswalk-dataset")

# Filter to a specific framework
nist = ds.filter(lambda x: x["framework_id"] == "nist_csf")

# Get only human-reviewed model predictions
reviewed = ds.filter(lambda x: x["assignment_type"].startswith("model_"))
```

## Dataset Structure

Each row in `crosswalk_v1.0.jsonl` represents a single control-to-hub assignment:

| Field | Type | Description |
|-------|------|-------------|
| `control_id` | string | Unique control identifier |
| `framework_id` | string | Framework identifier |
| `framework_name` | string | Human-readable framework name |
| `section_id` | string | Control section ID within framework |
| `control_title` | string | Control title |
| `hub_id` | string | CRE hub identifier |
| `hub_name` | string | Hub name |
| `hub_path` | string | Hub hierarchy path |
| `assignment_type` | string | How this assignment was made (see below) |
| `confidence` | float/null | Model calibrated confidence (null for GT and reassigned) |
| `provenance` | string | Data source provenance |
| `review_status` | string | Review status (accepted/rejected/pending) |
| `reviewer_notes` | string/null | Reviewer notes (if any) |

### Assignment Types

| Value | Meaning |
|-------|---------|
| `ground_truth_linked` | Manually linked in OpenCRE by domain experts |
| `ground_truth_auto` | Automatically linked via CAPEC→CWE→CRE transitive chain |
| `model_accepted` | Model prediction accepted by human reviewer |
| `model_reassigned` | Model prediction reassigned to a different hub by reviewer |
| `model_rejected` | Model prediction rejected by human reviewer |

## Framework Coverage

| Framework | Controls | Assignments | Coverage Type |
|-----------|----------|-------------|---------------|
{framework_table_rows}

## How It Was Made

1. **Ground truth import:** {total_rows:,} OpenCRE-linked assignments from 26 frameworks were imported as the foundation.
2. **Model inference:** A fine-tuned BGE-large-v1.5 bi-encoder (TRACT) predicted hub assignments for controls in 5 uncovered AI security frameworks.
3. **Human review:** An outsourced cybersecurity expert reviewed all model predictions, accepting, reassigning, or rejecting each one. 20 hidden calibration items measured reviewer quality.
4. **Deduplication:** Overlapping assignments were deduplicated by provenance priority (ground truth > model prediction).

## Review Methodology

- **Total reviewed:** {total_reviewed:,} predictions
- **Acceptance rate:** {accepted_pct:.1f}%
- **Rejection rate:** {rejected_pct:.1f}%
- **Reassignment rate:** {reassigned_pct:.1f}%{quality_line}

## Limitations

- **Training bias:** The model was trained on existing OpenCRE links, which may bias predictions toward well-represented hubs and underrepresent newer or niche security topics.
- **Text quality variance:** Some frameworks provide only short control titles without descriptions. Predictions for these controls (flagged `text_quality: "low"`) may be less reliable.
- **Calibration scope:** The 20 calibration items test reviewer consistency but do not exhaustively cover all hub categories or edge cases.
- **Single reviewer:** All model predictions were reviewed by a single expert. Inter-rater reliability is not measured.

## License

This dataset is licensed under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/). You are free to share and adapt the material for any purpose, provided you give appropriate credit and distribute contributions under the same license.

## Citation

```bibtex
@dataset{{tract_crosswalk_2026,
  title = {{TRACT Crosswalk Dataset}},
  author = {{Lambros, Rock}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/rockCO78/tract-crosswalk-dataset}}
}}
```
"""

    staging_dir.mkdir(parents=True, exist_ok=True)
    target = staging_dir / "README.md"

    fd, tmp_path = tempfile.mkstemp(
        dir=staging_dir, prefix=".README.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(card)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Dataset card written to %s", target)
    return target


def _build_framework_table(framework_metadata: list[dict]) -> str:
    """Build markdown table rows from framework metadata."""
    lines: list[str] = []
    for fw in sorted(framework_metadata, key=lambda x: x.get("framework_name", "")):
        name = fw.get("framework_name", fw.get("framework_id", ""))
        controls = fw.get("total_controls", 0)
        assignments = fw.get("assignment_count", 0)
        coverage = fw.get("coverage_type", "unknown")
        lines.append(f"| {name} | {controls} | {assignments} | {coverage} |")
    return "\n".join(lines)
