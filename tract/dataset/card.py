"""TRACT dataset card — HuggingFace Datasets README.md generation."""
from __future__ import annotations

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
  - zero-shot-classification
size_categories:
  - 1K<n<10K
tags:
  - security
  - crosswalk
  - CRE
  - AI-security
  - framework-mapping
  - compliance
  - governance
  - risk-management
  - NIST
  - MITRE-ATLAS
  - OWASP
  - ISO-27001
pretty_name: TRACT Security Framework Crosswalk
---"""


def generate_dataset_card(
    staging_dir: Path,
    framework_metadata: list[dict],
    review_metrics: dict,
    bundle_stats: dict,
) -> Path:
    """Generate HuggingFace Datasets card as README.md.

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
    calibration = review_metrics.get("calibration", review_metrics.get("reviewer_quality", {}))
    confidence = review_metrics.get("confidence_analysis", {})
    per_fw = review_metrics.get("per_framework", {})

    accepted = overall.get("accepted", 0)
    rejected = overall.get("rejected", 0)
    reassigned = overall.get("reassigned", 0)
    accepted_pct = overall.get("accepted_rate", overall.get("accepted_pct", 0))
    rejected_pct = overall.get("rejected_rate", overall.get("rejected_pct", 0))
    reassigned_pct = overall.get("reassigned_rate", overall.get("reassigned_pct", 0))
    total_reviewed = coverage.get("reviewed", 0)
    quality_score = calibration.get("quality_score")
    cal_total = calibration.get("reviewed", calibration.get("total_reviewed", 20))

    gt_count = sum(
        fw.get("assignment_count", 0)
        for fw in framework_metadata
        if fw.get("coverage_type") == "ground_truth"
    )
    model_count = sum(
        fw.get("assignment_count", 0)
        for fw in framework_metadata
        if fw.get("coverage_type") in ("model_prediction", "mixed")
    )

    framework_table_rows = _build_framework_table(framework_metadata)
    review_fw_table = _build_review_framework_table(per_fw)

    high_conf = confidence.get("high_confidence", {})
    low_conf = confidence.get("low_confidence", {})
    ood = confidence.get("ood_items", {})

    card = f"""{_YAML_FRONTMATTER}

# TRACT Crosswalk Dataset

A human-reviewed crosswalk mapping **{total_rows:,} security controls** from **{n_frameworks} frameworks** to **522 CRE hubs** — the first large-scale, multi-framework security taxonomy alignment dataset with expert review and calibrated quality metrics.

---

## Table of Contents

1. [What Is This Dataset?](#what-is-this-dataset)
2. [Key Concepts (Start Here If You're New)](#key-concepts-start-here-if-youre-new)
3. [Quick Start](#quick-start)
4. [Files in This Dataset](#files-in-this-dataset)
5. [Dataset Structure](#dataset-structure)
6. [Assignment Types Explained](#assignment-types-explained)
7. [Framework Coverage](#framework-coverage)
8. [How It Was Made](#how-it-was-made)
9. [Review Methodology and Quality](#review-methodology-and-quality)
10. [Confidence Scores and Calibration](#confidence-scores-and-calibration)
11. [Who Should Use This](#who-should-use-this)
12. [Usage Examples](#usage-examples)
13. [Known Limitations](#known-limitations)
14. [Related Resources](#related-resources)
15. [License](#license)
16. [Citation](#citation)

---

## What Is This Dataset?

This dataset answers a deceptively hard question: **when two security frameworks both talk about "access control" or "data encryption," are they actually talking about the same thing?**

Security teams today face a maze of overlapping frameworks — NIST 800-53, ISO 27001, MITRE ATLAS, OWASP Top 10, the EU AI Act, and dozens more. Each defines its own controls using its own terminology. Compliance officers, auditors, and security architects spend thousands of hours manually mapping controls between frameworks to answer questions like:

- "We're compliant with ISO 27001 — which NIST 800-53 controls does that cover?"
- "The EU AI Act requires X — do we already handle this under our MITRE ATLAS controls?"
- "Which frameworks address AI model poisoning, and how do their controls compare?"

**TRACT solves this by mapping every control to a shared taxonomy: the Common Requirement Enumeration (CRE).** Instead of N-to-N pairwise comparisons between frameworks, each control maps to one or more CRE hubs, making cross-framework comparison trivial.

This dataset contains **{total_rows:,} control-to-hub assignments** combining:
- **~{gt_count:,} ground truth links** from OpenCRE (expert-curated by the CRE project maintainers)
- **~{model_count:,} model predictions** from a fine-tuned BGE-large-v1.5 bi-encoder, each **individually reviewed by a cybersecurity domain expert**

---

## Key Concepts (Start Here If You're New)

**What is CRE?**
The [Common Requirement Enumeration](https://www.opencre.org/) is a universal taxonomy of security topics. Think of it as a Dewey Decimal System for cybersecurity. It organizes ~522 security concepts ("hubs") into a hierarchy — from broad topics like "Authentication" down to specifics like "Multi-factor Authentication > Time-based OTP."

**What is a "hub"?**
A hub is a single node in the CRE hierarchy representing one security concept. For example, hub `615-663` is "Cryptography" and hub `206-830` is "Input validation." Each hub has a unique ID, a name, and a path showing where it sits in the hierarchy.

**What is a "control"?**
A control is a specific security requirement from a framework. For example, NIST 800-53 control "AC-2" is "Account Management" and ISO 27001 control "A.9.2.1" is "User registration and de-registration." Different frameworks describe similar security concepts using different terminology and granularity.

**What is a "crosswalk"?**
A crosswalk maps controls from one system to equivalent concepts in another. This dataset is a crosswalk that maps controls from {n_frameworks} different frameworks to CRE hubs. If two controls from different frameworks map to the same CRE hub, they address the same security concept.

**What is an "assignment"?**
Each row in this dataset is an assignment — one control mapped to one CRE hub. A single control may map to multiple hubs (e.g., a control about "encrypted authentication" maps to both "Cryptography" and "Authentication" hubs).

---

## Quick Start

```python
from datasets import load_dataset

ds = load_dataset("rockCO78/tract-crosswalk-dataset")
print(f"{{len(ds['train'])}} assignments across {{ds['train'].unique('framework_id')}} frameworks")

# Filter to a specific framework
nist_800_53 = ds["train"].filter(lambda x: x["framework_id"] == "nist_800_53")
print(f"NIST 800-53: {{len(nist_800_53)}} assignments")

# Get only human-reviewed model predictions
reviewed = ds["train"].filter(lambda x: x["assignment_type"].startswith("model_"))
print(f"Model predictions (reviewed): {{len(reviewed)}}")

# Find all controls that map to the same hub as a given control
target_hub = ds["train"][0]["hub_id"]
same_hub = ds["train"].filter(lambda x: x["hub_id"] == target_hub)
print(f"Controls sharing hub '{{target_hub}}': {{len(same_hub)}}")
```

### Loading as pandas

```python
import pandas as pd
df = pd.read_json("crosswalk_v1.0.jsonl", lines=True)
```

### Loading as raw JSONL

```python
import json
with open("crosswalk_v1.0.jsonl") as f:
    assignments = [json.loads(line) for line in f]
```

---

## Files in This Dataset

| File | Description |
|------|-------------|
| `crosswalk_v1.0.jsonl` | Main dataset — one JSON object per line, one assignment per row |
| `framework_metadata.json` | Per-framework statistics (control counts, assignment counts, coverage type) |
| `cre_hierarchy_v1.1.json` | Full CRE hub hierarchy tree (522 hubs with parent-child relationships) |
| `hub_descriptions_v1.0.json` | Hub descriptions and metadata for all 522 CRE hubs |
| `review_metrics.json` | Detailed review quality metrics (acceptance rates, calibration scores, per-framework breakdown) |
| `bridge_report.json` | Bridge relationships connecting CRE subtrees (from Phase 2B analysis) |
| `zenodo_metadata.json` | Metadata for Zenodo DOI registration |
| `LICENSE` | CC-BY-SA-4.0 license text |
| `README.md` | This file |

---

## Dataset Structure

Each row in `crosswalk_v1.0.jsonl` represents a single control-to-hub assignment:

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `control_id` | string | Unique control identifier (framework-scoped) | `"nist_800_53:AC-2"` |
| `framework_id` | string | Framework identifier | `"nist_800_53"` |
| `framework_name` | string | Human-readable framework name | `"NIST SP 800-53"` |
| `section_id` | string | Control section ID within the framework | `"AC-2"` |
| `control_title` | string | Control title or name | `"Account Management"` |
| `hub_id` | string | CRE hub identifier | `"615-663"` |
| `hub_name` | string | Hub display name | `"Cryptography"` |
| `hub_path` | string | Full hierarchy path (root > ... > leaf) | `"Root > Technical > Crypto"` |
| `assignment_type` | string | How this assignment was created (see below) | `"ground_truth_linked"` |
| `confidence` | float or null | Model's calibrated confidence (0.0-1.0). Null for ground truth and reassigned items | `0.872` |
| `provenance` | string | Data source identifier | `"opencre_ground_truth"` |
| `review_status` | string | Review outcome | `"accepted"` |
| `reviewer_notes` | string or null | Free-text notes from the expert reviewer | `null` |

---

## Assignment Types Explained

Every assignment in this dataset has an `assignment_type` field explaining its origin and quality level:

| Type | Count | What It Means | Trust Level |
|------|-------|---------------|-------------|
| `ground_truth_linked` | varies | **Manually curated** by OpenCRE project maintainers. These are expert-created links between framework controls and CRE hubs. | Highest |
| `ground_truth_auto` | varies | **Automatically derived** via transitive chains (e.g., CAPEC attack pattern -> CWE weakness -> CRE hub). Deterministic, not ML. | High |
| `model_accepted` | {accepted} | TRACT model predicted this mapping and a **human expert confirmed** it was correct. | High |
| `model_reassigned` | {reassigned} | TRACT model predicted a mapping but the **expert chose a different (better) hub**. The `confidence` field is null because the model's score applied to a different hub. | High (expert-corrected) |
| `model_rejected` | {rejected} | TRACT model predicted a mapping but the **expert determined no appropriate hub exists**. These are included for completeness but represent failed mappings. | N/A (no valid mapping) |

**For most use cases, filter out `model_rejected`:**
```python
valid = ds["train"].filter(lambda x: x["assignment_type"] != "model_rejected")
```

---

## Framework Coverage

This dataset covers {n_frameworks} security, AI safety, and compliance frameworks:

| Framework | Controls | Assignments | Coverage Type |
|-----------|----------|-------------|---------------|
{framework_table_rows}

**Coverage types:**
- **ground_truth** — All assignments come from expert-curated OpenCRE links. These frameworks have been mapped by the CRE project maintainers.
- **model_prediction** — All assignments are TRACT model predictions, each reviewed by a human expert. These are typically newer AI-specific frameworks not yet in OpenCRE.
- **mixed** — Combines ground truth links with model predictions for controls not covered by OpenCRE.

---

## How It Was Made

This dataset was built through a four-stage pipeline:

### Stage 1: Ground Truth Import
We imported **{gt_count:,} existing links** from the [OpenCRE project](https://www.opencre.org/) — a community-maintained mapping of security standards. These cover 26 established frameworks (NIST 800-53, ISO 27001, OWASP, CWE, CAPEC, etc.). Links are either manually curated by domain experts (`LinkedTo`) or derived through deterministic transitive chains (`AutomaticallyLinkedTo`, e.g., CAPEC->CWE->CRE).

### Stage 2: Model Inference
For 10 frameworks with no OpenCRE coverage (primarily newer AI security frameworks), we ran inference using **TRACT** — a fine-tuned [BGE-large-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) bi-encoder trained on the OpenCRE ground truth data. The model encodes both control text and hub descriptions into a shared 1024-dimensional space, then assigns each control to its nearest hub via calibrated cosine similarity.

Key model details:
- Architecture: BGE-large-v1.5 with LoRA adapters, contrastive fine-tuning
- Training data: OpenCRE ground truth links (leave-one-framework-out cross-validation)
- Performance: hit@1 = 0.531 on held-out frameworks
- Confidence: Platt-scaled (temperature calibration) — NOT raw cosine similarity

### Stage 3: Expert Human Review
A cybersecurity domain expert reviewed **all {total_reviewed:,} model predictions** (878 real predictions + 20 hidden calibration items). For each prediction, the reviewer:
1. Read the control's full text to understand its security intent
2. Evaluated whether the model's suggested CRE hub was semantically appropriate
3. Made one of three decisions: **accept** (correct hub), **reassign** (better hub exists), or **reject** (no hub fits)

### Stage 4: Deduplication and Publication
Where a control appeared in multiple data sources (e.g., both ground truth and model prediction for the same control-hub pair), we kept the **highest-authority source**: ground truth > model prediction. The final dataset contains one row per unique (control, hub) pair.

---

## Review Methodology and Quality

### Overall Review Statistics

| Metric | Value |
|--------|-------|
| Total predictions reviewed | {total_reviewed:,} |
| Accepted (model was correct) | {accepted} ({accepted_pct:.1f}%) |
| Reassigned (expert chose better hub) | {reassigned} ({reassigned_pct:.1f}%) |
| Rejected (no appropriate hub) | {rejected} ({rejected_pct:.1f}%) |
| Review completion | {coverage.get('completion_pct', 100):.0f}% |

### Per-Framework Review Breakdown

| Framework | Accepted | Reassigned | Rejected | Acceptance Rate |
|-----------|----------|------------|----------|-----------------|
{review_fw_table}

### Calibration Quality Check
To measure reviewer quality, we embedded **{cal_total} hidden calibration items** among the predictions. These were ground-truth assignments where the correct hub was already known — the reviewer did not know which items were calibration tests.

- **Agreement rate: {f"{quality_score:.0%}" if quality_score is not None else "N/A"}** ({calibration.get('agreed', 0)}/{cal_total} calibration items agreed with ground truth)
- All {len(calibration.get('disagreements', []))} disagreements were **reassignments** (the reviewer chose a different hub), not rejections — suggesting legitimate alternative interpretations rather than errors

---

## Confidence Scores and Calibration

The `confidence` field contains the model's **calibrated probability** that its hub assignment is correct. These are Platt-scaled (temperature-calibrated) softmax outputs, NOT raw cosine similarities.

**Interpreting confidence values:**

| Range | Meaning | Observed Acceptance Rate |
|-------|---------|-------------------------|
| > 0.5 (high confidence) | Model is fairly certain | {high_conf.get('acceptance_rate', 0):.1f}% ({high_conf.get('accepted', 0)}/{high_conf.get('total', 0)}) |
| <= 0.5 (low confidence) | Model is less certain — review more carefully | {low_conf.get('acceptance_rate', 0):.1f}% ({low_conf.get('accepted', 0)}/{low_conf.get('total', 0)}) |
| OOD items | Model flags these as outside training distribution | {ood.get('acceptance_rate', 0):.1f}% ({ood.get('accepted', 0)}/{ood.get('total', 0)}) |

**Important:** `confidence` is `null` for:
- Ground truth assignments (no model was involved)
- Reassigned predictions (the model's confidence was for a *different* hub than the one the expert chose)

---

## Who Should Use This

**Compliance teams:** "We meet ISO 27001 — what does that buy us for NIST 800-53 compliance?" Filter both frameworks to the same hub_ids and you have your mapping.

**Security architects:** Building a control library that spans multiple frameworks? Use this dataset as the backbone — it handles the taxonomy alignment so you can focus on implementation.

**AI security researchers:** Studying how AI-specific frameworks (MITRE ATLAS, NIST AI RMF, EU AI Act) relate to traditional security controls? This is the first dataset to systematically map them to a shared taxonomy.

**GRC tool builders:** Need a machine-readable crosswalk for your platform? This dataset provides structured JSONL with consistent identifiers across all {n_frameworks} frameworks.

**NLP/ML researchers:** Interested in security text classification, taxonomy alignment, or domain-specific fine-tuning? This dataset provides labeled training data with expert-reviewed quality annotations.

---

## Usage Examples

### Find controls equivalent to a given control

```python
# "Which controls across all frameworks are equivalent to NIST 800-53 AC-2?"
ac2 = df[df["control_id"] == "nist_800_53:AC-2"]
ac2_hubs = set(ac2["hub_id"])
equivalents = df[df["hub_id"].isin(ac2_hubs) & (df["framework_id"] != "nist_800_53")]
print(equivalents[["framework_name", "section_id", "control_title", "hub_name"]])
```

### Cross-framework coverage gap analysis

```python
# "Which CRE hubs does ISO 27001 cover that MITRE ATLAS doesn't?"
iso_hubs = set(df[df["framework_id"] == "iso_27001"]["hub_id"])
atlas_hubs = set(df[df["framework_id"] == "mitre_atlas"]["hub_id"])
gaps = iso_hubs - atlas_hubs
gap_names = df[df["hub_id"].isin(gaps)][["hub_id", "hub_name"]].drop_duplicates()
print(f"ISO 27001 covers {{len(gaps)}} hubs that MITRE ATLAS does not:")
print(gap_names.head(20))
```

### Filter by trust level

```python
# Only expert-curated and expert-reviewed assignments
high_trust = df[df["assignment_type"].isin([
    "ground_truth_linked", "model_accepted", "model_reassigned"
])]
```

### AI-specific framework analysis

```python
ai_frameworks = ["mitre_atlas", "nist_ai_rmf", "csa_aicm", "eu_ai_act", "owasp_llm_top10"]
ai_controls = df[df["framework_id"].isin(ai_frameworks)]
print(f"AI framework controls: {{len(ai_controls)}}")
print(ai_controls.groupby("framework_name")["hub_id"].nunique().sort_values(ascending=False))
```

---

## Known Limitations

1. **Training distribution bias.** The model was trained on existing OpenCRE links, which over-represent traditional IT security (NIST, OWASP, CWE) and under-represent newer AI safety concepts. Predictions for AI-specific frameworks may be less accurate — this is reflected in the higher reassignment rate for frameworks like AIUC-1 and CoSAI.

2. **Text quality variance.** Some frameworks provide only short control titles without descriptions (53 controls flagged as `text_quality: "low"` during export). Model confidence for these controls is less reliable.

3. **Single reviewer.** All model predictions were reviewed by one cybersecurity expert. Inter-rater reliability is not measured. The 65% calibration agreement rate suggests some ambiguity in mapping decisions, particularly for AI-specific hubs.

4. **CRE taxonomy completeness.** The CRE hub hierarchy may not perfectly cover all security concepts in newer AI frameworks. Some controls may not have an ideal hub match — these appear as `model_rejected` (only {rejected} in this dataset).

5. **Temporal snapshot.** This dataset reflects the state of frameworks and CRE as of May 2026. Frameworks are updated periodically, and new CRE hubs may be added.

6. **Provenance-priority dedup.** When the same (control, hub) pair exists in multiple sources, we keep the highest-priority source. This means a ground truth link always overrides a model prediction, even if the model had additional context.

---

## Related Resources

- **[OpenCRE](https://www.opencre.org/)** — The source taxonomy and ground truth links
- **[TRACT Model](https://huggingface.co/rockCO78/tract-cre-assignment)** — The BGE-large-v1.5 bi-encoder used for predictions
- **[CRE Project](https://github.com/OWASP/OpenCRE)** — The open-source CRE project
- **[TRACT Repository](https://github.com/rockCO78/TRACT)** — Source code for the TRACT pipeline

---

## License

This dataset is licensed under **[CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/)**. You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material for any purpose, including commercially

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made
- **ShareAlike** — If you remix or build upon the material, you must distribute your contributions under the same license

---

## Citation

```bibtex
@dataset{{tract_crosswalk_2026,
  title = {{TRACT Crosswalk Dataset: Human-Reviewed Security Framework Alignment via CRE Hub Taxonomy}},
  author = {{Lambros, Rock}},
  year = {{2026}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/datasets/rockCO78/tract-crosswalk-dataset}},
  note = {{{total_rows:,} assignments mapping {n_frameworks} frameworks to 522 CRE hubs, with {total_reviewed:,} expert-reviewed model predictions}}
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


def _build_review_framework_table(per_fw: dict[str, dict]) -> str:
    """Build per-framework review breakdown table."""
    lines: list[str] = []
    for fw_id in sorted(per_fw, key=lambda k: per_fw[k].get("framework_name", k)):
        fw = per_fw[fw_id]
        name = fw.get("framework_name", fw_id)
        acc = fw.get("accepted", 0)
        reas = fw.get("reassigned", 0)
        rej = fw.get("rejected", 0)
        total = acc + reas + rej
        rate = f"{acc / total * 100:.0f}%" if total > 0 else "N/A"
        lines.append(f"| {name} | {acc} | {reas} | {rej} | {rate} |")
    return "\n".join(lines)
