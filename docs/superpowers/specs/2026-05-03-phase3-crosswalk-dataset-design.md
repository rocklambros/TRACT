# Phase 3: Published Human-Reviewed Crosswalk Dataset — Design Spec

## Goal

Produce and publish a versioned, human-reviewed crosswalk dataset mapping security controls from 31 frameworks to 522 CRE hubs. The dataset includes 4,388 OpenCRE ground-truth links (18 unresolvable), ~878 model predictions reviewed by an outsourced expert, and 46 bridge relationships — published to HuggingFace Datasets with a Zenodo DOI.

## Architecture

Four new CLI commands, three new modules. Ground truth import populates crosswalk.db with all OpenCRE-linked assignments and runs model inference on the 5 uncovered AI frameworks. Review export generates a monolithic JSON file for an outsourced expert. Review import reads back the expert's decisions and computes quality metrics. Dataset publication bundles everything into a HuggingFace Datasets release.

```
hub_links_by_framework.json ──┐
                               ├──> tract import-ground-truth ──> crosswalk.db (populated)
5 uncovered AI frameworks ────┘                                        │
                                                                       ├──> tract review-export
                                                                       │        │
                                                              review_predictions.json
                                                              reviewer_guide.md
                                                              hub_reference.json
                                                                       │
                                                            [expert review, outside CLI]
                                                                       │
                                                              review_predictions.json (reviewed)
                                                                       │
                                                              tract review-import ──> crosswalk.db (reviewed)
                                                                       │                    │
                                                              review_metrics.json           │
                                                                                            │
                                                              tract publish-dataset ──> HuggingFace Datasets
                                                                       │
                                                              zenodo_metadata.json (manual upload)
```

## Tech Stack

- SQLite (crosswalk.db, existing schema)
- `TRACTPredictor.predict_batch()` from `tract/inference.py` for inference on uncovered frameworks
- `huggingface_hub` for dataset upload (`create_repo(repo_type="dataset")` + `upload_folder`)
- `pass huggingface/token` for credentials

---

## 1. Ground Truth Import + Inference (`tract import-ground-truth`)

### 1.1 Multi-Strategy Section ID Resolver

Ground truth links in `hub_links_by_framework.json` use human-readable section IDs (e.g., `"CM-2 BASELINE CONFIGURATION"`, `"Abuse Case Cheat Sheet"`, `"V1.1.1"`). The crosswalk.db `controls` table uses machine-readable section IDs (e.g., `"nist_800_53:cm-2-baseline-configuration"`, `"owasp_cheat_sheets:abuse-case-cheat-sheet"`, `"asvs:V1.1.1"`). These formats are inconsistent across frameworks.

**Empirically verified match rates (5-round adversarial review):**
- 5-strategy resolver achieves 99.59% (4,388/4,406)
- 18 permanently unresolvable links (OWASP AI Exchange — controls not in our parsed data)

The resolver tries these strategies in order, stopping at first match:

| Strategy | Example | Frameworks it catches |
|----------|---------|----------------------|
| 1. Direct match: `gt_section_id == db_section_id` | `"AML.M0008"` = `"AML.M0008"` | mitre_atlas, owasp_llm_top10, samm |
| 2. Prefixed: `f"{fw_id}:{gt_section_id}"` == `db_section_id` | `"asvs:V1.1.1"` = `"asvs:V1.1.1"` | asvs, capec, cwe, biml, enisa, etc. |
| 3. Title exact: `gt_section_id == db_title` | `"CM-2 BASELINE CONFIGURATION"` = title | nist_800_53, owasp_cheat_sheets |
| 4. Title case-insensitive: `gt_section_id.lower() == db_title.lower()` | `"5.1.1.2"` = `"5.1.1.2"` | nist_800_63 |
| 5. Normalized: strip non-alphanumeric + lowercase both sides | `"aiprogram"` = `"aiprogram"` | owasp_ai_exchange |

For each framework, the resolver builds three lookup dictionaries at once (section_id → control_id, title → control_id, normalized → control_id) and matches each GT link through the strategy chain. Unresolvable links are logged with framework_id + section_id and skipped.

### 1.2 Ground Truth Assignment Creation

For each resolved GT link:
- `control_id`: resolved from section_id via strategy chain
- `hub_id`: the `cre_id` from the GT link (all 463 unique cre_ids are valid hub IDs — verified)
- `confidence`: `1.0` (ground truth is certain)
- `provenance`: `"opencre_ground_truth"`
- `review_status`: `"ground_truth"`
- `source_link_id`: the `link_type` from GT data (`"LinkedTo"` or `"AutomaticallyLinkedTo"`)
- `model_version`: `NULL`

**Many-to-many handling:** One control can map to multiple hubs (CAPEC has up to 23 hubs per control). Each link becomes one assignment row. This is correct — the assignments table already supports this.

**Deduplication:** Before inserting, check if `(control_id, hub_id)` already exists in assignments (from either `active_learning_round_2` or `ground_truth_T1-AI`). If so, skip the insert and log it as a "model-GT agreement" data point. The existing `ground_truth_T1-AI` provenance (78 assignments for mitre_atlas + owasp_llm_top10) will overlap with the incoming GT import. These 78 should be skipped since they're already present.

**Existing duplicate handling:** 10 duplicate `(control_id, hub_id)` pairs already exist in the DB. The import must tolerate these gracefully — check before insert, never fail on duplicates.

### 1.3 Inference on Uncovered Frameworks

Five AI frameworks have controls in crosswalk.db but zero assignments:

| Framework | Controls | full_text available | Description quality |
|-----------|----------|--------------------|--------------------|
| AIUC-1 | 132 | 0/132 (all empty) | Good (139-193 chars) |
| CoSAI | 55 | 26/55 | Good (95-273 chars) |
| EU GPAI CoP | 40 | 12/40 | Strong (691-2575 chars) |
| NIST AI RMF | 72 | 3/72 | Mixed (13 controls <50 chars) |
| OWASP DSGAI | 21 | 21/21 | Strong (1992-11725 chars) |

**Text preparation** (matches `tract ingest` pipeline exactly):
```python
parts = [ctrl.title, ctrl.description]
if ctrl.full_text:
    parts.append(ctrl.full_text)
text = " ".join(p for p in parts if p)
```

**Inference:** Load `TRACTPredictor`, call `predict_batch()` with all control texts for each framework. Insert top-1 prediction per control as assignment:
- `provenance`: `"model_prediction"`
- `review_status`: `"pending"`
- `confidence`: `raw_similarity` from prediction
- `calibrated_confidence` is NOT stored in DB (computed at export time from the stored confidence + calibration parameters)
- `is_ood`: stored in DB
- `model_version`: git SHA of current deployment model

**Text quality warning:** NIST AI RMF has 13/72 controls with descriptions under 50 characters. These will produce unreliable predictions. The review export flags these with `text_quality: "low"` so the reviewer knows to scrutinize them.

### 1.4 CLI Interface

```
tract import-ground-truth [--dry-run]
```

- Reads `data/training/hub_links_by_framework.json`
- Resolves section IDs using multi-strategy resolver
- Imports ground truth assignments (skipping duplicates)
- Runs inference on uncovered frameworks
- Reports: imported count, skipped duplicates, unresolvable links, inference results
- `--dry-run`: report what would happen without modifying DB

**New file:** `tract/crosswalk/ground_truth.py`

---

## 2. Review Export (`tract review-export`)

### 2.1 Scope: ALL Model Predictions

The review JSON includes ALL assignments with `provenance IN ("active_learning_round_2", "model_prediction")` — regardless of current `review_status`. The 558 AL-accepted predictions are re-reviewed by the expert. This is the entire point of Phase 3: independent human validation supersedes ML pipeline acceptance.

Additionally, ~20 calibration items from `provenance="ground_truth_T1-AI"` are included, disguised as regular predictions (see Section 2.3).

### 2.2 Review JSON Structure

One monolithic file. Predictions sorted by framework (alphabetical), then by `calibrated_confidence` descending within each framework.

```json
{
  "metadata": {
    "generated_at": "2026-05-03T12:00:00Z",
    "total_predictions": 898,
    "calibration_items": 20,
    "frameworks": {
      "CSA AI Controls Matrix": 243,
      "MITRE ATLAS": 260
    },
    "calibration": {
      "temperature": 0.074,
      "ece": 0.079,
      "ood_threshold": 0.568
    },
    "model_version": "abc123",
    "instructions": "See reviewer_guide.md for detailed review instructions."
  },
  "predictions": [
    {
      "id": 42,
      "framework": "CSA AI Controls Matrix",
      "framework_id": "csa_aicm",
      "control_id": "csa_aicm:csa_aicm:AICM-01-01",
      "control_title": "AI Governance Policy",
      "control_text": "Establish and maintain an AI governance policy...",
      "text_quality": "high",
      "predicted_hub_id": "646-285",
      "predicted_hub_name": "AI governance and management",
      "predicted_hub_path": "Root > Governance > AI governance and management",
      "confidence": 0.87,
      "calibrated_confidence": 0.82,
      "is_ood": false,
      "review_priority": "routine",
      "alternative_hubs": [
        {"hub_id": "220-442", "hub_name": "Risk management", "hub_path": "Root > Governance > Risk management", "confidence": 0.12},
        {"hub_id": "818-434", "hub_name": "Compliance monitoring", "hub_path": "Root > Governance > Compliance monitoring", "confidence": 0.05}
      ],
      "status": "pending",
      "corrected_hub_id": null,
      "reviewer_notes": ""
    }
  ]
}
```

**Field specifications:**

| Field | Source | Purpose |
|-------|--------|---------|
| `id` | `assignments.id` (integer PK) | Stable reference for import back |
| `text_quality` | Computed from text length | `"high"` (>200 chars), `"medium"` (100-200), `"low"` (<100) |
| `review_priority` | Computed from confidence + OOD + text_quality | `"routine"` (high conf, not OOD), `"careful"` (low conf or OOD), `"critical"` (low conf + low text quality) |
| `calibrated_confidence` | `calibrate_similarities(confidence, T)` | Computed at export time, not stored in DB |
| `alternative_hubs` | Top-2 next-best predictions | Re-computed at export time via `predict()` |

### 2.3 Calibration Items

Include ~20 items from `ground_truth_T1-AI` disguised as regular predictions. These have known correct hub assignments. Selection:
- Draw proportionally from mitre_atlas (~15) and owasp_llm_top10 (~5)
- Include a mix of "easy" (high similarity to correct hub) and "hard" (lower similarity) items
- Assign them synthetic `id` values in a reserved range (e.g., negative IDs or IDs > 100000) so the import can identify and skip them
- Set `status: "pending"` like all other predictions

At import time, compare reviewer decisions on calibration items to known ground truth. Report as "reviewer quality score" in metrics. If the reviewer rejects or reassigns known-correct mappings, flag for investigation.

### 2.4 Hub Reference Document

Generated alongside the review JSON as `hub_reference.json`:
```json
[
  {
    "hub_id": "044-202",
    "name": "Prompt injection I/O handling",
    "path": "Root > Technical > Input handling > Prompt injection I/O handling",
    "parent_id": "843-475",
    "is_leaf": true
  }
]
```
Sorted by hierarchy path (alphabetical). The reviewer guide instructs: "If none of the 3 suggested hubs fit, search this file by keyword to find the right hub_id for reassignment."

### 2.5 Reviewer Guide Document

Generated as `reviewer_guide.md`. Contents:

**Role & Persona:**
> You are a cybersecurity domain expert reviewing AI-generated mappings between security framework controls and the Common Requirement Enumeration (CRE) hub taxonomy. Your review determines which mappings are published as a peer-reviewed research dataset.

**Background section:** What CRE is, what hubs are (522 security topics organized in a hierarchy), what TRACT does (maps controls from 31 frameworks to CRE hubs), what "assignment" means.

**Step-by-step process:**
1. Open `review_predictions.json` in your editor
2. For each prediction, read `control_text` to understand the control's security intent
3. Read `predicted_hub_name` and `predicted_hub_path` to understand the model's suggestion
4. Check `calibrated_confidence` — above 0.70 is fairly certain, below 0.30 needs careful attention
5. Check `review_priority` — "critical" items need the most care
6. **Accept:** Control belongs under this hub → `"status": "accepted"`
7. **Reassign:** Wrong hub, but a better one exists → `"status": "reassigned"`, set `"corrected_hub_id"` (find IDs in `hub_reference.json`)
8. **Reject:** No hub fits → `"status": "rejected"`, explain in `"reviewer_notes"`
9. Add `reviewer_notes` for any non-obvious decision

**Decision criteria:** The control's security PURPOSE should align with the hub's security DOMAIN. Not keyword matching — a control about "encrypting AI training data" maps to encryption, not AI training.

**Common pitfalls:**
- MITRE ATLAS hubs are the hardest — many sound similar. Read the full path.
- "Rejected" means no hub fits at all. Check `alternative_hubs` and `hub_reference.json` before rejecting.
- High `is_ood: true` means the model thinks this control is outside its training distribution — review more carefully.
- NIST AI RMF predictions are based on short control descriptions and may be less reliable.
- Items flagged `text_quality: "low"` had sparse input text — predictions may be unreliable.

**Saving progress:** Work in batches. Leave unreviewed items as `"status": "pending"`. Partial files can be imported.

**Time estimate:** ~900 predictions. Routine items (~60%) take ~1 min each. Careful/critical items (~40%) take 3-5 min. Estimated total: 25-40 hours.

### 2.6 CLI Interface

```
tract review-export --output results/review/review_predictions.json
```

- Queries all model predictions from crosswalk.db
- Inserts calibration items
- Generates review JSON, reviewer guide, and hub reference
- Reports: total predictions, per-framework counts, calibration item count

**New files:** `tract/review/__init__.py`, `tract/review/export.py`, `tract/review/guide.py`

---

## 3. Review Import (`tract review-import`)

### 3.1 Validation

Before any DB writes, validate:
- JSON structure matches schema (metadata + predictions array)
- Every non-pending prediction has valid `status` (accepted/reassigned/rejected)
- Every `corrected_hub_id` (on reassigned items) is a valid hub ID in DB
- Every `id` matches an existing assignment in DB (skip calibration items with synthetic IDs)
- Warn (don't fail) if some predictions are still `"pending"` — partial review is fine

### 3.2 DB Updates (Single Transaction)

All updates in one transaction — atomic rollback on any failure.

| Status | DB Update |
|--------|-----------|
| `accepted` | `review_status="accepted"`, `reviewer=<name>`, `review_date=now` |
| `reassigned` | `hub_id=corrected_hub_id`, `confidence=NULL`, `review_status="accepted"`, `reviewer=<name>`, `review_date=now`, `reviewer_notes` includes original hub_id and confidence |
| `rejected` | `review_status="rejected"`, `reviewer=<name>`, `review_date=now` |
| `pending` | Skip — no update |

**Idempotent:** Re-importing the same file produces the same DB state. Updates by assignment `id`, not inserts.

**Reassignment confidence handling:** When a reviewer reassigns to a different hub, the stored confidence was for the ORIGINAL hub. Set `confidence=NULL` and append to `reviewer_notes`: `"[Reassigned from hub {original_hub_id} (confidence={original_conf:.3f})]"`.

### 3.3 Review Metrics

Computed at import time and saved to `results/review/review_metrics.json`:

```json
{
  "generated_at": "2026-05-15T...",
  "import_round": 1,
  "coverage": {
    "total_predictions": 896,
    "reviewed": 896,
    "pending": 0,
    "completion_pct": 100.0
  },
  "overall": {
    "accepted": 780,
    "rejected": 42,
    "reassigned": 74,
    "acceptance_rate": 0.87,
    "rejection_rate": 0.047,
    "reassignment_rate": 0.083
  },
  "per_framework": {
    "CSA AI Controls Matrix": {"accepted": 220, "rejected": 8, "reassigned": 15, "total": 243},
    "MITRE ATLAS": {"accepted": 195, "rejected": 25, "reassigned": 40, "total": 260}
  },
  "reviewer_quality": {
    "calibration_items_total": 20,
    "calibration_agreed": 18,
    "calibration_disagreed": 2,
    "quality_score": 0.90,
    "disagreements": [
      {"id": -1, "expected_hub": "547-824", "reviewer_decision": "reassigned", "corrected_hub": "XXX"}
    ]
  },
  "confidence_analysis": {
    "high_conf_acceptance_rate": 0.95,
    "low_conf_acceptance_rate": 0.62,
    "ood_acceptance_rate": 0.45
  }
}
```

**Partial review caveat:** Track `import_round` number. Metrics include `completion_pct`. Final metrics are only authoritative at 100% completion.

### 3.4 CLI Interface

```
tract review-import --input results/review/review_predictions.json --reviewer "expert_1"
```

- Validates JSON structure and hub IDs
- Updates assignments in single transaction
- Computes and saves review metrics
- Reports: accepted/rejected/reassigned counts, reviewer quality score

**New files:** `tract/review/import_review.py`, `tract/review/metrics.py`

---

## 4. Dataset Publication (`tract publish-dataset`)

### 4.1 Published Dataset Structure

```
tract-crosswalk-dataset/
  crosswalk_v1.0.jsonl
  framework_metadata.json
  cre_hierarchy_v1.1.json
  hub_descriptions_v1.0.json
  bridge_report.json
  review_metrics.json
  README.md
  LICENSE
  zenodo_metadata.json
```

### 4.2 crosswalk_v1.0.jsonl

One JSON line per assignment. Includes ALL assignments (ground truth + accepted model predictions + rejected). Rejected predictions are valuable — they show where the model fails.

```json
{
  "control_id": "csa_aicm:csa_aicm:AICM-01-01",
  "framework": "CSA AI Controls Matrix",
  "framework_id": "csa_aicm",
  "section_id": "AICM-01-01",
  "control_title": "AI Governance Policy",
  "hub_id": "646-285",
  "hub_name": "AI governance and management",
  "hub_path": "Root > Governance > AI governance and management",
  "confidence": 0.87,
  "is_ood": false,
  "assignment_type": "model_accepted",
  "reviewer": "expert_1",
  "review_date": "2026-05-15"
}
```

**`assignment_type` values** (provenance + review outcome combined):

| Value | Meaning | Source |
|-------|---------|--------|
| `ground_truth_linked` | Expert-curated LinkedTo from OpenCRE | `source_link_id="LinkedTo"` |
| `ground_truth_auto` | Transitive chain AutomaticallyLinkedTo from OpenCRE | `source_link_id="AutomaticallyLinkedTo"` |
| `model_accepted` | Model prediction accepted by expert | `review_status="accepted"`, not reassigned |
| `model_reassigned` | Model prediction corrected by expert | `review_status="accepted"`, hub_id differs from original |
| `model_rejected` | Model prediction rejected by expert | `review_status="rejected"` |

### 4.3 framework_metadata.json

```json
[
  {
    "framework_id": "csa_aicm",
    "name": "CSA AI Controls Matrix",
    "version": "1.0",
    "source": "https://cloudsecurityalliance.org/research/artifacts/ai-controls-matrix",
    "control_count": 243,
    "assignment_count": 243,
    "coverage_type": "model_reviewed",
    "ground_truth_links": 0,
    "model_predictions": 243,
    "accepted": 220,
    "rejected": 8,
    "reassigned": 15
  }
]
```

**`coverage_type`:**
- `ground_truth` — only OpenCRE ground truth links (traditional frameworks)
- `model_reviewed` — model predictions reviewed by expert (AI frameworks)
- `both` — has both GT links and reviewed predictions (mitre_atlas, owasp_llm_top10)
- `no_assignments` — registered but no coverage (if any remain)

### 4.4 Dataset Card (README.md)

HuggingFace Datasets format. Novice-friendly, matching the model card quality. Sections:

1. **What Is This?** — Plain English: a mapping of 31 security frameworks to a common taxonomy
2. **Quick Start** — `datasets.load_dataset("rockCO78/tract-crosswalk-dataset")` with usage examples
3. **Dataset Structure** — field descriptions for each file
4. **Framework Coverage Table** — all 31 frameworks with control counts and coverage types
5. **How It Was Made** — model training (LOFO, bi-encoder), active learning, expert review, bridge analysis
6. **Review Methodology** — who reviewed, criteria, quality score, acceptance rates
7. **Limitations** — ATLAS hub disambiguation, short-text predictions, coverage gaps
8. **License** — CC-BY-SA-4.0 (pending OpenCRE license verification)
9. **Citation** — BibTeX

### 4.5 License

**Default: CC-BY-SA-4.0.** Ground truth data originates from OpenCRE. OpenCRE's code is Apache 2.0, but the DATA (CRE taxonomy + framework links) may carry ShareAlike requirements. Use CC-BY-SA-4.0 unless OpenCRE explicitly confirms CC-BY-4.0 is sufficient.

**Verification step (before publication):** Check OpenCRE's data license at opencre.org. If confirmed as CC-BY-4.0 compatible, switch to CC-BY-4.0. Document the decision in the dataset card.

### 4.6 Zenodo Metadata

Auto-generated `zenodo_metadata.json` for manual upload:

```json
{
  "title": "TRACT: Security Framework Crosswalk Dataset v1.0",
  "description": "Human-reviewed mappings of 31 security frameworks to 522 CRE hubs...",
  "creators": [{"name": "Lambros, Rock", "orcid": "..."}],
  "keywords": ["security", "crosswalk", "CRE", "AI security", "framework mapping"],
  "license": "cc-by-sa-4.0",
  "related_identifiers": [
    {"identifier": "https://huggingface.co/rockCO78/tract-cre-assignment", "relation": "isSupplementTo"},
    {"identifier": "https://huggingface.co/datasets/rockCO78/tract-crosswalk-dataset", "relation": "isIdenticalTo"}
  ],
  "upload_type": "dataset",
  "access_right": "open"
}
```

Manual upload via Zenodo web UI. No API client built — one-time operation.

### 4.7 CLI Interface

```
tract publish-dataset --repo-id rockCO78/tract-crosswalk-dataset [--dry-run] [--skip-upload]
```

- Validates all assignments reviewed (warns if pending remain)
- Assembles staging directory
- Generates dataset card and zenodo metadata
- Uploads to HuggingFace Datasets (unless --dry-run or --skip-upload)
- Token via `pass huggingface/token`

**New files:** `tract/dataset/__init__.py`, `tract/dataset/bundle.py`, `tract/dataset/card.py`, `tract/dataset/publish.py`

---

## 5. Module and File Structure

### New Modules

| File | Responsibility |
|------|---------------|
| `tract/crosswalk/ground_truth.py` | Multi-strategy resolver + GT import + inference orchestration |
| `tract/review/__init__.py` | Review workflow orchestrator |
| `tract/review/export.py` | Generate review JSON with calibration items |
| `tract/review/import_review.py` | Parse reviewed JSON, validate, update DB |
| `tract/review/metrics.py` | Compute acceptance/rejection/quality metrics |
| `tract/review/guide.py` | Generate reviewer guide markdown + hub reference JSON |
| `tract/dataset/__init__.py` | Dataset publication orchestrator |
| `tract/dataset/bundle.py` | Assemble staging directory with all files |
| `tract/dataset/card.py` | Generate HuggingFace Datasets card |
| `tract/dataset/publish.py` | Upload to HuggingFace Hub |

### CLI Additions (tract/cli.py)

| Command | Handler |
|---------|---------|
| `tract import-ground-truth` | `_cmd_import_ground_truth` |
| `tract review-export` | `_cmd_review_export` |
| `tract review-import` | `_cmd_review_import` |
| `tract publish-dataset` | `_cmd_publish_dataset` |

### Tests

| Test File | Covers |
|-----------|--------|
| `tests/test_ground_truth_import.py` | Resolver strategies, dedup, GT insert, inference trigger |
| `tests/test_review_export.py` | JSON structure, calibration items, text quality, priority |
| `tests/test_review_import.py` | Validation, DB updates, reassignment, idempotency |
| `tests/test_review_metrics.py` | Metric computation, partial review handling |
| `tests/test_dataset_bundle.py` | JSONL format, assignment_type, framework metadata |
| `tests/test_dataset_card.py` | Card generation, field completeness |

---

## 6. Adversarial Review Findings (Incorporated)

All findings from the 5-round adversarial review are incorporated into this spec:

### MUST CHANGE (correctness)

| # | Finding | Resolution | Spec Section |
|---|---------|------------|-------------|
| 1 | GT import needs multi-strategy section_id resolver | 5-strategy resolver with 99.59% match rate | 1.1 |
| 2 | AL-accepted predictions must be re-reviewed | Review export includes ALL model predictions | 2.1 |
| 3 | No reviewer quality assessment | ~20 calibration items from ground_truth_T1-AI | 2.3 |
| 4 | License may be CC-BY-SA not CC-BY | Default CC-BY-SA-4.0, verify before publish | 4.5 |

### SHOULD CHANGE (quality)

| # | Finding | Resolution | Spec Section |
|---|---------|------------|-------------|
| 5 | Reviewer can't find hub IDs for reassignment | hub_reference.json generated alongside review | 2.4 |
| 6 | No text quality signal | `text_quality` field (high/medium/low) | 2.2 |
| 7 | Reassignment breaks confidence semantics | Set confidence=NULL, preserve original in notes | 3.2 |
| 8 | JSONL missing provenance granularity | `assignment_type` field (5 values) | 4.2 |
| 9 | No triage signal for reviewer | `review_priority` field (routine/careful/critical) | 2.2 |

### NICE TO HAVE (clarity)

| # | Finding | Resolution | Spec Section |
|---|---------|------------|-------------|
| 10 | 10 existing duplicate pairs in DB | Import checks before insert, skips duplicates | 1.2 |
| 11 | Partial review metrics misleading | Track import_round, caveat at <100% completion | 3.3 |
| 12 | NIST AI RMF has 13 controls with <50 char text | Warning in reviewer guide | 2.5 |

---

## 7. Data Inventory

### Ground Truth Import Target

| Framework | GT Links | Match Strategy | Expected Match |
|-----------|---------|---------------|---------------|
| CAPEC | 1,799 | Prefixed | 100% |
| CWE | 613 | Prefixed | 100% |
| OWASP Cheat Sheets | 391 | Title | 100% |
| NIST 800-53 | 300 | Title | 100% |
| ASVS | 277 | Prefixed | 100% |
| DSOMM | 214 | Prefixed | 100% |
| WSTG | 118 | Prefixed + title | 97% |
| ISO 27001 | 94 | Title | 100% |
| NIST 800-63 | 79 | Title | 100% |
| OWASP Proactive Controls | 76 | Prefixed | 100% |
| ENISA | 68 | Prefixed | 85% |
| MITRE ATLAS | 65 | Direct | 100% |
| OWASP AI Exchange | 65 | Normalized | 5% (18 unresolvable) |
| NIST SSDF | 46 | Prefixed | 96% |
| NIST AI 100-2 | 45 | Prefixed | 100% |
| ETSI | 36 | Prefixed + title | 97% |
| SAMM | 30 | Direct | 100% |
| CSA CCM | 29 | Prefixed | 100% |
| BIML | 21 | Prefixed | 100% |
| OWASP Top 10 2021 | 17 | Prefixed | 100% |
| OWASP LLM Top 10 | 13 | Direct | 100% |
| OWASP ML Top 10 | 10 | Prefixed | 100% |
| **Total** | **4,406** | | **4,388 (99.59%)** |

### Inference Target

| Framework | Controls | Text Source | Estimated Quality |
|-----------|----------|------------|------------------|
| AIUC-1 | 132 | Description only | Medium |
| NIST AI RMF | 72 | Description only (13 <50 chars) | Low-Medium |
| CoSAI | 55 | Description + partial full_text | Medium |
| EU GPAI CoP | 40 | Description + partial full_text | Medium-High |
| OWASP DSGAI | 21 | Full text available | High |
| **Total** | **320** | | |

### Review Target

| Source | Count | Review Status |
|--------|-------|--------------|
| AL round 2 (accepted) | 558 | Re-review by expert |
| AL round 2 (inference on new frameworks) | ~320 | New expert review |
| Ground truth T1-AI (calibration items) | ~20 | Disguised quality check |
| **Total review items** | **~898** | |

### Published Dataset Target

| Source | Assignments | Assignment Types |
|--------|------------|-----------------|
| OpenCRE ground truth | ~4,388 | ground_truth_linked, ground_truth_auto |
| Model reviewed (accepted) | ~780 | model_accepted |
| Model reviewed (reassigned) | ~74 | model_reassigned |
| Model reviewed (rejected) | ~42 | model_rejected |
| **Total** | **~5,284** | |

---

## 8. Execution Order

### Phase 3A: Engineering (this session + next)
1. `tract import-ground-truth` — populate crosswalk.db
2. `tract review-export` — generate review package
3. `tract review-import` — ready for post-review
4. `tract publish-dataset` — ready for post-review

### Phase 3B: Expert Review (outside Claude Code, ~25-40 hours)
5. Expert reviews `review_predictions.json`
6. Import reviewed JSON via `tract review-import`
7. Verify review metrics and quality score

### Phase 3C: Publication (after review)
8. Verify OpenCRE license
9. Run `tract publish-dataset`
10. Manual Zenodo upload for DOI
11. Update PRD.md and docs/session-prompts.md

---

## 9. What We Are NOT Building

- Interactive CLI review interface
- Web UI for review
- Zenodo API client (manual upload with auto-generated metadata)
- Multi-reviewer support or conflict resolution
- Cohen's kappa (single reviewer)
- Automatic re-training from review results
- Conformal prediction sets in review JSON
