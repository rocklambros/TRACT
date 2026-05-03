# Phase 3: Published Human-Reviewed Crosswalk Dataset — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and publish a versioned, human-reviewed crosswalk dataset mapping 31 security frameworks to 522 CRE hubs. Import 4,388 OpenCRE ground-truth links, run inference on 5 uncovered AI frameworks (~320 controls), export ~868 model predictions for expert review (+ 20 calibration items), import review decisions, and bundle everything into a HuggingFace Datasets release with Zenodo DOI.

**Architecture:** Five CLI commands (`import-ground-truth`, `review-export`, `review-validate`, `review-import`, `publish-dataset`), three new modules (`tract/crosswalk/ground_truth.py`, `tract/review/`, `tract/dataset/`). Ground truth import uses a 5-strategy section ID resolver (99.59% match rate). Review export re-runs inference to avoid double-calibration. Review import uses UPDATE-in-place (not the existing INSERT-new-row pattern). Dataset publication deduplicates by provenance priority.

**Tech Stack:** Python 3.12, SQLite, sentence-transformers, huggingface_hub, pytest

**Spec:** `docs/superpowers/specs/2026-05-03-phase3-crosswalk-dataset-design.md`

---

## File Structure

### New Files

```
tract/crosswalk/ground_truth.py        — Multi-strategy section ID resolver + GT import + inference on uncovered frameworks
tract/review/__init__.py               — Review workflow package init
tract/review/export.py                 — Generate review JSON with calibration items + re-inference
tract/review/import_review.py          — Parse reviewed JSON, validate, apply_review_decisions() (UPDATE-in-place)
tract/review/metrics.py                — Compute acceptance/rejection/calibration quality metrics
tract/review/guide.py                  — Generate reviewer_guide.md + hub_reference.json
tract/review/validate.py               — Standalone JSON validation (shared with import)
tract/dataset/__init__.py              — Dataset publication package init
tract/dataset/bundle.py                — Assemble staging directory: JSONL, metadata, card, license
tract/dataset/card.py                  — Generate HuggingFace Datasets card (README.md)
tract/dataset/publish.py               — Upload to HuggingFace Hub (repo_type="dataset")

tests/test_ground_truth_import.py      — Resolver strategies, dedup, GT insert, inference trigger
tests/test_review_export.py            — JSON structure, calibration items, text_quality, review_priority, re-inference
tests/test_review_validate.py          — JSON validation edge cases
tests/test_review_import.py            — DB updates, reassignment via original_hub_id, idempotency, schema migration
tests/test_review_metrics.py           — Metric computation, partial review handling, calibration quality
tests/test_dataset_bundle.py           — JSONL format, assignment_type derivation, (control_id,hub_id) dedup
tests/test_dataset_card.py             — Card generation, field completeness
tests/test_dataset_publish.py          — HuggingFace upload (mocked)
tests/test_phase3_integration.py       — End-to-end: GT import → export → import → bundle

tests/fixtures/phase3_mini_hub_links.json        — Synthetic hub links for resolver tests
tests/fixtures/phase3_review_predictions.json    — Sample review JSON for import tests
```

### Modified Files

```
tract/config.py:321+                   — Add PHASE3_* constants
tract/crosswalk/schema.py:40-55        — Add reviewer_notes + original_hub_id columns to SCHEMA_SQL
tract/crosswalk/schema.py:72+          — Add migrate_schema() for ALTER TABLE on existing DBs
tract/cli.py:27-32,239+                — Add 5 subcommands: import-ground-truth, review-export, review-validate, review-import, publish-dataset
```

### Existing Files Referenced (read-only)

```
tract/inference.py                     — TRACTPredictor, HubPrediction, predict_batch()
tract/calibration/temperature.py       — calibrate_similarities(sims, temperature) → softmax output
tract/crosswalk/store.py               — insert_assignments(), get_assignments_by_provenance() (NOT update_review_status)
tract/crosswalk/schema.py              — get_connection(), SCHEMA_SQL, create_database()
tract/io.py                            — atomic_write_json(), load_json()
tract/sanitize.py                      — sanitize_text()
tract/hierarchy.py                     — CREHierarchy.load()
data/training/hub_links_by_framework.json    — 4,406 GT links (22 frameworks)
results/phase1c/crosswalk.db                 — Existing DB: 636 assignments, 31 frameworks, 1,084 controls
results/phase1c/deployment_model/            — SentenceTransformer + PEFT adapter + calibration.json + deployment_artifacts.npz
```

---

### Task 0: Config Constants + Schema Migration

**Files:**
- Modify: `tract/config.py:321+`
- Modify: `tract/crosswalk/schema.py:40-55,72+`
- Create: `tests/test_phase3_schema_migration.py` (temporary, folded into test_review_import.py later)

> **Why:** All Phase 3 modules depend on shared constants. The schema migration must run before review-import but is needed early for tests. The schema.py update ensures new databases get both columns, while migrate_schema() adds them idempotently to existing databases.

- [ ] **Step 1: Add Phase 3 constants to config.py**

Append after line 320 (after `PHASE1C_ECE_GATE_PATH`):

```python
# ── Phase 3: Crosswalk Dataset Publication ────────────────────────────

PHASE3_REVIEW_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "results" / "review"
PHASE3_DATASET_STAGING_DIR: Final[Path] = PROJECT_ROOT / "build" / "dataset"
PHASE3_DATASET_REPO_ID: Final[str] = "rockCO78/tract-crosswalk-dataset"

PHASE3_CALIBRATION_SEED: Final[int] = 42
PHASE3_CALIBRATION_N_ITEMS: Final[int] = 20
PHASE3_CALIBRATION_EASY_N: Final[int] = 5
PHASE3_CALIBRATION_HARD_N: Final[int] = 5

PHASE3_TEXT_QUALITY_HIGH_THRESHOLD: Final[int] = 500
PHASE3_TEXT_QUALITY_LOW_THRESHOLD: Final[int] = 100

PHASE3_UNCOVERED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "aiuc_1", "cosai", "eu_gpai_cop", "nist_ai_rmf", "owasp_dsgai",
})

PHASE3_GT_PROVENANCE: Final[str] = "opencre_ground_truth"
PHASE3_MODEL_PROVENANCE: Final[str] = "model_prediction"

PHASE3_PROVENANCE_PRIORITY: Final[list[str]] = [
    "opencre_ground_truth",
    "ground_truth_T1-AI",
    "active_learning_round_2",
    "model_prediction",
]
```

Also add this import at the top of config.py (already present: `from typing import Final`):

```python
PHASE3_REVIEW_OUTPUT_DIR  # verify it resolves with PROJECT_ROOT
```

- [ ] **Step 2: Update SCHEMA_SQL in schema.py to include new columns**

In `tract/crosswalk/schema.py`, update the assignments CREATE TABLE to add two columns after `review_date TEXT,`:

```sql
reviewer_notes TEXT,
original_hub_id TEXT,
```

The full assignments CREATE TABLE becomes:

```sql
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    control_id TEXT NOT NULL REFERENCES controls(id),
    hub_id TEXT NOT NULL REFERENCES hubs(id),
    confidence REAL,
    in_conformal_set INTEGER,
    is_ood INTEGER DEFAULT 0,
    provenance TEXT NOT NULL,
    source_link_id TEXT,
    model_version TEXT,
    review_status TEXT DEFAULT 'pending',
    reviewer TEXT,
    review_date TEXT,
    reviewer_notes TEXT,
    original_hub_id TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
```

- [ ] **Step 3: Add migrate_schema() function to schema.py**

After `get_connection()`, add:

```python
def migrate_schema(db_path: Path) -> list[str]:
    """Apply schema migrations idempotently. Returns list of migrations applied."""
    conn = get_connection(db_path)
    applied: list[str] = []
    try:
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(assignments)").fetchall()
        }
        if "reviewer_notes" not in existing:
            conn.execute("ALTER TABLE assignments ADD COLUMN reviewer_notes TEXT")
            applied.append("reviewer_notes")
            logger.info("Added reviewer_notes column to assignments")
        if "original_hub_id" not in existing:
            conn.execute("ALTER TABLE assignments ADD COLUMN original_hub_id TEXT")
            applied.append("original_hub_id")
            logger.info("Added original_hub_id column to assignments")
        conn.commit()
    finally:
        conn.close()
    return applied
```

- [ ] **Step 4: Write tests for schema migration**

Create `tests/test_phase3_schema_migration.py`:

Tests should verify:
1. `migrate_schema()` on a fresh DB (created with updated SCHEMA_SQL) returns empty list (no-op)
2. `migrate_schema()` on an old DB (without the columns) adds both columns and returns their names
3. `migrate_schema()` is idempotent — running twice returns empty list on second call
4. Existing assignments data is preserved after migration

Use `tmp_path` fixture to create temporary DBs.

- [ ] **Step 5: Verify**

```bash
python -c "from tract.config import PHASE3_REVIEW_OUTPUT_DIR, PHASE3_CALIBRATION_SEED; print('Constants loaded:', PHASE3_REVIEW_OUTPUT_DIR)"
python -m pytest tests/test_phase3_schema_migration.py -v
```

- [ ] **Step 6: Commit**

```bash
git add tract/config.py tract/crosswalk/schema.py tests/test_phase3_schema_migration.py
git commit -m "feat(phase3): add config constants and schema migration for reviewer_notes + original_hub_id columns"
```

---

### Task 1: Multi-Strategy Section ID Resolver

**Files:**
- Create: `tract/crosswalk/ground_truth.py` (resolver portion only)
- Create: `tests/test_ground_truth_import.py` (resolver tests only)
- Create: `tests/fixtures/phase3_mini_hub_links.json`

> **Why:** The resolver is the most complex pure-logic component — 5 strategies, 22 frameworks, 4,406 links. Building and testing it independently before the import logic reduces debugging surface.

> **Critical context:** Ground truth links use `standard_name` (e.g., "NIST 800-53") and `section_id` (e.g., "CM-2 BASELINE CONFIGURATION"). The crosswalk.db `controls` table has `id` (e.g., "nist_800_53:cm-2-baseline-configuration"), `section_id` (e.g., "CM-2 BASELINE CONFIGURATION"), and `title`. The resolver maps GT `section_id` → DB `controls.id` using multiple strategies because the format is inconsistent across frameworks.

- [ ] **Step 1: Create test fixture**

Create `tests/fixtures/phase3_mini_hub_links.json` with synthetic hub links covering all 5 resolver strategies:

```json
{
  "mitre_atlas": [
    {"cre_id": "364-516", "cre_name": "Adversarial machine learning", "framework_id": "mitre_atlas", "link_type": "LinkedTo", "section_id": "AML.M0008", "section_name": "AML.M0008", "standard_name": "MITRE ATLAS"}
  ],
  "asvs": [
    {"cre_id": "170-772", "cre_name": "Session management", "framework_id": "asvs", "link_type": "LinkedTo", "section_id": "V1.1.1", "section_name": "V1.1.1", "standard_name": "ASVS"}
  ],
  "nist_800_53": [
    {"cre_id": "555-001", "cre_name": "Config management", "framework_id": "nist_800_53", "link_type": "AutomaticallyLinkedTo", "section_id": "CM-2 BASELINE CONFIGURATION", "section_name": "CM-2 BASELINE CONFIGURATION", "standard_name": "NIST 800-53"}
  ],
  "nist_800_63": [
    {"cre_id": "666-001", "cre_name": "Authentication", "framework_id": "nist_800_63", "link_type": "LinkedTo", "section_id": "5.1.1.2", "section_name": "5.1.1.2", "standard_name": "NIST 800-63"}
  ],
  "owasp_ai_exchange": [
    {"cre_id": "777-001", "cre_name": "AI security", "framework_id": "owasp_ai_exchange", "link_type": "LinkedTo", "section_id": "aiprogram", "section_name": "AI Program", "standard_name": "OWASP AI Exchange"}
  ]
}
```

- [ ] **Step 2: Write resolver in ground_truth.py**

Create `tract/crosswalk/ground_truth.py` with the resolver:

```python
"""Ground truth import — multi-strategy section ID resolver and assignment creation."""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ResolverResult:
    resolved: dict[str, str]           # gt_key → control_id
    unresolved: list[dict]             # list of unresolvable GT links
    strategy_counts: dict[str, int]    # strategy_name → count


def build_control_lookups(
    conn: sqlite3.Connection,
    framework_id: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build three lookup dictionaries for a framework's controls.

    Returns (section_id_map, title_map, normalized_map) where each maps
    the lookup key → control_id.
    """
    rows = conn.execute(
        "SELECT id, section_id, title FROM controls WHERE framework_id = ?",
        (framework_id,),
    ).fetchall()

    section_id_map: dict[str, str] = {}
    title_map: dict[str, str] = {}
    normalized_map: dict[str, str] = {}

    for row in rows:
        ctrl_id = row["id"]
        sid = row["section_id"]
        title = row["title"] or ""

        section_id_map[sid] = ctrl_id
        if title:
            title_map[title] = ctrl_id
            title_map[title.lower()] = ctrl_id  # case-insensitive
        normalized_key = re.sub(r"[^a-z0-9]", "", sid.lower())
        if normalized_key:
            normalized_map[normalized_key] = ctrl_id

    return section_id_map, title_map, normalized_map


def resolve_section_id(
    gt_section_id: str,
    framework_id: str,
    section_id_map: dict[str, str],
    title_map: dict[str, str],
    normalized_map: dict[str, str],
) -> tuple[str | None, str]:
    """Try 5 strategies to resolve a GT section_id to a control_id.

    Returns (control_id, strategy_name) or (None, "unresolved").
    """
    # Strategy 1: Direct match
    if gt_section_id in section_id_map:
        return section_id_map[gt_section_id], "direct"

    # Strategy 2: Prefixed match
    prefixed = f"{framework_id}:{gt_section_id}"
    if prefixed in section_id_map:
        return section_id_map[prefixed], "prefixed"

    # Strategy 3: Title exact match
    if gt_section_id in title_map:
        return title_map[gt_section_id], "title_exact"

    # Strategy 4: Title case-insensitive
    if gt_section_id.lower() in title_map:
        return title_map[gt_section_id.lower()], "title_case_insensitive"

    # Strategy 5: Normalized
    gt_normalized = re.sub(r"[^a-z0-9]", "", gt_section_id.lower())
    if gt_normalized and gt_normalized in normalized_map:
        return normalized_map[gt_normalized], "normalized"

    return None, "unresolved"


def resolve_framework_links(
    conn: sqlite3.Connection,
    framework_id: str,
    links: list[dict],
) -> ResolverResult:
    """Resolve all GT links for a single framework.

    Returns ResolverResult with resolved mappings, unresolved links, and strategy counts.
    """
    section_id_map, title_map, normalized_map = build_control_lookups(conn, framework_id)

    resolved: dict[str, str] = {}
    unresolved: list[dict] = []
    strategy_counts: dict[str, int] = {}

    for link in links:
        gt_sid = link["section_id"]
        gt_key = f"{framework_id}:{gt_sid}"

        control_id, strategy = resolve_section_id(
            gt_sid, framework_id, section_id_map, title_map, normalized_map,
        )

        if control_id is not None:
            resolved[gt_key] = control_id
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        else:
            unresolved.append(link)
            logger.warning(
                "Unresolvable GT link: framework=%s section_id=%s",
                framework_id, gt_sid,
            )

    return ResolverResult(
        resolved=resolved,
        unresolved=unresolved,
        strategy_counts=strategy_counts,
    )
```

- [ ] **Step 3: Write resolver tests**

Create `tests/test_ground_truth_import.py` with resolver-specific tests:

1. Test each strategy individually (direct, prefixed, title_exact, title_case_insensitive, normalized)
2. Test strategy priority (earlier strategies win)
3. Test unresolvable links return None
4. Test `resolve_framework_links()` with the mini fixture
5. Test `build_control_lookups()` builds correct dictionaries
6. Use `tmp_path` with a temporary SQLite DB populated with synthetic controls

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_ground_truth_import.py -v -k "resolver"
```

- [ ] **Step 5: Commit**

```bash
git add tract/crosswalk/ground_truth.py tests/test_ground_truth_import.py tests/fixtures/phase3_mini_hub_links.json
git commit -m "feat(phase3): multi-strategy section ID resolver for ground truth import"
```

---

### Task 2: Ground Truth Assignment Import

**Files:**
- Modify: `tract/crosswalk/ground_truth.py` — add `import_ground_truth()` function
- Modify: `tests/test_ground_truth_import.py` — add import tests

> **Why:** This populates crosswalk.db with ~4,388 ground truth assignments. The resolver from Task 1 handles the hard part; this task handles insertion with dedup and transaction safety.

> **Critical context:**
> - Existing DB has 636 assignments: 558 `active_learning_round_2` + 78 `ground_truth_T1-AI`
> - 36 duplicate `(control_id, hub_id)` groups already exist (including one 7-way)
> - GT import must use EXISTS check before each insert (provenance-agnostic)
> - `source_link_id` stores the `link_type` value ("LinkedTo" or "AutomaticallyLinkedTo") — semantic repurpose
> - Single transaction: atomic rollback on any failure
> - DB backup: copy crosswalk.db to crosswalk.db.backup.{timestamp} before writes

- [ ] **Step 1: Add import_ground_truth() to ground_truth.py**

```python
def import_ground_truth(
    db_path: Path,
    hub_links_path: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Import OpenCRE ground truth links into crosswalk.db.

    Returns summary dict with counts: imported, skipped_duplicate, unresolved, per_framework.
    """
```

Key implementation details:
- Load `hub_links_by_framework.json`
- Map `standard_name` to `framework_id` using `OPENCRE_FRAMEWORK_ID_MAP` from config
- For each framework, call `resolve_framework_links()`
- Before inserting, check `EXISTS(SELECT 1 FROM assignments WHERE control_id=? AND hub_id=?)`
- Insert with: `provenance="opencre_ground_truth"`, `confidence=1.0`, `review_status="ground_truth"`, `source_link_id=link["link_type"]`, `model_version=NULL`
- Wrap all inserts in a single transaction
- On dry_run, compute counts but don't write
- Backup DB before writing (unless dry_run)

- [ ] **Step 2: Add backup helper**

```python
def _backup_database(db_path: Path) -> Path:
    """Copy database to timestamped backup. Returns backup path."""
    import shutil
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = db_path.with_suffix(f".backup.{ts}")
    shutil.copy2(db_path, backup)
    logger.info("Database backed up to %s", backup)
    return backup
```

- [ ] **Step 3: Add import tests**

Add to `tests/test_ground_truth_import.py`:
1. Test basic import — synthetic fixture → correct row count
2. Test dedup — pre-populate with assignment, import same `(control_id, hub_id)` → skipped
3. Test all 78 ground_truth_T1-AI overlaps are skipped (synthetic version)
4. Test atomic rollback — inject error mid-transaction → no partial writes
5. Test backup creation
6. Test dry_run returns counts without modifying DB
7. Test source_link_id stores link_type correctly

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_ground_truth_import.py -v -k "import"
```

- [ ] **Step 5: Commit**

```bash
git add tract/crosswalk/ground_truth.py tests/test_ground_truth_import.py
git commit -m "feat(phase3): ground truth assignment import with dedup and atomic transactions"
```

---

### Task 3: Inference on Uncovered Frameworks

**Files:**
- Modify: `tract/crosswalk/ground_truth.py` — add `run_uncovered_inference()`
- Modify: `tests/test_ground_truth_import.py` — add inference tests (mocked TRACTPredictor)

> **Why:** Five AI frameworks (AIUC-1, CoSAI, EU GPAI CoP, NIST AI RMF, OWASP DSGAI) have controls in the DB but zero assignments. This runs inference on all ~320 controls to generate initial model predictions.

> **Critical context:**
> - Text preparation: `" ".join([title, description, full_text])` — matching `tract ingest` pipeline
> - Top-1 prediction per control stored as assignment
> - `provenance="model_prediction"`, `review_status="pending"`
> - `confidence` stores calibrated value (softmax output from predict_batch), matching AL convention
> - `model_version`: git SHA of deployment model (from artifacts metadata)
> - NIST AI RMF has 13 controls with <50 char descriptions — predictions may be unreliable
> - TRACTPredictor loads from `PHASE1D_DEPLOYMENT_MODEL_DIR`

- [ ] **Step 1: Add run_uncovered_inference()**

```python
def run_uncovered_inference(
    db_path: Path,
    model_dir: Path,
    *,
    dry_run: bool = False,
) -> dict:
    """Run inference on uncovered framework controls and insert as model_prediction.

    Returns summary dict with per-framework counts and text quality warnings.
    """
```

Key implementation:
- Query controls for each framework in `PHASE3_UNCOVERED_FRAMEWORK_IDS`
- Build text as `" ".join([title, description, full_text])` (filter empty parts)
- Load `TRACTPredictor(model_dir)`, call `predict_batch()` with top_k=1
- For each prediction, insert assignment:
  - `confidence` = `prediction.calibrated_confidence` (matching AL convention)
  - `is_ood` = `prediction.is_ood`
  - `in_conformal_set` = `prediction.in_conformal_set`
  - `model_version` = git SHA from artifacts
- Log warning for controls with combined text length < `PHASE3_TEXT_QUALITY_LOW_THRESHOLD`
- Separate transaction from GT import (failure here shouldn't roll back GT)

- [ ] **Step 2: Add inference tests (mocked model)**

Add tests with a mock TRACTPredictor that returns synthetic HubPrediction objects:
1. Test correct text preparation (title + description + full_text)
2. Test top-1 selection from predict_batch result
3. Test confidence stores calibrated value
4. Test is_ood flag stored correctly
5. Test text quality warning logged for short controls
6. Test model_version set from artifacts

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_ground_truth_import.py -v -k "inference"
```

- [ ] **Step 4: Commit**

```bash
git add tract/crosswalk/ground_truth.py tests/test_ground_truth_import.py
git commit -m "feat(phase3): inference on uncovered AI frameworks for model predictions"
```

---

### Task 4: Review Export — Core JSON Generation

**Files:**
- Create: `tract/review/__init__.py`
- Create: `tract/review/export.py`
- Create: `tests/test_review_export.py`

> **Why:** The review export is the most complex module — it re-runs inference on all in-scope controls, computes fresh confidence values (avoiding double-calibration), and generates the monolithic review JSON.

> **Critical context — DOUBLE-CALIBRATION BUG:**
> The existing 558 AL assignments store CALIBRATED confidence (softmax output from `accept.py:108`) in the DB `confidence` column. If we read these values and re-calibrate, we'd apply softmax twice → garbage. Solution: re-run inference at export time for ALL predictions to get fresh, consistent raw_similarity + calibrated_confidence values. This also gives us alternative_hubs (top-3 from predict_batch) for free.

> **Critical context — GT-confirmed exclusion:**
> ~30 AL predictions overlap with GT for the same (control_id, hub_id). These are already confirmed by a stronger source and should NOT go to the reviewer. Exclude via `NOT EXISTS (SELECT 1 FROM assignments a2 WHERE a2.control_id = a.control_id AND a2.hub_id = a.hub_id AND a2.provenance = 'opencre_ground_truth')`.

- [ ] **Step 1: Create tract/review/__init__.py**

```python
"""TRACT review workflow — export, validate, import, and metrics."""
```

- [ ] **Step 2: Create tract/review/export.py**

Core function:

```python
def generate_review_export(
    db_path: Path,
    model_dir: Path,
    output_dir: Path,
    calibration_path: Path,
) -> dict:
    """Generate review JSON, reviewer guide, and hub reference.

    Re-runs inference on all in-scope controls to get fresh confidence values.
    Returns metadata dict.
    """
```

Implementation details:

1. **Query in-scope assignments:**
   ```sql
   SELECT a.id, a.control_id, a.hub_id, a.provenance, a.is_ood,
          c.title, c.description, c.full_text, c.framework_id,
          f.name as framework_name
   FROM assignments a
   JOIN controls c ON a.control_id = c.id
   JOIN frameworks f ON c.framework_id = f.id
   WHERE a.provenance IN ('active_learning_round_2', 'model_prediction')
     AND a.reviewer IS NULL
     AND NOT EXISTS (
       SELECT 1 FROM assignments a2
       WHERE a2.control_id = a.control_id
         AND a2.hub_id = a.hub_id
         AND a2.provenance = 'opencre_ground_truth'
     )
   ORDER BY f.name, a.id
   ```

2. **Re-run inference:** Load TRACTPredictor, call predict_batch() with all control texts. For each assignment, find the prediction matching its hub_id to get fresh confidence + raw_similarity + alternative_hubs (top-3).

3. **Compute text_quality** from combined inference text length:
   - `"high"` if len > `PHASE3_TEXT_QUALITY_HIGH_THRESHOLD` (500)
   - `"medium"` if 100-500
   - `"low"` if < `PHASE3_TEXT_QUALITY_LOW_THRESHOLD` (100)

4. **Compute review_priority** using `global_threshold` from calibration.json:
   - `"critical"`: calibrated_conf ≤ global_threshold AND text_quality="low"
   - `"careful"`: calibrated_conf ≤ global_threshold OR is_ood
   - `"routine"`: calibrated_conf > global_threshold AND not OOD

5. **Build JSON structure** matching spec Section 2.2

- [ ] **Step 3: Write tests**

Create `tests/test_review_export.py`:
1. Test JSON structure matches schema (metadata, predictions array)
2. Test GT-confirmed AL predictions excluded
3. Test re-inference provides fresh values (mock TRACTPredictor)
4. Test text_quality computation from combined text length
5. Test review_priority logic (all three values)
6. Test alternative_hubs populated (top-2 next-best)
7. Test predictions sorted by framework (alpha) then confidence (desc)
8. Test re-export after partial import excludes reviewed items (reviewer IS NOT NULL)

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_review_export.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tract/review/__init__.py tract/review/export.py tests/test_review_export.py
git commit -m "feat(phase3): review export with re-inference and GT-confirmed exclusion"
```

---

### Task 5: Calibration Items

**Files:**
- Modify: `tract/review/export.py` — add calibration item generation
- Modify: `tests/test_review_export.py` — add calibration tests

> **Why:** 20 items from ground_truth_T1-AI are disguised as regular predictions to measure reviewer quality. They use negative IDs (-1 to -20) and inference-derived confidence to be indistinguishable from real predictions.

> **Critical context:**
> - All 78 ground_truth_T1-AI have NULL confidence in DB — can't use DB values
> - Must run inference on each calibration control's text to get model's genuine confidence
> - Selection: sort by model's confidence for known-correct hub, take top-5 (easy) + bottom-5 (hard) + 10 random middle (seed=42)
> - Negative IDs never collide with SQLite AUTOINCREMENT (always positive)
> - Set status="pending" like all others — no visible difference to reviewer

- [ ] **Step 1: Add calibration item generation to export.py**

```python
def _generate_calibration_items(
    db_path: Path,
    predictor: TRACTPredictor,
    calibration_path: Path,
) -> list[dict]:
    """Generate 20 calibration items from ground_truth_T1-AI.

    Runs inference to get model's genuine confidence for known-correct hubs.
    Uses stratified selection: 5 easy + 5 hard + 10 random middle.
    Returns list of prediction dicts with negative IDs.
    """
```

Key implementation:
- Query all `ground_truth_T1-AI` assignments (78 items)
- Join with controls table to get text
- Run inference on all 78 texts via predict_batch(top_k=5)
- For each, find the confidence for the known-correct hub_id in the predictions
  - If hub_id matches top-1: use that confidence
  - If hub_id not in top-K: use the raw cosine similarity for that hub from the full similarity matrix (need to handle this edge case)
- Sort by this confidence, take 5 from top (easy), 5 from bottom (hard), 10 random from middle (np.random.default_rng(42))
- Assign negative IDs: -1 to -20
- Format as prediction dicts matching the main predictions schema

- [ ] **Step 2: Add calibration tests**

Add to `tests/test_review_export.py`:
1. Test exactly 20 calibration items generated
2. Test all have negative IDs (-1 to -20)
3. Test selection is stratified (5 easy + 5 hard + 10 middle)
4. Test reproducibility with same seed
5. Test calibration items have status="pending" (indistinguishable)
6. Test confidence is from inference, not NULL

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_review_export.py -v -k "calibration"
```

- [ ] **Step 4: Commit**

```bash
git add tract/review/export.py tests/test_review_export.py
git commit -m "feat(phase3): calibration items with inference-derived confidence and stratified selection"
```

---

### Task 6: Reviewer Guide + Hub Reference

**Files:**
- Create: `tract/review/guide.py`
- Create: `tests/test_review_guide.py` (lightweight)

> **Why:** The reviewer guide provides step-by-step instructions for the outsourced expert. The hub reference (hub_reference.json) gives the reviewer a searchable lookup when the model's suggestions don't fit.

- [ ] **Step 1: Create tract/review/guide.py**

Two functions:

```python
def generate_reviewer_guide(output_dir: Path, metadata: dict) -> Path:
    """Generate reviewer_guide.md with instructions, decision criteria, and common pitfalls."""

def generate_hub_reference(db_path: Path, output_dir: Path) -> Path:
    """Generate hub_reference.json — all 522 hubs with id, name, path, parent_id, is_leaf."""
```

Guide content matches spec Section 2.5 exactly:
- Role & persona paragraph
- Background section (CRE, hubs, TRACT, assignments)
- Step-by-step process (9 steps)
- Decision criteria
- Common pitfalls (5 items)
- Editor requirement + validation tip
- Saving progress section
- Time estimate

Hub reference: query all hubs, sort by `path` (alphabetical), output JSON array.

- [ ] **Step 2: Write tests**

Create `tests/test_review_guide.py`:
1. Test guide.md generated and contains key sections ("Role", "Step-by-step", "Decision criteria")
2. Test hub_reference.json contains all hubs from DB
3. Test hub_reference.json sorted by path
4. Test each hub has required fields (hub_id, name, path, parent_id, is_leaf)

- [ ] **Step 3: Commit**

```bash
git add tract/review/guide.py tests/test_review_guide.py
git commit -m "feat(phase3): reviewer guide and hub reference generation"
```

---

### Task 7: Review Validation

**Files:**
- Create: `tract/review/validate.py`
- Create: `tests/test_review_validate.py`

> **Why:** JSON editing is error-prone. A standalone validation command (`tract review-validate`) catches syntax and semantic errors before the import modifies the DB. The same validation logic is reused by `review-import`.

- [ ] **Step 1: Create tract/review/validate.py**

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]      # hard errors (block import)
    warnings: list[str]    # soft warnings (allow import)

def validate_review_json(
    review_path: Path,
    db_path: Path,
) -> ValidationResult:
    """Validate a reviewed predictions JSON file.

    Checks:
    - JSON parse succeeds (report line context on failure)
    - Top-level structure: metadata + predictions array
    - Every non-pending status is valid (accepted/reassigned/rejected)
    - Every corrected_hub_id (on reassigned) is a valid hub ID in DB
    - Every id matches an existing assignment (skip calibration items with id < 0)
    - Warns (not fails) if pending items remain
    """
```

- [ ] **Step 2: Write tests**

Create `tests/test_review_validate.py`:
1. Test valid file passes
2. Test JSON syntax error reports line context
3. Test invalid status caught
4. Test invalid corrected_hub_id caught
5. Test non-existent assignment ID caught
6. Test calibration items (id < 0) skipped in ID validation
7. Test pending items produce warning, not error
8. Test missing metadata or predictions key caught

- [ ] **Step 3: Commit**

```bash
git add tract/review/validate.py tests/test_review_validate.py
git commit -m "feat(phase3): standalone review JSON validation"
```

---

### Task 8: Review Import — apply_review_decisions()

**Files:**
- Create: `tract/review/import_review.py`
- Create: `tests/test_review_import.py`

> **Why:** This is where the expert's decisions modify crosswalk.db. Uses UPDATE-in-place semantics, NOT the existing `update_review_status()` which creates new rows on correction.

> **Critical context — DO NOT USE update_review_status():**
> `store.py:update_review_status()` uses "corrected" status and INSERT-new-row pattern. This spec requires UPDATE-in-place with `original_hub_id` tracking. Write a new `apply_review_decisions()` function.

- [ ] **Step 1: Create tract/review/import_review.py**

```python
def apply_review_decisions(
    db_path: Path,
    review_path: Path,
    reviewer: str,
) -> dict:
    """Apply expert review decisions to crosswalk.db.

    Updates assignments in a single transaction. Returns summary dict.
    Skips calibration items (id < 0).
    """
```

Update logic per status:

| Status | DB Update |
|--------|-----------|
| `accepted` | `review_status='accepted'`, `reviewer=<name>`, `review_date=now`, `reviewer_notes=<notes>` |
| `reassigned` | `original_hub_id=<old hub_id>`, `hub_id=corrected_hub_id`, `confidence=NULL`, `review_status='accepted'`, `reviewer=<name>`, `review_date=now`, `reviewer_notes="[Reassigned from hub {old_id} (confidence={old_conf:.3f})]" + notes` |
| `rejected` | `review_status='rejected'`, `reviewer=<name>`, `review_date=now`, `reviewer_notes=<notes>` |
| `pending` | Skip — no update |

Key details:
- Run `migrate_schema(db_path)` first (idempotent)
- Call `validate_review_json()` first — fail on any errors
- Backup DB before writes
- Single transaction — atomic rollback
- Idempotent: re-importing same file produces same DB state
- If assignment already has reviewer and new reviewer differs, log warning but allow override

- [ ] **Step 2: Write tests**

Create `tests/test_review_import.py`:
1. Test accepted updates review_status + reviewer + review_date
2. Test reassigned: original_hub_id set, hub_id changed, confidence NULL, review_status='accepted'
3. Test rejected updates review_status + reviewer_notes
4. Test pending items skipped
5. Test calibration items (id < 0) skipped
6. Test idempotent: import twice → same DB state
7. Test atomic rollback on error mid-transaction
8. Test reviewer override warning
9. Test migrate_schema() called before import
10. Test validation errors block import

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_review_import.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tract/review/import_review.py tests/test_review_import.py
git commit -m "feat(phase3): review import with apply_review_decisions (UPDATE-in-place)"
```

---

### Task 9: Review Metrics

**Files:**
- Create: `tract/review/metrics.py`
- Create: `tests/test_review_metrics.py`

> **Why:** Metrics track acceptance/rejection/reassignment rates overall and per-framework, plus reviewer quality via calibration item agreement. Saved to `review_metrics.json` for dataset card.

- [ ] **Step 1: Create tract/review/metrics.py**

```python
def compute_review_metrics(
    db_path: Path,
    review_data: dict,
    output_path: Path,
) -> dict:
    """Compute review metrics and save to JSON.

    Includes: coverage, overall rates, per-framework breakdown, 
    reviewer quality (calibration items), confidence analysis.
    """
```

Key computations:
- **Coverage:** total_predictions, reviewed, pending, completion_pct
- **Overall:** accepted, rejected, reassigned counts + rates
- **Per-framework:** breakdown by framework_name
- **Reviewer quality:** Compare calibration item decisions to known GT hub_ids. quality_score = agreed/total. List disagreements.
- **Confidence analysis:** acceptance rate for high-conf (> global_threshold), low-conf (≤ threshold), and OOD items
- **import_round:** Increment counter (check for existing metrics file)

- [ ] **Step 2: Write tests**

Create `tests/test_review_metrics.py`:
1. Test overall rates computation
2. Test per-framework breakdown
3. Test calibration quality score (all agree, some disagree)
4. Test confidence analysis bins
5. Test partial review (pending > 0, completion_pct < 100)
6. Test import_round increments

- [ ] **Step 3: Commit**

```bash
git add tract/review/metrics.py tests/test_review_metrics.py
git commit -m "feat(phase3): review metrics computation with calibration quality assessment"
```

---

### Task 10: Dataset Bundle — JSONL + Metadata

**Files:**
- Create: `tract/dataset/__init__.py`
- Create: `tract/dataset/bundle.py`
- Create: `tests/test_dataset_bundle.py`

> **Why:** The published dataset assembles all assignments into a deduplicated JSONL with derived `assignment_type` values, plus framework metadata. This is the core research output.

> **Critical context — DEDUPLICATION:**
> 36 `(control_id, hub_id)` groups have multiple rows. Dedup by provenance priority: `opencre_ground_truth` > `ground_truth_T1-AI` > `active_learning_round_2` > `model_prediction`. Output ONE row per unique pair.

> **Critical context — assignment_type DERIVATION:**
> Uses `original_hub_id IS NOT NULL` for `model_reassigned` (queryable column, no text parsing).
> | assignment_type | Derivation |
> |----------------|------------|
> | ground_truth_linked | provenance="opencre_ground_truth" AND source_link_id="LinkedTo" |
> | ground_truth_auto | provenance="opencre_ground_truth" AND source_link_id="AutomaticallyLinkedTo" |
> | model_accepted | review_status="accepted" AND original_hub_id IS NULL |
> | model_reassigned | review_status="accepted" AND original_hub_id IS NOT NULL |
> | model_rejected | review_status="rejected" |

- [ ] **Step 1: Create tract/dataset/__init__.py**

```python
"""TRACT dataset publication — bundle, card, and upload."""
```

- [ ] **Step 2: Create tract/dataset/bundle.py**

```python
def bundle_dataset(
    db_path: Path,
    staging_dir: Path,
    hierarchy_path: Path,
    hub_descriptions_path: Path,
    bridge_report_path: Path,
    review_metrics_path: Path,
) -> dict:
    """Assemble staging directory with all dataset files.

    Creates: crosswalk_v1.0.jsonl, framework_metadata.json,
    cre_hierarchy_v1.1.json, hub_descriptions_v1.0.json,
    bridge_report.json, review_metrics.json, LICENSE, zenodo_metadata.json
    """

def _build_crosswalk_jsonl(db_path: Path, output_path: Path) -> int:
    """Query all assignments, dedup by (control_id, hub_id), write JSONL.

    Returns row count.
    """

def _build_framework_metadata(db_path: Path, output_path: Path) -> list[dict]:
    """Generate per-framework statistics."""

def _build_zenodo_metadata(output_path: Path) -> None:
    """Generate zenodo_metadata.json for manual upload."""
```

JSONL dedup query:
```sql
SELECT a.*, c.title, c.section_id, c.framework_id, f.name as framework_name,
       h.name as hub_name, h.path as hub_path
FROM assignments a
JOIN controls c ON a.control_id = c.id
JOIN frameworks f ON c.framework_id = f.id
JOIN hubs h ON a.hub_id = h.id
ORDER BY a.control_id, a.hub_id,
         CASE a.provenance
           WHEN 'opencre_ground_truth' THEN 1
           WHEN 'ground_truth_T1-AI' THEN 2
           WHEN 'active_learning_round_2' THEN 3
           WHEN 'model_prediction' THEN 4
           ELSE 5
         END
```
Then deduplicate in Python: keep first row per `(control_id, hub_id)` group (already priority-ordered).

- [ ] **Step 3: Write tests**

Create `tests/test_dataset_bundle.py`:
1. Test JSONL has one row per unique (control_id, hub_id)
2. Test dedup keeps highest-priority provenance
3. Test assignment_type derivation for all 5 values
4. Test framework_metadata coverage_type computation
5. Test zenodo_metadata structure
6. Test LICENSE file contents (CC-BY-SA-4.0)
7. Test JSONL rows have all required fields

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_dataset_bundle.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tract/dataset/__init__.py tract/dataset/bundle.py tests/test_dataset_bundle.py
git commit -m "feat(phase3): dataset bundle with provenance-priority dedup and assignment_type derivation"
```

---

### Task 11: Dataset Card

**Files:**
- Create: `tract/dataset/card.py`
- Create: `tests/test_dataset_card.py`

> **Why:** The HuggingFace Datasets card (README.md) is the primary documentation for dataset consumers. Matches model card quality standards. Uses YAML frontmatter for HuggingFace auto-discovery.

- [ ] **Step 1: Create tract/dataset/card.py**

```python
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
    """
```

YAML frontmatter:
```yaml
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
---
```

- [ ] **Step 2: Write tests**

Create `tests/test_dataset_card.py`:
1. Test YAML frontmatter present and valid
2. Test all 9 sections present
3. Test framework coverage table has all frameworks
4. Test Quick Start section has load_dataset example
5. Test Citation section has BibTeX

- [ ] **Step 3: Commit**

```bash
git add tract/dataset/card.py tests/test_dataset_card.py
git commit -m "feat(phase3): HuggingFace dataset card generation"
```

---

### Task 12: Dataset Publication (Upload)

**Files:**
- Create: `tract/dataset/publish.py`
- Create: `tests/test_dataset_publish.py`

> **Why:** Upload staging directory to HuggingFace Hub as a dataset repository. Uses `create_repo(repo_type="dataset")` + `upload_folder()`. Token via `pass huggingface/token`.

- [ ] **Step 1: Create tract/dataset/publish.py**

```python
def publish_dataset(
    repo_id: str,
    staging_dir: Path,
    *,
    dry_run: bool = False,
    skip_upload: bool = False,
) -> None:
    """Upload dataset staging directory to HuggingFace Hub.

    Uses repo_type="dataset" (not "model").
    Token retrieved via `pass huggingface/token`.
    """
```

Pattern matches `tract/publish/__init__.py:_upload_to_hub()` but uses `repo_type="dataset"`.

- [ ] **Step 2: Write tests (mocked HfApi)**

Create `tests/test_dataset_publish.py`:
1. Test dry_run skips upload
2. Test skip_upload skips upload
3. Test HfApi.create_repo called with repo_type="dataset"
4. Test HfApi.upload_folder called with correct paths
5. Test token cleanup (del token in finally block)

- [ ] **Step 3: Commit**

```bash
git add tract/dataset/publish.py tests/test_dataset_publish.py
git commit -m "feat(phase3): HuggingFace dataset upload (repo_type=dataset)"
```

---

### Task 13: CLI Commands (5 subcommands)

**Files:**
- Modify: `tract/cli.py:27-32,239+`
- Create: `tests/test_phase3_cli.py`

> **Why:** Five CLI commands wire up all the Phase 3 modules. Pattern matches existing subcommand structure in cli.py (e.g., bridge, publish-hf).

- [ ] **Step 1: Add imports to cli.py**

Add to the imports section (lines 15-26):
```python
from tract.config import (
    # ... existing imports ...
    PHASE1D_DEPLOYMENT_MODEL_DIR,   # already imported
    PHASE3_DATASET_REPO_ID,
    PHASE3_DATASET_STAGING_DIR,
    PHASE3_REVIEW_OUTPUT_DIR,
    TRAINING_DIR,                    # already imported
)
```

- [ ] **Step 2: Add 5 subcommand parsers in build_parser()**

Add after the publish-hf parser (line 237), before `return parser`:

```python
# ── import-ground-truth ─────────────────────────────────────────
p_import_gt = subparsers.add_parser(
    "import-ground-truth",
    help="Import OpenCRE ground truth links and run inference on uncovered frameworks",
    epilog=(
        "Examples:\n"
        "  tract import-ground-truth\n"
        "  tract import-ground-truth --dry-run\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
p_import_gt.add_argument("--dry-run", action="store_true", help="Report counts without modifying DB")

# ── review-export ──────────────────────────────────────────────
p_review_export = subparsers.add_parser(
    "review-export",
    help="Generate review JSON for expert review",
    epilog=(
        "Examples:\n"
        "  tract review-export --output results/review/review_predictions.json\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
p_review_export.add_argument(
    "--output", default=str(PHASE3_REVIEW_OUTPUT_DIR / "review_predictions.json"),
    help="Output file path",
)
p_review_export.add_argument(
    "--model-dir", default=str(PHASE1D_DEPLOYMENT_MODEL_DIR),
    help="Path to deployment model directory",
)

# ── review-validate ─────────────────────────────────────────────
p_review_validate = subparsers.add_parser(
    "review-validate",
    help="Validate a reviewed predictions JSON file",
    epilog=(
        "Examples:\n"
        "  tract review-validate --input results/review/review_predictions.json\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
p_review_validate.add_argument("--input", required=True, help="Path to reviewed JSON file")

# ── review-import ──────────────────────────────────────────────
p_review_import = subparsers.add_parser(
    "review-import",
    help="Import expert review decisions into crosswalk.db",
    epilog=(
        "Examples:\n"
        "  tract review-import --input results/review/review_predictions.json --reviewer expert_1\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
p_review_import.add_argument("--input", required=True, help="Path to reviewed JSON file")
p_review_import.add_argument("--reviewer", required=True, help="Reviewer name/identifier")

# ── publish-dataset ──────────────────────────────────────────────
p_pub_dataset = subparsers.add_parser(
    "publish-dataset",
    help="Publish crosswalk dataset to HuggingFace Datasets",
    epilog=(
        "Examples:\n"
        "  tract publish-dataset --repo-id rockCO78/tract-crosswalk-dataset --dry-run\n"
        "  tract publish-dataset --repo-id rockCO78/tract-crosswalk-dataset\n"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
p_pub_dataset.add_argument("--repo-id", default=PHASE3_DATASET_REPO_ID, help="HuggingFace repo ID")
p_pub_dataset.add_argument("--staging-dir", default=str(PHASE3_DATASET_STAGING_DIR), help="Local build dir")
p_pub_dataset.add_argument("--dry-run", action="store_true", help="Build without upload")
p_pub_dataset.add_argument("--skip-upload", action="store_true", help="Build only, no upload")
```

- [ ] **Step 3: Add 5 handler functions**

Add after existing `_cmd_publish_hf()` handler:

```python
def _cmd_import_ground_truth(args: argparse.Namespace) -> None:
    from tract.crosswalk.ground_truth import import_ground_truth, run_uncovered_inference
    
    summary = import_ground_truth(
        PHASE1C_CROSSWALK_DB_PATH,
        TRAINING_DIR / "hub_links_by_framework.json",
        dry_run=args.dry_run,
    )
    # Print summary...
    
    if not args.dry_run:
        inf_summary = run_uncovered_inference(
            PHASE1C_CROSSWALK_DB_PATH,
            PHASE1D_DEPLOYMENT_MODEL_DIR,
            dry_run=args.dry_run,
        )
        # Print inference summary...


def _cmd_review_export(args: argparse.Namespace) -> None:
    from tract.review.export import generate_review_export
    # ...


def _cmd_review_validate(args: argparse.Namespace) -> None:
    from tract.review.validate import validate_review_json
    # ...


def _cmd_review_import(args: argparse.Namespace) -> None:
    from tract.review.import_review import apply_review_decisions
    from tract.review.metrics import compute_review_metrics
    # ...


def _cmd_publish_dataset(args: argparse.Namespace) -> None:
    from tract.dataset.bundle import bundle_dataset
    from tract.dataset.card import generate_dataset_card
    from tract.dataset.publish import publish_dataset
    # ...
```

- [ ] **Step 4: Add command dispatch**

In the `main()` function, add command routing:
```python
elif args.command == "import-ground-truth":
    _cmd_import_ground_truth(args)
elif args.command == "review-export":
    _cmd_review_export(args)
elif args.command == "review-validate":
    _cmd_review_validate(args)
elif args.command == "review-import":
    _cmd_review_import(args)
elif args.command == "publish-dataset":
    _cmd_publish_dataset(args)
```

- [ ] **Step 5: Write CLI tests**

Create `tests/test_phase3_cli.py`:
1. Test all 5 subcommands parse arguments correctly
2. Test --help for each command doesn't error
3. Test import-ground-truth --dry-run argument
4. Test review-export --model-dir argument
5. Test review-import requires --reviewer

- [ ] **Step 6: Run tests**

```bash
python -m pytest tests/test_phase3_cli.py -v
python -m pytest tests/test_cli.py -v  # existing tests still pass
```

- [ ] **Step 7: Commit**

```bash
git add tract/cli.py tests/test_phase3_cli.py
git commit -m "feat(phase3): add 5 CLI subcommands (import-ground-truth, review-export, review-validate, review-import, publish-dataset)"
```

---

### Task 14: Integration Test

**Files:**
- Create: `tests/test_phase3_integration.py`
- Create: `tests/fixtures/phase3_review_predictions.json`

> **Why:** End-to-end test exercises the full pipeline: GT import → review export → review import → dataset bundle. Uses synthetic data in a temp DB — no GPU, no real model.

- [ ] **Step 1: Create review predictions fixture**

Create `tests/fixtures/phase3_review_predictions.json` with 5 synthetic predictions (3 accepted, 1 reassigned, 1 rejected) + 2 calibration items (id < 0).

- [ ] **Step 2: Write integration test**

Create `tests/test_phase3_integration.py`:

```python
def test_full_pipeline(tmp_path):
    """GT import → review export → review import → bundle."""
    # 1. Create temp DB with synthetic controls + hubs
    # 2. Import GT from mini fixture
    # 3. Mock TRACTPredictor, generate review export
    # 4. Simulate expert review (modify exported JSON)
    # 5. Import review decisions
    # 6. Bundle dataset
    # 7. Verify JSONL row count, dedup, assignment_types
```

Key assertions:
- GT assignments created with correct provenance
- Review export excludes GT-confirmed overlaps
- Calibration items have negative IDs
- Review import updates assignments (accepted, reassigned, rejected)
- Reassigned assignments have original_hub_id set
- JSONL deduplication correct
- Framework metadata coverage_type correct

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/test_phase3_integration.py -v
python -m pytest tests/ -q --tb=short  # all 553+ tests still pass
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_phase3_integration.py tests/fixtures/phase3_review_predictions.json
git commit -m "test(phase3): end-to-end integration test for GT import → export → import → bundle"
```

---

### Task 15: Live Run — import-ground-truth

**Files:**
- No new files — runs against real crosswalk.db

> **Why:** This is the first live execution. Validates the resolver against all 4,406 GT links and populates crosswalk.db with ~4,388 assignments.

- [ ] **Step 1: Run import-ground-truth with dry-run**

```bash
tract import-ground-truth --dry-run
```

Expected output:
- 4,388 GT links resolved (99.59%)
- 18 unresolvable (OWASP AI Exchange)
- ~78 skipped duplicates (ground_truth_T1-AI overlap)
- Per-framework breakdown

- [ ] **Step 2: Review dry-run output**

Verify:
- Strategy counts match expectations (spec Section 7)
- Unresolvable links are all OWASP AI Exchange
- No unexpected errors

- [ ] **Step 3: Run for real**

```bash
tract import-ground-truth
```

Verify:
- DB backup created
- Row count increased by ~4,310 (4,388 GT - 78 T1-AI overlap)
- Inference completed on ~320 uncovered framework controls

- [ ] **Step 4: Validate DB state**

```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('results/phase1c/crosswalk.db')
for row in conn.execute('SELECT provenance, count(*) FROM assignments GROUP BY provenance ORDER BY count(*) DESC'):
    print(f'{row[0]:30s} {row[1]:>6d}')
print()
total = conn.execute('SELECT count(*) FROM assignments').fetchone()[0]
print(f'Total assignments: {total}')
conn.close()
"
```

Expected: ~5,200+ total assignments.

- [ ] **Step 5: Commit data changes**

```bash
git add results/phase1c/crosswalk.db
git commit -m "data(phase3): import 4,388 GT links + 320 model predictions for uncovered frameworks"
```

---

### Task 16: Live Run — review-export

**Files:**
- No new files — generates output to results/review/

> **Why:** Generates the actual review package for the outsourced expert. Re-runs inference on all ~868 in-scope controls for fresh confidence values.

- [ ] **Step 1: Run review-export**

```bash
tract review-export --output results/review/review_predictions.json
```

Expected output:
- ~848 model predictions exported (excluding GT-confirmed overlaps)
- 20 calibration items inserted
- Total: ~868 predictions
- Per-framework counts
- reviewer_guide.md and hub_reference.json generated

- [ ] **Step 2: Validate export**

```bash
python3 -c "
import json
with open('results/review/review_predictions.json') as f:
    data = json.load(f)
print('Total predictions:', data['metadata']['total_predictions'])
print('Calibration items:', data['metadata']['calibration_items'])
print('Frameworks:', json.dumps(data['metadata']['frameworks'], indent=2))

# Check calibration items
cal = [p for p in data['predictions'] if p['id'] < 0]
print(f'Calibration items found: {len(cal)}')
print(f'All negative IDs: {all(p[\"id\"] < 0 for p in cal)}')

# Check no NULL confidence
nulls = [p for p in data['predictions'] if p['confidence'] is None]
print(f'NULL confidence: {len(nulls)}')
"
```

- [ ] **Step 3: Spot-check a few predictions**

Manually inspect 3-5 predictions to verify:
- confidence and calibrated_confidence are different values
- alternative_hubs populated
- text_quality computed correctly
- review_priority set

- [ ] **Step 4: Commit export outputs**

```bash
git add results/review/
git commit -m "data(phase3): generate review export with 868 predictions + 20 calibration items"
```

---

## Execution Dependencies

```
Task 0 (config + schema migration)
    │
    ├── Task 1 (section ID resolver) ─── Task 2 (GT import) ─── Task 3 (inference)
    │                                                                │
    │                                                        Task 15 (live GT import)
    │
    ├── Task 4 (review export core) ─── Task 5 (calibration items) ─── Task 6 (guide)
    │                                                                      │
    │                                                              Task 16 (live export)
    │
    ├── Task 7 (review validate) ─── Task 8 (review import) ─── Task 9 (metrics)
    │
    ├── Task 10 (dataset bundle) ─── Task 11 (dataset card) ─── Task 12 (dataset publish)
    │
    └── Task 13 (CLI) ─── depends on: Tasks 2, 4, 7, 8, 10
                │
                Task 14 (integration test) ─── depends on: all above
```

**Parallelizable groups:**
- Tasks 1-3 (resolver → GT import → inference) are sequential
- Tasks 4-6 (export core → calibration → guide) are sequential
- Tasks 7-9 (validate → import → metrics) are sequential
- Tasks 10-12 (bundle → card → publish) are sequential
- Groups can be parallelized across groups after Task 0 completes

---

## Adversarial Review Findings — Implementation Constraints

All 17 findings from the spec's adversarial review are incorporated as constraints in the tasks above. Quick reference:

| ID | Constraint | Task |
|----|-----------|------|
| M1 | reviewer_notes + original_hub_id columns via ALTER TABLE | Task 0 |
| M2 | Re-inference at export avoids double-calibration; confidence stores calibrated | Tasks 3, 4 |
| M3 | 36 duplicate groups — provenance-agnostic EXISTS check | Task 2 |
| M4 | New apply_review_decisions(), NOT update_review_status() | Task 8 |
| M5 | Exclude GT-confirmed AL predictions from review export | Task 4 |
| M6 | Provenance-priority dedup for dataset JSONL | Task 10 |
| S1 | text_quality + review_priority thresholds defined | Task 4 |
| S2 | Calibration selection with inference + stratification + seed=42 | Task 5 |
| S3 | Negative IDs only (-1 to -20) | Task 5 |
| S4 | --model-dir argument for review-export | Task 13 |
| S5 | DB backup before writes | Tasks 2, 8 |
| S6 | Single transaction for GT import | Task 2 |
| S7 | Validation with line-level errors | Task 7 |
| N1 | Re-export excludes reviewer IS NOT NULL | Task 4 |
| N2 | source_link_id stores link_type | Task 2 |
| N3 | model_version NULL for 636 existing — documented | Task 11 |
| N4 | Reviewer override warning | Task 8 |
