# Phase 1C: Guardrails, Active Learning & Crosswalk DB — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build calibration pipeline (temperature scaling, conformal prediction, OOD detection), active learning loop (deployment model + expert review), and crosswalk SQLite database — bridging Phase 1B's trained contrastive model to a usable crosswalk.

**Architecture:** Three new packages (`tract/calibration/`, `tract/active_learning/`, `tract/crosswalk/`) plus small modifications to existing training modules. Calibration gates active learning; AL populates the crosswalk DB. Single RunPod H100 for GPU work, local CPU for calibration math and DB operations.

**Tech Stack:** Python 3.11, numpy, scipy (softmax, KS-test), sqlite3 (stdlib), sentence-transformers + PEFT (existing), pytest

**Spec:** `docs/superpowers/specs/2026-04-29-phase1c-design.md`

---

## File Structure

### New Files

```
tract/calibration/__init__.py          — Package init
tract/calibration/temperature.py       — Multi-label NLL, T_lofo fitting, T_deploy fitting
tract/calibration/conformal.py         — Conformal prediction sets
tract/calibration/ood.py               — OOD detection (max-cosine threshold)
tract/calibration/diagnostics.py       — ECE, KS-test, full-recall coverage

tract/active_learning/__init__.py      — Package init
tract/active_learning/deploy.py        — Model loading, deployment training, inference
tract/active_learning/review.py        — review.json generation + ingestion
tract/active_learning/canary.py        — Canary item management
tract/active_learning/stopping.py      — AL stopping criteria evaluation

tract/crosswalk/__init__.py            — Package init
tract/crosswalk/schema.py             — SQLite DDL, migration, WAL mode
tract/crosswalk/store.py              — CRUD operations (atomic transactions)
tract/crosswalk/export.py             — JSON/CSV export of accepted assignments
tract/crosswalk/snapshot.py           — Pre-round DB snapshots (SHA-256)

tests/test_calibration_temperature.py  — Multi-label NLL, temperature fitting
tests/test_calibration_conformal.py    — Conformal prediction sets
tests/test_calibration_ood.py          — OOD threshold + synthetic validation
tests/test_calibration_diagnostics.py  — ECE, KS-test
tests/test_crosswalk_schema.py         — DDL, table creation, WAL mode
tests/test_crosswalk_store.py          — CRUD operations, transactions
tests/test_crosswalk_export.py         — JSON/CSV export
tests/test_crosswalk_snapshot.py       — Snapshot creation, hash verification
tests/test_active_learning_deploy.py   — Model load, holdout selection, adapter
tests/test_active_learning_review.py   — review.json generation + ingestion
tests/test_active_learning_canary.py   — Canary management
tests/test_active_learning_stopping.py — Stopping criteria evaluation

tests/fixtures/ood_synthetic_texts.json — 30 non-security texts for OOD validation
```

### Modified Files

```
tract/training/data_quality.py:44-48   — Add QualityTier.AL
tract/training/data.py:74-78           — Add "AL": 3 to TIER_PRIORITY
tract/training/evaluate.py             — Add extract_similarity_matrix()
tract/config.py                        — Add PHASE1C_* constants
```

---

### Task 1: Phase 1C Constants in config.py

**Files:**
- Modify: `tract/config.py:186` (append after Phase 1B section)

- [ ] **Step 1: Add Phase 1C constants**

```python
# Add after line 186 in tract/config.py (after PHASE1B_MIN_SECTION_TEXT_LENGTH)

# ── Phase 1C: Guardrails, Active Learning & Crosswalk DB ─────────────

PHASE1C_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1c"
PHASE1C_SIMILARITIES_DIR: Final[Path] = PHASE1C_RESULTS_DIR / "similarities"
PHASE1C_DEPLOYMENT_MODEL_DIR: Final[Path] = PHASE1C_RESULTS_DIR / "deployment_model"
PHASE1C_CROSSWALK_DB_PATH: Final[Path] = PHASE1C_RESULTS_DIR / "crosswalk.db"

PHASE1C_HOLDOUT_TOTAL: Final[int] = 440
PHASE1C_HOLDOUT_CALIBRATION: Final[int] = 420
PHASE1C_HOLDOUT_CANARY: Final[int] = 20
PHASE1C_N_AI_CANARIES: Final[int] = 20

PHASE1C_T_GRID_N: Final[int] = 200
PHASE1C_T_GRID_MIN: Final[float] = 0.01
PHASE1C_T_GRID_MAX: Final[float] = 5.0

PHASE1C_ECE_N_BINS: Final[int] = 5
PHASE1C_ECE_THRESHOLD: Final[float] = 0.10
PHASE1C_ECE_BOOTSTRAP_N: Final[int] = 1000

PHASE1C_CONFORMAL_ALPHA: Final[float] = 0.10
PHASE1C_CONFORMAL_COVERAGE_GATE: Final[float] = 0.90

PHASE1C_OOD_PERCENTILE: Final[int] = 5
PHASE1C_OOD_SEPARATION_GATE: Final[float] = 0.90

PHASE1C_AL_ACCEPTANCE_GATE: Final[float] = 0.80
PHASE1C_AL_CANARY_ACCURACY_GATE: Final[float] = 0.85
PHASE1C_AL_HUB_DIVERSITY_GATE: Final[int] = 50
PHASE1C_AL_MAX_ROUNDS: Final[int] = 3

PHASE1C_T_GAP_WARNING: Final[float] = 0.5
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from tract.config import PHASE1C_RESULTS_DIR, PHASE1C_HOLDOUT_TOTAL; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tract/config.py
git commit -m "feat: add Phase 1C constants to config"
```

---

### Task 2: QualityTier.AL and TIER_PRIORITY Update

**Files:**
- Modify: `tract/training/data_quality.py:44-48`
- Modify: `tract/training/data.py:74-78`
- Test: `tests/test_data_quality.py` (existing)
- Test: `tests/test_training_data.py` (existing)

- [ ] **Step 1: Write failing test for QualityTier.AL**

Add to `tests/test_data_quality.py`:

```python
def test_quality_tier_al_exists() -> None:
    from tract.training.data_quality import QualityTier
    assert QualityTier.AL.value == "AL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_quality.py::test_quality_tier_al_exists -v`
Expected: FAIL with `AttributeError: AL`

- [ ] **Step 3: Add QualityTier.AL to data_quality.py**

In `tract/training/data_quality.py`, change the enum (lines 44-48):

```python
class QualityTier(enum.Enum):
    T1 = "T1"
    T1_AI = "T1-AI"
    T3 = "T3"
    AL = "AL"
    DROPPED = "DROPPED"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_data_quality.py::test_quality_tier_al_exists -v`
Expected: PASS

- [ ] **Step 5: Write failing test for TIER_PRIORITY["AL"]**

Add to `tests/test_training_data.py`:

```python
def test_tier_priority_includes_al() -> None:
    from tract.training.data import TIER_PRIORITY
    assert "AL" in TIER_PRIORITY
    assert TIER_PRIORITY["AL"] == 3
```

- [ ] **Step 6: Run test to verify it fails**

Run: `python -m pytest tests/test_training_data.py::test_tier_priority_includes_al -v`
Expected: FAIL with `KeyError` or `AssertionError`

- [ ] **Step 7: Add "AL" to TIER_PRIORITY in data.py**

In `tract/training/data.py`, change TIER_PRIORITY (line 74):

```python
TIER_PRIORITY: dict[str, int] = {
    "T1": 0,
    "T1-AI": 1,
    "T3": 2,
    "AL": 3,
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest tests/test_training_data.py::test_tier_priority_includes_al -v`
Expected: PASS

- [ ] **Step 9: Run full existing test suite to verify no regressions**

Run: `python -m pytest tests/test_data_quality.py tests/test_training_data.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add tract/training/data_quality.py tract/training/data.py tests/test_data_quality.py tests/test_training_data.py
git commit -m "feat: add QualityTier.AL and TIER_PRIORITY for active learning"
```

---

### Task 3: Crosswalk Database Schema

**Files:**
- Create: `tract/crosswalk/__init__.py`
- Create: `tract/crosswalk/schema.py`
- Test: `tests/test_crosswalk_schema.py`

- [ ] **Step 1: Create package init**

```python
# tract/crosswalk/__init__.py
```

(Empty file — package marker only.)

- [ ] **Step 2: Write failing test for schema creation**

Create `tests/test_crosswalk_schema.py`:

```python
"""Tests for crosswalk database schema."""
from __future__ import annotations

import sqlite3

import pytest


class TestCreateSchema:
    def test_creates_all_tables(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "frameworks" in tables
        assert "controls" in tables
        assert "hubs" in tables
        assert "assignments" in tables
        assert "snapshots" in tables

    def test_wal_mode_enabled(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(str(db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()

        assert mode == "wal"

    def test_foreign_keys_enabled(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(str(db_path))
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        conn.close()

        assert fk == 1

    def test_assignments_has_expected_columns(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("PRAGMA table_info(assignments)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        expected = {
            "id", "control_id", "hub_id", "confidence",
            "in_conformal_set", "is_ood", "provenance",
            "source_link_id", "model_version", "review_status",
            "reviewer", "review_date", "created_at",
        }
        assert expected.issubset(columns)

    def test_idempotent_creation(self, tmp_path) -> None:
        from tract.crosswalk.schema import create_database

        db_path = tmp_path / "test.db"
        create_database(db_path)
        create_database(db_path)  # Should not raise
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_crosswalk_schema.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement schema.py**

Create `tract/crosswalk/schema.py`:

```python
"""Crosswalk database schema and initialization.

SQLite database with 5 tables: frameworks, controls, hubs, assignments, snapshots.
Uses WAL mode for concurrent read access. All tables use IF NOT EXISTS for
idempotent creation.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS frameworks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT,
    fetch_date TEXT,
    control_count INTEGER
);

CREATE TABLE IF NOT EXISTS controls (
    id TEXT PRIMARY KEY,
    framework_id TEXT NOT NULL REFERENCES frameworks(id),
    section_id TEXT NOT NULL,
    title TEXT,
    description TEXT,
    full_text TEXT
);

CREATE TABLE IF NOT EXISTS hubs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    path TEXT,
    parent_id TEXT REFERENCES hubs(id)
);

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
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number INTEGER NOT NULL,
    snapshot_date TEXT DEFAULT (datetime('now')),
    db_hash TEXT NOT NULL,
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_assignments_control ON assignments(control_id);
CREATE INDEX IF NOT EXISTS idx_assignments_hub ON assignments(hub_id);
CREATE INDEX IF NOT EXISTS idx_assignments_provenance ON assignments(provenance);
CREATE INDEX IF NOT EXISTS idx_assignments_review ON assignments(review_status);
"""


def create_database(db_path: Path) -> None:
    """Create the crosswalk database with all tables and indices.

    Idempotent — safe to call on an existing database.
    Enables WAL mode and foreign key enforcement.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        logger.info("Crosswalk database initialized at %s", db_path)
    finally:
        conn.close()


def get_connection(db_path: Path) -> sqlite3.Connection:
    """Open a connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_crosswalk_schema.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tract/crosswalk/__init__.py tract/crosswalk/schema.py tests/test_crosswalk_schema.py
git commit -m "feat: crosswalk database schema with 5 tables and WAL mode"
```

---

### Task 4: Crosswalk Store (CRUD Operations)

**Files:**
- Create: `tract/crosswalk/store.py`
- Test: `tests/test_crosswalk_store.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_crosswalk_store.py`:

```python
"""Tests for crosswalk store CRUD operations."""
from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def db_path(tmp_path):
    from tract.crosswalk.schema import create_database
    path = tmp_path / "test.db"
    create_database(path)
    return path


class TestInsertHub:
    def test_insert_and_retrieve(self, db_path) -> None:
        from tract.crosswalk.store import insert_hubs, get_hub

        hubs = [{"id": "236-712", "name": "Access Control", "path": "Root > Access", "parent_id": None}]
        insert_hubs(db_path, hubs)

        hub = get_hub(db_path, "236-712")
        assert hub is not None
        assert hub["name"] == "Access Control"
        assert hub["path"] == "Root > Access"


class TestInsertFramework:
    def test_insert_and_count(self, db_path) -> None:
        from tract.crosswalk.store import insert_frameworks, count_frameworks

        fws = [{"id": "owasp_ai_exchange", "name": "OWASP AI Exchange", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 54}]
        insert_frameworks(db_path, fws)

        assert count_frameworks(db_path) == 1


class TestInsertControl:
    def test_insert_with_framework_fk(self, db_path) -> None:
        from tract.crosswalk.store import insert_frameworks, insert_controls, get_controls_by_framework

        insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
        insert_controls(db_path, [{"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "Control 1", "description": "Desc", "full_text": None}])

        controls = get_controls_by_framework(db_path, "fw1")
        assert len(controls) == 1
        assert controls[0]["title"] == "Control 1"


class TestInsertAssignment:
    def test_insert_and_query(self, db_path) -> None:
        from tract.crosswalk.store import (
            insert_assignments,
            insert_controls,
            insert_frameworks,
            insert_hubs,
            get_assignments_by_control,
        )

        insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
        insert_hubs(db_path, [{"id": "hub1", "name": "Hub 1", "path": "R > H1", "parent_id": None}])
        insert_controls(db_path, [{"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": None, "full_text": None}])

        assignments = [{
            "control_id": "fw1:c1",
            "hub_id": "hub1",
            "confidence": 0.85,
            "in_conformal_set": 1,
            "is_ood": 0,
            "provenance": "active_learning_round_1",
            "source_link_id": None,
            "model_version": "abc123",
            "review_status": "pending",
        }]
        insert_assignments(db_path, assignments)

        result = get_assignments_by_control(db_path, "fw1:c1")
        assert len(result) == 1
        assert result[0]["confidence"] == 0.85
        assert result[0]["provenance"] == "active_learning_round_1"


class TestUpdateReviewStatus:
    def test_update_status(self, db_path) -> None:
        from tract.crosswalk.store import (
            insert_assignments,
            insert_controls,
            insert_frameworks,
            insert_hubs,
            update_review_status,
            get_assignments_by_control,
        )

        insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
        insert_hubs(db_path, [{"id": "hub1", "name": "Hub 1", "path": "R", "parent_id": None}])
        insert_controls(db_path, [{"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": None, "full_text": None}])
        insert_assignments(db_path, [{"control_id": "fw1:c1", "hub_id": "hub1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0, "provenance": "al_r1", "source_link_id": None, "model_version": "v1", "review_status": "pending"}])

        assignments = get_assignments_by_control(db_path, "fw1:c1")
        update_review_status(db_path, assignments[0]["id"], "accepted", reviewer="expert1")

        updated = get_assignments_by_control(db_path, "fw1:c1")
        assert updated[0]["review_status"] == "accepted"
        assert updated[0]["reviewer"] == "expert1"


class TestTransactionAtomicity:
    def test_failed_insert_rolls_back(self, db_path) -> None:
        from tract.crosswalk.store import insert_hubs, count_hubs

        insert_hubs(db_path, [{"id": "h1", "name": "Hub 1", "path": "R", "parent_id": None}])

        with pytest.raises(sqlite3.IntegrityError):
            insert_hubs(db_path, [
                {"id": "h2", "name": "Hub 2", "path": "R", "parent_id": None},
                {"id": "h1", "name": "Duplicate", "path": "R", "parent_id": None},
            ])

        assert count_hubs(db_path) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crosswalk_store.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement store.py**

Create `tract/crosswalk/store.py`:

```python
"""Crosswalk database CRUD operations.

All write operations use explicit transactions for atomicity.
Read operations use Row factory for dict-like access.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


def insert_hubs(db_path: Path, hubs: list[dict]) -> int:
    """Insert hub records. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
                [(h["id"], h["name"], h["path"], h["parent_id"]) for h in hubs],
            )
        return len(hubs)
    finally:
        conn.close()


def get_hub(db_path: Path, hub_id: str) -> dict | None:
    """Get a single hub by ID."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM hubs WHERE id = ?", (hub_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def count_hubs(db_path: Path) -> int:
    """Count total hubs."""
    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM hubs").fetchone()[0]
    finally:
        conn.close()


def insert_frameworks(db_path: Path, frameworks: list[dict]) -> int:
    """Insert framework records. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
                "VALUES (?, ?, ?, ?, ?)",
                [(f["id"], f["name"], f["version"], f["fetch_date"], f["control_count"]) for f in frameworks],
            )
        return len(frameworks)
    finally:
        conn.close()


def count_frameworks(db_path: Path) -> int:
    """Count total frameworks."""
    conn = get_connection(db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM frameworks").fetchone()[0]
    finally:
        conn.close()


def insert_controls(db_path: Path, controls: list[dict]) -> int:
    """Insert control records. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO controls (id, framework_id, section_id, title, description, full_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(c["id"], c["framework_id"], c["section_id"], c["title"], c["description"], c["full_text"]) for c in controls],
            )
        return len(controls)
    finally:
        conn.close()


def get_controls_by_framework(db_path: Path, framework_id: str) -> list[dict]:
    """Get all controls for a framework."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM controls WHERE framework_id = ?", (framework_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def insert_assignments(db_path: Path, assignments: list[dict]) -> int:
    """Insert assignment records atomically. Returns count inserted."""
    conn = get_connection(db_path)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO assignments "
                "(control_id, hub_id, confidence, in_conformal_set, is_ood, "
                "provenance, source_link_id, model_version, review_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (a["control_id"], a["hub_id"], a["confidence"],
                     a["in_conformal_set"], a["is_ood"], a["provenance"],
                     a["source_link_id"], a["model_version"], a["review_status"])
                    for a in assignments
                ],
            )
        return len(assignments)
    finally:
        conn.close()


def get_assignments_by_control(db_path: Path, control_id: str) -> list[dict]:
    """Get all assignments for a control."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM assignments WHERE control_id = ?", (control_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_assignments_by_provenance(db_path: Path, provenance: str) -> list[dict]:
    """Get all assignments with a given provenance."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM assignments WHERE provenance = ?", (provenance,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_assignments_by_status(db_path: Path, status: str) -> list[dict]:
    """Get all assignments with a given review status."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT * FROM assignments WHERE review_status = ?", (status,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_review_status(
    db_path: Path,
    assignment_id: int,
    status: str,
    reviewer: str | None = None,
    corrected_hub_id: str | None = None,
) -> None:
    """Update review status of an assignment.

    If status is 'corrected' and corrected_hub_id is provided,
    inserts a new assignment with the corrected hub.
    """
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection(db_path)
    try:
        with conn:
            conn.execute(
                "UPDATE assignments SET review_status = ?, reviewer = ?, review_date = ? "
                "WHERE id = ?",
                (status, reviewer, now, assignment_id),
            )
            if status == "corrected" and corrected_hub_id:
                row = conn.execute(
                    "SELECT * FROM assignments WHERE id = ?", (assignment_id,)
                ).fetchone()
                if row:
                    conn.execute(
                        "INSERT INTO assignments "
                        "(control_id, hub_id, confidence, in_conformal_set, is_ood, "
                        "provenance, source_link_id, model_version, review_status, "
                        "reviewer, review_date) "
                        "VALUES (?, ?, NULL, NULL, 0, ?, NULL, ?, 'accepted', ?, ?)",
                        (row["control_id"], corrected_hub_id,
                         row["provenance"] + "_corrected", row["model_version"],
                         reviewer, now),
                    )
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crosswalk_store.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/crosswalk/store.py tests/test_crosswalk_store.py
git commit -m "feat: crosswalk store with CRUD operations and atomic transactions"
```

---

### Task 5: Crosswalk Snapshot

**Files:**
- Create: `tract/crosswalk/snapshot.py`
- Test: `tests/test_crosswalk_snapshot.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_crosswalk_snapshot.py`:

```python
"""Tests for crosswalk database snapshots."""
from __future__ import annotations

import pytest


@pytest.fixture
def populated_db(tmp_path):
    from tract.crosswalk.schema import create_database
    from tract.crosswalk.store import insert_frameworks, insert_hubs

    db_path = tmp_path / "test.db"
    create_database(db_path)
    insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": None, "fetch_date": None, "control_count": 1}])
    insert_hubs(db_path, [{"id": "h1", "name": "Hub 1", "path": "R", "parent_id": None}])
    return db_path


class TestSnapshot:
    def test_creates_snapshot_record(self, populated_db) -> None:
        from tract.crosswalk.snapshot import take_snapshot

        snap = take_snapshot(populated_db, round_number=0, description="initial")
        assert snap["round_number"] == 0
        assert len(snap["db_hash"]) == 64  # SHA-256 hex

    def test_consecutive_snapshots_differ_after_change(self, populated_db) -> None:
        from tract.crosswalk.snapshot import take_snapshot
        from tract.crosswalk.store import insert_hubs

        snap1 = take_snapshot(populated_db, round_number=0, description="before")
        insert_hubs(populated_db, [{"id": "h2", "name": "Hub 2", "path": "R", "parent_id": None}])
        snap2 = take_snapshot(populated_db, round_number=1, description="after")

        assert snap1["db_hash"] != snap2["db_hash"]

    def test_identical_state_produces_same_hash(self, populated_db) -> None:
        from tract.crosswalk.snapshot import compute_db_hash

        hash1 = compute_db_hash(populated_db)
        hash2 = compute_db_hash(populated_db)
        assert hash1 == hash2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crosswalk_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement snapshot.py**

Create `tract/crosswalk/snapshot.py`:

```python
"""Pre-round database snapshots for crosswalk integrity tracking."""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)

SNAPSHOT_TABLES = ["frameworks", "controls", "hubs", "assignments"]


def compute_db_hash(db_path: Path) -> str:
    """Compute SHA-256 hash of all data in snapshot-tracked tables.

    Reads all rows from each table in deterministic order and hashes
    the JSON-serialized result.
    """
    conn = get_connection(db_path)
    try:
        state: dict[str, list[list]] = {}
        for table in SNAPSHOT_TABLES:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY rowid"  # noqa: S608
            ).fetchall()
            state[table] = [list(row) for row in rows]

        canonical = json.dumps(state, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    finally:
        conn.close()


def take_snapshot(db_path: Path, round_number: int, description: str) -> dict:
    """Record a snapshot of the current database state.

    Returns the snapshot record as a dict.
    """
    db_hash = compute_db_hash(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                "INSERT INTO snapshots (round_number, db_hash, description) "
                "VALUES (?, ?, ?)",
                (round_number, db_hash, description),
            )
            snap_id = cursor.lastrowid

        row = conn.execute(
            "SELECT * FROM snapshots WHERE id = ?", (snap_id,)
        ).fetchone()
        result = dict(row)
        logger.info(
            "Snapshot %d: round=%d, hash=%s, desc=%s",
            snap_id, round_number, db_hash[:16], description,
        )
        return result
    finally:
        conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crosswalk_snapshot.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/crosswalk/snapshot.py tests/test_crosswalk_snapshot.py
git commit -m "feat: crosswalk DB snapshots with SHA-256 hash tracking"
```

---

### Task 6: Crosswalk Export

**Files:**
- Create: `tract/crosswalk/export.py`
- Test: `tests/test_crosswalk_export.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_crosswalk_export.py`:

```python
"""Tests for crosswalk export."""
from __future__ import annotations

import csv
import json

import pytest


@pytest.fixture
def populated_db(tmp_path):
    from tract.crosswalk.schema import create_database
    from tract.crosswalk.store import (
        insert_assignments,
        insert_controls,
        insert_frameworks,
        insert_hubs,
        update_review_status,
    )

    db_path = tmp_path / "test.db"
    create_database(db_path)
    insert_frameworks(db_path, [{"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2}])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": "Desc1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2", "title": "C2", "description": "Desc2", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0, "provenance": "training_T1", "source_link_id": "link1", "model_version": "v1", "review_status": "pending"},
        {"control_id": "fw1:c2", "hub_id": "h2", "confidence": 0.7, "in_conformal_set": 0, "is_ood": 0, "provenance": "al_r1", "source_link_id": None, "model_version": "v1", "review_status": "pending"},
    ])
    update_review_status(db_path, 1, "accepted", reviewer="expert")
    return db_path


class TestExportJSON:
    def test_exports_accepted_only(self, populated_db, tmp_path) -> None:
        from tract.crosswalk.export import export_crosswalk

        out = export_crosswalk(populated_db, tmp_path / "out.json", fmt="json")
        data = json.loads(out.read_text(encoding="utf-8"))

        assert "FW1" in data
        assert len(data["FW1"]) == 1
        assert "fw1:c1" in data["FW1"]


class TestExportCSV:
    def test_exports_all_with_status(self, populated_db, tmp_path) -> None:
        from tract.crosswalk.export import export_crosswalk

        out = export_crosswalk(populated_db, tmp_path / "out.csv", fmt="csv")
        with open(out, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["review_status"] in ("accepted", "pending")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crosswalk_export.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement export.py**

Create `tract/crosswalk/export.py`:

```python
"""Export crosswalk assignments to JSON or CSV."""
from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from collections import defaultdict
from pathlib import Path

from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


def export_crosswalk(db_path: Path, output_path: Path, fmt: str = "json") -> Path:
    """Export assignments from the crosswalk database.

    JSON format exports only accepted assignments grouped by framework.
    CSV format exports all assignments with full metadata.
    """
    if fmt == "json":
        return _export_json(db_path, output_path)
    elif fmt == "csv":
        return _export_csv(db_path, output_path)
    else:
        raise ValueError(f"Unsupported format: {fmt!r}. Use 'json' or 'csv'.")


def _export_json(db_path: Path, output_path: Path) -> Path:
    """Export accepted assignments as JSON grouped by framework name."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, a.hub_id, a.confidence, a.provenance, "
            "f.name AS framework_name "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            "WHERE a.review_status = 'accepted' "
            "ORDER BY f.name, a.control_id, a.hub_id"
        ).fetchall()
    finally:
        conn.close()

    result: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        result[row["framework_name"]][row["control_id"]].append({
            "hub_id": row["hub_id"],
            "confidence": row["confidence"],
            "provenance": row["provenance"],
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, sort_keys=True, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Exported %d accepted assignments to %s", len(rows), output_path)
    return output_path


def _export_csv(db_path: Path, output_path: Path) -> Path:
    """Export all assignments as CSV with full metadata."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT a.control_id, f.name AS framework, a.hub_id, "
            "a.confidence, a.provenance, a.review_status, "
            "a.reviewer, a.review_date "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN frameworks f ON c.framework_id = f.id "
            "ORDER BY f.name, a.control_id"
        ).fetchall()
    finally:
        conn.close()

    fieldnames = ["control_id", "framework", "hub_id", "confidence",
                  "provenance", "review_status", "reviewer", "review_date"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Exported %d assignments to %s", len(rows), output_path)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_crosswalk_export.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/crosswalk/export.py tests/test_crosswalk_export.py
git commit -m "feat: crosswalk export to JSON and CSV formats"
```

---

### Task 7: Multi-Label NLL and Temperature Fitting

**Files:**
- Create: `tract/calibration/__init__.py`
- Create: `tract/calibration/temperature.py`
- Test: `tests/test_calibration_temperature.py`

- [ ] **Step 1: Create package init**

```python
# tract/calibration/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_calibration_temperature.py`:

```python
"""Tests for multi-label temperature scaling calibration."""
from __future__ import annotations

import numpy as np
import pytest


class TestMultiLabelNLL:
    def test_single_label_matches_standard_nll(self) -> None:
        from tract.calibration.temperature import multi_label_nll

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(10), size=5)
        single_label_indices = [[3], [7], [1], [0], [9]]

        nll = multi_label_nll(probs, single_label_indices)
        expected = -np.mean([np.log(probs[i, idx[0]] + 1e-10) for i, idx in enumerate(single_label_indices)])
        np.testing.assert_allclose(nll, expected, atol=1e-6)

    def test_multi_label_lower_nll_than_single(self) -> None:
        from tract.calibration.temperature import multi_label_nll

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(10), size=5)
        single = [[3], [7], [1], [0], [9]]
        multi = [[3, 5], [7, 2], [1, 4], [0, 8], [9, 6]]

        nll_single = multi_label_nll(probs, single)
        nll_multi = multi_label_nll(probs, multi)
        assert nll_multi < nll_single

    def test_empty_valid_raises(self) -> None:
        from tract.calibration.temperature import multi_label_nll

        probs = np.array([[0.5, 0.5]])
        with pytest.raises(ValueError, match="empty"):
            multi_label_nll(probs, [[]])


class TestCalibrateSimilarities:
    def test_probabilities_sum_to_one(self) -> None:
        from tract.calibration.temperature import calibrate_similarities

        sims = np.array([[0.3, 0.7, 0.1], [0.9, 0.2, 0.4]])
        probs = calibrate_similarities(sims, temperature=1.0)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)


class TestFitTemperature:
    def test_finds_temperature_in_range(self) -> None:
        from tract.calibration.temperature import fit_temperature

        rng = np.random.default_rng(42)
        n, n_hubs = 30, 50
        sims = rng.uniform(-0.3, 0.3, size=(n, n_hubs))
        gt_indices = rng.integers(0, n_hubs, size=n).tolist()
        for i in range(n):
            sims[i, gt_indices[i]] += 0.5
        valid_hub_indices = [[idx] for idx in gt_indices]

        result = fit_temperature(sims, valid_hub_indices)
        assert 0.01 <= result["temperature"] <= 5.0
        assert "nll" in result

    def test_uses_log_spaced_grid(self) -> None:
        from tract.calibration.temperature import fit_temperature

        rng = np.random.default_rng(42)
        sims = rng.uniform(0, 1, size=(10, 5))
        valid = [[0]] * 10

        result = fit_temperature(sims, valid, n_grid=10, t_min=0.01, t_max=5.0)
        assert result["temperature"] > 0


class TestFitTLofo:
    def test_sqrt_n_weighting(self) -> None:
        from tract.calibration.temperature import fit_t_lofo

        rng = np.random.default_rng(42)
        fold_sims = {}
        fold_gt = {}
        for name, n in [("A", 60), ("B", 40), ("C", 10)]:
            s = rng.uniform(-0.3, 0.3, size=(n, 50))
            gt = rng.integers(0, 50, size=n).tolist()
            for i in range(n):
                s[i, gt[i]] += 0.5
            fold_sims[name] = s
            fold_gt[name] = [[idx] for idx in gt]

        result = fit_t_lofo(fold_sims, fold_gt)
        assert 0.01 <= result["temperature"] <= 5.0
        assert "per_fold_nll" in result
        assert len(result["per_fold_nll"]) == 3
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_calibration_temperature.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement temperature.py**

Create `tract/calibration/temperature.py`:

```python
"""Temperature scaling calibration with multi-label NLL.

Two temperatures:
- T_lofo: diagnostic, fitted on pooled LOFO fold similarities with sqrt(n) weighting
- T_deploy: production, fitted on held-out traditional links from deployment model
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from tract.config import (
    PHASE1C_T_GRID_MAX,
    PHASE1C_T_GRID_MIN,
    PHASE1C_T_GRID_N,
)

logger = logging.getLogger(__name__)


def calibrate_similarities(
    similarities: NDArray[np.floating],
    temperature: float,
) -> NDArray[np.floating]:
    """Convert cosine similarities to calibrated probabilities via softmax."""
    scaled = similarities / temperature
    return softmax(scaled, axis=1)


def multi_label_nll(
    probs: NDArray[np.floating],
    valid_hub_indices: list[list[int]],
) -> float:
    """Multi-label NLL: -log(sum(P(hub) for hub in valid_hubs)) per item.

    For single-label items (list of length 1), this reduces to standard NLL.
    """
    if len(probs) != len(valid_hub_indices):
        raise ValueError(
            f"probs rows ({len(probs)}) != valid_hub_indices length ({len(valid_hub_indices)})"
        )

    nll = 0.0
    for i, valid_indices in enumerate(valid_hub_indices):
        if not valid_indices:
            raise ValueError(f"Item {i} has empty valid_hub_indices")
        p_valid = sum(probs[i, j] for j in valid_indices)
        nll -= np.log(p_valid + 1e-10)
    return float(nll / len(valid_hub_indices))


def fit_temperature(
    similarities: NDArray[np.floating],
    valid_hub_indices: list[list[int]],
    n_grid: int = PHASE1C_T_GRID_N,
    t_min: float = PHASE1C_T_GRID_MIN,
    t_max: float = PHASE1C_T_GRID_MAX,
    weights: NDArray[np.floating] | None = None,
) -> dict:
    """Find temperature T that minimizes (weighted) multi-label NLL via log-spaced grid search.

    Args:
        similarities: (n, n_hubs) cosine similarity matrix.
        valid_hub_indices: Per-item list of valid hub column indices.
        n_grid: Number of grid points.
        t_min: Minimum temperature.
        t_max: Maximum temperature.
        weights: Optional per-item weights (for fold weighting).

    Returns:
        Dict with keys: temperature, nll, grid_min_t, grid_max_t.
    """
    temperatures = np.logspace(np.log10(t_min), np.log10(t_max), n_grid)
    best_t = 1.0
    best_nll = float("inf")

    for t in temperatures:
        probs = calibrate_similarities(similarities, float(t))
        if weights is not None:
            item_nlls = []
            for i, valid_indices in enumerate(valid_hub_indices):
                p_valid = sum(probs[i, j] for j in valid_indices)
                item_nlls.append(-np.log(p_valid + 1e-10))
            nll = float(np.average(item_nlls, weights=weights))
        else:
            nll = multi_label_nll(probs, valid_hub_indices)

        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    logger.info(
        "Optimal temperature: %.4f (NLL=%.4f, %d points log-spaced in [%.3f, %.1f])",
        best_t, best_nll, n_grid, t_min, t_max,
    )
    return {"temperature": best_t, "nll": best_nll, "grid_min_t": t_min, "grid_max_t": t_max}


def fit_t_lofo(
    fold_sims: dict[str, NDArray[np.floating]],
    fold_valid_indices: dict[str, list[list[int]]],
    n_grid: int = PHASE1C_T_GRID_N,
    t_min: float = PHASE1C_T_GRID_MIN,
    t_max: float = PHASE1C_T_GRID_MAX,
) -> dict:
    """Fit T_lofo on pooled LOFO fold similarities with sqrt(n) weighting.

    Returns dict with: temperature, nll, per_fold_nll, fold_weights.
    """
    fold_names = sorted(fold_sims.keys())
    fold_sizes = {name: len(fold_sims[name]) for name in fold_names}
    sqrt_sizes = {name: math.sqrt(n) for name, n in fold_sizes.items()}
    total_sqrt = sum(sqrt_sizes.values())
    fold_weights_map = {name: sqrt_sizes[name] / total_sqrt for name in fold_names}

    all_sims = np.concatenate([fold_sims[name] for name in fold_names], axis=0)
    all_valid = []
    per_item_weights = []
    for name in fold_names:
        n = fold_sizes[name]
        w = fold_weights_map[name] / n
        all_valid.extend(fold_valid_indices[name])
        per_item_weights.extend([w] * n)

    weights_arr = np.array(per_item_weights)
    weights_arr = weights_arr / weights_arr.sum() * len(weights_arr)

    result = fit_temperature(all_sims, all_valid, n_grid, t_min, t_max, weights=weights_arr)

    per_fold_nll: dict[str, float] = {}
    for name in fold_names:
        probs = calibrate_similarities(fold_sims[name], result["temperature"])
        per_fold_nll[name] = multi_label_nll(probs, fold_valid_indices[name])

    result["per_fold_nll"] = per_fold_nll
    result["fold_weights"] = fold_weights_map

    logger.info("T_lofo=%.4f, per-fold NLL: %s", result["temperature"],
                {k: f"{v:.4f}" for k, v in per_fold_nll.items()})
    return result
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration_temperature.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tract/calibration/__init__.py tract/calibration/temperature.py tests/test_calibration_temperature.py
git commit -m "feat: multi-label NLL temperature scaling with sqrt(n) fold weighting"
```

---

### Task 8: Calibration Diagnostics (ECE, KS-test, Coverage)

**Files:**
- Create: `tract/calibration/diagnostics.py`
- Test: `tests/test_calibration_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration_diagnostics.py`:

```python
"""Tests for calibration diagnostic metrics."""
from __future__ import annotations

import numpy as np
import pytest


class TestECE:
    def test_perfect_calibration_ece_near_zero(self) -> None:
        from tract.calibration.diagnostics import expected_calibration_error

        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9] * 20)
        accuracies = np.array([0, 0, 1, 1, 1] * 20)
        ece = expected_calibration_error(confidences, accuracies, n_bins=5)
        assert ece < 0.25

    def test_ece_in_valid_range(self) -> None:
        from tract.calibration.diagnostics import expected_calibration_error

        rng = np.random.default_rng(42)
        confidences = rng.uniform(0, 1, size=100)
        accuracies = rng.integers(0, 2, size=100).astype(float)
        ece = expected_calibration_error(confidences, accuracies, n_bins=5)
        assert 0.0 <= ece <= 1.0


class TestBootstrapECE:
    def test_returns_ci(self) -> None:
        from tract.calibration.diagnostics import bootstrap_ece

        rng = np.random.default_rng(42)
        confidences = rng.uniform(0, 1, size=50)
        accuracies = rng.integers(0, 2, size=50).astype(float)

        result = bootstrap_ece(confidences, accuracies, n_bins=5, n_bootstrap=100, seed=42)
        assert "ece" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert result["ci_low"] <= result["ece"] <= result["ci_high"]


class TestKSTest:
    def test_identical_distributions_high_p(self) -> None:
        from tract.calibration.diagnostics import ks_test_similarity_distributions

        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, size=100)
        b = rng.uniform(0, 1, size=100)

        result = ks_test_similarity_distributions(a, b)
        assert result["p_value"] > 0.01

    def test_different_distributions_low_p(self) -> None:
        from tract.calibration.diagnostics import ks_test_similarity_distributions

        a = np.ones(100) * 0.9
        b = np.ones(100) * 0.1

        result = ks_test_similarity_distributions(a, b)
        assert result["p_value"] < 0.01


class TestFullRecallCoverage:
    def test_all_hubs_covered(self) -> None:
        from tract.calibration.diagnostics import full_recall_coverage

        prediction_sets = [{"h1", "h2"}, {"h3"}]
        valid_hub_sets = [frozenset({"h1", "h2"}), frozenset({"h3"})]
        assert full_recall_coverage(prediction_sets, valid_hub_sets) == 1.0

    def test_partial_coverage(self) -> None:
        from tract.calibration.diagnostics import full_recall_coverage

        prediction_sets = [{"h1"}, {"h3"}]
        valid_hub_sets = [frozenset({"h1", "h2"}), frozenset({"h3"})]
        assert full_recall_coverage(prediction_sets, valid_hub_sets) == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calibration_diagnostics.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement diagnostics.py**

Create `tract/calibration/diagnostics.py`:

```python
"""Calibration diagnostic metrics: ECE, KS-test, coverage."""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.stats import ks_2samp

from tract.config import PHASE1C_ECE_BOOTSTRAP_N, PHASE1C_ECE_N_BINS

logger = logging.getLogger(__name__)


def expected_calibration_error(
    confidences: NDArray[np.floating],
    accuracies: NDArray[np.floating],
    n_bins: int = PHASE1C_ECE_N_BINS,
) -> float:
    """Equal-width binned Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_conf = confidences[mask].mean()
        avg_acc = accuracies[mask].mean()
        ece += (n_bin / n) * abs(avg_acc - avg_conf)

    return float(ece)


def bootstrap_ece(
    confidences: NDArray[np.floating],
    accuracies: NDArray[np.floating],
    n_bins: int = PHASE1C_ECE_N_BINS,
    n_bootstrap: int = PHASE1C_ECE_BOOTSTRAP_N,
    seed: int = 42,
) -> dict:
    """Bootstrap 95% CI for ECE."""
    rng = np.random.default_rng(seed)
    n = len(confidences)
    ece_samples = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ece_samples[b] = expected_calibration_error(
            confidences[idx], accuracies[idx], n_bins=n_bins
        )

    point_ece = expected_calibration_error(confidences, accuracies, n_bins=n_bins)
    return {
        "ece": point_ece,
        "ci_low": float(np.percentile(ece_samples, 2.5)),
        "ci_high": float(np.percentile(ece_samples, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def ks_test_similarity_distributions(
    traditional_max_sims: NDArray[np.floating],
    ai_max_sims: NDArray[np.floating],
) -> dict:
    """Two-sample KS test between traditional and AI max-cosine similarity distributions."""
    stat, p_value = ks_2samp(traditional_max_sims, ai_max_sims)
    result = {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "n_traditional": len(traditional_max_sims),
        "n_ai": len(ai_max_sims),
    }

    if p_value < 0.01:
        logger.warning(
            "Domain mismatch: KS p=%.4f (stat=%.3f) between traditional (n=%d) "
            "and AI (n=%d) similarity distributions",
            p_value, stat, len(traditional_max_sims), len(ai_max_sims),
        )
    return result


def full_recall_coverage(
    prediction_sets: list[set[str]],
    valid_hub_sets: list[frozenset[str]],
) -> float:
    """Fraction of multi-label items where ALL valid hubs are in the prediction set."""
    if len(prediction_sets) != len(valid_hub_sets):
        raise ValueError("Length mismatch between prediction_sets and valid_hub_sets")
    n = len(prediction_sets)
    if n == 0:
        return 0.0
    covered = sum(
        1 for pset, vset in zip(prediction_sets, valid_hub_sets)
        if vset.issubset(pset)
    )
    return covered / n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration_diagnostics.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/calibration/diagnostics.py tests/test_calibration_diagnostics.py
git commit -m "feat: calibration diagnostics — ECE, bootstrap CI, KS-test, full-recall coverage"
```

---

### Task 9: Conformal Prediction

**Files:**
- Create: `tract/calibration/conformal.py`
- Test: `tests/test_calibration_conformal.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibration_conformal.py`:

```python
"""Tests for conformal prediction sets."""
from __future__ import annotations

import math

import numpy as np
import pytest


class TestConformalQuantile:
    def test_quantile_in_valid_range(self) -> None:
        from tract.calibration.conformal import compute_conformal_quantile

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(50), size=30)
        valid = [[rng.integers(0, 50)] for _ in range(30)]

        q = compute_conformal_quantile(probs, valid, alpha=0.10)
        assert 0.0 <= q <= 1.0

    def test_higher_alpha_yields_lower_quantile(self) -> None:
        from tract.calibration.conformal import compute_conformal_quantile

        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(20), size=50)
        valid = [[rng.integers(0, 20)] for _ in range(50)]

        q_strict = compute_conformal_quantile(probs, valid, alpha=0.05)
        q_loose = compute_conformal_quantile(probs, valid, alpha=0.20)
        assert q_strict >= q_loose


class TestPredictionSets:
    def test_prediction_set_covers_ground_truth(self) -> None:
        from tract.calibration.conformal import build_prediction_sets

        probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        hub_ids = ["h1", "h2", "h3"]

        sets = build_prediction_sets(probs, hub_ids, quantile=0.5)
        assert len(sets) == 2
        for s in sets:
            assert isinstance(s, set)


class TestConformalCoverage:
    def test_empirical_coverage(self) -> None:
        from tract.calibration.conformal import compute_conformal_coverage

        prediction_sets = [{"h1", "h2"}, {"h2", "h3"}, {"h1"}]
        valid_hub_sets = [frozenset({"h1"}), frozenset({"h2"}), frozenset({"h3"})]

        coverage = compute_conformal_coverage(prediction_sets, valid_hub_sets)
        assert coverage == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_calibration_conformal.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement conformal.py**

Create `tract/calibration/conformal.py`:

```python
"""Conformal prediction sets for hub assignment.

Provides empirical coverage guarantees (not mathematical — LOFO models != deployment model
violates exchangeability). Conservative bias is expected and documented.
"""
from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from tract.config import PHASE1C_CONFORMAL_ALPHA

logger = logging.getLogger(__name__)


def compute_conformal_quantile(
    probs: NDArray[np.floating],
    valid_hub_indices: list[list[int]],
    alpha: float = PHASE1C_CONFORMAL_ALPHA,
) -> float:
    """Compute conformal quantile from multi-label nonconformity scores.

    Nonconformity score: 1 - sum(P(hub) for hub in valid_hubs)
    Quantile: ceil((n+1)*(1-alpha))/n percentile of scores.
    """
    n = len(probs)
    scores = np.empty(n)
    for i, valid_indices in enumerate(valid_hub_indices):
        p_valid = sum(probs[i, j] for j in valid_indices)
        scores[i] = 1.0 - p_valid

    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    quantile = float(np.quantile(scores, q_level))

    logger.info(
        "Conformal quantile: %.4f (alpha=%.2f, n=%d, q_level=%.4f)",
        quantile, alpha, n, q_level,
    )
    return quantile


def build_prediction_sets(
    probs: NDArray[np.floating],
    hub_ids: list[str],
    quantile: float,
) -> list[set[str]]:
    """Build prediction sets: {hub : P(hub) >= 1-quantile}.

    Args:
        probs: (n, n_hubs) calibrated probabilities.
        hub_ids: Ordered hub IDs matching probs columns.
        quantile: Conformal quantile from compute_conformal_quantile.

    Returns:
        List of sets, one per item.
    """
    threshold = 1.0 - quantile
    result: list[set[str]] = []
    for i in range(len(probs)):
        pset = {hub_ids[j] for j in range(len(hub_ids)) if probs[i, j] >= threshold}
        result.append(pset)
    return result


def compute_conformal_coverage(
    prediction_sets: list[set[str]],
    valid_hub_sets: list[frozenset[str]],
) -> float:
    """Empirical coverage: fraction where ANY valid hub is in prediction set."""
    n = len(prediction_sets)
    if n == 0:
        return 0.0
    covered = sum(
        1 for pset, vset in zip(prediction_sets, valid_hub_sets)
        if pset & vset
    )
    return covered / n
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration_conformal.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/calibration/conformal.py tests/test_calibration_conformal.py
git commit -m "feat: conformal prediction sets with multi-label coverage"
```

---

### Task 10: OOD Detection

**Files:**
- Create: `tract/calibration/ood.py`
- Create: `tests/fixtures/ood_synthetic_texts.json`
- Test: `tests/test_calibration_ood.py`

- [ ] **Step 1: Create OOD synthetic fixture**

Create `tests/fixtures/ood_synthetic_texts.json`:

```json
[
  "The best way to prepare a sourdough starter involves feeding flour and water daily for at least seven days",
  "Jupiter's Great Red Spot is a persistent anticyclonic storm larger than Earth",
  "The goalkeeper blocked three penalty kicks during the championship match",
  "Photosynthesis converts carbon dioxide and water into glucose using sunlight",
  "The Renaissance period saw unprecedented advances in painting and sculpture across Europe",
  "Salmon migrate upstream to their natal rivers to spawn each autumn",
  "The Cretaceous-Paleogene extinction event eliminated approximately 75% of all species",
  "A standard chess opening involves controlling the center squares with pawns",
  "The process of fermentation converts sugars into alcohol and carbon dioxide",
  "Mount Everest was first summited by Edmund Hillary and Tenzing Norgay in 1953",
  "Knitting a basic scarf requires casting on approximately 30 stitches",
  "The speed of light in a vacuum is approximately 299,792 kilometers per second",
  "Beethoven composed his Ninth Symphony while almost completely deaf",
  "Plate tectonics describes the movement of Earth's lithospheric plates",
  "A traditional risotto requires constant stirring while gradually adding warm broth",
  "The International Space Station orbits Earth at approximately 28,000 km/h",
  "Victorian era literature often explored themes of social class and morality",
  "Dolphins use echolocation to navigate and find prey in murky waters",
  "The triple jump combines a hop, step, and jump in sequence",
  "Sedimentary rocks form through the accumulation and compaction of mineral particles",
  "A properly brewed espresso should take between 25 and 30 seconds to extract",
  "The Hubble Space Telescope has observed galaxies over 13 billion light-years away",
  "Origami involves folding a single sheet of paper without cutting or gluing",
  "The human skeleton contains 206 bones in adults and 270 in infants",
  "Traditional Thai cuisine balances sweet, sour, salty, and spicy flavors",
  "Comets are composed primarily of ice, dust, and rocky debris",
  "The waltz originated in the ballrooms of Vienna in the 18th century",
  "Coral reefs support approximately 25% of all known marine species",
  "A marathon covers a distance of exactly 42.195 kilometers",
  "The periodic table organizes elements by atomic number and chemical properties"
]
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_calibration_ood.py`:

```python
"""Tests for OOD detection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestOODThreshold:
    def test_threshold_is_5th_percentile(self) -> None:
        from tract.calibration.ood import compute_ood_threshold

        max_sims = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65,
                             0.55, 0.45, 0.35, 0.42, 0.52, 0.62, 0.72, 0.82, 0.88, 0.92])

        threshold = compute_ood_threshold(max_sims)
        expected = float(np.percentile(max_sims, 5))
        assert threshold == pytest.approx(expected)


class TestOODValidation:
    def test_validates_against_synthetic(self) -> None:
        from tract.calibration.ood import validate_ood_threshold

        ood_max_sims = np.array([0.05, 0.03, 0.08, 0.02, 0.04] * 6)
        threshold = 0.20

        result = validate_ood_threshold(ood_max_sims, threshold)
        assert result["separation_rate"] >= 0.9
        assert result["n_below"] == 30
        assert result["n_total"] == 30

    def test_fails_if_ood_above_threshold(self) -> None:
        from tract.calibration.ood import validate_ood_threshold

        ood_max_sims = np.array([0.5, 0.6, 0.7, 0.8, 0.9] * 6)
        threshold = 0.20

        result = validate_ood_threshold(ood_max_sims, threshold)
        assert result["separation_rate"] < 0.9


class TestFlagOOD:
    def test_flags_low_similarity_items(self) -> None:
        from tract.calibration.ood import flag_ood_items

        max_sims = np.array([0.1, 0.5, 0.05, 0.9, 0.02])
        threshold = 0.15

        flags = flag_ood_items(max_sims, threshold)
        assert flags == [True, False, True, False, True]


class TestOODFixture:
    def test_fixture_has_30_texts(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "ood_synthetic_texts.json"
        texts = json.loads(fixture.read_text(encoding="utf-8"))
        assert len(texts) == 30
        assert all(isinstance(t, str) for t in texts)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_calibration_ood.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement ood.py**

Create `tract/calibration/ood.py`:

```python
"""Out-of-distribution detection for hub assignment.

OOD items (non-security content) get flagged and routed to
hub proposal pipeline (Phase 1D) instead of receiving an assignment.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from tract.config import PHASE1C_OOD_PERCENTILE, PHASE1C_OOD_SEPARATION_GATE

logger = logging.getLogger(__name__)


def compute_ood_threshold(
    max_sims: NDArray[np.floating],
    percentile: int = PHASE1C_OOD_PERCENTILE,
) -> float:
    """Compute OOD threshold as the p-th percentile of in-distribution max cosine similarities."""
    threshold = float(np.percentile(max_sims, percentile))
    logger.info(
        "OOD threshold: %.4f (%dth percentile of %d in-distribution items)",
        threshold, percentile, len(max_sims),
    )
    return threshold


def validate_ood_threshold(
    ood_max_sims: NDArray[np.floating],
    threshold: float,
    gate: float = PHASE1C_OOD_SEPARATION_GATE,
) -> dict:
    """Validate OOD threshold against synthetic non-security texts.

    Returns dict with separation_rate, n_below, n_total, gate_passed.
    """
    n_below = int((ood_max_sims < threshold).sum())
    n_total = len(ood_max_sims)
    separation_rate = n_below / n_total if n_total > 0 else 0.0
    passed = separation_rate >= gate

    if not passed:
        logger.warning(
            "OOD gate FAILED: %.1f%% separation (need ≥%.0f%%), threshold=%.4f",
            separation_rate * 100, gate * 100, threshold,
        )
    else:
        logger.info(
            "OOD gate passed: %.1f%% separation, threshold=%.4f",
            separation_rate * 100, threshold,
        )

    return {
        "separation_rate": separation_rate,
        "n_below": n_below,
        "n_total": n_total,
        "threshold": threshold,
        "gate_passed": passed,
    }


def flag_ood_items(
    max_sims: NDArray[np.floating],
    threshold: float,
) -> list[bool]:
    """Flag items as OOD if their max cosine similarity is below threshold."""
    return [bool(sim < threshold) for sim in max_sims]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration_ood.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tract/calibration/ood.py tests/fixtures/ood_synthetic_texts.json tests/test_calibration_ood.py
git commit -m "feat: OOD detection with synthetic text validation fixture"
```

---

### Task 11: extract_similarity_matrix() in evaluate.py

**Files:**
- Modify: `tract/training/evaluate.py`
- Test: `tests/test_evaluate.py` (existing, add new tests)

- [ ] **Step 1: Write failing test**

Add to `tests/test_evaluate.py`:

```python
class TestExtractSimilarityMatrix:
    def test_returns_correct_shape(self) -> None:
        from unittest.mock import MagicMock
        from tract.training.evaluate import extract_similarity_matrix

        n_items, n_hubs, dim = 5, 10, 64
        mock_model = MagicMock()
        rng = np.random.default_rng(42)
        query_embs = rng.standard_normal((n_items, dim)).astype(np.float32)
        query_embs /= np.linalg.norm(query_embs, axis=1, keepdims=True)
        mock_model.encode.return_value = query_embs

        hub_ids = [f"hub_{i}" for i in range(n_hubs)]
        hub_embs = rng.standard_normal((n_hubs, dim)).astype(np.float32)
        hub_embs /= np.linalg.norm(hub_embs, axis=1, keepdims=True)

        EvalItem = type("EvalItem", (), {})
        items = []
        for i in range(n_items):
            item = EvalItem()
            item.control_text = f"text_{i}"
            item.valid_hub_ids = frozenset({hub_ids[i % n_hubs]})
            item.framework_name = "test_fw"
            items.append(item)

        result = extract_similarity_matrix(mock_model, items, hub_ids, hub_embs)
        assert result["sims"].shape == (n_items, n_hubs)
        assert len(result["hub_ids"]) == n_hubs
        assert len(result["gt_json"]) == n_items
        assert len(result["frameworks"]) == n_items

    def test_gt_json_is_valid_json(self) -> None:
        import json
        from unittest.mock import MagicMock
        from tract.training.evaluate import extract_similarity_matrix

        mock_model = MagicMock()
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((2, 32)).astype(np.float32)
        embs /= np.linalg.norm(embs, axis=1, keepdims=True)
        mock_model.encode.return_value = embs

        hub_ids = ["h1", "h2", "h3"]
        hub_embs = rng.standard_normal((3, 32)).astype(np.float32)
        hub_embs /= np.linalg.norm(hub_embs, axis=1, keepdims=True)

        EvalItem = type("EvalItem", (), {})
        item1 = EvalItem()
        item1.control_text = "ctrl1"
        item1.valid_hub_ids = frozenset({"h1", "h2"})
        item1.framework_name = "fw1"
        item2 = EvalItem()
        item2.control_text = "ctrl2"
        item2.valid_hub_ids = frozenset({"h3"})
        item2.framework_name = "fw1"

        result = extract_similarity_matrix(mock_model, [item1, item2], hub_ids, hub_embs)

        for gt_str in result["gt_json"]:
            parsed = json.loads(gt_str)
            assert isinstance(parsed, list)
            assert all(isinstance(h, str) for h in parsed)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py::TestExtractSimilarityMatrix -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement extract_similarity_matrix()**

Add to `tract/training/evaluate.py` after the `rank_hubs_by_similarity` function (after line 60):

```python
def extract_similarity_matrix(
    model: Any,
    eval_items: list[Any],
    hub_ids: list[str],
    hub_embs: NDArray[np.floating[Any]],
) -> dict[str, Any]:
    """Extract full similarity matrix for calibration.

    Args:
        model: SentenceTransformer (or anything with .encode()).
        eval_items: Items with .control_text, .valid_hub_ids, .framework_name.
        hub_ids: Canonical sorted hub IDs matching hub_embs rows.
        hub_embs: Pre-computed hub embeddings, shape (n_hubs, dim), L2-normalized.

    Returns:
        Dict with keys: sims (n_eval, n_hubs), hub_ids, gt_json, frameworks.
    """
    import json

    control_texts = [item.control_text for item in eval_items]
    query_embs = model.encode(
        control_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=128,
    )

    sims = query_embs @ hub_embs.T

    gt_json = [
        json.dumps(sorted(item.valid_hub_ids))
        for item in eval_items
    ]
    frameworks = [item.framework_name for item in eval_items]

    return {
        "sims": sims.astype(np.float32),
        "hub_ids": hub_ids,
        "gt_json": gt_json,
        "frameworks": frameworks,
    }
```

Also add `import json` to the top of the file if not present (it's not currently imported). Actually, move the `import json` inside the function to keep it local since it's only used here.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_evaluate.py::TestExtractSimilarityMatrix -v`
Expected: All PASS

- [ ] **Step 5: Run full evaluate test suite**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tract/training/evaluate.py tests/test_evaluate.py
git commit -m "feat: extract_similarity_matrix for calibration pipeline"
```

---

### Task 12: Deployment Model — Loading, Holdout Selection, Adapter

**Files:**
- Create: `tract/active_learning/__init__.py`
- Create: `tract/active_learning/deploy.py`
- Test: `tests/test_active_learning_deploy.py`

- [ ] **Step 1: Create package init**

```python
# tract/active_learning/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_active_learning_deploy.py`:

```python
"""Tests for deployment model utilities."""
from __future__ import annotations

import numpy as np
import pytest

from tract.training.data_quality import QualityTier, TieredLink


class TestSelectHoldout:
    def test_returns_correct_counts(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(500)
        ]

        cal, canary, remaining = select_holdout(links, n_cal=420, n_canary=20, seed=42)
        assert len(cal) == 420
        assert len(canary) == 20
        assert len(remaining) == 60

    def test_deterministic_with_seed(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "CWE", "section_name": f"s{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(500)
        ]

        cal1, _, _ = select_holdout(links, seed=42)
        cal2, _, _ = select_holdout(links, seed=42)
        assert [l.link["cre_id"] for l in cal1] == [l.link["cre_id"] for l in cal2]

    def test_no_overlap(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(500)
        ]

        cal, canary, remaining = select_holdout(links, seed=42)
        cal_ids = {l.link["cre_id"] for l in cal}
        canary_ids = {l.link["cre_id"] for l in canary}
        remaining_ids = {l.link["cre_id"] for l in remaining}

        assert cal_ids & canary_ids == set()
        assert cal_ids & remaining_ids == set()
        assert canary_ids & remaining_ids == set()

    def test_excludes_ai_frameworks(self) -> None:
        from tract.active_learning.deploy import select_holdout

        links = [
            TieredLink(link={"cre_id": "h1", "standard_name": "MITRE ATLAS", "section_name": "s1", "link_type": "LinkedTo"}, tier=QualityTier.T1_AI),
            TieredLink(link={"cre_id": "h2", "standard_name": "ASVS", "section_name": "s2", "link_type": "LinkedTo"}, tier=QualityTier.T1),
        ]

        cal, canary, remaining = select_holdout(links, n_cal=1, n_canary=0, seed=42)
        assert all(l.link["standard_name"] not in {"MITRE ATLAS"} for l in cal)


class TestHoldoutToEval:
    def test_converts_to_eval_dict(self) -> None:
        from tract.active_learning.deploy import holdout_to_eval

        link = TieredLink(
            link={"cre_id": "236-712", "standard_name": "ASVS", "section_name": "Validate all input", "section_id": "V5.1.1", "link_type": "LinkedTo"},
            tier=QualityTier.T1,
        )

        result = holdout_to_eval(link)
        assert result["valid_hub_ids"] == frozenset({"236-712"})
        assert "Validate all input" in result["control_text"]
        assert result["framework"] == "ASVS"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_active_learning_deploy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement deploy.py**

Create `tract/active_learning/deploy.py`:

```python
"""Deployment model training, loading, and inference utilities."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from tract.config import (
    PHASE1C_HOLDOUT_CALIBRATION,
    PHASE1C_HOLDOUT_CANARY,
)
from tract.training.data_quality import AI_FRAMEWORK_NAMES, TieredLink

logger = logging.getLogger(__name__)


def select_holdout(
    tiered_links: list[TieredLink],
    n_cal: int = PHASE1C_HOLDOUT_CALIBRATION,
    n_canary: int = PHASE1C_HOLDOUT_CANARY,
    seed: int = 42,
) -> tuple[list[TieredLink], list[TieredLink], list[TieredLink]]:
    """Select holdout links for calibration and canaries.

    Only traditional (non-AI) links are eligible for holdout.

    Returns:
        (calibration_links, canary_links, remaining_links)
    """
    traditional = [l for l in tiered_links if l.link.get("standard_name", "") not in AI_FRAMEWORK_NAMES]
    ai_links = [l for l in tiered_links if l.link.get("standard_name", "") in AI_FRAMEWORK_NAMES]

    n_total = n_cal + n_canary
    if len(traditional) < n_total:
        raise ValueError(
            f"Not enough traditional links for holdout: need {n_total}, have {len(traditional)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(traditional))

    holdout_indices = set(indices[:n_total].tolist())
    cal_indices = indices[:n_cal]
    canary_indices = indices[n_cal:n_total]

    calibration = [traditional[i] for i in cal_indices]
    canaries = [traditional[i] for i in canary_indices]
    remaining_trad = [traditional[i] for i in range(len(traditional)) if i not in holdout_indices]

    remaining = remaining_trad + ai_links

    fw_counts: dict[str, int] = {}
    for link in calibration:
        fw = link.link.get("standard_name", "unknown")
        fw_counts[fw] = fw_counts.get(fw, 0) + 1
    max_fw_pct = max(fw_counts.values()) / n_cal if fw_counts else 0
    if max_fw_pct > 0.5:
        logger.warning(
            "Holdout dominated by single framework: %s (%.0f%%)",
            max(fw_counts, key=fw_counts.get), max_fw_pct * 100,
        )

    logger.info(
        "Holdout: %d calibration + %d canary from %d traditional links, %d remaining",
        len(calibration), len(canaries), len(traditional), len(remaining),
    )
    return calibration, canaries, remaining


def holdout_to_eval(link: TieredLink) -> dict:
    """Convert a TieredLink to an eval-compatible record.

    Returns dict with keys: control_text, framework, valid_hub_ids.
    """
    control_text = link.link.get("section_name") or link.link.get("section_id", "")
    section_id = link.link.get("section_id", "")
    if section_id and section_id not in control_text:
        control_text = f"{section_id}: {control_text}"

    return {
        "control_text": control_text,
        "framework": link.link.get("standard_name", ""),
        "valid_hub_ids": frozenset({link.link["cre_id"]}),
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_active_learning_deploy.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add tract/active_learning/__init__.py tract/active_learning/deploy.py tests/test_active_learning_deploy.py
git commit -m "feat: holdout selection and TieredLink-to-eval adapter for deployment model"
```

---

### Task 13: Global Threshold (Multi-Label F1)

**Files:**
- Modify: `tract/calibration/temperature.py` (add function)
- Test: `tests/test_calibration_temperature.py` (add tests)

- [ ] **Step 1: Write failing test**

Add to `tests/test_calibration_temperature.py`:

```python
class TestGlobalThreshold:
    def test_threshold_in_valid_range(self) -> None:
        from tract.calibration.temperature import find_global_threshold

        rng = np.random.default_rng(42)
        n, n_hubs = 30, 10
        sims = rng.uniform(0, 1, size=(n, n_hubs))
        gt_indices = rng.integers(0, n_hubs, size=n).tolist()
        for i in range(n):
            sims[i, gt_indices[i]] += 0.3
        valid = [[idx] for idx in gt_indices]

        result = find_global_threshold(sims, valid, temperature=1.0)
        assert 0.0 < result["threshold"] < 1.0
        assert result["f1"] >= 0.0

    def test_multi_label_tp_if_any_valid(self) -> None:
        from tract.calibration.temperature import find_global_threshold

        sims = np.array([[0.9, 0.8, 0.1], [0.1, 0.1, 0.9]])
        valid = [[0, 1], [2]]

        result = find_global_threshold(sims, valid, temperature=1.0)
        assert result["f1"] > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_calibration_temperature.py::TestGlobalThreshold -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement find_global_threshold**

Add to `tract/calibration/temperature.py`:

```python
def find_global_threshold(
    similarities: NDArray[np.floating],
    valid_hub_indices: list[list[int]],
    temperature: float,
    n_thresholds: int = 200,
) -> dict:
    """Find global probability threshold at max-F1 for multi-label assignment.

    TP: predicted hub in valid_hubs
    FP: predicted hub not in valid_hubs
    FN: no predicted hub in valid_hubs
    """
    probs = calibrate_similarities(similarities, temperature)
    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5

    for t in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for i, valid_indices in enumerate(valid_hub_indices):
            valid_set = set(valid_indices)
            any_hit = False
            for j in range(probs.shape[1]):
                predicted = probs[i, j] >= t
                is_valid = j in valid_set
                if predicted and is_valid:
                    tp += 1
                    any_hit = True
                elif predicted and not is_valid:
                    fp += 1
            if not any_hit:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    logger.info("Global threshold: %.4f (F1=%.4f)", best_threshold, best_f1)
    return {"threshold": best_threshold, "f1": best_f1}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_calibration_temperature.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/calibration/temperature.py tests/test_calibration_temperature.py
git commit -m "feat: multi-label global threshold at max-F1"
```

---

### Task 14: Review JSON Generation and Ingestion

**Files:**
- Create: `tract/active_learning/review.py`
- Test: `tests/test_active_learning_review.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_active_learning_review.py`:

```python
"""Tests for review JSON generation and ingestion."""
from __future__ import annotations

import json

import numpy as np
import pytest


class TestGenerateReviewJSON:
    def test_generates_valid_schema(self, tmp_path) -> None:
        from tract.active_learning.review import generate_review_json

        items = [
            {"control_id": "fw1:c1", "framework": "FW1", "control_text": "Control text 1"},
            {"control_id": "fw1:c2", "framework": "FW1", "control_text": "Control text 2"},
        ]
        hub_ids = ["h1", "h2", "h3"]
        sims = np.array([[0.9, 0.3, 0.1], [0.2, 0.8, 0.05]])
        probs = np.array([[0.7, 0.2, 0.1], [0.15, 0.75, 0.10]])
        conformal_sets = [{"h1"}, {"h2"}]
        ood_flags = [False, False]

        out = tmp_path / "review.json"
        generate_review_json(
            items=items,
            hub_ids=hub_ids,
            probs=probs,
            conformal_sets=conformal_sets,
            ood_flags=ood_flags,
            threshold=0.5,
            temperature=0.42,
            model_version="abc123",
            round_number=1,
            output_path=out,
        )

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["round"] == 1
        assert data["temperature"] == 0.42
        assert len(data["items"]) == 2

        item = data["items"][0]
        assert "control_id" in item
        assert "predictions" in item
        assert "is_ood" in item
        assert "auto_accept_candidate" in item
        assert item["review"] is None

    def test_auto_accept_if_above_threshold_and_in_conformal(self, tmp_path) -> None:
        from tract.active_learning.review import generate_review_json

        items = [{"control_id": "c1", "framework": "F", "control_text": "T"}]
        hub_ids = ["h1"]
        probs = np.array([[0.9]])
        conformal_sets = [{"h1"}]
        ood_flags = [False]

        out = tmp_path / "r.json"
        generate_review_json(items, hub_ids, probs, conformal_sets, ood_flags,
                             threshold=0.5, temperature=1.0, model_version="v1",
                             round_number=1, output_path=out)

        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["items"][0]["auto_accept_candidate"] is True


class TestIngestReviews:
    def test_ingest_accepted(self) -> None:
        from tract.active_learning.review import ingest_reviews
        from tract.training.data_quality import QualityTier

        review_data = {
            "round": 1,
            "items": [
                {
                    "control_id": "fw1:c1",
                    "framework": "FW1",
                    "control_text": "Control text",
                    "predictions": [{"hub_id": "h1", "confidence": 0.9, "in_conformal_set": True}],
                    "is_ood": False,
                    "review": {"status": "accepted", "corrected_hub_id": None, "notes": ""},
                },
                {
                    "control_id": "fw1:c2",
                    "framework": "FW1",
                    "control_text": "Control text 2",
                    "predictions": [{"hub_id": "h2", "confidence": 0.7, "in_conformal_set": False}],
                    "is_ood": False,
                    "review": {"status": "rejected", "corrected_hub_id": None, "notes": "wrong"},
                },
            ],
        }

        accepted = ingest_reviews(review_data)
        assert len(accepted) == 1
        assert accepted[0].tier == QualityTier.AL
        assert accepted[0].link["cre_id"] == "h1"
        assert accepted[0].link["link_type"] == "active_learning"

    def test_ingest_corrected(self) -> None:
        from tract.active_learning.review import ingest_reviews

        review_data = {
            "round": 1,
            "items": [
                {
                    "control_id": "fw1:c1",
                    "framework": "FW1",
                    "control_text": "Control text",
                    "predictions": [{"hub_id": "h1", "confidence": 0.9, "in_conformal_set": True}],
                    "is_ood": False,
                    "review": {"status": "corrected", "corrected_hub_id": "h5", "notes": ""},
                },
            ],
        }

        accepted = ingest_reviews(review_data)
        assert len(accepted) == 1
        assert accepted[0].link["cre_id"] == "h5"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_active_learning_review.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement review.py**

Create `tract/active_learning/review.py`:

```python
"""Review JSON generation and ingestion for active learning."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from tract.training.data_quality import QualityTier, TieredLink

logger = logging.getLogger(__name__)


def generate_review_json(
    items: list[dict],
    hub_ids: list[str],
    probs: NDArray[np.floating],
    conformal_sets: list[set[str]],
    ood_flags: list[bool],
    threshold: float,
    temperature: float,
    model_version: str,
    round_number: int,
    output_path: Path,
    top_k: int = 10,
) -> Path:
    """Generate review.json for expert review.

    Canary items should be pre-interleaved in the items list — this function
    does not distinguish canaries from real predictions.
    """
    review_items = []
    for i, item in enumerate(items):
        predictions = []
        ranked_indices = np.argsort(probs[i])[::-1][:top_k]
        for j in ranked_indices:
            hub_id = hub_ids[j]
            predictions.append({
                "hub_id": hub_id,
                "confidence": round(float(probs[i, j]), 4),
                "in_conformal_set": hub_id in conformal_sets[i],
            })

        top_pred = predictions[0] if predictions else None
        auto_accept = (
            not ood_flags[i]
            and top_pred is not None
            and top_pred["confidence"] >= threshold
            and top_pred["in_conformal_set"]
        )

        review_items.append({
            "control_id": item["control_id"],
            "framework": item["framework"],
            "control_text": item["control_text"],
            "predictions": predictions,
            "is_ood": ood_flags[i],
            "auto_accept_candidate": auto_accept,
            "review": None,
        })

    data = {
        "round": round_number,
        "model_version": model_version,
        "temperature": temperature,
        "threshold": threshold,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "items": review_items,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=False)
            f.write("\n")
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Generated review.json: %d items, round %d", len(review_items), round_number)
    return output_path


def ingest_reviews(review_data: dict) -> list[TieredLink]:
    """Ingest reviewed predictions, returning accepted/corrected as TieredLinks.

    Accepted items use the model's top prediction hub.
    Corrected items use the expert's corrected_hub_id.
    Rejected items are excluded.
    """
    round_number = review_data["round"]
    accepted: list[TieredLink] = []

    for item in review_data["items"]:
        review = item.get("review")
        if review is None:
            continue

        status = review["status"]
        if status == "rejected":
            continue

        if status == "corrected":
            hub_id = review["corrected_hub_id"]
            if not hub_id:
                logger.warning("Corrected item %s has no corrected_hub_id, skipping", item["control_id"])
                continue
        elif status == "accepted":
            preds = item.get("predictions", [])
            if not preds:
                continue
            hub_id = preds[0]["hub_id"]
        else:
            logger.warning("Unknown review status: %s", status)
            continue

        link_dict = {
            "standard_name": item["framework"],
            "section_name": item["control_text"],
            "section_id": item["control_id"].split(":", 1)[-1] if ":" in item["control_id"] else item["control_id"],
            "cre_id": hub_id,
            "link_type": "active_learning",
            "al_round": str(round_number),
        }
        accepted.append(TieredLink(link=link_dict, tier=QualityTier.AL))

    logger.info(
        "Ingested round %d: %d accepted/corrected out of %d reviewed",
        round_number, len(accepted), len(review_data["items"]),
    )
    return accepted
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_active_learning_review.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/active_learning/review.py tests/test_active_learning_review.py
git commit -m "feat: review JSON generation and ingestion for active learning"
```

---

### Task 15: Canary Item Management

**Files:**
- Create: `tract/active_learning/canary.py`
- Test: `tests/test_active_learning_canary.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_active_learning_canary.py`:

```python
"""Tests for canary item management."""
from __future__ import annotations

import pytest


class TestSelectAICanaries:
    def test_selects_correct_count(self) -> None:
        from tract.active_learning.canary import select_ai_canaries

        controls = [{"control_id": f"c{i}", "framework": "CSA AICM", "control_text": f"text {i}"} for i in range(100)]
        canaries = select_ai_canaries(controls, n=20, seed=42)
        assert len(canaries) == 20

    def test_deterministic(self) -> None:
        from tract.active_learning.canary import select_ai_canaries

        controls = [{"control_id": f"c{i}", "framework": "CSA AICM", "control_text": f"text {i}"} for i in range(100)]
        c1 = select_ai_canaries(controls, seed=42)
        c2 = select_ai_canaries(controls, seed=42)
        assert [c["control_id"] for c in c1] == [c["control_id"] for c in c2]


class TestEvaluateCanaries:
    def test_perfect_accuracy(self) -> None:
        from tract.active_learning.canary import evaluate_canary_accuracy

        canary_labels = {"c1": frozenset({"h1"}), "c2": frozenset({"h2"})}
        review_items = [
            {"control_id": "c1", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h1"}]},
            {"control_id": "c2", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h2"}]},
        ]

        accuracy = evaluate_canary_accuracy(canary_labels, review_items)
        assert accuracy == 1.0

    def test_partial_accuracy(self) -> None:
        from tract.active_learning.canary import evaluate_canary_accuracy

        canary_labels = {"c1": frozenset({"h1"}), "c2": frozenset({"h2"})}
        review_items = [
            {"control_id": "c1", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h1"}]},
            {"control_id": "c2", "review": {"status": "accepted"}, "predictions": [{"hub_id": "h_wrong"}]},
        ]

        accuracy = evaluate_canary_accuracy(canary_labels, review_items)
        assert accuracy == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_active_learning_canary.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement canary.py**

Create `tract/active_learning/canary.py`:

```python
"""Canary item management for active learning quality measurement.

Two canary types:
1. Pre-labeled AI controls (from unmapped pool, expert labels BEFORE seeing predictions)
2. Traditional holdout canaries (from 20-item partition, ground truth from OpenCRE)
"""
from __future__ import annotations

import logging

import numpy as np

from tract.config import PHASE1C_N_AI_CANARIES

logger = logging.getLogger(__name__)


def select_ai_canaries(
    unmapped_controls: list[dict],
    n: int = PHASE1C_N_AI_CANARIES,
    seed: int = 42,
) -> list[dict]:
    """Select n controls from the unmapped pool for canary pre-labeling.

    Returns the selected controls (expert must label these before AL begins).
    """
    if len(unmapped_controls) < n:
        raise ValueError(
            f"Not enough unmapped controls for canaries: need {n}, have {len(unmapped_controls)}"
        )

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(unmapped_controls), size=n, replace=False)
    selected = [unmapped_controls[i] for i in sorted(indices)]

    logger.info("Selected %d AI canary controls for pre-labeling", len(selected))
    return selected


def evaluate_canary_accuracy(
    canary_labels: dict[str, frozenset[str]],
    review_items: list[dict],
) -> float:
    """Evaluate expert accuracy on canary items.

    Compares expert's accepted/corrected hub against the pre-labeled ground truth.
    Only counts items the expert actually reviewed (not skipped).
    """
    canary_ids = set(canary_labels.keys())
    correct = 0
    total = 0

    for item in review_items:
        cid = item["control_id"]
        if cid not in canary_ids:
            continue

        review = item.get("review")
        if review is None or review["status"] == "rejected":
            continue

        total += 1
        if review["status"] == "corrected":
            assigned_hub = review.get("corrected_hub_id", "")
        else:
            preds = item.get("predictions", [])
            assigned_hub = preds[0]["hub_id"] if preds else ""

        if assigned_hub in canary_labels[cid]:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    logger.info("Canary accuracy: %.1f%% (%d/%d)", accuracy * 100, correct, total)
    return accuracy
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_active_learning_canary.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/active_learning/canary.py tests/test_active_learning_canary.py
git commit -m "feat: canary item selection and accuracy evaluation"
```

---

### Task 16: AL Stopping Criteria

**Files:**
- Create: `tract/active_learning/stopping.py`
- Test: `tests/test_active_learning_stopping.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_active_learning_stopping.py`:

```python
"""Tests for active learning stopping criteria."""
from __future__ import annotations

import pytest


class TestEvaluateStopping:
    def test_all_met_returns_stop(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.85,
            canary_accuracy=0.90,
            unique_hubs_accepted=60,
        )
        assert result["should_stop"] is True
        assert all(result["criteria_met"].values())

    def test_low_acceptance_returns_continue(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.70,
            canary_accuracy=0.90,
            unique_hubs_accepted=60,
        )
        assert result["should_stop"] is False
        assert result["criteria_met"]["acceptance_rate"] is False

    def test_low_canary_returns_continue(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.85,
            canary_accuracy=0.80,
            unique_hubs_accepted=60,
        )
        assert result["should_stop"] is False

    def test_low_diversity_returns_continue(self) -> None:
        from tract.active_learning.stopping import evaluate_stopping_criteria

        result = evaluate_stopping_criteria(
            acceptance_rate=0.85,
            canary_accuracy=0.90,
            unique_hubs_accepted=30,
        )
        assert result["should_stop"] is False
        assert result["criteria_met"]["hub_diversity"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_active_learning_stopping.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement stopping.py**

Create `tract/active_learning/stopping.py`:

```python
"""Active learning stopping criteria evaluation."""
from __future__ import annotations

import logging

from tract.config import (
    PHASE1C_AL_ACCEPTANCE_GATE,
    PHASE1C_AL_CANARY_ACCURACY_GATE,
    PHASE1C_AL_HUB_DIVERSITY_GATE,
)

logger = logging.getLogger(__name__)


def evaluate_stopping_criteria(
    acceptance_rate: float,
    canary_accuracy: float,
    unique_hubs_accepted: int,
    acceptance_gate: float = PHASE1C_AL_ACCEPTANCE_GATE,
    canary_gate: float = PHASE1C_AL_CANARY_ACCURACY_GATE,
    diversity_gate: int = PHASE1C_AL_HUB_DIVERSITY_GATE,
) -> dict:
    """Evaluate whether AL loop should stop.

    All three criteria must be met:
    - Acceptance rate > gate (expert trusts predictions)
    - AI canary accuracy >= gate (expert applies judgment)
    - Hub diversity >= gate (model not concentrating on easy hubs)
    """
    criteria_met = {
        "acceptance_rate": acceptance_rate > acceptance_gate,
        "canary_accuracy": canary_accuracy >= canary_gate,
        "hub_diversity": unique_hubs_accepted >= diversity_gate,
    }
    should_stop = all(criteria_met.values())

    logger.info(
        "Stopping criteria: acceptance=%.1f%% (%s), canary=%.1f%% (%s), "
        "diversity=%d (%s) → %s",
        acceptance_rate * 100, "PASS" if criteria_met["acceptance_rate"] else "FAIL",
        canary_accuracy * 100, "PASS" if criteria_met["canary_accuracy"] else "FAIL",
        unique_hubs_accepted, "PASS" if criteria_met["hub_diversity"] else "FAIL",
        "STOP" if should_stop else "CONTINUE",
    )

    return {
        "should_stop": should_stop,
        "criteria_met": criteria_met,
        "values": {
            "acceptance_rate": acceptance_rate,
            "canary_accuracy": canary_accuracy,
            "unique_hubs_accepted": unique_hubs_accepted,
        },
        "gates": {
            "acceptance_rate": acceptance_gate,
            "canary_accuracy": canary_gate,
            "hub_diversity": diversity_gate,
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_active_learning_stopping.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add tract/active_learning/stopping.py tests/test_active_learning_stopping.py
git commit -m "feat: AL stopping criteria — acceptance, canary accuracy, hub diversity"
```

---

### Task 17: Full Test Suite Verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run complete test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests PASS (existing + new)

- [ ] **Step 2: Type check new modules**

Run: `python -m mypy tract/calibration/ tract/active_learning/ tract/crosswalk/ --ignore-missing-imports`
Expected: No errors (or only pre-existing)

- [ ] **Step 3: Verify all imports work end-to-end**

Run:
```bash
python -c "
from tract.calibration.temperature import calibrate_similarities, multi_label_nll, fit_temperature, fit_t_lofo, find_global_threshold
from tract.calibration.conformal import compute_conformal_quantile, build_prediction_sets, compute_conformal_coverage
from tract.calibration.ood import compute_ood_threshold, validate_ood_threshold, flag_ood_items
from tract.calibration.diagnostics import expected_calibration_error, bootstrap_ece, ks_test_similarity_distributions, full_recall_coverage
from tract.active_learning.deploy import select_holdout, holdout_to_eval
from tract.active_learning.review import generate_review_json, ingest_reviews
from tract.active_learning.canary import select_ai_canaries, evaluate_canary_accuracy
from tract.active_learning.stopping import evaluate_stopping_criteria
from tract.crosswalk.schema import create_database, get_connection
from tract.crosswalk.store import insert_hubs, insert_frameworks, insert_controls, insert_assignments
from tract.crosswalk.export import export_crosswalk
from tract.crosswalk.snapshot import take_snapshot, compute_db_hash
from tract.training.data_quality import QualityTier
from tract.training.data import TIER_PRIORITY
from tract.training.evaluate import extract_similarity_matrix
print('All imports OK')
print(f'QualityTier.AL = {QualityTier.AL.value}')
print(f'TIER_PRIORITY = {TIER_PRIORITY}')
"
```
Expected: `All imports OK`, `QualityTier.AL = AL`, `TIER_PRIORITY = {'T1': 0, 'T1-AI': 1, 'T3': 2, 'AL': 3}`

- [ ] **Step 4: Commit any fixes**

If any fixes were needed, commit them with a descriptive message.

---

## Self-Review Checklist

### Spec Coverage

| Spec Section | Task(s) |
|---|---|
| 2.1 New Modules (calibration/) | 7, 8, 9, 10 |
| 2.1 New Modules (active_learning/) | 12, 14, 15, 16 |
| 2.1 New Modules (crosswalk/) | 3, 4, 5, 6 |
| 2.2 Modified: data_quality.py QualityTier.AL | 2 |
| 2.2 Modified: data.py TIER_PRIORITY | 2 |
| 2.2 Modified: evaluate.py extract_similarity_matrix | 11 |
| 2.3 T_lofo/T_deploy | 7 (fit_t_lofo, fit_temperature) |
| 3 Similarity extraction | 11 |
| 4.1 Multi-label NLL | 7 |
| 4.2 T_lofo fitting | 7 |
| 4.3 T_deploy fitting | 7 (same fit_temperature function) |
| 4.4 ECE gate | 8 |
| 4.5 Conformal prediction | 9 |
| 4.6 Global threshold | 13 |
| 4.7 OOD detection | 10 |
| 5.1-5.2 Deployment model training | 12 (holdout selection); training itself reuses existing loop.py |
| 5.4 Unmapped controls | 12 (adapter); control loading from existing parsers |
| 5.5 Holdout adapter | 12 |
| 5.6 review.json schema | 14 |
| 6.1 Canaries | 15 |
| 6.2 Round structure | 14 (generate + ingest) |
| 6.3 Stopping criteria | 16 |
| 7.1 Schema | 3 |
| 7.2 Population | 4 |
| 7.3 Atomic operations | 4 |
| 7.4 Export | 6 |
| 8 Execution flow | Orchestration scripts (T0-T4) — built from these modules at runtime |
| 9 Quality gates | Constants in Task 1; logic in Tasks 7-10, 16 |
| Snapshots | 5 |

### Not Covered (by design)

- **Orchestration scripts (T0-T4 GPU/CPU flow)**: These compose the modules built here into RunPod scripts. They are integration code that runs on the GPU pod and depends on the deployed environment. They should be written after this library code is complete and tested.
- **train_deployment_model() in orchestrate.py**: Composes existing `build_training_pairs()`, `pairs_to_dataset()`, and `train_model()` with `select_holdout()`. Thin wrapper — best written during integration.
- **load_fold_model()**: Loads saved LoRA adapters from Phase 1B results. Depends on actual saved model paths. Best written during integration with real artifacts.
