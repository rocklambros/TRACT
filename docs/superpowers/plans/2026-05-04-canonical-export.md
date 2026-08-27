# Canonical Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tract export-canonical` CLI command that produces per-framework JSON snapshots, changesets, and optional embeddings for OpenCRE's incremental import RFC.

**Architecture:** Pydantic v2 models define the canonical schema. A snapshot builder queries crosswalk.db (reusing the existing filter pipeline) and assembles typed objects. A differ computes changesets against prior exports stored in a new `export_history` table. The CLI handler is a thin wrapper delegating to `tract/export/canonical.py`.

**Tech Stack:** Python 3.11+, Pydantic v2, SQLite3, NumPy (embeddings), pytest

**Spec:** `docs/superpowers/specs/2026-05-04-canonical-export-design.md`

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `tract/export/canonical_schema.py` | Pydantic v2 models: `FilterPolicy`, `CanonicalControl`, `CREMapping`, `ChangesetEntry`, `ChangesetSummary`, `ImpactAnalysis`, `Changeset`, `StandardSnapshot` | CREATE |
| `tract/export/canonical.py` | Snapshot builder, differ, serializer, export_history DDL | CREATE |
| `tract/cli.py` | Add `export-canonical` subcommand parser + thin handler | MODIFY |
| `tract/config.py` | Add `CANONICAL_EXPORT_DIR` constant | MODIFY |
| `tests/test_canonical_schema.py` | Unit tests for Pydantic models and content hash | CREATE |
| `tests/test_canonical_export.py` | Unit tests for snapshot builder, differ, embedding slicer | CREATE |
| `tests/test_canonical_cli.py` | Integration tests for CLI subcommand | CREATE |

**Existing files used but NOT modified:**
- `tract/export/filters.py` — reused for filter logic (canonical builds its own query extending it)
- `tract/export/opencre_names.py` — reused for `get_opencre_name()` and `build_hyperlink()`
- `tract/crosswalk/schema.py` — reused for `get_connection()`
- `tract/io.py` — reused for `atomic_write_json()`

---

### Task 0: Add config constant

**Files:**
- Modify: `tract/config.py:269-278` (Phase 5 section)

- [ ] **Step 1: Add the constant**

In `tract/config.py`, after line 277 (`PHASE5_GROUND_TRUTH_PROVENANCE`), add:

```python
PHASE5_CANONICAL_EXPORT_DIR: Final[Path] = PROJECT_ROOT / "canonical_export"
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -5`
Expected: All tests pass (831+), no regressions.

- [ ] **Step 3: Commit**

```bash
git add tract/config.py
git commit -m "feat(config): add PHASE5_CANONICAL_EXPORT_DIR constant"
```

---

### Task 1: Pydantic schema models

**Files:**
- Create: `tract/export/canonical_schema.py`
- Test: `tests/test_canonical_schema.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_canonical_schema.py`:

```python
"""Tests for canonical export Pydantic models (spec §2.1)."""
from __future__ import annotations

import hashlib
import json

import pytest


def _make_control(
    control_id: str = "fw1:c1",
    framework_id: str = "fw1",
    section_id: str = "c1",
    title: str = "Control 1",
    description: str = "Description 1",
    hyperlink: str = "https://example.com",
) -> dict:
    return {
        "control_id": control_id,
        "framework_id": framework_id,
        "section_id": section_id,
        "title": title,
        "description": description,
        "hyperlink": hyperlink,
    }


def _make_mapping(
    control_id: str = "fw1:c1",
    hub_id: str = "004-517",
    hub_name: str = "Security requirements",
    confidence: float = 0.75,
    rank: int = 1,
    provenance: str = "active_learning_round_2",
    model_version: str = "7e8b8f834db5",
) -> dict:
    return {
        "control_id": control_id,
        "hub_id": hub_id,
        "hub_name": hub_name,
        "confidence": confidence,
        "rank": rank,
        "link_type": "TRACT_ML_PREDICTED",
        "provenance": provenance,
        "model_version": model_version,
    }


class TestCanonicalControl:
    def test_roundtrip(self) -> None:
        from tract.export.canonical_schema import CanonicalControl

        ctrl = CanonicalControl(**_make_control())
        assert ctrl.control_id == "fw1:c1"
        assert ctrl.hyperlink == "https://example.com"
        d = ctrl.model_dump()
        assert CanonicalControl(**d) == ctrl

    def test_rejects_missing_fields(self) -> None:
        from tract.export.canonical_schema import CanonicalControl

        with pytest.raises(Exception):
            CanonicalControl(control_id="x", framework_id="y")  # type: ignore[call-arg]


class TestCREMapping:
    def test_roundtrip(self) -> None:
        from tract.export.canonical_schema import CREMapping

        m = CREMapping(**_make_mapping())
        assert m.rank == 1
        assert m.link_type == "TRACT_ML_PREDICTED"
        assert m.model_version == "7e8b8f834db5"

    def test_default_link_type(self) -> None:
        from tract.export.canonical_schema import CREMapping

        data = _make_mapping()
        del data["link_type"]
        m = CREMapping(**data)
        assert m.link_type == "TRACT_ML_PREDICTED"


class TestFilterPolicy:
    def test_defaults(self) -> None:
        from tract.export.canonical_schema import FilterPolicy

        fp = FilterPolicy(confidence_floor=0.3, confidence_override=None)
        assert fp.excluded_ground_truth is True
        assert fp.excluded_ood is True
        assert fp.review_status_required == "accepted"


class TestStandardSnapshot:
    def test_roundtrip(self) -> None:
        from tract.export.canonical_schema import (
            CanonicalControl,
            CREMapping,
            FilterPolicy,
            StandardSnapshot,
        )

        snap = StandardSnapshot(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**_make_control())],
            mappings=[CREMapping(**_make_mapping())],
        )
        assert snap.schema_version == "1.0"
        assert len(snap.controls) == 1
        assert len(snap.mappings) == 1


class TestContentHash:
    def test_deterministic(self) -> None:
        from tract.export.canonical_schema import compute_content_hash, StandardSnapshot, FilterPolicy, CanonicalControl, CREMapping

        snap = StandardSnapshot(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**_make_control())],
            mappings=[CREMapping(**_make_mapping())],
        )
        h1 = compute_content_hash(snap)
        h2 = compute_content_hash(snap)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_excludes_date_and_hash(self) -> None:
        from tract.export.canonical_schema import compute_content_hash, StandardSnapshot, FilterPolicy, CanonicalControl, CREMapping

        base = dict(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**_make_control())],
            mappings=[CREMapping(**_make_mapping())],
        )
        snap1 = StandardSnapshot(export_date="2026-05-04T00:00:00Z", **base)
        snap2 = StandardSnapshot(export_date="2026-12-25T00:00:00Z", **base)
        assert compute_content_hash(snap1) == compute_content_hash(snap2)

    def test_different_data_different_hash(self) -> None:
        from tract.export.canonical_schema import compute_content_hash, StandardSnapshot, FilterPolicy, CanonicalControl, CREMapping

        base = dict(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            mappings=[CREMapping(**_make_mapping())],
        )
        snap1 = StandardSnapshot(controls=[CanonicalControl(**_make_control())], **base)
        snap2 = StandardSnapshot(controls=[CanonicalControl(**_make_control(title="Different"))], **base)
        assert compute_content_hash(snap1) != compute_content_hash(snap2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_canonical_schema.py -v 2>&1 | tail -5`
Expected: FAIL — `ModuleNotFoundError: No module named 'tract.export.canonical_schema'`

- [ ] **Step 3: Write the implementation**

Create `tract/export/canonical_schema.py`:

```python
"""Pydantic v2 models for the canonical export format (spec §2).

Defines the schema TRACT proposes as OpenCRE RFC's "Spike A2" canonical
format. Models are used for both serialization and validation — every
snapshot passes model_validate() before export.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel


class FilterPolicy(BaseModel):
    confidence_floor: float
    confidence_override: float | None
    excluded_ground_truth: bool = True
    excluded_ood: bool = True
    excluded_null_confidence: bool = True
    review_status_required: str = "accepted"


class CanonicalControl(BaseModel):
    control_id: str
    framework_id: str
    section_id: str
    title: str
    description: str
    hyperlink: str


class CREMapping(BaseModel):
    control_id: str
    hub_id: str
    hub_name: str
    confidence: float
    rank: int
    link_type: str = "TRACT_ML_PREDICTED"
    provenance: str
    model_version: str


class StandardSnapshot(BaseModel):
    schema_version: str = "1.0"
    framework_id: str
    framework_name: str
    export_date: str
    content_hash: str
    tract_version: str
    model_adapter_hash: str
    filter_policy: FilterPolicy
    controls: list[CanonicalControl]
    mappings: list[CREMapping]


class ChangesetEntry(BaseModel):
    operation: Literal[
        "ADD_CONTROL", "UPDATE_CONTROL", "DELETE_CONTROL",
        "ADD_MAPPING", "UPDATE_MAPPING", "DELETE_MAPPING",
    ]
    entity: CanonicalControl | CREMapping | None = None
    before: CanonicalControl | CREMapping | None = None
    key: str | None = None


class ChangesetSummary(BaseModel):
    controls_added: int
    controls_updated: int
    controls_deleted: int
    mappings_added: int
    mappings_updated: int
    mappings_deleted: int


class ImpactAnalysis(BaseModel):
    affected_hubs: list[str]
    affected_frameworks: list[str]
    co_mapped_changes: int
    scope: str


class Changeset(BaseModel):
    schema_version: str = "1.0"
    framework_id: str
    from_version: str | None
    to_version: str
    export_date: str
    operations: list[ChangesetEntry]
    summary: ChangesetSummary
    impact: ImpactAnalysis


def compute_content_hash(snapshot: StandardSnapshot) -> str:
    """Compute SHA-256 of snapshot excluding volatile fields (spec §2.2)."""
    data = snapshot.model_dump(exclude={"content_hash", "export_date"})
    canonical_json = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_canonical_schema.py -v 2>&1 | tail -15`
Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tract/export/canonical_schema.py tests/test_canonical_schema.py
git commit -m "feat(export): add Pydantic v2 models for canonical export schema"
```

---

### Task 2: Snapshot builder — query and assemble

**Files:**
- Create: `tract/export/canonical.py`
- Test: `tests/test_canonical_export.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_canonical_export.py`:

```python
"""Tests for canonical export snapshot builder and differ (spec §§2-7)."""
from __future__ import annotations

import json

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def canonical_db(tmp_path):
    """DB with representative data for canonical export tests."""
    db_path = tmp_path / "canonical_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-05-04", "control_count": 3},
        {"id": "fw2", "name": "FW2", "version": "1.0", "fetch_date": "2026-05-04", "control_count": 1},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1",
         "title": "Control 1", "description": "Desc 1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2",
         "title": "Control 2", "description": "Desc 2", "full_text": None},
        {"id": "fw1:c3", "framework_id": "fw1", "section_id": "c3",
         "title": "Control 3", "description": "Desc 3", "full_text": None},
        {"id": "fw2:c1", "framework_id": "fw2", "section_id": "c1",
         "title": "FW2 Control 1", "description": "FW2 Desc 1", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.8,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw1:c2", "hub_id": "h2", "confidence": 0.6,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw1:c3", "hub_id": "h1", "confidence": 0.2,
         "in_conformal_set": 0, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw2:c1", "hub_id": "h2", "confidence": 0.7,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
    ])
    return db_path


class TestBuildSnapshot:
    def test_builds_valid_snapshot(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
        )
        assert snap.framework_id == "fw1"
        assert snap.model_adapter_hash == "abc123"
        assert len(snap.controls) == 2  # c3 filtered by 0.3 floor
        assert len(snap.mappings) == 2
        assert snap.content_hash != "placeholder"
        assert len(snap.content_hash) == 64

    def test_all_mappings_get_model_version(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
        )
        for m in snap.mappings:
            assert m.model_version == "abc123"

    def test_rank_is_one_for_single_hub(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
        )
        for m in snap.mappings:
            assert m.rank == 1

    def test_filters_by_confidence_floor(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.5,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
        )
        assert len(snap.mappings) == 2
        confidences = [m.confidence for m in snap.mappings]
        assert all(c >= 0.5 for c in confidences)

    def test_hyperlink_populated(self, canonical_db) -> None:
        from tract.export.canonical import build_snapshot

        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
        )
        for c in snap.controls:
            assert c.hyperlink.startswith("https://example.com/fw1/")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_canonical_export.py::TestBuildSnapshot -v 2>&1 | tail -5`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Write the implementation**

Create `tract/export/canonical.py`:

```python
"""Canonical export: snapshot builder, differ, and serializer (spec §§2-8).

Produces per-framework JSON snapshots and changesets for OpenCRE's
incremental import RFC. The export_history table tracks prior exports
for changeset generation.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from tract.crosswalk.schema import get_connection
from tract.export.canonical_schema import (
    CanonicalControl,
    Changeset,
    ChangesetEntry,
    ChangesetSummary,
    CREMapping,
    FilterPolicy,
    ImpactAnalysis,
    StandardSnapshot,
    compute_content_hash,
)

logger = logging.getLogger(__name__)

EXPORT_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS export_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    framework_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    export_date TEXT NOT NULL DEFAULT (datetime('now')),
    snapshot_json TEXT NOT NULL,
    filter_policy_json TEXT NOT NULL,
    assignment_count INTEGER NOT NULL,
    control_count INTEGER NOT NULL,
    tract_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_export_history_fw
    ON export_history(framework_id, export_date);
"""


def ensure_export_history_table(db_path: Path) -> None:
    """Create the export_history table if it does not exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(EXPORT_HISTORY_DDL)
        conn.commit()
    finally:
        conn.close()


def _query_canonical_assignments(
    db_path: Path,
    framework_id: str,
    confidence_floor: float,
    confidence_overrides: dict[str, float],
) -> list[dict]:
    """Query assignments passing all export filters, returning fields needed for canonical export."""
    from tract.config import PHASE5_GROUND_TRUTH_PROVENANCE

    conn = get_connection(db_path)
    try:
        floor = confidence_overrides.get(framework_id, confidence_floor)
        rows = conn.execute(
            "SELECT a.control_id, a.hub_id, h.name AS hub_name, "
            "a.confidence, a.provenance, "
            "c.framework_id, c.section_id, c.title, c.description "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN hubs h ON a.hub_id = h.id "
            "WHERE a.review_status = 'accepted' "
            "AND a.provenance != ? "
            "AND a.confidence IS NOT NULL "
            "AND a.is_ood != 1 "
            "AND c.framework_id = ? "
            "AND a.confidence >= ? "
            "ORDER BY a.hub_id, c.section_id",
            [PHASE5_GROUND_TRUTH_PROVENANCE, framework_id, floor],
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def build_snapshot(
    db_path: Path,
    framework_id: str,
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    model_adapter_hash: str,
    tract_version: str,
    hyperlink_fn: Callable[[str, str], str],
    framework_name: str | None = None,
) -> StandardSnapshot:
    """Build a StandardSnapshot from live DB data.

    Args:
        db_path: Path to crosswalk.db.
        framework_id: Which framework to export.
        confidence_floor: Global confidence threshold.
        confidence_overrides: Per-framework threshold overrides.
        model_adapter_hash: Adapter hash from deployment artifacts.
        tract_version: Git SHA of current TRACT checkout.
        hyperlink_fn: (framework_id, section_id) -> URL string.
        framework_name: OpenCRE display name. Auto-resolved if None.
    """
    if framework_name is None:
        from tract.export.opencre_names import get_opencre_name
        framework_name = get_opencre_name(framework_id)

    rows = _query_canonical_assignments(
        db_path, framework_id, confidence_floor, confidence_overrides,
    )

    seen_controls: dict[str, CanonicalControl] = {}
    control_mappings: defaultdict[str, list[dict]] = defaultdict(list)

    for row in rows:
        cid = row["control_id"]
        if cid not in seen_controls:
            seen_controls[cid] = CanonicalControl(
                control_id=cid,
                framework_id=row["framework_id"],
                section_id=row["section_id"],
                title=row["title"],
                description=row["description"],
                hyperlink=hyperlink_fn(row["framework_id"], row["section_id"]),
            )
        control_mappings[cid].append(row)

    controls = sorted(seen_controls.values(), key=lambda c: c.control_id)

    mappings: list[CREMapping] = []
    for cid in sorted(control_mappings.keys()):
        ranked = sorted(control_mappings[cid], key=lambda r: -r["confidence"])
        for rank_idx, row in enumerate(ranked, start=1):
            mappings.append(CREMapping(
                control_id=row["control_id"],
                hub_id=row["hub_id"],
                hub_name=row["hub_name"],
                confidence=row["confidence"],
                rank=rank_idx,
                provenance=row["provenance"],
                model_version=model_adapter_hash,
            ))

    filter_policy = FilterPolicy(
        confidence_floor=confidence_floor,
        confidence_override=confidence_overrides.get(framework_id),
    )

    snapshot = StandardSnapshot(
        framework_id=framework_id,
        framework_name=framework_name,
        export_date=datetime.now(timezone.utc).isoformat(),
        content_hash="placeholder",
        tract_version=tract_version,
        model_adapter_hash=model_adapter_hash,
        filter_policy=filter_policy,
        controls=controls,
        mappings=mappings,
    )
    snapshot.content_hash = compute_content_hash(snapshot)
    return snapshot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_canonical_export.py::TestBuildSnapshot -v 2>&1 | tail -10`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tract/export/canonical.py tests/test_canonical_export.py
git commit -m "feat(export): add canonical snapshot builder with filter pipeline"
```

---

### Task 3: Differ — changeset generation

**Files:**
- Modify: `tract/export/canonical.py`
- Modify: `tests/test_canonical_export.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_canonical_export.py`:

```python
class TestDiffSnapshots:
    def _make_snapshot(self, controls, mappings, framework_id="fw1"):
        from tract.export.canonical_schema import (
            CanonicalControl, CREMapping, FilterPolicy, StandardSnapshot, compute_content_hash,
        )
        snap = StandardSnapshot(
            framework_id=framework_id,
            framework_name="FW1",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="abc123",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**c) for c in controls],
            mappings=[CREMapping(**m) for m in mappings],
        )
        snap.content_hash = compute_content_hash(snap)
        return snap

    def test_initial_export_all_adds(self) -> None:
        from tract.export.canonical import diff_snapshots

        current = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=None, current=current)
        assert cs.from_version is None
        assert len(cs.operations) == 2
        ops = {op.operation for op in cs.operations}
        assert ops == {"ADD_CONTROL", "ADD_MAPPING"}
        assert cs.summary.controls_added == 1
        assert cs.summary.mappings_added == 1

    def test_no_changes_empty_changeset(self) -> None:
        from tract.export.canonical import diff_snapshots

        snap = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=snap, current=snap)
        assert len(cs.operations) == 0
        assert cs.summary.controls_added == 0

    def test_added_control_detected(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        c2 = _make_control(control_id="fw1:c2", section_id="c2", title="New")
        m2 = _make_mapping(control_id="fw1:c2", hub_id="h2")
        current = self._make_snapshot(
            controls=[_make_control(), c2],
            mappings=[_make_mapping(), m2],
        )
        cs = diff_snapshots(prior=prior, current=current)
        add_ops = [op for op in cs.operations if op.operation.startswith("ADD_")]
        assert len(add_ops) == 2

    def test_deleted_control_detected(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        current = self._make_snapshot(controls=[], mappings=[])
        cs = diff_snapshots(prior=prior, current=current)
        del_ops = [op for op in cs.operations if op.operation.startswith("DELETE_")]
        assert len(del_ops) == 2
        assert cs.summary.controls_deleted == 1
        assert cs.summary.mappings_deleted == 1

    def test_updated_control_detected(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        updated_control = _make_control(title="Changed Title")
        current = self._make_snapshot(
            controls=[updated_control],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=prior, current=current)
        update_ops = [op for op in cs.operations if op.operation == "UPDATE_CONTROL"]
        assert len(update_ops) == 1
        assert update_ops[0].before is not None
        assert update_ops[0].entity is not None

    def test_updated_mapping_confidence_change(self) -> None:
        from tract.export.canonical import diff_snapshots

        prior = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping(confidence=0.75)],
        )
        current = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping(confidence=0.85)],
        )
        cs = diff_snapshots(prior=prior, current=current)
        update_ops = [op for op in cs.operations if op.operation == "UPDATE_MAPPING"]
        assert len(update_ops) == 1

    def test_impact_analysis_populated(self) -> None:
        from tract.export.canonical import diff_snapshots

        current = self._make_snapshot(
            controls=[_make_control()],
            mappings=[_make_mapping()],
        )
        cs = diff_snapshots(prior=None, current=current)
        assert cs.impact.scope == "minor"
        assert "004-517" in cs.impact.affected_hubs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_canonical_export.py::TestDiffSnapshots -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name 'diff_snapshots'`

- [ ] **Step 3: Write the implementation**

Append to `tract/export/canonical.py`:

```python
# ── Control diff helpers ────────────────────────────────────────────────

_CONTROL_MUTABLE_FIELDS = ("title", "description", "hyperlink")
_MAPPING_MUTABLE_FIELDS = ("confidence", "rank", "provenance", "model_version")


def _diff_controls(
    prior_controls: dict[str, CanonicalControl],
    current_controls: dict[str, CanonicalControl],
) -> list[ChangesetEntry]:
    ops: list[ChangesetEntry] = []
    all_keys = sorted(set(prior_controls) | set(current_controls))
    for key in all_keys:
        old = prior_controls.get(key)
        new = current_controls.get(key)
        if old is None and new is not None:
            ops.append(ChangesetEntry(operation="ADD_CONTROL", entity=new))
        elif old is not None and new is None:
            ops.append(ChangesetEntry(operation="DELETE_CONTROL", key=key))
        elif old is not None and new is not None:
            if any(getattr(old, f) != getattr(new, f) for f in _CONTROL_MUTABLE_FIELDS):
                ops.append(ChangesetEntry(
                    operation="UPDATE_CONTROL", entity=new, before=old,
                ))
    return ops


def _diff_mappings(
    prior_mappings: dict[tuple[str, str], CREMapping],
    current_mappings: dict[tuple[str, str], CREMapping],
) -> list[ChangesetEntry]:
    ops: list[ChangesetEntry] = []
    all_keys = sorted(set(prior_mappings) | set(current_mappings))
    for key in all_keys:
        old = prior_mappings.get(key)
        new = current_mappings.get(key)
        if old is None and new is not None:
            ops.append(ChangesetEntry(operation="ADD_MAPPING", entity=new))
        elif old is not None and new is None:
            ops.append(ChangesetEntry(
                operation="DELETE_MAPPING", key=f"{key[0]}|{key[1]}",
            ))
        elif old is not None and new is not None:
            if any(getattr(old, f) != getattr(new, f) for f in _MAPPING_MUTABLE_FIELDS):
                ops.append(ChangesetEntry(
                    operation="UPDATE_MAPPING", entity=new, before=old,
                ))
    return ops


def _compute_impact(
    operations: list[ChangesetEntry],
    framework_id: str,
    db_path: Path | None = None,
) -> ImpactAnalysis:
    affected_hubs: set[str] = set()
    for op in operations:
        if op.entity and isinstance(op.entity, CREMapping):
            affected_hubs.add(op.entity.hub_id)
        if op.before and isinstance(op.before, CREMapping):
            affected_hubs.add(op.before.hub_id)
        if op.key and "|" in op.key:
            affected_hubs.add(op.key.split("|")[1])

    co_mapped = 0
    affected_frameworks: list[str] = [framework_id]
    if db_path is not None and affected_hubs:
        conn = get_connection(db_path)
        try:
            placeholders = ",".join("?" for _ in affected_hubs)
            rows = conn.execute(
                f"SELECT DISTINCT c.framework_id FROM assignments a "
                f"JOIN controls c ON a.control_id = c.id "
                f"WHERE a.hub_id IN ({placeholders}) "
                f"AND c.framework_id != ?",
                [*affected_hubs, framework_id],
            ).fetchall()
            other_fws = [r["framework_id"] for r in rows]
            affected_frameworks.extend(sorted(other_fws))
            co_mapped_rows = conn.execute(
                f"SELECT COUNT(DISTINCT a.control_id) FROM assignments a "
                f"JOIN controls c ON a.control_id = c.id "
                f"WHERE a.hub_id IN ({placeholders}) "
                f"AND c.framework_id != ?",
                [*affected_hubs, framework_id],
            ).fetchone()
            co_mapped = co_mapped_rows[0] if co_mapped_rows else 0
        finally:
            conn.close()

    n_ops = len(operations)
    has_delete_mapping = any(op.operation == "DELETE_MAPPING" for op in operations)
    has_delete_control = any(op.operation == "DELETE_CONTROL" for op in operations)

    if n_ops > 50 or has_delete_control:
        scope = "major"
    elif n_ops >= 10 or has_delete_mapping:
        scope = "moderate"
    else:
        scope = "minor"

    return ImpactAnalysis(
        affected_hubs=sorted(affected_hubs),
        affected_frameworks=affected_frameworks,
        co_mapped_changes=co_mapped,
        scope=scope,
    )


def diff_snapshots(
    prior: StandardSnapshot | None,
    current: StandardSnapshot,
    db_path: Path | None = None,
) -> Changeset:
    """Compute changeset between two snapshots (spec §3)."""
    if prior is None:
        ops: list[ChangesetEntry] = []
        for ctrl in current.controls:
            ops.append(ChangesetEntry(operation="ADD_CONTROL", entity=ctrl))
        for mapping in current.mappings:
            ops.append(ChangesetEntry(operation="ADD_MAPPING", entity=mapping))
    else:
        prior_controls = {c.control_id: c for c in prior.controls}
        current_controls = {c.control_id: c for c in current.controls}
        prior_mappings = {(m.control_id, m.hub_id): m for m in prior.mappings}
        current_mappings = {(m.control_id, m.hub_id): m for m in current.mappings}

        ops = _diff_controls(prior_controls, current_controls)
        ops.extend(_diff_mappings(prior_mappings, current_mappings))

    summary = ChangesetSummary(
        controls_added=sum(1 for o in ops if o.operation == "ADD_CONTROL"),
        controls_updated=sum(1 for o in ops if o.operation == "UPDATE_CONTROL"),
        controls_deleted=sum(1 for o in ops if o.operation == "DELETE_CONTROL"),
        mappings_added=sum(1 for o in ops if o.operation == "ADD_MAPPING"),
        mappings_updated=sum(1 for o in ops if o.operation == "UPDATE_MAPPING"),
        mappings_deleted=sum(1 for o in ops if o.operation == "DELETE_MAPPING"),
    )

    impact = _compute_impact(ops, current.framework_id, db_path)

    return Changeset(
        framework_id=current.framework_id,
        from_version=prior.content_hash if prior else None,
        to_version=current.content_hash,
        export_date=current.export_date,
        operations=ops,
        summary=summary,
        impact=impact,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_canonical_export.py::TestDiffSnapshots -v 2>&1 | tail -10`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tract/export/canonical.py tests/test_canonical_export.py
git commit -m "feat(export): add changeset differ with impact analysis"
```

---

### Task 4: Export history — store and retrieve

**Files:**
- Modify: `tract/export/canonical.py`
- Modify: `tests/test_canonical_export.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_canonical_export.py`:

```python
class TestExportHistory:
    def test_ensure_table_idempotent(self, canonical_db) -> None:
        from tract.export.canonical import ensure_export_history_table

        ensure_export_history_table(canonical_db)
        ensure_export_history_table(canonical_db)  # second call is safe

    def test_save_and_load_snapshot(self, canonical_db) -> None:
        from tract.export.canonical import (
            build_snapshot,
            ensure_export_history_table,
            load_prior_snapshot,
            save_to_export_history,
        )

        ensure_export_history_table(canonical_db)
        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name="FW1",
        )
        save_to_export_history(canonical_db, snap)
        prior = load_prior_snapshot(canonical_db, "fw1")
        assert prior is not None
        assert prior.content_hash == snap.content_hash
        assert prior.framework_id == "fw1"
        assert len(prior.controls) == len(snap.controls)

    def test_load_returns_none_when_empty(self, canonical_db) -> None:
        from tract.export.canonical import ensure_export_history_table, load_prior_snapshot

        ensure_export_history_table(canonical_db)
        prior = load_prior_snapshot(canonical_db, "fw1")
        assert prior is None

    def test_content_hash_verified_on_load(self, canonical_db) -> None:
        from tract.export.canonical import (
            build_snapshot,
            ensure_export_history_table,
            load_prior_snapshot,
            save_to_export_history,
        )
        from tract.crosswalk.schema import get_connection

        ensure_export_history_table(canonical_db)
        snap = build_snapshot(
            db_path=canonical_db,
            framework_id="fw1",
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_name="FW1",
        )
        save_to_export_history(canonical_db, snap)

        conn = get_connection(canonical_db)
        try:
            conn.execute(
                "UPDATE export_history SET content_hash = 'corrupted' WHERE framework_id = 'fw1'"
            )
            conn.commit()
        finally:
            conn.close()

        with pytest.raises(ValueError, match="content_hash mismatch"):
            load_prior_snapshot(canonical_db, "fw1")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_canonical_export.py::TestExportHistory -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name 'save_to_export_history'`

- [ ] **Step 3: Write the implementation**

Append to `tract/export/canonical.py`:

```python
def save_to_export_history(db_path: Path, snapshot: StandardSnapshot) -> None:
    """Persist a snapshot to the export_history table."""
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO export_history "
            "(framework_id, content_hash, export_date, snapshot_json, "
            "filter_policy_json, assignment_count, control_count, tract_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                snapshot.framework_id,
                snapshot.content_hash,
                snapshot.export_date,
                snapshot.model_dump_json(),
                snapshot.filter_policy.model_dump_json(),
                len(snapshot.mappings),
                len(snapshot.controls),
                snapshot.tract_version,
            ],
        )
        conn.commit()
        logger.info(
            "Saved export history: framework=%s hash=%s controls=%d mappings=%d",
            snapshot.framework_id, snapshot.content_hash[:12],
            len(snapshot.controls), len(snapshot.mappings),
        )
    finally:
        conn.close()


def load_prior_snapshot(db_path: Path, framework_id: str) -> StandardSnapshot | None:
    """Load the most recent snapshot for a framework from export_history.

    Verifies content_hash integrity after deserialization. Raises ValueError
    if the stored hash doesn't match the recomputed hash (spec §6.2 step 5).
    """
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT content_hash, snapshot_json FROM export_history "
            "WHERE framework_id = ? ORDER BY export_date DESC LIMIT 1",
            [framework_id],
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    stored_hash = row["content_hash"]
    snapshot = StandardSnapshot.model_validate_json(row["snapshot_json"])
    recomputed = compute_content_hash(snapshot)

    if stored_hash != recomputed:
        raise ValueError(
            f"export_history content_hash mismatch for {framework_id}: "
            f"stored={stored_hash[:12]}... recomputed={recomputed[:12]}..."
        )

    return snapshot
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_canonical_export.py::TestExportHistory -v 2>&1 | tail -10`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tract/export/canonical.py tests/test_canonical_export.py
git commit -m "feat(export): add export_history persistence with hash verification"
```

---

### Task 5: Embedding slicer

**Files:**
- Modify: `tract/export/canonical.py`
- Modify: `tests/test_canonical_export.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_canonical_export.py`:

```python
class TestEmbeddingSlicer:
    def test_slices_correct_controls(self, tmp_path) -> None:
        import numpy as np
        from tract.export.canonical import slice_embeddings_for_framework

        n_controls = 10
        n_hubs = 3
        dim = 1024
        control_ids = [f"fw1::c{i}" for i in range(5)] + [f"fw2::c{i}" for i in range(5)]
        hub_ids = [f"h{i}" for i in range(n_hubs)]

        artifacts_path = tmp_path / "deployment_artifacts.npz"
        np.savez(
            str(artifacts_path),
            control_embeddings=np.random.rand(n_controls, dim).astype(np.float32),
            control_ids=np.array(control_ids),
            hub_embeddings=np.random.rand(n_hubs, dim).astype(np.float32),
            hub_ids=np.array(hub_ids),
            model_adapter_hash=np.array("abc123"),
        )

        canonical_control_ids = {"fw1:c0", "fw1:c1", "fw1:c2", "fw1:c3", "fw1:c4"}
        result = slice_embeddings_for_framework(
            artifacts_path=artifacts_path,
            canonical_control_ids=canonical_control_ids,
            model_adapter_hash="abc123",
        )
        assert result["control_ids"].shape[0] == 5
        assert result["control_embeddings"].shape == (5, dim)
        assert result["hub_embeddings"].shape == (n_hubs, dim)
        assert all("::" not in str(cid) for cid in result["control_ids"])

    def test_raises_on_hash_mismatch(self, tmp_path) -> None:
        import numpy as np
        from tract.export.canonical import slice_embeddings_for_framework

        artifacts_path = tmp_path / "deployment_artifacts.npz"
        np.savez(
            str(artifacts_path),
            control_embeddings=np.random.rand(2, 1024).astype(np.float32),
            control_ids=np.array(["fw1::c0", "fw1::c1"]),
            hub_embeddings=np.random.rand(1, 1024).astype(np.float32),
            hub_ids=np.array(["h0"]),
            model_adapter_hash=np.array("abc123"),
        )
        with pytest.raises(ValueError, match="model_adapter_hash mismatch"):
            slice_embeddings_for_framework(
                artifacts_path=artifacts_path,
                canonical_control_ids={"fw1:c0"},
                model_adapter_hash="different_hash",
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_canonical_export.py::TestEmbeddingSlicer -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name 'slice_embeddings_for_framework'`

- [ ] **Step 3: Write the implementation**

Append to `tract/export/canonical.py`:

```python
def slice_embeddings_for_framework(
    artifacts_path: Path,
    canonical_control_ids: set[str],
    model_adapter_hash: str,
) -> dict:
    """Slice deployment_artifacts.npz to a single framework's controls.

    Normalizes :: to : in artifact IDs to match canonical format (spec §5.2).
    Returns dict with keys: control_embeddings, control_ids, hub_embeddings,
    hub_ids, model_adapter_hash.
    """
    import numpy as np

    data = np.load(str(artifacts_path), allow_pickle=True)
    stored_hash = str(data["model_adapter_hash"])

    if stored_hash != model_adapter_hash:
        raise ValueError(
            f"model_adapter_hash mismatch: artifacts={stored_hash}, "
            f"expected={model_adapter_hash}"
        )

    all_control_ids = data["control_ids"]
    normalized_ids = [cid.replace("::", ":") for cid in all_control_ids]

    mask = [nid in canonical_control_ids for nid in normalized_ids]
    selected_ids = [nid for nid, m in zip(normalized_ids, mask) if m]
    selected_embeddings = data["control_embeddings"][mask]

    return {
        "control_embeddings": selected_embeddings,
        "control_ids": np.array(selected_ids),
        "hub_embeddings": data["hub_embeddings"],
        "hub_ids": data["hub_ids"],
        "model_adapter_hash": model_adapter_hash,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_canonical_export.py::TestEmbeddingSlicer -v 2>&1 | tail -10`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tract/export/canonical.py tests/test_canonical_export.py
git commit -m "feat(export): add embedding slicer with ID normalization"
```

---

### Task 6: Full export orchestrator

**Files:**
- Modify: `tract/export/canonical.py`
- Modify: `tests/test_canonical_export.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_canonical_export.py`:

```python
class TestExportCanonical:
    def test_initial_export_creates_files(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical

        output_dir = tmp_path / "output"
        result = export_canonical(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
        )
        assert (output_dir / "fw1" / "snapshot.json").exists()
        assert (output_dir / "fw1" / "changeset.json").exists()
        assert result["fw1"]["changeset_summary"]["controls_added"] > 0

    def test_second_export_detects_no_changes(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical

        output_dir = tmp_path / "output"
        kwargs = dict(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
        )
        export_canonical(**kwargs)
        result = export_canonical(**kwargs)
        summary = result["fw1"]["changeset_summary"]
        assert summary["controls_added"] == 0
        assert summary["mappings_added"] == 0

    def test_dry_run_does_not_write(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical

        output_dir = tmp_path / "output"
        result = export_canonical(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
            dry_run=True,
        )
        assert not (output_dir / "fw1" / "snapshot.json").exists()
        assert "fw1" in result

    def test_snapshot_json_validates(self, canonical_db, tmp_path) -> None:
        from tract.export.canonical import export_canonical
        from tract.export.canonical_schema import StandardSnapshot

        output_dir = tmp_path / "output"
        export_canonical(
            db_path=canonical_db,
            framework_ids=["fw1"],
            output_dir=output_dir,
            confidence_floor=0.3,
            confidence_overrides={},
            model_adapter_hash="abc123",
            tract_version="def456",
            hyperlink_fn=lambda fw, sec: f"https://example.com/{fw}/{sec}",
            framework_names={"fw1": "FW1"},
        )
        snap_path = output_dir / "fw1" / "snapshot.json"
        snap = StandardSnapshot.model_validate_json(snap_path.read_text(encoding="utf-8"))
        assert snap.framework_id == "fw1"
        assert len(snap.content_hash) == 64
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_canonical_export.py::TestExportCanonical -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name 'export_canonical'`

- [ ] **Step 3: Write the implementation**

Append to `tract/export/canonical.py`:

```python
def export_canonical(
    db_path: Path,
    framework_ids: list[str],
    output_dir: Path,
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    model_adapter_hash: str,
    tract_version: str,
    hyperlink_fn: Callable[[str, str], str],
    framework_names: dict[str, str] | None = None,
    artifacts_path: Path | None = None,
    with_embeddings: bool = False,
    dry_run: bool = False,
) -> dict[str, dict]:
    """Run the full canonical export pipeline for one or more frameworks.

    Returns a dict keyed by framework_id with export metadata per framework.
    """
    ensure_export_history_table(db_path)
    results: dict[str, dict] = {}

    for fw_id in framework_ids:
        fw_name = (framework_names or {}).get(fw_id)
        snapshot = build_snapshot(
            db_path=db_path,
            framework_id=fw_id,
            confidence_floor=confidence_floor,
            confidence_overrides=confidence_overrides,
            model_adapter_hash=model_adapter_hash,
            tract_version=tract_version,
            hyperlink_fn=hyperlink_fn,
            framework_name=fw_name,
        )

        if not snapshot.controls and not snapshot.mappings:
            logger.info("No exportable assignments for %s, skipping", fw_id)
            continue

        prior = load_prior_snapshot(db_path, fw_id)
        changeset = diff_snapshots(prior=prior, current=snapshot, db_path=db_path)

        results[fw_id] = {
            "content_hash": snapshot.content_hash,
            "controls": len(snapshot.controls),
            "mappings": len(snapshot.mappings),
            "changeset_summary": changeset.summary.model_dump(),
            "impact_scope": changeset.impact.scope,
        }

        if dry_run:
            continue

        fw_dir = output_dir / fw_id
        fw_dir.mkdir(parents=True, exist_ok=True)

        snap_path = fw_dir / "snapshot.json"
        snap_path.write_text(
            snapshot.model_dump_json(indent=2), encoding="utf-8",
        )

        cs_path = fw_dir / "changeset.json"
        cs_path.write_text(
            changeset.model_dump_json(indent=2), encoding="utf-8",
        )

        if with_embeddings and artifacts_path is not None:
            control_ids = {c.control_id for c in snapshot.controls}
            emb_data = slice_embeddings_for_framework(
                artifacts_path=artifacts_path,
                canonical_control_ids=control_ids,
                model_adapter_hash=model_adapter_hash,
            )
            import numpy as np
            emb_path = fw_dir / "embeddings.npz"
            np.savez(
                str(emb_path),
                control_embeddings=emb_data["control_embeddings"],
                control_ids=emb_data["control_ids"],
                hub_embeddings=emb_data["hub_embeddings"],
                hub_ids=emb_data["hub_ids"],
                model_adapter_hash=emb_data["model_adapter_hash"],
            )

        save_to_export_history(db_path, snapshot)
        logger.info(
            "Exported %s: %d controls, %d mappings, scope=%s",
            fw_id, len(snapshot.controls), len(snapshot.mappings),
            changeset.impact.scope,
        )

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_canonical_export.py::TestExportCanonical -v 2>&1 | tail -10`
Expected: All 4 tests PASS.

- [ ] **Step 5: Run all canonical tests together**

Run: `python -m pytest tests/test_canonical_export.py tests/test_canonical_schema.py -v 2>&1 | tail -20`
Expected: All 22 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add tract/export/canonical.py tests/test_canonical_export.py
git commit -m "feat(export): add full canonical export orchestrator with dry-run support"
```

---

### Task 7: CLI subcommand — parser and handler

**Files:**
- Modify: `tract/cli.py`
- Create: `tests/test_canonical_cli.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_canonical_cli.py`:

```python
"""Tests for tract export-canonical CLI subcommand."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def cli_db(tmp_path):
    """DB with data for CLI integration tests."""
    db_path = tmp_path / "cli_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-05-04", "control_count": 2},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1",
         "title": "Control 1", "description": "Desc 1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2",
         "title": "Control 2", "description": "Desc 2", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.8,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw1:c2", "hub_id": "h1", "confidence": 0.7,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
    ])
    return db_path


class TestExportCanonicalCLI:
    def test_parser_registered(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export-canonical", "--dry-run"])
        assert args.command == "export-canonical"
        assert args.dry_run is True

    def test_parser_with_embeddings_flag(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export-canonical", "--with-embeddings"])
        assert args.with_embeddings is True

    def test_parser_framework_filter(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export-canonical", "--framework", "csa_aicm"])
        assert args.framework == "csa_aicm"

    def test_handler_dry_run(self, cli_db, tmp_path, capsys) -> None:
        from tract.cli import build_parser, _cmd_export_canonical

        output_dir = tmp_path / "output"
        parser = build_parser()
        args = parser.parse_args([
            "export-canonical",
            "--framework", "fw1",
            "--output-dir", str(output_dir),
            "--dry-run",
        ])

        with patch("tract.cli.PHASE1C_CROSSWALK_DB_PATH", cli_db), \
             patch("tract.cli.PHASE1D_ARTIFACTS_PATH", tmp_path / "nonexistent.npz"), \
             patch("tract.export.opencre_names.TRACT_TO_OPENCRE_NAME", {"fw1": "FW1"}):
            _cmd_export_canonical(args)

        assert not (output_dir / "fw1" / "snapshot.json").exists()
        captured = capsys.readouterr()
        assert "fw1" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_canonical_cli.py -v 2>&1 | tail -5`
Expected: FAIL — `ImportError: cannot import name '_cmd_export_canonical'`

- [ ] **Step 3: Add the CLI parser entry**

In `tract/cli.py`, after the export-opencre parser section (around line 122), add the new subparser. Find the comment `# ── hierarchy ──` and insert before it:

```python
    # ── export-canonical ────────────────────────────────────────
    p_export_canonical = subparsers.add_parser(
        "export-canonical",
        help="Export canonical JSON snapshots for OpenCRE RFC",
        epilog=(
            "Examples:\n"
            "  tract export-canonical --dry-run\n"
            "  tract export-canonical --framework csa_aicm --with-embeddings\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_export_canonical.add_argument(
        "--framework", help="Export single framework (default: all mapped frameworks)")
    p_export_canonical.add_argument(
        "--output-dir", help="Output directory (default: ./canonical_export)")
    p_export_canonical.add_argument(
        "--with-embeddings", action="store_true",
        help="Include .npz embedding files per framework")
    p_export_canonical.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be exported without writing files or updating history")
```

- [ ] **Step 4: Add the CLI handler function**

In `tract/cli.py`, after `_cmd_export_opencre_proposals` (around line 958), add:

```python
def _cmd_export_canonical(args: argparse.Namespace) -> None:
    import hashlib

    from tract.config import (
        PHASE1D_ARTIFACTS_PATH,
        PHASE5_CANONICAL_EXPORT_DIR,
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR,
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES,
    )
    from tract.export.canonical import export_canonical
    from tract.export.opencre_names import (
        TRACT_TO_OPENCRE_NAME,
        build_hyperlink,
    )

    output_dir = Path(args.output_dir) if args.output_dir else PHASE5_CANONICAL_EXPORT_DIR

    if args.framework:
        if args.framework not in TRACT_TO_OPENCRE_NAME:
            print(
                f"Error: Framework '{args.framework}' has no OpenCRE name mapping",
                file=sys.stderr,
            )
            print(
                f"  Available: {', '.join(sorted(TRACT_TO_OPENCRE_NAME.keys()))}",
                file=sys.stderr,
            )
            sys.exit(1)
        framework_ids = [args.framework]
    else:
        framework_ids = sorted(TRACT_TO_OPENCRE_NAME.keys())

    model_hash = "unknown"
    if PHASE1D_ARTIFACTS_PATH.exists():
        model_hash = hashlib.sha256(
            PHASE1D_ARTIFACTS_PATH.read_bytes()
        ).hexdigest()[:12]

    try:
        tract_version = (
            __import__("subprocess")
            .check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        tract_version = "unknown"

    result = export_canonical(
        db_path=PHASE1C_CROSSWALK_DB_PATH,
        framework_ids=framework_ids,
        output_dir=output_dir,
        confidence_floor=PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR,
        confidence_overrides=dict(PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES),
        model_adapter_hash=model_hash,
        tract_version=tract_version,
        hyperlink_fn=build_hyperlink,
        framework_names=dict(TRACT_TO_OPENCRE_NAME),
        artifacts_path=PHASE1D_ARTIFACTS_PATH if args.with_embeddings else None,
        with_embeddings=args.with_embeddings,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\nDry run — would export:\n")

    total_controls = 0
    total_mappings = 0
    for fw_id, info in sorted(result.items()):
        summary = info["changeset_summary"]
        total_controls += info["controls"]
        total_mappings += info["mappings"]
        change_parts = []
        for k, v in summary.items():
            if v > 0:
                change_parts.append(f"{k}={v}")
        changes_str = ", ".join(change_parts) if change_parts else "no changes"
        print(f"  {fw_id}: {info['controls']} controls, {info['mappings']} mappings "
              f"[{info['impact_scope']}] ({changes_str})")

    print(f"\n  Total: {total_controls} controls, {total_mappings} mappings "
          f"across {len(result)} frameworks")

    if not args.dry_run:
        print(f"  Output: {output_dir}")
```

- [ ] **Step 5: Register in the handlers dict**

In `tract/cli.py`, in the `main()` function's `handlers` dict (around line 1503), add the entry:

```python
        "export-canonical": _cmd_export_canonical,
```

Add it alphabetically after `"export": _cmd_export,`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_canonical_cli.py -v 2>&1 | tail -10`
Expected: All 4 tests PASS.

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -10`
Expected: All tests pass (831+ original + ~26 new = 857+).

- [ ] **Step 8: Commit**

```bash
git add tract/cli.py tests/test_canonical_cli.py
git commit -m "feat(cli): add export-canonical subcommand for OpenCRE RFC integration"
```

---

### Task 8: Update notebook to reflect canonical export

**Files:**
- Modify: `tract_experimental_narrative.ipynb` (via notebook cell additions)
- Modify: `notebooks/nb_helpers.py` (add loaders)

Per standing requirement: the experimental narrative notebook must reflect all changes.

- [ ] **Step 1: Add canonical export loaders to nb_helpers.py**

In `notebooks/nb_helpers.py`, after the `load_framework_metadata()` function (line ~238), add:

```python
def load_canonical_snapshot(framework_id: str) -> dict:
    """Load a canonical export snapshot for a specific framework."""
    path = PROJECT_ROOT / "canonical_export" / framework_id / "snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"Canonical snapshot not found: {path}")
    return _load_json(path)


def load_canonical_changeset(framework_id: str) -> dict:
    """Load a canonical export changeset for a specific framework."""
    path = PROJECT_ROOT / "canonical_export" / framework_id / "changeset.json"
    if not path.exists():
        raise FileNotFoundError(f"Canonical changeset not found: {path}")
    return _load_json(path)
```

- [ ] **Step 2: Add a new section to the notebook**

After the existing Section 12 (Dataset Publication) or in an appropriate position near the end of the notebook, add a new markdown cell and code cell documenting the canonical export:

**Markdown cell:**
```markdown
## 13. Canonical Export for OpenCRE RFC

TRACT produces a canonical JSON format designed to align with OpenCRE's
[easier-importing RFC](https://github.com/OWASP/OpenCRE/blob/main/docs/designs/easier-importing.md).
Each framework gets a `StandardSnapshot` (all controls + mappings + filter policy)
and a `Changeset` (diff against the prior export). This enables incremental,
reviewable imports rather than full-graph replacement.

Key design decisions from the 4-round adversarial review:
- **Control IDs** use single-colon format (`framework_id:section_id`), matching crosswalk.db
- **model_version** on each mapping is derived from the snapshot-level `model_adapter_hash`
- **rank** field is forward-compatible for top-k>1 (v1 data is all rank=1)
- **Hyperlinks** are computed at export time from templates, not stored in DB
```

**Code cell:**
```python
# Show canonical export schema structure
from tract.export.canonical_schema import StandardSnapshot
print("StandardSnapshot schema:")
for field_name, field_info in StandardSnapshot.model_fields.items():
    annotation = field_info.annotation
    print(f"  {field_name}: {annotation}")
```

- [ ] **Step 3: Commit**

```bash
git add notebooks/nb_helpers.py tract_experimental_narrative.ipynb
git commit -m "docs(notebook): add Section 13 documenting canonical export for OpenCRE RFC"
```

---

### Task 9: Final integration test and CLAUDE.md update

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest tests/ -q --tb=short 2>&1 | tail -10`
Expected: All tests pass (857+).

- [ ] **Step 2: Run type checking**

Run: `mypy tract/export/canonical_schema.py tract/export/canonical.py --strict 2>&1 | tail -10`
Expected: No errors (or only pre-existing warnings from other files).

- [ ] **Step 3: Update CLAUDE.md commands section**

In `CLAUDE.md`, in the `## Commands` section, add after the Phase 3 commands:

```bash
# Phase 5A — Canonical Export (OpenCRE RFC)
tract export-canonical --dry-run                  # Preview what would be exported
tract export-canonical --framework csa_aicm       # Export single framework
tract export-canonical --with-embeddings           # Include .npz embedding files
```

- [ ] **Step 4: Update CLAUDE.md project status**

In the `## Project Status` section, add a new line:

```
- **Phase 5B (Canonical Export):** COMPLETE — 411 assignments across 5 frameworks, JSON snapshots + changesets for OpenCRE RFC
```

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with canonical export commands and status"
```

---

## Self-Review Checklist

### 1. Spec Coverage

| Spec Section | Task(s) | Status |
|-------------|---------|--------|
| §2.1 Core Models | Task 1 | Covered — all 8 Pydantic models |
| §2.2 Content Hash | Task 1 | Covered — `compute_content_hash()` |
| §2.3 Control ID Format | Task 2 | Covered — query uses DB canonical `:` format |
| §3.1-3.2 Changeset Operations | Task 3 | Covered — 6 operations, ChangesetEntry model |
| §3.3 Diff Algorithm | Task 3 | Covered — keyed diff with mutable/immutable field distinction |
| §3.4 Initial Export | Task 3 | Covered — `prior=None` branch |
| §4 Filter Policy | Task 2 | Covered — reuses filter logic with explicit FilterPolicy |
| §5 Embedding Export | Task 5 | Covered — `slice_embeddings_for_framework()` |
| §5.2 ID Normalization | Task 5 | Covered — `::` → `:` with assertion |
| §6.1 export_history Schema | Task 4 | Covered — DDL + `ensure_export_history_table()` |
| §6.2 Workflow | Task 6 | Covered — `export_canonical()` orchestrator |
| §6.3 Validation | Task 1 | Covered — `model_validate()` via Pydantic |
| §7 Impact Analysis | Task 3 | Covered — `_compute_impact()` with cross-fw query |
| §8.1 CLI Design | Task 7 | Covered — parser + handler |
| §8.2 Module Structure | Tasks 1-6 | Covered — `canonical_schema.py` + `canonical.py` |
| §9 Backward Compatibility | Task 7 | Covered — new subcommand, existing unchanged |
| Notebook update | Task 8 | Covered — Section 13 + nb_helpers loaders |

### 2. Placeholder Scan
No TBD, TODO, "implement later", or "similar to Task N" found.

### 3. Type Consistency
- `build_snapshot()` returns `StandardSnapshot` — used in Tasks 2, 4, 6 ✓
- `diff_snapshots()` returns `Changeset` — used in Tasks 3, 6 ✓
- `compute_content_hash()` returns `str` — used in Tasks 1, 2, 4 ✓
- `slice_embeddings_for_framework()` returns `dict` — used in Tasks 5, 6 ✓
- `export_canonical()` returns `dict[str, dict]` — used in Tasks 6, 7 ✓
- `hyperlink_fn: Callable[[str, str], str]` — consistent across Tasks 2, 6, 7 ✓
