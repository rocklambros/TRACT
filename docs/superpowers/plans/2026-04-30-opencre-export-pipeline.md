# OpenCRE Export Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `tract export --opencre` to generate OpenCRE-compatible CSV from TRACT's crosswalk.db, with staleness checks, confidence filtering, export manifest, and a post-import verification script.

**Architecture:** Filter pipeline reads crosswalk.db → excludes ground truth / low-confidence / OOD → maps TRACT framework IDs to OpenCRE `FAMILY_NAME` strings → generates one CSV per framework in OpenCRE's `parse_export_format()` layout → writes `export_manifest.json` with provenance. A separate staleness check diffs TRACT's hub IDs against upstream OpenCRE before export. Hub proposals are exported as a separate JSON document.

**Tech Stack:** Python 3.11, SQLite3 (crosswalk.db), `csv` stdlib, `requests` (staleness check), `pytest` (testing). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-04-30-opencre-export-design.md`

---

## File Structure

### New files

| File | Responsibility |
|------|---------------|
| `tract/export/__init__.py` | Package marker |
| `tract/export/opencre_names.py` | TRACT→OpenCRE framework name mapping dict + hyperlink URL templates |
| `tract/export/filters.py` | Ground truth exclusion, NULL/OOD exclusion, confidence floor filtering |
| `tract/export/opencre_csv.py` | CSV generation in OpenCRE `parse_export_format()` layout |
| `tract/export/staleness.py` | Pre-export CRE ID diff against upstream OpenCRE API |
| `tract/export/manifest.py` | Export manifest JSON generation |
| `tests/test_opencre_names.py` | Name mapping completeness + KeyError on unknown |
| `tests/test_export_filters.py` | Filter pipeline unit tests |
| `tests/test_opencre_csv.py` | CSV format correctness |
| `tests/test_staleness.py` | Staleness check logic |
| `tests/test_export_manifest.py` | Manifest generation |
| `tests/test_opencre_export_e2e.py` | End-to-end: DB → filters → CSV → manifest |
| `scripts/opencre_import.sh` | Fork import helper (start app + curl upload) |
| `scripts/verify_opencre_import.py` | Post-import verification against fork API |

### Modified files

| File | Change |
|------|--------|
| `tract/config.py` | Add `PHASE5_*` constants (confidence floor, overrides, staleness URL) |
| `tract/cli.py` | Add `--opencre`, `--opencre-proposals`, `--dry-run`, `--framework` flags to `export` subcommand |

---

### Task 1: Config Constants

**Files:**
- Modify: `tract/config.py`
- Test: `tests/test_opencre_names.py` (created in Task 2, but config is tested implicitly)

- [ ] **Step 1: Add Phase 5 constants to config.py**

Append to the end of `tract/config.py`:

```python
# ── Phase 5: OpenCRE Export Pipeline ─────────────────────────────────

PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR: Final[float] = 0.30
PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES: Final[dict[str, float]] = {
    "mitre_atlas": 0.35,
}
PHASE5_OPENCRE_STALENESS_URL: Final[str] = "https://opencre.org/rest/v1/root_cres"
PHASE5_OPENCRE_STALENESS_TIMEOUT_S: Final[int] = 30
PHASE5_GROUND_TRUTH_PROVENANCE: Final[str] = "ground_truth_T1-AI"
```

- [ ] **Step 2: Verify config imports cleanly**

Run: `python -c "from tract.config import PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tract/config.py
git commit -m "feat(export): add Phase 5 OpenCRE export config constants"
```

---

### Task 2: Framework Name Mapping

**Files:**
- Create: `tract/export/__init__.py`
- Create: `tract/export/opencre_names.py`
- Test: `tests/test_opencre_names.py`

- [ ] **Step 1: Create the export package**

Create `tract/export/__init__.py` as an empty file.

- [ ] **Step 2: Write the failing test for name mapping**

Create `tests/test_opencre_names.py`:

```python
"""Tests for TRACT→OpenCRE framework name mapping."""
from __future__ import annotations

import pytest


class TestOpenCRENameMapping:
    def test_all_exportable_frameworks_have_mapping(self) -> None:
        from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME

        exportable = [
            "mitre_atlas",
            "owasp_llm_top10",
            "nist_ai_600_1",
            "csa_aicm",
            "eu_ai_act",
            "owasp_agentic_top10",
        ]
        for fw_id in exportable:
            assert fw_id in TRACT_TO_OPENCRE_NAME, f"Missing mapping for {fw_id}"

    def test_known_names_are_exact(self) -> None:
        from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME

        assert TRACT_TO_OPENCRE_NAME["mitre_atlas"] == "MITRE ATLAS"
        assert TRACT_TO_OPENCRE_NAME["owasp_llm_top10"] == "OWASP Top10 for LLM"
        assert TRACT_TO_OPENCRE_NAME["nist_ai_600_1"] == "NIST AI 600-1"
        assert TRACT_TO_OPENCRE_NAME["csa_aicm"] == "CSA AI Controls Matrix"
        assert TRACT_TO_OPENCRE_NAME["eu_ai_act"] == "EU AI Act"
        assert TRACT_TO_OPENCRE_NAME["owasp_agentic_top10"] == "OWASP Agentic AI Top 10"

    def test_get_opencre_name_raises_on_unknown(self) -> None:
        from tract.export.opencre_names import get_opencre_name

        with pytest.raises(KeyError, match="no_such_framework"):
            get_opencre_name("no_such_framework")

    def test_get_opencre_name_returns_correct_value(self) -> None:
        from tract.export.opencre_names import get_opencre_name

        assert get_opencre_name("mitre_atlas") == "MITRE ATLAS"


class TestHyperlinkTemplates:
    def test_all_exportable_frameworks_have_hyperlink_template(self) -> None:
        from tract.export.opencre_names import HYPERLINK_TEMPLATES

        exportable = [
            "mitre_atlas",
            "owasp_llm_top10",
            "nist_ai_600_1",
            "csa_aicm",
            "eu_ai_act",
            "owasp_agentic_top10",
        ]
        for fw_id in exportable:
            assert fw_id in HYPERLINK_TEMPLATES, f"Missing hyperlink template for {fw_id}"

    def test_build_hyperlink_mitre_atlas(self) -> None:
        from tract.export.opencre_names import build_hyperlink

        url = build_hyperlink("mitre_atlas", "AML.T0000")
        assert url == "https://atlas.mitre.org/techniques/AML.T0000"

    def test_build_hyperlink_nist_ai_600_1(self) -> None:
        from tract.export.opencre_names import build_hyperlink

        url = build_hyperlink("nist_ai_600_1", "GAI-CBRN")
        assert url == "https://airc.nist.gov/Docs/1"

    def test_build_hyperlink_unknown_framework_raises(self) -> None:
        from tract.export.opencre_names import build_hyperlink

        with pytest.raises(KeyError):
            build_hyperlink("unknown_fw", "X01")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_opencre_names.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tract.export'`

- [ ] **Step 4: Implement the name mapping module**

Create `tract/export/opencre_names.py`:

```python
"""TRACT framework_id → OpenCRE FAMILY_NAME mapping.

CRITICAL (spec §4): A mismatch creates duplicate standard entries in
OpenCRE's database. These names are verified against OpenCRE's parser
source code (spreadsheet_mitre_atlas.py FAMILY_NAME, etc).
"""
from __future__ import annotations

from typing import Final

TRACT_TO_OPENCRE_NAME: Final[dict[str, str]] = {
    "mitre_atlas": "MITRE ATLAS",
    "owasp_llm_top10": "OWASP Top10 for LLM",
    "nist_ai_600_1": "NIST AI 600-1",
    "csa_aicm": "CSA AI Controls Matrix",
    "eu_ai_act": "EU AI Act",
    "owasp_agentic_top10": "OWASP Agentic AI Top 10",
}

HYPERLINK_TEMPLATES: Final[dict[str, str]] = {
    "mitre_atlas": "https://atlas.mitre.org/techniques/{section_id}",
    "owasp_llm_top10": "https://genai.owasp.org/llmrisk/{section_id}",
    "nist_ai_600_1": "https://airc.nist.gov/Docs/1",
    "csa_aicm": "https://cloudsecurityalliance.org/artifacts/ai-controls-matrix",
    "eu_ai_act": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
    "owasp_agentic_top10": "https://genai.owasp.org",
}


def get_opencre_name(framework_id: str) -> str:
    """Return the exact OpenCRE FAMILY_NAME for a TRACT framework_id.

    Raises KeyError if the framework_id has no mapping — no silent fallthrough.
    """
    return TRACT_TO_OPENCRE_NAME[framework_id]


def build_hyperlink(framework_id: str, section_id: str) -> str:
    """Build a hyperlink URL for a control section.

    For frameworks with per-control URLs (ATLAS, OWASP LLM), the section_id
    is interpolated. For others (NIST, CSA, EU AI Act), returns the
    framework-level URL since no per-section deep links exist.
    """
    template = HYPERLINK_TEMPLATES[framework_id]
    return template.format(section_id=section_id)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_opencre_names.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tract/export/__init__.py tract/export/opencre_names.py tests/test_opencre_names.py
git commit -m "feat(export): add TRACT→OpenCRE framework name mapping"
```

---

### Task 3: Filter Pipeline

**Files:**
- Create: `tract/export/filters.py`
- Test: `tests/test_export_filters.py`

- [ ] **Step 1: Write the failing tests for filters**

Create `tests/test_export_filters.py`:

```python
"""Tests for OpenCRE export filter pipeline."""
from __future__ import annotations

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def filter_db(tmp_path):
    """DB with mixed provenance, confidence, and OOD assignments."""
    db_path = tmp_path / "filter_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 5},
        {"id": "fw2", "name": "FW2", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
        {"id": "h2", "name": "Hub 2", "path": "R > H2", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1", "title": "C1", "description": "D1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2", "title": "C2", "description": "D2", "full_text": None},
        {"id": "fw1:c3", "framework_id": "fw1", "section_id": "c3", "title": "C3", "description": "D3", "full_text": None},
        {"id": "fw1:c4", "framework_id": "fw1", "section_id": "c4", "title": "C4", "description": "D4", "full_text": None},
        {"id": "fw1:c5", "framework_id": "fw1", "section_id": "c5", "title": "C5", "description": "D5", "full_text": None},
        {"id": "fw2:c1", "framework_id": "fw2", "section_id": "c1", "title": "C1", "description": "D1", "full_text": None},
        {"id": "fw2:c2", "framework_id": "fw2", "section_id": "c2", "title": "C2", "description": "D2", "full_text": None},
    ])
    insert_assignments(db_path, [
        # fw1:c1 — ground_truth, should be excluded
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "ground_truth_T1-AI", "source_link_id": None, "model_version": "v1", "review_status": "ground_truth"},
        # fw1:c2 — accepted, good confidence
        {"control_id": "fw1:c2", "hub_id": "h1", "confidence": 0.6, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        # fw1:c3 — accepted but below floor (0.25 < 0.30)
        {"control_id": "fw1:c3", "hub_id": "h2", "confidence": 0.25, "in_conformal_set": 0, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        # fw1:c4 — accepted, OOD flagged
        {"control_id": "fw1:c4", "hub_id": "h1", "confidence": 0.5, "in_conformal_set": 0, "is_ood": 1,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        # fw1:c5 — accepted, NULL confidence
        {"control_id": "fw1:c5", "hub_id": "h1", "confidence": None, "in_conformal_set": 0, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        # fw2:c1 — accepted, good confidence
        {"control_id": "fw2:c1", "hub_id": "h2", "confidence": 0.7, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        # fw2:c2 — pending (not accepted)
        {"control_id": "fw2:c2", "hub_id": "h1", "confidence": 0.8, "in_conformal_set": 1, "is_ood": 0,
         "provenance": "active_learning_round_2", "source_link_id": None, "model_version": "v1", "review_status": "pending"},
    ])
    return db_path


class TestFilterPipeline:
    def test_excludes_ground_truth(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c1" not in control_ids

    def test_excludes_null_confidence(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c5" not in control_ids

    def test_excludes_ood(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c4" not in control_ids

    def test_excludes_below_global_floor(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(filter_db, confidence_floor=0.30, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c3" not in control_ids

    def test_keeps_above_global_floor(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(filter_db, confidence_floor=0.30, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw1:c2" in control_ids
        assert "fw2:c1" in control_ids

    def test_excludes_non_accepted(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        control_ids = {r["control_id"] for r in rows}
        assert "fw2:c2" not in control_ids

    def test_per_framework_override(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(
            filter_db, confidence_floor=0.30,
            confidence_overrides={"fw1": 0.65},
        )
        control_ids = {r["control_id"] for r in rows}
        # fw1:c2 has confidence 0.6 which is below fw1's override of 0.65
        assert "fw1:c2" not in control_ids
        # fw2:c1 has confidence 0.7 which is above global floor 0.30
        assert "fw2:c1" in control_ids

    def test_returns_required_columns(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(filter_db, confidence_floor=0.0, confidence_overrides={})
        assert len(rows) > 0
        row = rows[0]
        required_keys = {"control_id", "hub_id", "hub_name", "confidence",
                         "framework_id", "section_id", "title", "description"}
        assert required_keys.issubset(set(row.keys()))

    def test_filter_stats_counts(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments, compute_filter_stats

        rows = query_exportable_assignments(filter_db, confidence_floor=0.30, confidence_overrides={})
        stats = compute_filter_stats(filter_db, rows, confidence_floor=0.30, confidence_overrides={})
        assert stats["fw1"]["exported"] == 1  # only fw1:c2 survives
        assert stats["fw1"]["excluded_ground_truth"] == 1
        assert stats["fw1"]["excluded_confidence"] >= 1
        assert stats["fw1"]["excluded_ood"] == 1

    def test_framework_filter(self, filter_db) -> None:
        from tract.export.filters import query_exportable_assignments

        rows = query_exportable_assignments(
            filter_db, confidence_floor=0.0, confidence_overrides={},
            framework_filter="fw2",
        )
        framework_ids = {r["framework_id"] for r in rows}
        assert framework_ids == {"fw2"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_export_filters.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the filter module**

Create `tract/export/filters.py`:

```python
"""OpenCRE export filter pipeline (spec §5).

Filters are applied in SQL for efficiency:
1. Ground truth exclusion (provenance != 'ground_truth_T1-AI')
2. NULL confidence exclusion
3. OOD exclusion (is_ood != 1)
4. Only accepted review_status
5. Per-framework confidence floor applied in Python (SQL can't do per-framework overrides cleanly)
"""
from __future__ import annotations

import logging
from pathlib import Path

from tract.config import PHASE5_GROUND_TRUTH_PROVENANCE
from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


def query_exportable_assignments(
    db_path: Path,
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    framework_filter: str | None = None,
) -> list[dict]:
    """Query assignments passing all export filters.

    Returns list of dicts with keys: control_id, hub_id, hub_name,
    confidence, framework_id, section_id, title, description.
    Sorted by (hub_id, framework_id, section_id) per spec §3.
    """
    conn = get_connection(db_path)
    try:
        query = (
            "SELECT a.control_id, a.hub_id, h.name AS hub_name, "
            "a.confidence, a.is_ood, a.provenance, "
            "c.framework_id, c.section_id, c.title, c.description "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id "
            "JOIN hubs h ON a.hub_id = h.id "
            "WHERE a.review_status = 'accepted' "
            "AND a.provenance != ? "
            "AND a.confidence IS NOT NULL "
            "AND a.is_ood != 1 "
        )
        params: list = [PHASE5_GROUND_TRUTH_PROVENANCE]

        if framework_filter:
            query += "AND c.framework_id = ? "
            params.append(framework_filter)

        query += "ORDER BY a.hub_id, c.framework_id, c.section_id"

        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        fw_id = row["framework_id"]
        floor = confidence_overrides.get(fw_id, confidence_floor)
        if row["confidence"] < floor:
            logger.debug(
                "Excluded %s: confidence %.3f < floor %.3f (framework=%s)",
                row["control_id"], row["confidence"], floor, fw_id,
            )
            continue
        results.append(dict(row))

    logger.info(
        "Export filter: %d assignments passed (%d excluded by confidence floor)",
        len(results), len(rows) - len(results),
    )
    return results


def compute_filter_stats(
    db_path: Path,
    exported_rows: list[dict],
    confidence_floor: float,
    confidence_overrides: dict[str, float],
) -> dict[str, dict[str, int]]:
    """Compute per-framework filter statistics for the export manifest."""
    conn = get_connection(db_path)
    try:
        all_rows = conn.execute(
            "SELECT a.control_id, a.hub_id, a.confidence, a.is_ood, "
            "a.provenance, a.review_status, c.framework_id "
            "FROM assignments a "
            "JOIN controls c ON a.control_id = c.id"
        ).fetchall()
    finally:
        conn.close()

    exported_keys = {(r["control_id"], r["hub_id"]) for r in exported_rows}

    stats: dict[str, dict[str, int]] = {}
    for row in all_rows:
        fw_id = row["framework_id"]
        if fw_id not in stats:
            stats[fw_id] = {
                "exported": 0,
                "excluded_ground_truth": 0,
                "excluded_confidence": 0,
                "excluded_ood": 0,
                "excluded_null_confidence": 0,
                "excluded_not_accepted": 0,
            }
        s = stats[fw_id]
        key = (row["control_id"], row["hub_id"])

        if key in exported_keys:
            s["exported"] += 1
        elif row["provenance"] == PHASE5_GROUND_TRUTH_PROVENANCE:
            s["excluded_ground_truth"] += 1
        elif row["review_status"] != "accepted":
            s["excluded_not_accepted"] += 1
        elif row["confidence"] is None:
            s["excluded_null_confidence"] += 1
        elif row["is_ood"] == 1:
            s["excluded_ood"] += 1
        else:
            floor = confidence_overrides.get(fw_id, confidence_floor)
            if row["confidence"] < floor:
                s["excluded_confidence"] += 1

    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_export_filters.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tract/export/filters.py tests/test_export_filters.py
git commit -m "feat(export): add filter pipeline for OpenCRE export"
```

---

### Task 4: OpenCRE CSV Generator

**Files:**
- Create: `tract/export/opencre_csv.py`
- Test: `tests/test_opencre_csv.py`

- [ ] **Step 1: Write the failing tests for CSV generation**

Create `tests/test_opencre_csv.py`:

```python
"""Tests for OpenCRE CSV format generation."""
from __future__ import annotations

import csv
from io import StringIO

import pytest


class TestOpenCRECSVFormat:
    def test_header_columns(self) -> None:
        from tract.export.opencre_csv import generate_opencre_csv

        rows = [
            {
                "hub_id": "607-671", "hub_name": "Protect against JS injection",
                "framework_id": "csa_aicm", "section_id": "A&A-01",
                "title": "Audit Policy", "description": "Establish audit policies.",
                "confidence": 0.6, "control_id": "csa_aicm:A&A-01",
            },
        ]
        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        fieldnames = reader.fieldnames
        assert "CRE 0" in fieldnames
        assert "CSA AI Controls Matrix|name" in fieldnames
        assert "CSA AI Controls Matrix|id" in fieldnames
        assert "CSA AI Controls Matrix|description" in fieldnames
        assert "CSA AI Controls Matrix|hyperlink" in fieldnames

    def test_cre0_pipe_delimited_format(self) -> None:
        from tract.export.opencre_csv import generate_opencre_csv

        rows = [
            {
                "hub_id": "607-671", "hub_name": "Protect against JS injection",
                "framework_id": "csa_aicm", "section_id": "A&A-01",
                "title": "Audit Policy", "description": "Establish audit policies.",
                "confidence": 0.6, "control_id": "csa_aicm:A&A-01",
            },
        ]
        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        row = next(reader)
        cre0 = row["CRE 0"]
        assert "|" in cre0
        parts = cre0.split("|")
        assert parts[0] == "607-671"
        assert parts[1] == "Protect against JS injection"

    def test_no_hierarchy_columns(self) -> None:
        from tract.export.opencre_csv import generate_opencre_csv

        rows = [
            {
                "hub_id": "607-671", "hub_name": "Hub",
                "framework_id": "csa_aicm", "section_id": "A&A-01",
                "title": "T", "description": "D",
                "confidence": 0.6, "control_id": "csa_aicm:A&A-01",
            },
        ]
        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        fieldnames = reader.fieldnames
        for fn in fieldnames:
            assert not fn.startswith("CRE 1"), f"Unexpected hierarchy column: {fn}"
            assert not fn.startswith("CRE 2"), f"Unexpected hierarchy column: {fn}"

    def test_multiple_hubs_same_control_produces_multiple_rows(self) -> None:
        from tract.export.opencre_csv import generate_opencre_csv

        rows = [
            {
                "hub_id": "111-222", "hub_name": "Hub A",
                "framework_id": "csa_aicm", "section_id": "A&A-01",
                "title": "Audit Policy", "description": "Desc.",
                "confidence": 0.6, "control_id": "csa_aicm:A&A-01",
            },
            {
                "hub_id": "333-444", "hub_name": "Hub B",
                "framework_id": "csa_aicm", "section_id": "A&A-01",
                "title": "Audit Policy", "description": "Desc.",
                "confidence": 0.5, "control_id": "csa_aicm:A&A-01",
            },
        ]
        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        csv_rows = list(reader)
        assert len(csv_rows) == 2
        cre0_values = {r["CRE 0"] for r in csv_rows}
        assert len(cre0_values) == 2

    def test_rows_sorted_by_hub_framework_section(self) -> None:
        from tract.export.opencre_csv import generate_opencre_csv

        rows = [
            {
                "hub_id": "333-444", "hub_name": "Hub B",
                "framework_id": "csa_aicm", "section_id": "B-02",
                "title": "T2", "description": "D2",
                "confidence": 0.5, "control_id": "csa_aicm:B-02",
            },
            {
                "hub_id": "111-222", "hub_name": "Hub A",
                "framework_id": "csa_aicm", "section_id": "A-01",
                "title": "T1", "description": "D1",
                "confidence": 0.6, "control_id": "csa_aicm:A-01",
            },
        ]
        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        csv_rows = list(reader)
        # 111-222 should come before 333-444
        assert csv_rows[0]["CRE 0"].startswith("111-222")
        assert csv_rows[1]["CRE 0"].startswith("333-444")

    def test_standard_columns_populated(self) -> None:
        from tract.export.opencre_csv import generate_opencre_csv

        rows = [
            {
                "hub_id": "607-671", "hub_name": "Hub",
                "framework_id": "csa_aicm", "section_id": "A&A-01",
                "title": "Audit Policy", "description": "Establish audit policies.",
                "confidence": 0.6, "control_id": "csa_aicm:A&A-01",
            },
        ]
        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        row = next(reader)
        assert row["CSA AI Controls Matrix|name"] == "Audit Policy"
        assert row["CSA AI Controls Matrix|id"] == "A&A-01"
        assert row["CSA AI Controls Matrix|description"] == "Establish audit policies."
        assert row["CSA AI Controls Matrix|hyperlink"] != ""

    def test_empty_rows_produces_header_only(self) -> None:
        from tract.export.opencre_csv import generate_opencre_csv

        csv_text = generate_opencre_csv([], "csa_aicm")
        lines = csv_text.strip().split("\n")
        assert len(lines) == 1  # header only
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_opencre_csv.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the CSV generator**

Create `tract/export/opencre_csv.py`:

```python
"""Generate OpenCRE-compatible CSV from filtered assignments (spec §3).

Output targets OpenCRE's parse_export_format() parser. Each row has:
- CRE 0: "hub_id|hub_name" (pipe-delimited)
- StandardName|name: control title
- StandardName|id: section_id
- StandardName|description: control description
- StandardName|hyperlink: URL to official source

CRE 0 ONLY — no hierarchy columns. See spec §3 for parser safety proof.
"""
from __future__ import annotations

import csv
import logging
import os
import tempfile
from io import StringIO
from pathlib import Path

from tract.export.opencre_names import build_hyperlink, get_opencre_name

logger = logging.getLogger(__name__)


def generate_opencre_csv(rows: list[dict], framework_id: str) -> str:
    """Generate OpenCRE CSV string from filtered assignment rows.

    Args:
        rows: List of assignment dicts from filters.query_exportable_assignments().
        framework_id: TRACT framework_id (used to determine column names).

    Returns:
        CSV string ready to write to file.
    """
    opencre_name = get_opencre_name(framework_id)

    fieldnames = [
        "CRE 0",
        f"{opencre_name}|name",
        f"{opencre_name}|id",
        f"{opencre_name}|description",
        f"{opencre_name}|hyperlink",
    ]

    sorted_rows = sorted(rows, key=lambda r: (r["hub_id"], r["framework_id"], r["section_id"]))

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for row in sorted_rows:
        cre0 = f"{row['hub_id']}|{row['hub_name']}"
        hyperlink = build_hyperlink(framework_id, row["section_id"])

        writer.writerow({
            "CRE 0": cre0,
            f"{opencre_name}|name": row["title"],
            f"{opencre_name}|id": row["section_id"],
            f"{opencre_name}|description": row["description"],
            f"{opencre_name}|hyperlink": hyperlink,
        })

    return output.getvalue()


def write_opencre_csv(
    rows: list[dict],
    framework_id: str,
    output_dir: Path,
) -> Path:
    """Generate and atomically write OpenCRE CSV to output_dir.

    Returns path to the written CSV file.
    """
    csv_text = generate_opencre_csv(rows, framework_id)
    opencre_name = get_opencre_name(framework_id)
    safe_name = opencre_name.replace(" ", "_").replace("/", "_")
    output_path = output_dir / f"{safe_name}.csv"

    output_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=f".{output_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(csv_text)
        os.replace(tmp, output_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Wrote %d rows to %s", len(rows), output_path)
    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_opencre_csv.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tract/export/opencre_csv.py tests/test_opencre_csv.py
git commit -m "feat(export): add OpenCRE CSV generator (CRE 0 only format)"
```

---

### Task 5: Pre-Export Staleness Check

**Files:**
- Create: `tract/export/staleness.py`
- Test: `tests/test_staleness.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_staleness.py`:

```python
"""Tests for pre-export CRE ID staleness check."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import insert_hubs


@pytest.fixture
def staleness_db(tmp_path):
    db_path = tmp_path / "staleness.db"
    create_database(db_path)
    insert_hubs(db_path, [
        {"id": "111-222", "name": "Hub A", "path": "R > A", "parent_id": None},
        {"id": "333-444", "name": "Hub B", "path": "R > B", "parent_id": None},
        {"id": "555-666", "name": "Hub C", "path": "R > C", "parent_id": None},
    ])
    return db_path


class TestStalenessCheck:
    def test_all_match_returns_pass(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness

        upstream_ids = {"111-222", "333-444", "555-666"}
        with patch("tract.export.staleness._fetch_upstream_cre_ids", return_value=upstream_ids):
            result = check_staleness(staleness_db)
        assert result["status"] == "pass"
        assert result["stale_ids"] == []

    def test_tract_has_extra_ids_returns_warn(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness

        upstream_ids = {"111-222", "333-444"}  # missing 555-666
        with patch("tract.export.staleness._fetch_upstream_cre_ids", return_value=upstream_ids):
            result = check_staleness(staleness_db)
        assert result["status"] == "warn"
        assert "555-666" in result["stale_ids"]

    def test_upstream_has_extra_ids_returns_pass(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness

        upstream_ids = {"111-222", "333-444", "555-666", "999-999"}
        with patch("tract.export.staleness._fetch_upstream_cre_ids", return_value=upstream_ids):
            result = check_staleness(staleness_db)
        assert result["status"] == "pass"
        assert result["upstream_only"] == ["999-999"]

    def test_network_error_returns_error(self, staleness_db) -> None:
        from tract.export.staleness import check_staleness

        with patch("tract.export.staleness._fetch_upstream_cre_ids", side_effect=ConnectionError("timeout")):
            result = check_staleness(staleness_db)
        assert result["status"] == "error"
        assert "timeout" in result["message"]

    def test_get_tract_hub_ids(self, staleness_db) -> None:
        from tract.export.staleness import _get_tract_hub_ids

        ids = _get_tract_hub_ids(staleness_db)
        assert ids == {"111-222", "333-444", "555-666"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_staleness.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the staleness check**

Create `tract/export/staleness.py`:

```python
"""Pre-export CRE ID staleness check (spec §6).

Fetches current CRE IDs from upstream OpenCRE, diffs against TRACT's
hub snapshot. Warns if TRACT references hub IDs that no longer exist
upstream (may have been removed/merged).
"""
from __future__ import annotations

import logging
from pathlib import Path

import requests

from tract.config import (
    PHASE5_OPENCRE_STALENESS_TIMEOUT_S,
    PHASE5_OPENCRE_STALENESS_URL,
)
from tract.crosswalk.schema import get_connection

logger = logging.getLogger(__name__)


def _get_tract_hub_ids(db_path: Path) -> set[str]:
    """Get all hub IDs from TRACT's crosswalk database."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT id FROM hubs").fetchall()
        return {row["id"] for row in rows}
    finally:
        conn.close()


def _fetch_upstream_cre_ids() -> set[str]:
    """Fetch all CRE IDs from upstream OpenCRE API.

    Walks the root_cres endpoint and recursively collects all CRE IDs.
    """
    resp = requests.get(
        PHASE5_OPENCRE_STALENESS_URL,
        timeout=PHASE5_OPENCRE_STALENESS_TIMEOUT_S,
    )
    resp.raise_for_status()
    data = resp.json()

    ids: set[str] = set()
    _collect_ids(data, ids)
    return ids


def _collect_ids(node: dict | list, ids: set[str]) -> None:
    """Recursively collect CRE IDs from the API response tree."""
    if isinstance(node, list):
        for item in node:
            _collect_ids(item, ids)
    elif isinstance(node, dict):
        if "id" in node:
            ids.add(node["id"])
        for key in ("links", "children", "contains"):
            if key in node and isinstance(node[key], list):
                for child in node[key]:
                    if isinstance(child, dict) and "document" in child:
                        _collect_ids(child["document"], ids)
                    elif isinstance(child, dict):
                        _collect_ids(child, ids)


def check_staleness(db_path: Path) -> dict:
    """Run pre-export staleness check.

    Returns dict with:
        status: "pass" | "warn" | "error"
        stale_ids: list of TRACT hub IDs not found upstream
        upstream_only: list of upstream IDs not in TRACT (info only)
        upstream_hub_count: total upstream CRE count
        message: human-readable summary
    """
    tract_ids = _get_tract_hub_ids(db_path)

    try:
        upstream_ids = _fetch_upstream_cre_ids()
    except Exception as e:
        logger.error("Staleness check failed: %s", e)
        return {
            "status": "error",
            "stale_ids": [],
            "upstream_only": [],
            "upstream_hub_count": 0,
            "message": str(e),
        }

    stale = sorted(tract_ids - upstream_ids)
    upstream_only = sorted(upstream_ids - tract_ids)

    if stale:
        logger.warning(
            "Staleness check: %d TRACT hub IDs not found upstream: %s",
            len(stale), stale,
        )
        status = "warn"
        message = f"{len(stale)} TRACT hub IDs not found in upstream OpenCRE"
    else:
        status = "pass"
        message = "All TRACT hub IDs found in upstream OpenCRE"

    if upstream_only:
        logger.info(
            "Staleness check: %d upstream IDs not in TRACT (new hubs, not our concern)",
            len(upstream_only),
        )

    return {
        "status": status,
        "stale_ids": stale,
        "upstream_only": upstream_only,
        "upstream_hub_count": len(upstream_ids),
        "message": message,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_staleness.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tract/export/staleness.py tests/test_staleness.py
git commit -m "feat(export): add pre-export CRE ID staleness check"
```

---

### Task 6: Export Manifest

**Files:**
- Create: `tract/export/manifest.py`
- Test: `tests/test_export_manifest.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export_manifest.py`:

```python
"""Tests for export manifest generation."""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest


class TestExportManifest:
    def test_manifest_has_required_fields(self) -> None:
        from tract.export.manifest import build_manifest

        per_framework = {
            "csa_aicm": {"exported": 10, "excluded_confidence": 5, "excluded_ood": 0,
                         "excluded_ground_truth": 0, "excluded_null_confidence": 0, "excluded_not_accepted": 0},
        }
        staleness = {"status": "pass", "upstream_hub_count": 522}

        manifest = build_manifest(
            per_framework_stats=per_framework,
            confidence_floor=0.30,
            confidence_overrides={"mitre_atlas": 0.35},
            staleness_result=staleness,
            model_adapter_hash="abc123",
        )

        assert manifest["confidence_floor"] == 0.30
        assert manifest["confidence_overrides"] == {"mitre_atlas": 0.35}
        assert manifest["staleness_check"]["status"] == "pass"
        assert manifest["total_exported"] == 10
        assert manifest["total_excluded"] == 5
        assert "export_date" in manifest
        assert "tract_git_sha" in manifest
        assert manifest["model_adapter_hash"] == "abc123"

    def test_manifest_sums_across_frameworks(self) -> None:
        from tract.export.manifest import build_manifest

        per_framework = {
            "fw1": {"exported": 10, "excluded_confidence": 5, "excluded_ood": 1,
                    "excluded_ground_truth": 2, "excluded_null_confidence": 0, "excluded_not_accepted": 0},
            "fw2": {"exported": 20, "excluded_confidence": 3, "excluded_ood": 0,
                    "excluded_ground_truth": 0, "excluded_null_confidence": 1, "excluded_not_accepted": 0},
        }

        manifest = build_manifest(
            per_framework_stats=per_framework,
            confidence_floor=0.30,
            confidence_overrides={},
            staleness_result={"status": "pass", "upstream_hub_count": 100},
            model_adapter_hash="abc",
        )

        assert manifest["total_exported"] == 30
        assert manifest["total_excluded"] == 12

    def test_manifest_is_json_serializable(self) -> None:
        from tract.export.manifest import build_manifest

        manifest = build_manifest(
            per_framework_stats={},
            confidence_floor=0.30,
            confidence_overrides={},
            staleness_result={"status": "pass", "upstream_hub_count": 0},
            model_adapter_hash="abc",
        )
        serialized = json.dumps(manifest, sort_keys=True, indent=2)
        roundtripped = json.loads(serialized)
        assert roundtripped["confidence_floor"] == 0.30

    def test_manifest_git_sha_format(self) -> None:
        from tract.export.manifest import build_manifest

        manifest = build_manifest(
            per_framework_stats={},
            confidence_floor=0.30,
            confidence_overrides={},
            staleness_result={"status": "pass", "upstream_hub_count": 0},
            model_adapter_hash="abc",
        )
        sha = manifest["tract_git_sha"]
        assert isinstance(sha, str)
        assert len(sha) >= 7 or sha == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_export_manifest.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement the manifest module**

Create `tract/export/manifest.py`:

```python
"""Export manifest generation (spec §7).

Every export produces export_manifest.json alongside the CSVs,
capturing provenance, filter settings, and per-framework counts.
"""
from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    """Get current git SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def build_manifest(
    per_framework_stats: dict[str, dict[str, int]],
    confidence_floor: float,
    confidence_overrides: dict[str, float],
    staleness_result: dict,
    model_adapter_hash: str,
) -> dict:
    """Build the export manifest dict.

    Args:
        per_framework_stats: Output of filters.compute_filter_stats().
        confidence_floor: Global confidence floor used.
        confidence_overrides: Per-framework overrides used.
        staleness_result: Output of staleness.check_staleness().
        model_adapter_hash: SHA256 hash of the model adapter file.
    """
    total_exported = sum(
        s.get("exported", 0) for s in per_framework_stats.values()
    )
    total_excluded = sum(
        s.get("excluded_confidence", 0) + s.get("excluded_ood", 0) +
        s.get("excluded_ground_truth", 0) + s.get("excluded_null_confidence", 0) +
        s.get("excluded_not_accepted", 0)
        for s in per_framework_stats.values()
    )

    return {
        "tract_version": "0.1.0",
        "model_adapter_hash": model_adapter_hash,
        "confidence_floor": confidence_floor,
        "confidence_overrides": confidence_overrides,
        "export_date": datetime.now(timezone.utc).isoformat(),
        "tract_git_sha": _get_git_sha(),
        "staleness_check": {
            "status": staleness_result.get("status", "unknown"),
            "upstream_hub_count": staleness_result.get("upstream_hub_count", 0),
        },
        "per_framework": per_framework_stats,
        "total_exported": total_exported,
        "total_excluded": total_excluded,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_export_manifest.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tract/export/manifest.py tests/test_export_manifest.py
git commit -m "feat(export): add export manifest generator"
```

---

### Task 7: CLI Integration

**Files:**
- Modify: `tract/cli.py`
- Test: `tests/test_cli.py` (add new tests)

- [ ] **Step 1: Write the failing tests for CLI export --opencre**

Append to `tests/test_cli.py` (after existing test classes):

```python
class TestExportOpenCRECLI:
    def test_opencre_flag_recognized(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre", "--dry-run"])
        assert args.opencre is True
        assert args.dry_run is True

    def test_opencre_with_framework_filter(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre", "--framework", "nist_ai_600_1"])
        assert args.opencre is True
        assert args.framework == "nist_ai_600_1"

    def test_opencre_with_output_dir(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre", "--output-dir", "/tmp/test"])
        assert args.output_dir == "/tmp/test"

    def test_opencre_proposals_flag_recognized(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre-proposals", "--output-dir", "/tmp/test"])
        assert args.opencre_proposals is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli.py::TestExportOpenCRECLI -v`
Expected: FAIL — `AttributeError: ...no attribute 'opencre'`

- [ ] **Step 3: Add new CLI flags to the export subparser in cli.py**

In `tract/cli.py`, find the `# ── export ───` section and add these flags to `p_export`:

```python
    p_export.add_argument("--opencre", action="store_true",
                          help="Export in OpenCRE CSV format (one CSV per framework)")
    p_export.add_argument("--opencre-proposals", action="store_true",
                          help="Export hub proposals document for OpenCRE")
    p_export.add_argument("--output-dir",
                          help="Output directory for OpenCRE export (default: ./opencre_export/)")
    p_export.add_argument("--dry-run", action="store_true",
                          help="Show what would be exported without writing files")
    p_export.add_argument("--skip-staleness", action="store_true",
                          help="Skip pre-export staleness check (offline mode)")
```

- [ ] **Step 4: Run tests to verify parser changes pass**

Run: `pytest tests/test_cli.py::TestExportOpenCRECLI -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Implement _cmd_export_opencre handler**

Add to `tract/cli.py` after the existing `_cmd_export` function:

```python
def _cmd_export_opencre(args: argparse.Namespace) -> None:
    from tract.config import (
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR,
        PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES,
    )
    from tract.export.filters import compute_filter_stats, query_exportable_assignments
    from tract.export.manifest import build_manifest
    from tract.export.opencre_csv import generate_opencre_csv, write_opencre_csv
    from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME
    from tract.io import atomic_write_json

    output_dir = Path(args.output_dir) if args.output_dir else Path("./opencre_export")

    confidence_floor = PHASE5_OPENCRE_EXPORT_CONFIDENCE_FLOOR
    confidence_overrides = dict(PHASE5_OPENCRE_EXPORT_CONFIDENCE_OVERRIDES)

    # Staleness check
    staleness_result = {"status": "skipped", "upstream_hub_count": 0, "message": "skipped"}
    if not args.skip_staleness and not args.dry_run:
        from tract.export.staleness import check_staleness

        print("Running pre-export staleness check...")
        staleness_result = check_staleness(PHASE1C_CROSSWALK_DB_PATH)
        if staleness_result["status"] == "warn":
            print(f"WARNING: {staleness_result['message']}")
            print(f"  Stale IDs: {staleness_result['stale_ids']}")
        elif staleness_result["status"] == "error":
            print(f"ERROR: Staleness check failed: {staleness_result['message']}")
            print("  Use --skip-staleness to bypass (offline mode)")
            sys.exit(1)
        else:
            print(f"Staleness check passed ({staleness_result['upstream_hub_count']} upstream hubs)")

    # Determine which frameworks to export
    if args.framework:
        if args.framework not in TRACT_TO_OPENCRE_NAME:
            print(f"Error: Framework '{args.framework}' has no OpenCRE name mapping", file=sys.stderr)
            print(f"  Available: {', '.join(sorted(TRACT_TO_OPENCRE_NAME.keys()))}", file=sys.stderr)
            sys.exit(1)
        frameworks = [args.framework]
    else:
        frameworks = sorted(TRACT_TO_OPENCRE_NAME.keys())

    # Query and filter
    all_rows: dict[str, list[dict]] = {}
    for fw_id in frameworks:
        rows = query_exportable_assignments(
            PHASE1C_CROSSWALK_DB_PATH,
            confidence_floor=confidence_floor,
            confidence_overrides=confidence_overrides,
            framework_filter=fw_id,
        )
        if rows:
            all_rows[fw_id] = rows

    if not all_rows:
        print("No assignments survived filters. Nothing to export.")
        return

    # Dry run — show summary and exit
    if args.dry_run:
        print("\nDry run — would export:\n")
        total = 0
        for fw_id, rows in sorted(all_rows.items()):
            print(f"  {fw_id}: {len(rows)} assignments")
            total += len(rows)
        print(f"\n  Total: {total} assignments")
        return

    # Write CSVs
    written_files: list[Path] = []
    total_exported = 0
    for fw_id, rows in sorted(all_rows.items()):
        csv_path = write_opencre_csv(rows, fw_id, output_dir)
        written_files.append(csv_path)
        total_exported += len(rows)
        print(f"  {fw_id}: {len(rows)} assignments → {csv_path}")

    # Build and write manifest
    all_exported = []
    for rows in all_rows.values():
        all_exported.extend(rows)

    stats = compute_filter_stats(
        PHASE1C_CROSSWALK_DB_PATH, all_exported,
        confidence_floor, confidence_overrides,
    )

    model_hash = "unknown"
    from tract.config import PHASE1D_ARTIFACTS_PATH
    if PHASE1D_ARTIFACTS_PATH.exists():
        import numpy as np
        data = np.load(str(PHASE1D_ARTIFACTS_PATH), allow_pickle=True)
        model_hash = str(data["model_adapter_hash"])

    manifest = build_manifest(
        per_framework_stats=stats,
        confidence_floor=confidence_floor,
        confidence_overrides=confidence_overrides,
        staleness_result=staleness_result,
        model_adapter_hash=model_hash,
    )

    manifest_path = output_dir / "export_manifest.json"
    atomic_write_json(manifest, manifest_path)
    print(f"\n  Manifest: {manifest_path}")
    print(f"  Total exported: {total_exported} assignments across {len(written_files)} frameworks")
```

- [ ] **Step 6: Wire the handler into the export command**

In `_cmd_export`, add an early check at the top of the function:

```python
def _cmd_export(args: argparse.Namespace) -> None:
    if getattr(args, "opencre", False):
        _cmd_export_opencre(args)
        return
    if getattr(args, "opencre_proposals", False):
        _cmd_export_opencre_proposals(args)
        return

    from tract.crosswalk.export import export_crosswalk
    # ... rest of existing code unchanged
```

- [ ] **Step 7: Add stub for opencre-proposals handler**

Add to `tract/cli.py`:

```python
def _cmd_export_opencre_proposals(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir) if args.output_dir else Path("./opencre_export")
    output_dir.mkdir(parents=True, exist_ok=True)

    from tract.io import load_json

    proposals_dir = HUB_PROPOSALS_DIR
    if not proposals_dir.exists():
        print("No hub proposals found. Run 'tract propose-hubs' first.")
        return

    rounds = sorted(proposals_dir.glob("round_*"))
    if not rounds:
        print("No proposal rounds found.")
        return

    latest_round = rounds[-1]
    summary_path = latest_round / "summary.json"
    if not summary_path.exists():
        print(f"No summary.json in {latest_round}")
        return

    summary = load_json(summary_path)
    proposals_output = {
        "source": "TRACT hub proposal pipeline",
        "round": latest_round.name,
        "proposals": summary.get("proposals", []),
    }

    from tract.io import atomic_write_json

    out_path = output_dir / "hub_proposals_for_opencre.json"
    atomic_write_json(proposals_output, out_path)
    print(f"Hub proposals written to {out_path}")
```

- [ ] **Step 8: Run all CLI tests**

Run: `pytest tests/test_cli.py -v`
Expected: All tests PASS (existing + new)

- [ ] **Step 9: Commit**

```bash
git add tract/cli.py tests/test_cli.py
git commit -m "feat(export): add tract export --opencre CLI command"
```

---

### Task 8: End-to-End Export Test

**Files:**
- Create: `tests/test_opencre_export_e2e.py`

- [ ] **Step 1: Write the e2e test**

Create `tests/test_opencre_export_e2e.py`:

```python
"""End-to-end test: crosswalk.db → filters → CSV → manifest."""
from __future__ import annotations

import csv
import json
from io import StringIO

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def e2e_db(tmp_path):
    """Realistic mini-DB with multiple frameworks and mixed assignments."""
    db_path = tmp_path / "e2e.db"
    create_database(db_path)

    insert_frameworks(db_path, [
        {"id": "csa_aicm", "name": "CSA AI Controls Matrix", "version": "1.0",
         "fetch_date": "2026-04-30", "control_count": 3},
        {"id": "mitre_atlas", "name": "MITRE ATLAS", "version": "1.0",
         "fetch_date": "2026-04-30", "control_count": 2},
    ])
    insert_hubs(db_path, [
        {"id": "217-168", "name": "Audit & accountability", "path": "R > Audit", "parent_id": None},
        {"id": "607-671", "name": "Protect against injection", "path": "R > Injection", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "csa_aicm:A&A-01", "framework_id": "csa_aicm", "section_id": "A&A-01",
         "title": "Audit Policy", "description": "Establish audit policies.", "full_text": None},
        {"id": "csa_aicm:A&A-02", "framework_id": "csa_aicm", "section_id": "A&A-02",
         "title": "Independent Assessments", "description": "Conduct assessments.", "full_text": None},
        {"id": "csa_aicm:A&A-03", "framework_id": "csa_aicm", "section_id": "A&A-03",
         "title": "Risk Planning", "description": "Risk-based plans.", "full_text": None},
        {"id": "mitre_atlas:AML.T0000", "framework_id": "mitre_atlas", "section_id": "AML.T0000",
         "title": "Search Databases", "description": "Search technical databases.", "full_text": None},
        {"id": "mitre_atlas:AML.M0015", "framework_id": "mitre_atlas", "section_id": "AML.M0015",
         "title": "Adversarial Input Detection", "description": "Detect adversarial inputs.", "full_text": None},
    ])
    insert_assignments(db_path, [
        # CSA — 2 accepted above floor, 1 below floor
        {"control_id": "csa_aicm:A&A-01", "hub_id": "217-168", "confidence": 0.60, "in_conformal_set": 1,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        {"control_id": "csa_aicm:A&A-02", "hub_id": "217-168", "confidence": 0.33, "in_conformal_set": 0,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        {"control_id": "csa_aicm:A&A-03", "hub_id": "607-671", "confidence": 0.20, "in_conformal_set": 0,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        # ATLAS — 1 accepted above override floor, 1 ground_truth
        {"control_id": "mitre_atlas:AML.T0000", "hub_id": "607-671", "confidence": 0.50, "in_conformal_set": 1,
         "is_ood": 0, "provenance": "active_learning_round_2", "source_link_id": None,
         "model_version": "v1", "review_status": "accepted"},
        {"control_id": "mitre_atlas:AML.M0015", "hub_id": "217-168", "confidence": 0.90, "in_conformal_set": 1,
         "is_ood": 0, "provenance": "ground_truth_T1-AI", "source_link_id": None,
         "model_version": "v1", "review_status": "ground_truth"},
    ])
    return db_path


class TestOpenCREExportE2E:
    def test_full_pipeline_csa(self, e2e_db, tmp_path) -> None:
        from tract.export.filters import query_exportable_assignments
        from tract.export.opencre_csv import generate_opencre_csv, write_opencre_csv

        rows = query_exportable_assignments(
            e2e_db, confidence_floor=0.30, confidence_overrides={},
            framework_filter="csa_aicm",
        )
        # A&A-01 (0.60) and A&A-02 (0.33) survive; A&A-03 (0.20) excluded
        assert len(rows) == 2

        csv_text = generate_opencre_csv(rows, "csa_aicm")
        reader = csv.DictReader(StringIO(csv_text))
        csv_rows = list(reader)
        assert len(csv_rows) == 2

        assert "CSA AI Controls Matrix|name" in reader.fieldnames
        assert csv_rows[0]["CRE 0"].startswith("217-168|")

    def test_full_pipeline_atlas_with_override(self, e2e_db, tmp_path) -> None:
        from tract.export.filters import query_exportable_assignments
        from tract.export.opencre_csv import generate_opencre_csv

        rows = query_exportable_assignments(
            e2e_db, confidence_floor=0.30,
            confidence_overrides={"mitre_atlas": 0.35},
            framework_filter="mitre_atlas",
        )
        # AML.T0000 (0.50 >= 0.35) survives; AML.M0015 excluded (ground_truth)
        assert len(rows) == 1
        assert rows[0]["section_id"] == "AML.T0000"

        csv_text = generate_opencre_csv(rows, "mitre_atlas")
        reader = csv.DictReader(StringIO(csv_text))
        csv_rows = list(reader)
        assert len(csv_rows) == 1
        assert "MITRE ATLAS|name" in reader.fieldnames

    def test_full_pipeline_writes_files(self, e2e_db, tmp_path) -> None:
        from tract.export.filters import compute_filter_stats, query_exportable_assignments
        from tract.export.manifest import build_manifest
        from tract.export.opencre_csv import write_opencre_csv
        from tract.io import atomic_write_json

        output_dir = tmp_path / "export"
        all_exported = []

        for fw_id in ["csa_aicm", "mitre_atlas"]:
            rows = query_exportable_assignments(
                e2e_db, confidence_floor=0.30,
                confidence_overrides={"mitre_atlas": 0.35},
                framework_filter=fw_id,
            )
            if rows:
                write_opencre_csv(rows, fw_id, output_dir)
                all_exported.extend(rows)

        assert (output_dir / "CSA_AI_Controls_Matrix.csv").exists()
        assert (output_dir / "MITRE_ATLAS.csv").exists()

        stats = compute_filter_stats(
            e2e_db, all_exported, 0.30, {"mitre_atlas": 0.35},
        )
        manifest = build_manifest(
            per_framework_stats=stats,
            confidence_floor=0.30,
            confidence_overrides={"mitre_atlas": 0.35},
            staleness_result={"status": "skipped", "upstream_hub_count": 0},
            model_adapter_hash="test_hash",
        )
        manifest_path = output_dir / "export_manifest.json"
        atomic_write_json(manifest, manifest_path)

        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert loaded["total_exported"] == 3  # 2 CSA + 1 ATLAS
        assert loaded["confidence_floor"] == 0.30
```

- [ ] **Step 2: Run e2e tests**

Run: `pytest tests/test_opencre_export_e2e.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_opencre_export_e2e.py
git commit -m "test(export): add end-to-end OpenCRE export tests"
```

---

### Task 9: Fork Import Helper Script

**Files:**
- Create: `scripts/opencre_import.sh`

- [ ] **Step 1: Create the import helper script**

Create `scripts/opencre_import.sh`:

```bash
#!/usr/bin/env bash
# opencre_import.sh — Import TRACT-generated CSV into a local OpenCRE fork.
#
# Usage: ./scripts/opencre_import.sh <csv_file>
#
# Prerequisites:
#   - OpenCRE fork at ~/github_projects/OpenCRE
#   - Fork initialized: CRE_ALLOW_IMPORT=1 python cre.py --upstream_sync --cache_file cre.db
#
# This script:
#   1. Starts the OpenCRE Flask app (fork) on port 5001
#   2. Waits for it to be ready
#   3. Uploads the CSV via POST /rest/v1/cre_csv_import
#   4. Reports the result
#   5. Stops the Flask app

set -euo pipefail

CSV_FILE="${1:?Usage: $0 <csv_file>}"
OPENCRE_DIR="${HOME}/github_projects/OpenCRE"
PORT=5001
PID_FILE="/tmp/opencre_import_flask.pid"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

if [ ! -d "$OPENCRE_DIR" ]; then
    echo "Error: OpenCRE fork not found at $OPENCRE_DIR"
    exit 1
fi

cleanup() {
    if [ -f "$PID_FILE" ]; then
        kill "$(cat "$PID_FILE")" 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
}
trap cleanup EXIT

echo "Starting OpenCRE Flask app on port $PORT..."
cd "$OPENCRE_DIR"
CRE_ALLOW_IMPORT=1 FLASK_APP=cre.py flask run --port "$PORT" &
echo $! > "$PID_FILE"

echo "Waiting for app to be ready..."
for i in $(seq 1 30); do
    if curl -s "http://localhost:$PORT/rest/v1/root_cres" > /dev/null 2>&1; then
        echo "App ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Error: App did not start within 30 seconds"
        exit 1
    fi
    sleep 1
done

echo "Uploading CSV: $CSV_FILE"
RESPONSE=$(curl -s -w "\n%{http_code}" \
    -X POST "http://localhost:$PORT/rest/v1/cre_csv_import" \
    -F "cre_csv=@${CSV_FILE}")

HTTP_CODE=$(echo "$RESPONSE" | tail -1)
BODY=$(echo "$RESPONSE" | sed '$d')

echo "HTTP Status: $HTTP_CODE"
echo "Response: $BODY"

if [ "$HTTP_CODE" -eq 200 ]; then
    echo "Import successful."
else
    echo "Import failed (HTTP $HTTP_CODE)."
    exit 1
fi
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x scripts/opencre_import.sh`

- [ ] **Step 3: Commit**

```bash
git add scripts/opencre_import.sh
git commit -m "feat(export): add fork import helper script"
```

---

### Task 10: Post-Import Verification Script

**Files:**
- Create: `scripts/verify_opencre_import.py`

- [ ] **Step 1: Create the verification script**

Create `scripts/verify_opencre_import.py`:

```python
#!/usr/bin/env python3
"""Verify OpenCRE import by comparing fork API against export manifest (spec §10.1).

Usage:
    python scripts/verify_opencre_import.py --manifest opencre_export/export_manifest.json

Queries the local OpenCRE fork's API and compares link counts
against the manifest's per-framework exported counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import quote

import requests

FORK_BASE_URL = "http://localhost:5001/rest/v1"


def verify_framework(
    opencre_name: str,
    expected_count: int,
    base_url: str = FORK_BASE_URL,
) -> dict:
    """Query the fork API for a standard and count its CRE links."""
    url = f"{base_url}/standard/{quote(opencre_name)}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 404:
            return {"name": opencre_name, "status": "missing", "expected": expected_count, "actual": 0}
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return {"name": opencre_name, "status": "error", "message": str(e),
                "expected": expected_count, "actual": 0}

    linked_cre_count = 0
    if isinstance(data, dict):
        standards = [data]
    elif isinstance(data, list):
        standards = data
    else:
        standards = []

    for std in standards:
        for link in std.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") == "CRE":
                linked_cre_count += 1

    status = "match" if linked_cre_count >= expected_count else "mismatch"
    return {
        "name": opencre_name,
        "status": status,
        "expected": expected_count,
        "actual": linked_cre_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify OpenCRE import against manifest")
    parser.add_argument("--manifest", required=True, help="Path to export_manifest.json")
    parser.add_argument("--base-url", default=FORK_BASE_URL, help="Fork API base URL")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Error: Manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    from tract.export.opencre_names import TRACT_TO_OPENCRE_NAME

    print(f"Verifying import against manifest ({manifest.get('export_date', 'unknown')})")
    print(f"Fork API: {args.base_url}")
    print("=" * 60)

    all_ok = True
    for fw_id, fw_stats in sorted(manifest.get("per_framework", {}).items()):
        expected = fw_stats.get("exported", 0)
        if expected == 0:
            continue

        opencre_name = TRACT_TO_OPENCRE_NAME.get(fw_id, fw_id)
        result = verify_framework(opencre_name, expected, args.base_url)

        icon = "✓" if result["status"] == "match" else "✗"
        print(f"  {icon} {opencre_name}: expected={expected}, actual={result['actual']} ({result['status']})")

        if result["status"] not in ("match",):
            all_ok = False

    print()
    if all_ok:
        print("All frameworks verified successfully.")
    else:
        print("Some frameworks have mismatches. Review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x scripts/verify_opencre_import.py`

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_opencre_import.py
git commit -m "feat(export): add post-import verification script"
```

---

### Task 11: Run Full Test Suite

**Files:** None (verification only)

- [ ] **Step 1: Run all existing tests to check for regressions**

Run: `pytest tests/ -v --tb=short -q`
Expected: All existing tests PASS, no regressions

- [ ] **Step 2: Run only the new export tests**

Run: `pytest tests/test_opencre_names.py tests/test_export_filters.py tests/test_opencre_csv.py tests/test_staleness.py tests/test_export_manifest.py tests/test_opencre_export_e2e.py tests/test_cli.py -v`
Expected: All new tests PASS

- [ ] **Step 3: Run type checking**

Run: `mypy tract/export/ --strict`
Expected: No errors (or only pre-existing warnings from dependencies)

- [ ] **Step 4: Run linter**

Run: `ruff check tract/export/ tests/test_opencre_names.py tests/test_export_filters.py tests/test_opencre_csv.py tests/test_staleness.py tests/test_export_manifest.py tests/test_opencre_export_e2e.py`
Expected: No errors

- [ ] **Step 5: Fix any issues and commit**

```bash
git add -A
git commit -m "chore(export): fix lint/type issues from export pipeline"
```

---

### Task 12: Integration Test with Real Database

**Files:** None (manual verification with actual crosswalk.db)

- [ ] **Step 1: Dry-run against real crosswalk.db**

Run: `python -m tract.cli export --opencre --dry-run`

Expected output should show approximately:
```
Dry run — would export:

  csa_aicm: ~184 assignments
  eu_ai_act: ~84 assignments
  mitre_atlas: ~120 assignments
  nist_ai_600_1: ~7 assignments
  owasp_agentic_top10: ~8 assignments

  Total: ~403 assignments
```

- [ ] **Step 2: Single framework pilot export**

Run: `python -m tract.cli export --opencre --framework nist_ai_600_1 --output-dir /tmp/opencre_pilot --skip-staleness`

Expected:
- Creates `/tmp/opencre_pilot/NIST_AI_600-1.csv`
- Creates `/tmp/opencre_pilot/export_manifest.json`
- CSV has 5 columns: `CRE 0`, `NIST AI 600-1|name`, `NIST AI 600-1|id`, `NIST AI 600-1|description`, `NIST AI 600-1|hyperlink`
- ~7 data rows

- [ ] **Step 3: Inspect the generated CSV**

Run: `head -5 /tmp/opencre_pilot/NIST_AI_600-1.csv`

Verify:
- First column header is `CRE 0`
- CRE 0 values are pipe-delimited (`123-456|Hub Name`)
- No `CRE 1`, `CRE 2`, etc. columns

- [ ] **Step 4: Inspect the manifest**

Run: `python -c "import json; print(json.dumps(json.load(open('/tmp/opencre_pilot/export_manifest.json')), indent=2))"`

Verify:
- `confidence_floor` is `0.30`
- `confidence_overrides` includes `mitre_atlas: 0.35`
- `per_framework` has `nist_ai_600_1` with correct counts
- `tract_git_sha` is populated

- [ ] **Step 5: Full export (all frameworks)**

Run: `python -m tract.cli export --opencre --output-dir /tmp/opencre_full --skip-staleness`

Verify:
- CSVs generated for each framework with assignments
- No CSV for `owasp_llm_top10` (0 surviving assignments)
- Manifest shows ~403 total exported

- [ ] **Step 6: Commit any fixes**

If any issues were found and fixed during integration testing:

```bash
git add -A
git commit -m "fix(export): fixes from integration testing with real crosswalk.db"
```
