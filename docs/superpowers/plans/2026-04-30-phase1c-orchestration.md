# Phase 1C Orchestration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the tested Phase 1C building blocks into end-to-end orchestration scripts that run on real data — extracting similarities from LOFO fold models, fitting calibration, training a deployment model, running inference on unmapped controls, and populating the crosswalk database.

**Architecture:** Four scripts in `scripts/phase1c/` compose the library functions from `tract/calibration/`, `tract/active_learning/`, and `tract/crosswalk/`. Two new library functions (`load_fold_model`, `train_deployment_model`) bridge the saved Phase 1B artifacts to the new pipeline. All GPU work runs locally on Jetson Orin AGX (61.4 GB unified memory). Scripts are sequential: T0 → T1 → T2 → T4 (T3 is human review, not a script).

**Tech Stack:** Python 3.12, sentence-transformers, PEFT (LoRA), numpy, scipy, sqlite3, pytest

**Spec:** `docs/superpowers/specs/2026-04-29-phase1c-design.md` (Section 8: Execution Flow)

---

## File Structure

### New Files

```
tract/active_learning/model_io.py        — load_fold_model(), load_deployment_model()
tract/active_learning/train_deploy.py    — train_deployment_model() wrapper
tract/active_learning/unmapped.py        — load_unmapped_controls() from parsed frameworks

scripts/phase1c/__init__.py              — Package marker
scripts/phase1c/t0_extract_similarities.py   — Load 5 fold models, extract similarity NPZs, populate crosswalk.db
scripts/phase1c/t1_calibrate_and_train.py    — Fit T_lofo, select holdout, train deployment model
scripts/phase1c/t2_inference_and_calibrate.py — Deployment inference, T_deploy, gates, review.json
scripts/phase1c/t4_retrain_round.py          — Ingest reviews, retrain, re-infer (per AL round)

tests/test_active_learning_model_io.py       — Tests for model loading
tests/test_active_learning_train_deploy.py   — Tests for deployment model training wrapper
tests/test_active_learning_unmapped.py       — Tests for unmapped control loading
```

### Modified Files

```
tract/config.py               — Add PHASE1C_UNMAPPED_FRAMEWORKS mapping
```

### Existing Files Referenced (read-only)

```
results/phase1b/phase1b_textaware/fold_*/model/model/  — Saved LoRA adapters
results/phase1b/phase1b_textaware/fold_*/predictions.json  — For verification
data/processed/cre_hierarchy.json             — Hub hierarchy
data/processed/frameworks/*.json              — Parsed framework controls
data/training/hub_links_curated.jsonl         — Training links
data/canary_items_for_labeling.json           — Expert-labeled canary items
tests/fixtures/ood_synthetic_texts.json       — 30 synthetic OOD texts
```

---

### Task 1: Unmapped Control Loader

**Files:**
- Create: `tract/active_learning/unmapped.py`
- Create: `tests/test_active_learning_unmapped.py`
- Modify: `tract/config.py`

- [ ] **Step 1: Add PHASE1C_UNMAPPED_FRAMEWORKS to config.py**

Add after line ~218 (after `PHASE1C_T_GAP_WARNING`):

```python
PHASE1C_UNMAPPED_FRAMEWORKS: Final[dict[str, str]] = {
    "csa_aicm": "CSA AI Controls Matrix",
    "eu_ai_act": "EU AI Act — Regulation (EU) 2024/1689",
    "mitre_atlas": "MITRE ATLAS",
    "nist_ai_600_1": "NIST AI 600-1 Generative AI Profile",
    "owasp_agentic_top10": "OWASP Top 10 for Agentic Applications 2026",
}
```

- [ ] **Step 2: Write the failing test**

```python
"""Tests for unmapped control loading."""
from __future__ import annotations

import json
import pytest
from pathlib import Path


class TestLoadUnmappedControls:
    def test_returns_list_of_dicts(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        fw_file = fw_dir / "csa_aicm.json"
        fw_file.write_text(json.dumps({
            "framework_name": "CSA AI Controls Matrix",
            "framework_id": "csa_aicm",
            "controls": [
                {"control_id": "A01", "title": "Audit", "description": "Audit desc", "full_text": "Full audit text"},
                {"control_id": "A02", "title": "Access", "description": "Access desc", "full_text": ""},
            ],
        }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm"],
            framework_display_names={"csa_aicm": "CSA AI Controls Matrix"},
        )
        assert len(controls) == 2
        assert controls[0]["control_id"] == "csa_aicm:A01"
        assert controls[0]["framework"] == "CSA AI Controls Matrix"
        assert "control_text" in controls[0]
        assert len(controls[0]["control_text"]) > 0

    def test_control_text_uses_full_text_when_available(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        fw_file = fw_dir / "csa_aicm.json"
        fw_file.write_text(json.dumps({
            "framework_name": "CSA AI Controls Matrix",
            "framework_id": "csa_aicm",
            "controls": [
                {"control_id": "A01", "title": "Audit", "description": "Short", "full_text": "Very long full text here"},
            ],
        }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm"],
            framework_display_names={"csa_aicm": "CSA AI Controls Matrix"},
        )
        assert "Very long full text here" in controls[0]["control_text"]

    def test_control_text_falls_back_to_description(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        fw_file = fw_dir / "csa_aicm.json"
        fw_file.write_text(json.dumps({
            "framework_name": "CSA AI Controls Matrix",
            "framework_id": "csa_aicm",
            "controls": [
                {"control_id": "A01", "title": "Audit", "description": "Description text here", "full_text": ""},
            ],
        }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm"],
            framework_display_names={"csa_aicm": "CSA AI Controls Matrix"},
        )
        assert "Description text here" in controls[0]["control_text"]

    def test_multiple_frameworks(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        for fid, count in [("csa_aicm", 3), ("mitre_atlas", 2)]:
            fw_file = fw_dir / f"{fid}.json"
            fw_file.write_text(json.dumps({
                "framework_name": f"FW {fid}",
                "framework_id": fid,
                "controls": [
                    {"control_id": f"C{i}", "title": f"T{i}", "description": f"D{i}", "full_text": ""}
                    for i in range(count)
                ],
            }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm", "mitre_atlas"],
            framework_display_names={"csa_aicm": "FW csa_aicm", "mitre_atlas": "FW mitre_atlas"},
        )
        assert len(controls) == 5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_active_learning_unmapped.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write minimal implementation**

```python
"""Load unmapped AI controls from parsed framework JSON files."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_unmapped_controls(
    frameworks_dir: Path,
    framework_file_ids: list[str],
    framework_display_names: dict[str, str],
) -> list[dict]:
    """Load all controls from unmapped AI frameworks.

    Each control gets a composite control_id (framework_file_id:original_id)
    and a control_text built from title + full_text (or description as fallback).
    """
    all_controls: list[dict] = []

    for fid in sorted(framework_file_ids):
        fw_path = frameworks_dir / f"{fid}.json"
        with open(fw_path, encoding="utf-8") as f:
            data = json.load(f)

        display_name = framework_display_names.get(fid, data.get("framework_name", fid))

        for ctrl in data.get("controls", []):
            cid = ctrl.get("control_id", "")
            title = ctrl.get("title", "")
            full_text = ctrl.get("full_text", "")
            description = ctrl.get("description", "")

            body = full_text if full_text.strip() else description
            control_text = f"{cid}: {title}. {body}" if title else body

            all_controls.append({
                "control_id": f"{fid}:{cid}",
                "framework": display_name,
                "control_text": control_text,
                "title": title,
                "description": description,
            })

        logger.info("Loaded %d controls from %s", len(data.get("controls", [])), display_name)

    logger.info("Total unmapped controls: %d from %d frameworks", len(all_controls), len(framework_file_ids))
    return all_controls
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_active_learning_unmapped.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add tract/config.py tract/active_learning/unmapped.py tests/test_active_learning_unmapped.py
git commit -m "feat: add unmapped control loader for AI frameworks"
```

---

### Task 2: Model I/O (load_fold_model, load_deployment_model)

**Files:**
- Create: `tract/active_learning/model_io.py`
- Create: `tests/test_active_learning_model_io.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for model loading utilities."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLoadFoldModel:
    def test_loads_from_valid_path(self) -> None:
        from tract.active_learning.model_io import load_fold_model

        fold_path = Path("results/phase1b/phase1b_textaware/fold_MITRE_ATLAS")
        model = load_fold_model(fold_path)
        emb = model.encode(["test input"], normalize_embeddings=True)
        assert emb.shape == (1, 1024)

    def test_raises_on_missing_path(self) -> None:
        from tract.active_learning.model_io import load_fold_model

        with pytest.raises(FileNotFoundError):
            load_fold_model(Path("/nonexistent/path"))

    def test_smoke_test_embedding(self) -> None:
        from tract.active_learning.model_io import load_fold_model
        import numpy as np

        fold_path = Path("results/phase1b/phase1b_textaware/fold_MITRE_ATLAS")
        model = load_fold_model(fold_path)
        emb = model.encode(["Implement encryption for data at rest"], normalize_embeddings=True)
        assert emb.shape == (1, 1024)
        assert abs(float(np.linalg.norm(emb[0])) - 1.0) < 1e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_active_learning_model_io.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Model loading utilities for Phase 1C orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EXPECTED_DIM = 1024


def load_fold_model(fold_path: Path) -> SentenceTransformer:
    """Load a saved LOFO fold model with LoRA adapters.

    Args:
        fold_path: Path to fold directory (e.g., results/.../fold_MITRE_ATLAS).
                   Expects model/model/ subdirectory with adapter files.
    """
    model_dir = fold_path / "model" / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = SentenceTransformer(str(model_dir))
    model.max_seq_length = 512

    emb = model.encode(["smoke test"], normalize_embeddings=True, show_progress_bar=False)
    if emb.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected dim={EXPECTED_DIM}, got {emb.shape[1]}")

    logger.info("Loaded fold model from %s (dim=%d)", fold_path.name, emb.shape[1])
    return model


def load_deployment_model(model_dir: Path) -> SentenceTransformer:
    """Load a saved deployment model.

    Args:
        model_dir: Path containing the saved model (with adapter files or full weights).
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = SentenceTransformer(str(model_dir))
    model.max_seq_length = 512

    emb = model.encode(["smoke test"], normalize_embeddings=True, show_progress_bar=False)
    if emb.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected dim={EXPECTED_DIM}, got {emb.shape[1]}")

    logger.info("Loaded deployment model from %s (dim=%d)", model_dir, emb.shape[1])
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_active_learning_model_io.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add tract/active_learning/model_io.py tests/test_active_learning_model_io.py
git commit -m "feat: model I/O for fold and deployment model loading"
```

---

### Task 3: Deployment Model Training Wrapper

**Files:**
- Create: `tract/active_learning/train_deploy.py`
- Create: `tests/test_active_learning_train_deploy.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for deployment model training wrapper."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from tract.training.data_quality import TieredLink, QualityTier


class TestTrainDeploymentModel:
    def test_removes_holdout_from_training(self) -> None:
        from tract.active_learning.train_deploy import prepare_deployment_training_data

        all_links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "section_id": f"id{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(100)
        ]
        holdout = all_links[:10]
        remaining = prepare_deployment_training_data(all_links, holdout)
        assert len(remaining) == 90
        holdout_ids = {(l.link["section_name"], l.link["cre_id"]) for l in holdout}
        for link in remaining:
            assert (link.link["section_name"], link.link["cre_id"]) not in holdout_ids

    def test_preserves_ai_links(self) -> None:
        from tract.active_learning.train_deploy import prepare_deployment_training_data

        trad_links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "ASVS", "section_name": f"s{i}", "section_id": f"id{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1)
            for i in range(50)
        ]
        ai_links = [
            TieredLink(link={"cre_id": f"h{i}", "standard_name": "MITRE ATLAS", "section_name": f"a{i}", "section_id": f"aid{i}", "link_type": "LinkedTo"}, tier=QualityTier.T1_AI)
            for i in range(10)
        ]
        all_links = trad_links + ai_links
        holdout = trad_links[:5]
        remaining = prepare_deployment_training_data(all_links, holdout)
        ai_count = sum(1 for l in remaining if l.tier == QualityTier.T1_AI)
        assert ai_count == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_active_learning_train_deploy.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Deployment model training wrapper for Phase 1C."""
from __future__ import annotations

import logging
from pathlib import Path

from tract.training.data_quality import TieredLink

logger = logging.getLogger(__name__)


def prepare_deployment_training_data(
    all_links: list[TieredLink],
    holdout_links: list[TieredLink],
) -> list[TieredLink]:
    """Remove holdout links from training data.

    Uses (section_name, cre_id) as the identity key for matching.
    """
    holdout_keys = {
        (l.link.get("section_name", ""), l.link.get("cre_id", ""))
        for l in holdout_links
    }

    remaining = [
        l for l in all_links
        if (l.link.get("section_name", ""), l.link.get("cre_id", "")) not in holdout_keys
    ]

    logger.info(
        "Deployment training data: %d links (%d removed as holdout)",
        len(remaining), len(all_links) - len(remaining),
    )
    return remaining
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_active_learning_train_deploy.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add tract/active_learning/train_deploy.py tests/test_active_learning_train_deploy.py
git commit -m "feat: deployment model training data preparation"
```

---

### Task 4: Crosswalk DB Population Helper

**Files:**
- Create: `tract/crosswalk/populate.py`
- Create: `tests/test_crosswalk_populate.py`

This task creates helper functions that transform hierarchy data, framework data, and tiered links into the dict format expected by `insert_hubs()`, `insert_frameworks()`, `insert_controls()`, and `insert_assignments()`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for crosswalk DB population helpers."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from tract.training.data_quality import TieredLink, QualityTier


class TestBuildHubRecords:
    def test_converts_hierarchy_to_hub_dicts(self) -> None:
        from tract.crosswalk.populate import build_hub_records

        hubs_data = {
            "h1": {"name": "Hub One", "path": "/Root/Hub One", "parent": "root1"},
            "h2": {"name": "Hub Two", "path": "/Root/Hub Two", "parent": "root1"},
        }
        records = build_hub_records(hubs_data)
        assert len(records) == 2
        assert records[0]["id"] == "h1"
        assert records[0]["name"] == "Hub One"
        assert records[0]["path"] == "/Root/Hub One"
        assert records[0]["parent_id"] == "root1"


class TestBuildFrameworkRecords:
    def test_converts_framework_metadata(self) -> None:
        from tract.crosswalk.populate import build_framework_records

        fw_data = [
            {
                "framework_id": "csa_aicm",
                "framework_name": "CSA AI Controls Matrix",
                "version": "1.0",
                "fetched_date": "2026-04-28",
                "controls": [{"control_id": "A01"}, {"control_id": "A02"}],
            }
        ]
        records = build_framework_records(fw_data)
        assert len(records) == 1
        assert records[0]["id"] == "csa_aicm"
        assert records[0]["control_count"] == 2


class TestBuildControlRecords:
    def test_converts_controls_with_framework_prefix(self) -> None:
        from tract.crosswalk.populate import build_control_records

        fw_data = [
            {
                "framework_id": "csa_aicm",
                "framework_name": "CSA AI Controls Matrix",
                "controls": [
                    {"control_id": "A01", "title": "Audit", "description": "Audit desc", "full_text": "Full text"},
                ],
            }
        ]
        records = build_control_records(fw_data)
        assert len(records) == 1
        assert records[0]["id"] == "csa_aicm:A01"
        assert records[0]["framework_id"] == "csa_aicm"
        assert records[0]["section_id"] == "A01"


class TestBuildTrainingAssignments:
    def test_converts_tiered_links(self) -> None:
        from tract.crosswalk.populate import build_training_assignments

        links = [
            TieredLink(
                link={"cre_id": "h1", "standard_name": "ASVS", "section_name": "s1", "section_id": "4.1.1", "link_type": "LinkedTo"},
                tier=QualityTier.T1,
            ),
        ]
        # Need a mapping from (framework, section_id) -> control_id in DB
        control_id_map = {("ASVS", "4.1.1"): "asvs:4.1.1"}
        records = build_training_assignments(links, control_id_map)
        assert len(records) == 1
        assert records[0]["control_id"] == "asvs:4.1.1"
        assert records[0]["hub_id"] == "h1"
        assert records[0]["provenance"] == "training_T1"
        assert records[0]["review_status"] == "ground_truth"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_crosswalk_populate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
"""Transform hierarchy/framework/link data into crosswalk DB record format."""
from __future__ import annotations

import logging

from tract.training.data_quality import TieredLink

logger = logging.getLogger(__name__)


def build_hub_records(hubs_data: dict[str, dict]) -> list[dict]:
    """Convert hierarchy hubs dict to insert_hubs() format."""
    records = []
    for hub_id in sorted(hubs_data.keys()):
        hub = hubs_data[hub_id]
        records.append({
            "id": hub_id,
            "name": hub.get("name", ""),
            "path": hub.get("path", ""),
            "parent_id": hub.get("parent"),
        })
    return records


def build_framework_records(frameworks_data: list[dict]) -> list[dict]:
    """Convert parsed framework JSON metadata to insert_frameworks() format."""
    records = []
    for fw in frameworks_data:
        records.append({
            "id": fw["framework_id"],
            "name": fw.get("framework_name", fw["framework_id"]),
            "version": fw.get("version"),
            "fetch_date": fw.get("fetched_date"),
            "control_count": len(fw.get("controls", [])),
        })
    return records


def build_control_records(frameworks_data: list[dict]) -> list[dict]:
    """Convert parsed framework controls to insert_controls() format.

    Each control gets a composite ID: framework_id:control_id.
    """
    records = []
    for fw in frameworks_data:
        fid = fw["framework_id"]
        for ctrl in fw.get("controls", []):
            cid = ctrl.get("control_id", "")
            records.append({
                "id": f"{fid}:{cid}",
                "framework_id": fid,
                "section_id": cid,
                "title": ctrl.get("title", ""),
                "description": ctrl.get("description", ""),
                "full_text": ctrl.get("full_text", ""),
            })
    return records


def build_training_assignments(
    tiered_links: list[TieredLink],
    control_id_map: dict[tuple[str, str], str],
) -> list[dict]:
    """Convert TieredLinks to insert_assignments() format.

    Args:
        tiered_links: Filtered training links.
        control_id_map: Maps (standard_name, section_id) -> composite control_id in DB.
    """
    records = []
    skipped = 0
    for link in tiered_links:
        fw = link.link.get("standard_name", "")
        sid = link.link.get("section_id", "")
        key = (fw, sid)
        control_id = control_id_map.get(key)
        if control_id is None:
            skipped += 1
            continue

        records.append({
            "control_id": control_id,
            "hub_id": link.link["cre_id"],
            "confidence": None,
            "in_conformal_set": None,
            "is_ood": 0,
            "provenance": f"training_{link.tier.value}",
            "source_link_id": link.link.get("link_id"),
            "model_version": None,
            "review_status": "ground_truth",
        })

    if skipped:
        logger.warning("Skipped %d links with no matching control in DB", skipped)
    logger.info("Built %d training assignments from %d links", len(records), len(tiered_links))
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_crosswalk_populate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add tract/crosswalk/populate.py tests/test_crosswalk_populate.py
git commit -m "feat: crosswalk DB population helpers for hubs, frameworks, controls, assignments"
```

---

### Task 5: T0 Script — Extract Similarities + Populate Crosswalk DB

**Files:**
- Create: `scripts/phase1c/__init__.py`
- Create: `scripts/phase1c/t0_extract_similarities.py`

This script does two things:
1. Load each of the 5 LOFO fold models, encode eval items + hub embeddings, save similarity NPZ files.
2. Populate crosswalk.db with hubs, frameworks, controls, and training assignments.

- [ ] **Step 1: Create package init**

```python
# scripts/phase1c/__init__.py — empty package marker
```

- [ ] **Step 2: Write T0 script**

```python
"""T0: Extract similarity matrices from LOFO fold models + populate crosswalk.db.

Reads:
  - results/phase1b/phase1b_textaware/fold_*/model/model/ (5 fold models)
  - data/processed/cre_hierarchy.json (hub hierarchy)
  - data/processed/frameworks/*.json (all parsed frameworks)
  - data/training/hub_links_curated.jsonl (training links)

Writes:
  - results/phase1c/similarities/fold_{name}.npz (5 NPZ files)
  - results/phase1c/crosswalk.db (populated with hubs, frameworks, controls, training assignments)

Usage:
  python -m scripts.phase1c.t0_extract_similarities
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from scripts.phase0.common import AI_FRAMEWORK_NAMES, build_evaluation_corpus, load_curated_links
from tract.active_learning.model_io import load_fold_model
from tract.config import (
    PHASE1B_RESULTS_DIR,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1C_SIMILARITIES_DIR,
    PROCESSED_DIR,
)
from tract.crosswalk.populate import (
    build_control_records,
    build_framework_records,
    build_hub_records,
    build_training_assignments,
)
from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    count_frameworks,
    count_hubs,
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.evaluate import extract_similarity_matrix
from tract.training.firewall import build_all_hub_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FOLD_DIR = PHASE1B_RESULTS_DIR / "phase1b_textaware"


def _extract_fold_similarities(hierarchy: CREHierarchy) -> None:
    """Load each fold model, extract similarity matrices, save as NPZ."""
    PHASE1C_SIMILARITIES_DIR.mkdir(parents=True, exist_ok=True)

    hub_ids = sorted(hierarchy.hubs.keys())
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    eval_by_fw: dict[str, list] = {}
    for item in corpus:
        eval_by_fw.setdefault(item.framework_name, []).append(item)

    fold_dirs = sorted(FOLD_DIR.glob("fold_*"))
    fold_dirs = [d for d in fold_dirs if d.is_dir()]

    for fold_dir in fold_dirs:
        fw_name = fold_dir.name.replace("fold_", "").replace("_", " ")
        eval_items = eval_by_fw.get(fw_name, [])
        if not eval_items:
            logger.warning("No eval items for fold %s, skipping", fw_name)
            continue

        logger.info("Extracting similarities for fold %s (%d items)", fw_name, len(eval_items))
        t0 = time.time()

        model = load_fold_model(fold_dir)

        hub_texts = build_all_hub_texts(hierarchy, excluded_framework=fw_name)
        hub_texts_ordered = [hub_texts[hid] for hid in hub_ids]
        hub_embs = model.encode(
            hub_texts_ordered,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=128,
        )

        sim_data = extract_similarity_matrix(model, eval_items, hub_ids, hub_embs)

        npz_path = PHASE1C_SIMILARITIES_DIR / f"fold_{fold_dir.name.replace('fold_', '')}.npz"
        np.savez_compressed(
            npz_path,
            sims=sim_data["sims"],
            hub_ids=np.array(sim_data["hub_ids"]),
            gt_json=np.array(sim_data["gt_json"]),
            frameworks=np.array(sim_data["frameworks"]),
        )

        del model
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

        logger.info("Fold %s: saved %s (%.1fs)", fw_name, npz_path.name, time.time() - t0)


def _populate_crosswalk_db(hierarchy: CREHierarchy) -> None:
    """Create and populate crosswalk.db with hubs, frameworks, controls, training assignments."""
    db_path = PHASE1C_CROSSWALK_DB_PATH
    create_database(db_path)

    # 1. Insert hubs
    hub_records = build_hub_records(hierarchy.hubs_raw)
    insert_hubs(db_path, hub_records)
    logger.info("Inserted %d hubs", count_hubs(db_path))

    # 2. Load all frameworks and insert
    fw_dir = PROCESSED_DIR / "frameworks"
    all_fw_data = []
    for fw_file in sorted(fw_dir.glob("*.json")):
        with open(fw_file, encoding="utf-8") as f:
            all_fw_data.append(json.load(f))

    fw_records = build_framework_records(all_fw_data)
    insert_frameworks(db_path, fw_records)
    logger.info("Inserted %d frameworks", count_frameworks(db_path))

    # 3. Insert controls
    ctrl_records = build_control_records(all_fw_data)
    insert_controls(db_path, ctrl_records)
    logger.info("Inserted %d controls", len(ctrl_records))

    # 4. Insert training assignments
    tiered_links, _ = load_and_filter_curated_links()
    control_id_map: dict[tuple[str, str], str] = {}
    for rec in ctrl_records:
        fw_id = rec["framework_id"]
        # Find framework display name from the loaded data
        fw_name = ""
        for fw in all_fw_data:
            if fw["framework_id"] == fw_id:
                fw_name = fw.get("framework_name", fw_id)
                break
        control_id_map[(fw_name, rec["section_id"])] = rec["id"]

    assignment_records = build_training_assignments(tiered_links, control_id_map)
    if assignment_records:
        insert_assignments(db_path, assignment_records)
    logger.info("Inserted %d training assignments", len(assignment_records))


def main() -> None:
    logger.info("=== T0: Extract Similarities + Populate Crosswalk DB ===")
    t0 = time.time()

    hierarchy_data = load_json(PROCESSED_DIR / "cre_hierarchy.json")
    hierarchy = CREHierarchy.model_validate(hierarchy_data)
    logger.info("Loaded hierarchy: %d hubs", len(hierarchy.hubs))

    _extract_fold_similarities(hierarchy)
    _populate_crosswalk_db(hierarchy)

    logger.info("T0 complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script**

Run: `python -m scripts.phase1c.t0_extract_similarities 2>&1 | tee results/phase1c/t0_log.txt`
Expected: 5 NPZ files in `results/phase1c/similarities/`, crosswalk.db created with hubs, frameworks, controls, assignments.

- [ ] **Step 4: Verify outputs**

```bash
ls results/phase1c/similarities/*.npz
python3 -c "
import numpy as np
for name in ['MITRE_ATLAS','NIST_AI_100-2','OWASP_AI_Exchange','OWASP_Top10_for_LLM','OWASP_Top10_for_ML']:
    d = np.load(f'results/phase1c/similarities/fold_{name}.npz')
    print(f'{name}: sims={d[\"sims\"].shape}, hubs={len(d[\"hub_ids\"])}, items={len(d[\"gt_json\"])}')
"
python3 -c "
import sqlite3
conn = sqlite3.connect('results/phase1c/crosswalk.db')
for table in ['hubs','frameworks','controls','assignments']:
    count = conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
    print(f'{table}: {count}')
conn.close()
"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/phase1c/__init__.py scripts/phase1c/t0_extract_similarities.py
git commit -m "feat: T0 script — extract fold similarity matrices and populate crosswalk DB"
```

---

### Task 6: T1 Script — Calibrate T_lofo + Train Deployment Model

**Files:**
- Create: `scripts/phase1c/t1_calibrate_and_train.py`

This script:
1. Loads fold NPZ files, fits T_lofo (diagnostic, sqrt(n)-weighted).
2. Computes diagnostic ECE on LOFO predictions.
3. Selects 440 holdout (420 cal + 20 canary) from traditional links.
4. Trains deployment model on remaining data.

- [ ] **Step 1: Write T1 script**

```python
"""T1: Fit T_lofo diagnostic + train deployment model.

Reads:
  - results/phase1c/similarities/fold_*.npz (from T0)
  - data/training/hub_links_curated.jsonl (training links)
  - data/processed/cre_hierarchy.json

Writes:
  - results/phase1c/calibration/t_lofo_result.json
  - results/phase1c/calibration/diagnostic_ece.json
  - results/phase1c/holdout/calibration_links.json
  - results/phase1c/holdout/canary_links.json
  - results/phase1c/deployment_model/ (trained model + metadata)

Usage:
  python -m scripts.phase1c.t1_calibrate_and_train
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from tract.active_learning.deploy import select_holdout, holdout_to_eval
from tract.active_learning.train_deploy import prepare_deployment_training_data
from tract.calibration.diagnostics import bootstrap_ece, expected_calibration_error
from tract.calibration.temperature import calibrate_similarities, fit_t_lofo
from tract.config import (
    PHASE1C_DEPLOYMENT_MODEL_DIR,
    PHASE1C_ECE_BOOTSTRAP_N,
    PHASE1C_ECE_N_BINS,
    PHASE1C_RESULTS_DIR,
    PHASE1C_SIMILARITIES_DIR,
    PROCESSED_DIR,
)
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.config import TrainingConfig
from tract.training.data import build_training_pairs, pairs_to_dataset
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.firewall import build_all_hub_texts
from tract.training.loop import save_checkpoint, train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_gt_json(gt_json: np.ndarray, hub_ids: list[str]) -> list[list[int]]:
    """Parse JSON-encoded ground truth into hub column indices."""
    hub_id_to_idx = {hid: i for i, hid in enumerate(hub_ids)}
    valid_indices = []
    for gt_str in gt_json:
        valid_hub_ids = json.loads(str(gt_str))
        indices = [hub_id_to_idx[hid] for hid in valid_hub_ids if hid in hub_id_to_idx]
        if not indices:
            raise ValueError(f"No valid hub indices for ground truth: {gt_str}")
        valid_indices.append(indices)
    return valid_indices


def _fit_diagnostic_temperature() -> dict:
    """Fit T_lofo on pooled LOFO fold similarities."""
    cal_dir = PHASE1C_RESULTS_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(PHASE1C_SIMILARITIES_DIR.glob("fold_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No fold NPZ files in {PHASE1C_SIMILARITIES_DIR}")

    fold_sims: dict[str, np.ndarray] = {}
    fold_valid_indices: dict[str, list[list[int]]] = {}

    for npz_path in npz_files:
        fold_name = npz_path.stem.replace("fold_", "").replace("_", " ")
        data = np.load(npz_path, allow_pickle=False)
        hub_ids = list(data["hub_ids"])
        sims = data["sims"]
        gt_json = data["gt_json"]

        valid_indices = _parse_gt_json(gt_json, hub_ids)
        fold_sims[fold_name] = sims
        fold_valid_indices[fold_name] = valid_indices
        logger.info("Loaded fold %s: %d items, %d hubs", fold_name, len(sims), len(hub_ids))

    result = fit_t_lofo(fold_sims, fold_valid_indices)
    atomic_write_json(
        {k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
        cal_dir / "t_lofo_result.json",
    )

    # Compute diagnostic ECE
    all_sims = np.concatenate([fold_sims[n] for n in sorted(fold_sims.keys())])
    all_valid = []
    for n in sorted(fold_sims.keys()):
        all_valid.extend(fold_valid_indices[n])

    probs = calibrate_similarities(all_sims, result["temperature"])
    confidences = np.array([float(probs[i].max()) for i in range(len(probs))])
    accuracies = np.array([
        1.0 if int(probs[i].argmax()) in valid else 0.0
        for i, valid in enumerate(all_valid)
    ])

    ece = expected_calibration_error(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS)
    ece_ci = bootstrap_ece(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS, n_bootstrap=PHASE1C_ECE_BOOTSTRAP_N)

    diagnostic_ece = {"ece": ece, "ece_ci": ece_ci, "t_lofo": result["temperature"], "n_items": len(all_sims)}
    atomic_write_json(diagnostic_ece, cal_dir / "diagnostic_ece.json")
    logger.info("Diagnostic ECE: %.4f [%.4f, %.4f]", ece, ece_ci["ci_low"], ece_ci["ci_high"])

    return result


def _select_and_save_holdout() -> tuple[list, list, list]:
    """Select 440 holdout, save link records, return (cal, canary, remaining)."""
    holdout_dir = PHASE1C_RESULTS_DIR / "holdout"
    holdout_dir.mkdir(parents=True, exist_ok=True)

    tiered_links, _ = load_and_filter_curated_links()
    cal_links, canary_links, remaining = select_holdout(tiered_links)

    # Save holdout link records for reproducibility
    cal_records = [{"link": l.link, "tier": l.tier.value} for l in cal_links]
    canary_records = [{"link": l.link, "tier": l.tier.value} for l in canary_links]
    atomic_write_json(cal_records, holdout_dir / "calibration_links.json")
    atomic_write_json(canary_records, holdout_dir / "canary_links.json")

    logger.info("Holdout: %d calibration, %d canary, %d remaining for training", len(cal_links), len(canary_links), len(remaining))
    return cal_links, canary_links, remaining


def _train_deployment_model(
    remaining_links: list,
    hierarchy: CREHierarchy,
) -> None:
    """Train deployment model on all data minus holdout."""
    PHASE1C_DEPLOYMENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    hub_texts = build_all_hub_texts(hierarchy, excluded_framework=None)
    pairs = build_training_pairs(remaining_links, hub_texts, excluded_framework=None)
    dataset = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=3)

    config = TrainingConfig(name="phase1c_deployment")

    model = train_model(config, dataset, PHASE1C_DEPLOYMENT_MODEL_DIR)

    metrics = {"n_training_pairs": len(pairs), "n_links": len(remaining_links)}
    save_checkpoint(model, config, metrics, PHASE1C_DEPLOYMENT_MODEL_DIR / "model", "deployment")

    del model
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    logger.info("Deployment model saved to %s", PHASE1C_DEPLOYMENT_MODEL_DIR)


def main() -> None:
    logger.info("=== T1: Calibrate T_lofo + Train Deployment Model ===")
    t0 = time.time()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))

    t_lofo_result = _fit_diagnostic_temperature()
    logger.info("T_lofo = %.4f", t_lofo_result["temperature"])

    cal_links, canary_links, remaining = _select_and_save_holdout()

    _train_deployment_model(remaining, hierarchy)

    logger.info("T1 complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `python -m scripts.phase1c.t1_calibrate_and_train 2>&1 | tee results/phase1c/t1_log.txt`
Expected: T_lofo result saved, holdout links saved, deployment model trained and saved.

**Estimated time:** ~15-20 minutes (20 epochs × ~3,700 training pairs, batch=32).

- [ ] **Step 3: Verify outputs**

```bash
cat results/phase1c/calibration/t_lofo_result.json | python3 -m json.tool
cat results/phase1c/calibration/diagnostic_ece.json | python3 -m json.tool
python3 -c "import json; d=json.load(open('results/phase1c/holdout/calibration_links.json')); print(f'Cal holdout: {len(d)} links')"
python3 -c "import json; d=json.load(open('results/phase1c/holdout/canary_links.json')); print(f'Canary holdout: {len(d)} links')"
ls results/phase1c/deployment_model/model/model/
```

- [ ] **Step 4: Commit**

```bash
git add scripts/phase1c/t1_calibrate_and_train.py
git commit -m "feat: T1 script — T_lofo diagnostic calibration and deployment model training"
```

---

### Task 7: T2 Script — Deployment Inference + Production Calibration + Review JSON

**Files:**
- Create: `scripts/phase1c/t2_inference_and_calibrate.py`

The largest orchestration script. It:
1. Loads the deployment model.
2. Runs inference on 5 item sets: 552 unmapped, 420 holdout, 147 LOFO, 20 trad canaries, 30 OOD texts.
3. Fits T_deploy on 420 holdout similarities.
4. Computes ECE gate (< 0.10).
5. Computes conformal quantile + coverage gate (>= 0.90).
6. Computes OOD threshold + separation gate (>= 0.90).
7. Computes global threshold (max-F1).
8. KS-test diagnostic (traditional vs AI similarity distributions).
9. Generates round_1/review.json with 552 unmapped + 20 AI canaries + 20 trad canaries.
10. Populates crosswalk.db with model prediction assignments.

- [ ] **Step 1: Write T2 script**

```python
"""T2: Deployment model inference + production calibration + review.json generation.

Reads:
  - results/phase1c/deployment_model/model/model/ (trained deployment model)
  - results/phase1c/holdout/ (calibration and canary links)
  - data/processed/cre_hierarchy.json
  - data/processed/frameworks/*.json (unmapped controls)
  - data/canary_items_for_labeling.json (expert-labeled AI canaries)
  - tests/fixtures/ood_synthetic_texts.json

Writes:
  - results/phase1c/calibration/t_deploy_result.json
  - results/phase1c/calibration/ece_gate.json
  - results/phase1c/calibration/conformal.json
  - results/phase1c/calibration/ood.json
  - results/phase1c/calibration/ks_test.json
  - results/phase1c/calibration/global_threshold.json
  - results/phase1c/round_1/review.json
  - results/phase1c/similarities/deployment_*.npz

Usage:
  python -m scripts.phase1c.t2_inference_and_calibrate
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

from scripts.phase0.common import AI_FRAMEWORK_NAMES, build_evaluation_corpus, load_curated_links
from tract.active_learning.canary import select_ai_canaries
from tract.active_learning.deploy import holdout_to_eval
from tract.active_learning.model_io import load_deployment_model
from tract.active_learning.review import generate_review_json
from tract.active_learning.unmapped import load_unmapped_controls
from tract.calibration.conformal import build_prediction_sets, compute_conformal_coverage, compute_conformal_quantile
from tract.calibration.diagnostics import bootstrap_ece, expected_calibration_error, ks_test_similarity_distributions
from tract.calibration.ood import compute_ood_threshold, flag_ood_items, validate_ood_threshold
from tract.calibration.temperature import calibrate_similarities, find_global_threshold, fit_temperature
from tract.config import (
    PHASE1C_CONFORMAL_ALPHA,
    PHASE1C_CONFORMAL_COVERAGE_GATE,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1C_DEPLOYMENT_MODEL_DIR,
    PHASE1C_ECE_BOOTSTRAP_N,
    PHASE1C_ECE_N_BINS,
    PHASE1C_ECE_THRESHOLD,
    PHASE1C_OOD_SEPARATION_GATE,
    PHASE1C_RESULTS_DIR,
    PHASE1C_T_GAP_WARNING,
    PHASE1C_UNMAPPED_FRAMEWORKS,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from tract.crosswalk.store import insert_assignments
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.data_quality import TieredLink, QualityTier, load_and_filter_curated_links
from tract.training.firewall import build_all_hub_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class _SimpleEvalItem:
    """Lightweight eval item for extract_similarity_matrix compatibility."""
    def __init__(self, control_text: str, framework_name: str, valid_hub_ids: frozenset[str]):
        self.control_text = control_text
        self.framework_name = framework_name
        self.valid_hub_ids = valid_hub_ids


def _load_holdout_links(holdout_dir: Path) -> tuple[list[TieredLink], list[TieredLink]]:
    """Load saved holdout links from JSON."""
    cal_data = load_json(holdout_dir / "calibration_links.json")
    canary_data = load_json(holdout_dir / "canary_links.json")
    cal_links = [TieredLink(link=d["link"], tier=QualityTier(d["tier"])) for d in cal_data]
    canary_links = [TieredLink(link=d["link"], tier=QualityTier(d["tier"])) for d in canary_data]
    return cal_links, canary_links


def _run_inference(model, items: list, hub_ids: list[str], hub_embs: np.ndarray, label: str) -> np.ndarray:
    """Run inference on a set of items, return similarity matrix."""
    from tract.training.evaluate import extract_similarity_matrix

    eval_items = [
        _SimpleEvalItem(
            control_text=item.get("control_text", item.control_text if hasattr(item, "control_text") else ""),
            framework_name=item.get("framework", item.framework_name if hasattr(item, "framework_name") else ""),
            valid_hub_ids=item.get("valid_hub_ids", frozenset()) if isinstance(item, dict) else getattr(item, "valid_hub_ids", frozenset()),
        )
        if isinstance(item, dict) else item
        for item in items
    ]

    sim_data = extract_similarity_matrix(model, eval_items, hub_ids, hub_embs)
    logger.info("Inference %s: %d items × %d hubs", label, sim_data["sims"].shape[0], sim_data["sims"].shape[1])
    return sim_data["sims"]


def main() -> None:
    logger.info("=== T2: Deployment Inference + Production Calibration ===")
    t_start = time.time()

    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())

    # Load deployment model
    model = load_deployment_model(PHASE1C_DEPLOYMENT_MODEL_DIR / "model" / "model")

    # Encode hub representations (no framework excluded — deployment model)
    hub_texts = build_all_hub_texts(hierarchy, excluded_framework=None)
    hub_texts_ordered = [hub_texts[hid] for hid in hub_ids]
    hub_embs = model.encode(
        hub_texts_ordered, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False, batch_size=128,
    )

    # === 1. Load all inference item sets ===
    holdout_dir = PHASE1C_RESULTS_DIR / "holdout"
    cal_links, canary_trad_links = _load_holdout_links(holdout_dir)
    cal_eval = [holdout_to_eval(l) for l in cal_links]
    canary_trad_eval = [holdout_to_eval(l) for l in canary_trad_links]

    # Unmapped AI controls
    unmapped = load_unmapped_controls(
        frameworks_dir=PROCESSED_DIR / "frameworks",
        framework_file_ids=list(PHASE1C_UNMAPPED_FRAMEWORKS.keys()),
        framework_display_names=PHASE1C_UNMAPPED_FRAMEWORKS,
    )

    # LOFO eval items (for conformal coverage check)
    links_raw = load_curated_links()
    lofo_corpus = build_evaluation_corpus(links_raw, AI_FRAMEWORK_NAMES, {})

    # OOD synthetic texts
    ood_data = load_json(PROJECT_ROOT / "tests" / "fixtures" / "ood_synthetic_texts.json")
    ood_texts = [{"control_text": t, "framework": "synthetic_ood", "valid_hub_ids": frozenset()} for t in ood_data]

    # === 2. Run inference on all sets ===
    sims_unmapped = _run_inference(model, unmapped, hub_ids, hub_embs, "unmapped")
    sims_cal = _run_inference(model, cal_eval, hub_ids, hub_embs, "calibration_holdout")
    sims_lofo = _run_inference(model, lofo_corpus, hub_ids, hub_embs, "lofo_eval")
    sims_canary_trad = _run_inference(model, canary_trad_eval, hub_ids, hub_embs, "canary_traditional")
    sims_ood = _run_inference(model, ood_texts, hub_ids, hub_embs, "ood_synthetic")

    # Save similarity matrices
    sim_dir = PHASE1C_RESULTS_DIR / "similarities"
    sim_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sim_dir / "deployment_unmapped.npz", sims=sims_unmapped)
    np.savez_compressed(sim_dir / "deployment_calibration.npz", sims=sims_cal)
    np.savez_compressed(sim_dir / "deployment_lofo.npz", sims=sims_lofo)

    # Free GPU memory
    del model
    import gc; gc.collect()
    import torch; torch.cuda.empty_cache()

    # === 3. Fit T_deploy on 420 holdout ===
    cal_dir = PHASE1C_RESULTS_DIR / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)

    hub_id_to_idx = {hid: i for i, hid in enumerate(hub_ids)}
    cal_valid_indices = []
    for item in cal_eval:
        indices = [hub_id_to_idx[hid] for hid in item["valid_hub_ids"] if hid in hub_id_to_idx]
        cal_valid_indices.append(indices)

    t_deploy_result = fit_temperature(sims_cal, cal_valid_indices)
    t_deploy = t_deploy_result["temperature"]
    atomic_write_json(t_deploy_result, cal_dir / "t_deploy_result.json")
    logger.info("T_deploy = %.4f", t_deploy)

    # T_gap diagnostic
    t_lofo_data = load_json(cal_dir / "t_lofo_result.json")
    t_gap = abs(t_deploy - t_lofo_data["temperature"])
    if t_gap > PHASE1C_T_GAP_WARNING:
        logger.warning("T_gap = %.4f > %.2f warning threshold", t_gap, PHASE1C_T_GAP_WARNING)

    # === 4. ECE gate ===
    probs_cal = calibrate_similarities(sims_cal, t_deploy)
    confidences = np.array([float(probs_cal[i].max()) for i in range(len(probs_cal))])
    accuracies = np.array([
        1.0 if int(probs_cal[i].argmax()) in valid else 0.0
        for i, valid in enumerate(cal_valid_indices)
    ])
    ece = expected_calibration_error(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS)
    ece_ci = bootstrap_ece(confidences, accuracies, n_bins=PHASE1C_ECE_N_BINS, n_bootstrap=PHASE1C_ECE_BOOTSTRAP_N)
    ece_passed = ece < PHASE1C_ECE_THRESHOLD
    ece_result = {"ece": ece, "ece_ci": ece_ci, "threshold": PHASE1C_ECE_THRESHOLD, "passed": ece_passed}
    atomic_write_json(ece_result, cal_dir / "ece_gate.json")
    logger.info("ECE gate: %.4f %s (threshold=%.2f)", ece, "PASS" if ece_passed else "FAIL", PHASE1C_ECE_THRESHOLD)

    # === 5. Conformal prediction ===
    lofo_valid_indices = []
    for item in lofo_corpus:
        indices = [hub_id_to_idx[hid] for hid in item.valid_hub_ids if hid in hub_id_to_idx]
        lofo_valid_indices.append(indices)

    probs_lofo = calibrate_similarities(sims_lofo, t_deploy)
    conformal_quantile = compute_conformal_quantile(probs_lofo, lofo_valid_indices, alpha=PHASE1C_CONFORMAL_ALPHA)
    prediction_sets_lofo = build_prediction_sets(probs_lofo, hub_ids, conformal_quantile)
    valid_hub_sets_lofo = [item.valid_hub_ids for item in lofo_corpus]
    coverage = compute_conformal_coverage(prediction_sets_lofo, valid_hub_sets_lofo)
    coverage_passed = coverage >= PHASE1C_CONFORMAL_COVERAGE_GATE

    conformal_result = {
        "quantile": conformal_quantile, "coverage": coverage,
        "coverage_gate": PHASE1C_CONFORMAL_COVERAGE_GATE, "passed": coverage_passed,
        "alpha": PHASE1C_CONFORMAL_ALPHA, "n_items": len(lofo_corpus),
        "mean_set_size": float(np.mean([len(s) for s in prediction_sets_lofo])),
    }
    atomic_write_json(conformal_result, cal_dir / "conformal.json")
    logger.info("Conformal coverage: %.4f %s", coverage, "PASS" if coverage_passed else "FAIL")

    # === 6. OOD threshold ===
    max_sims_cal = np.array([float(sims_cal[i].max()) for i in range(len(sims_cal))])
    ood_threshold = compute_ood_threshold(max_sims_cal)
    max_sims_ood = np.array([float(sims_ood[i].max()) for i in range(len(sims_ood))])
    ood_result = validate_ood_threshold(max_sims_ood, ood_threshold)
    atomic_write_json(ood_result, cal_dir / "ood.json")
    logger.info("OOD gate: %s", "PASS" if ood_result["gate_passed"] else "FAIL")

    # === 7. Global threshold (max-F1) ===
    threshold_result = find_global_threshold(sims_cal, cal_valid_indices, t_deploy)
    atomic_write_json(threshold_result, cal_dir / "global_threshold.json")
    logger.info("Global threshold: %.4f (F1=%.4f)", threshold_result["threshold"], threshold_result["f1"])

    # === 8. KS-test diagnostic ===
    max_sims_trad = max_sims_cal
    max_sims_ai = np.array([float(sims_unmapped[i].max()) for i in range(len(sims_unmapped))])
    ks_result = ks_test_similarity_distributions(max_sims_trad, max_sims_ai)
    atomic_write_json(ks_result, cal_dir / "ks_test.json")
    logger.info("KS-test: statistic=%.4f, p=%.6f", ks_result["statistic"], ks_result["p_value"])

    # === 9. Generate review.json ===
    # Load AI canary labels
    canary_data = load_json(PROJECT_ROOT / "data" / "canary_items_for_labeling.json")
    canary_items = canary_data["canaries"]

    # Build unified item list for review: unmapped + AI canaries (interleaved)
    # AI canary items are already in the unmapped list — we just need to track which ones they are
    ai_canary_ids = {f"{c['framework']}:{c['control_id']}" for c in canary_items}

    # Calibrate unmapped predictions
    probs_unmapped = calibrate_similarities(sims_unmapped, t_deploy)
    conformal_sets_unmapped = build_prediction_sets(probs_unmapped, hub_ids, conformal_quantile)
    ood_flags_unmapped = flag_ood_items(
        np.array([float(sims_unmapped[i].max()) for i in range(len(sims_unmapped))]),
        ood_threshold,
    )

    # Build items list matching the unmapped controls order
    review_items = []
    for ctrl in unmapped:
        review_items.append({
            "control_id": ctrl["control_id"],
            "framework": ctrl["framework"],
            "control_text": ctrl["control_text"],
        })

    round_dir = PHASE1C_RESULTS_DIR / "round_1"
    round_dir.mkdir(parents=True, exist_ok=True)

    # Get model version
    import subprocess
    try:
        git_sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        git_sha = "unknown"

    generate_review_json(
        items=review_items,
        hub_ids=hub_ids,
        probs=probs_unmapped,
        conformal_sets=conformal_sets_unmapped,
        ood_flags=ood_flags_unmapped,
        threshold=threshold_result["threshold"],
        temperature=t_deploy,
        model_version=git_sha,
        round_number=1,
        output_path=round_dir / "review.json",
    )

    # === 10. Populate crosswalk.db with model prediction assignments ===
    db_path = PHASE1C_CROSSWALK_DB_PATH
    prediction_assignments = []
    for i, ctrl in enumerate(unmapped):
        top_idx = int(probs_unmapped[i].argmax())
        top_hub = hub_ids[top_idx]
        top_conf = float(probs_unmapped[i, top_idx])
        in_conformal = 1 if top_hub in conformal_sets_unmapped[i] else 0

        prediction_assignments.append({
            "control_id": ctrl["control_id"],
            "hub_id": top_hub,
            "confidence": round(top_conf, 4),
            "in_conformal_set": in_conformal,
            "is_ood": 1 if ood_flags_unmapped[i] else 0,
            "provenance": "active_learning_round_1",
            "source_link_id": None,
            "model_version": git_sha,
            "review_status": "pending",
        })

    insert_assignments(db_path, prediction_assignments)
    logger.info("Inserted %d prediction assignments into crosswalk.db", len(prediction_assignments))

    # === Summary ===
    logger.info("=== T2 QUALITY GATES ===")
    logger.info("ECE: %.4f %s", ece, "PASS" if ece_passed else "FAIL")
    logger.info("Conformal coverage: %.4f %s", coverage, "PASS" if coverage_passed else "FAIL")
    logger.info("OOD separation: %.1f%% %s", ood_result["separation_rate"] * 100, "PASS" if ood_result["gate_passed"] else "FAIL")
    logger.info("Global threshold: %.4f (F1=%.4f)", threshold_result["threshold"], threshold_result["f1"])
    logger.info("Review.json: %d items in %s", len(review_items), round_dir / "review.json")
    logger.info("T2 complete in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

Run: `python -m scripts.phase1c.t2_inference_and_calibrate 2>&1 | tee results/phase1c/t2_log.txt`
Expected: All calibration artifacts saved, gates evaluated, review.json generated.

**Estimated time:** ~5-10 minutes (inference only, no training).

- [ ] **Step 3: Verify gate results**

```bash
echo "=== QUALITY GATES ==="
python3 -c "
import json
for f in ['ece_gate', 'conformal', 'ood', 'global_threshold']:
    d = json.load(open(f'results/phase1c/calibration/{f}.json'))
    print(f'{f}: {json.dumps(d, indent=2)[:200]}')
"
echo "=== REVIEW.JSON ==="
python3 -c "
import json
d = json.load(open('results/phase1c/round_1/review.json'))
print(f'Round: {d[\"round\"]}, Items: {len(d[\"items\"])}, Auto-accept candidates: {sum(1 for i in d[\"items\"] if i[\"auto_accept_candidate\"])}')
ood_count = sum(1 for i in d['items'] if i['is_ood'])
print(f'OOD flagged: {ood_count}')
"
```

- [ ] **Step 4: Commit**

```bash
git add scripts/phase1c/t2_inference_and_calibrate.py
git commit -m "feat: T2 script — deployment inference, production calibration, quality gates, review.json"
```

---

### Task 8: T4 Script — Ingest Reviews + Retrain Round

**Files:**
- Create: `scripts/phase1c/t4_retrain_round.py`

This script runs after expert review (T3). It:
1. Loads the reviewed review.json.
2. Ingests accepted/corrected predictions as TieredLinks.
3. Evaluates canary accuracy.
4. Evaluates stopping criteria.
5. If not stopping: retrains deployment model, re-runs inference, generates next round review.json.

- [ ] **Step 1: Write T4 script**

```python
"""T4: Ingest expert reviews, evaluate stopping criteria, optionally retrain.

Reads:
  - results/phase1c/round_N/review.json (expert-reviewed)
  - data/canary_items_for_labeling.json
  - results/phase1c/deployment_model/ (current model)

Writes:
  - results/phase1c/round_N/round_summary.json
  - results/phase1c/round_{N+1}/review.json (if continuing)
  - results/phase1c/deployment_model_round_{N+1}/ (if retraining)

Usage:
  python -m scripts.phase1c.t4_retrain_round --round 1
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

from tract.active_learning.canary import evaluate_canary_accuracy
from tract.active_learning.review import ingest_reviews
from tract.active_learning.stopping import evaluate_stopping_criteria
from tract.config import (
    PHASE1C_AL_MAX_ROUNDS,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1C_RESULTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
)
from tract.crosswalk.snapshot import take_snapshot
from tract.crosswalk.store import update_review_status
from tract.io import atomic_write_json, load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest expert reviews and evaluate stopping criteria")
    parser.add_argument("--round", type=int, required=True, help="Round number to process")
    args = parser.parse_args()

    round_num = args.round
    logger.info("=== T4: Process Round %d Reviews ===", round_num)
    t_start = time.time()

    round_dir = PHASE1C_RESULTS_DIR / f"round_{round_num}"
    review_path = round_dir / "review.json"
    if not review_path.exists():
        raise FileNotFoundError(f"Review file not found: {review_path}")

    review_data = load_json(review_path)
    items = review_data["items"]
    logger.info("Loaded %d review items from round %d", len(items), round_num)

    # Count review statuses
    reviewed_items = [i for i in items if i.get("review") is not None]
    status_counts = Counter(i["review"]["status"] for i in reviewed_items)
    logger.info("Review status: %s (%d unreviewed)", dict(status_counts), len(items) - len(reviewed_items))

    if not reviewed_items:
        logger.error("No items have been reviewed. Cannot proceed.")
        return

    # Snapshot crosswalk DB before any changes
    take_snapshot(PHASE1C_CROSSWALK_DB_PATH, round_num, f"Pre-ingest snapshot for round {round_num}")

    # Ingest accepted/corrected predictions
    new_links = ingest_reviews(review_data)
    logger.info("Ingested %d accepted/corrected predictions", len(new_links))

    # Evaluate canary accuracy
    canary_data = load_json(PROJECT_ROOT / "data" / "canary_items_for_labeling.json")
    canary_labels = {}
    for c in canary_data["canaries"]:
        cid = f"{c['framework']}:{c['control_id']}"
        # Try matching by framework_file_id format too
        for fid, fname in [("csa_aicm", "CSA AI Controls Matrix"), ("eu_ai_act", "EU AI Act — Regulation (EU) 2024/1689"),
                           ("mitre_atlas", "MITRE ATLAS"), ("nist_ai_600_1", "NIST AI 600-1 Generative AI Profile"),
                           ("owasp_agentic_top10", "OWASP Top 10 for Agentic Applications 2026")]:
            if c["framework"] == fid or c["framework"] == fname:
                cid = f"{fid}:{c['control_id']}"
                break
        canary_labels[cid] = frozenset(c["expert_hub_ids"])

    canary_accuracy = evaluate_canary_accuracy(canary_labels, items)

    # Compute acceptance rate
    total_reviewed = len(reviewed_items)
    accepted = status_counts.get("accepted", 0) + status_counts.get("corrected", 0)
    acceptance_rate = accepted / total_reviewed if total_reviewed > 0 else 0.0

    # Compute hub diversity
    unique_hubs = set()
    for link in new_links:
        unique_hubs.add(link.link["cre_id"])

    # Evaluate stopping criteria
    stopping = evaluate_stopping_criteria(
        acceptance_rate=acceptance_rate,
        canary_accuracy=canary_accuracy,
        unique_hubs_accepted=len(unique_hubs),
    )

    # Save round summary
    summary = {
        "round": round_num,
        "total_items": len(items),
        "reviewed": total_reviewed,
        "status_counts": dict(status_counts),
        "acceptance_rate": acceptance_rate,
        "canary_accuracy": canary_accuracy,
        "unique_hubs_accepted": len(unique_hubs),
        "new_links_ingested": len(new_links),
        "stopping": stopping,
    }
    atomic_write_json(summary, round_dir / "round_summary.json")

    if stopping["should_stop"]:
        logger.info("=== STOPPING: All criteria met after round %d ===", round_num)
    elif round_num >= PHASE1C_AL_MAX_ROUNDS:
        logger.info("=== MAX ROUNDS REACHED (%d): Stopping ===", PHASE1C_AL_MAX_ROUNDS)
    else:
        logger.info("=== CONTINUE: Criteria not met, round %d needed ===", round_num + 1)
        logger.info("To retrain and generate next round, run T1 and T2 with updated training data.")

    logger.info("T4 complete in %.1fs", time.time() - t_start)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit** (execution happens after expert review in T3)

```bash
git add scripts/phase1c/t4_retrain_round.py
git commit -m "feat: T4 script — ingest expert reviews, evaluate stopping criteria"
```

---

### Task 9: CREHierarchy hubs_raw Property

The T0 script's `_populate_crosswalk_db()` calls `hierarchy.hubs_raw` to get the raw hub data for building DB records. We need to verify this property exists on `CREHierarchy`, and if not, add it.

**Files:**
- Modify: `tract/hierarchy.py` (if needed)

- [ ] **Step 1: Check if hubs_raw exists**

Run: `python3 -c "from tract.hierarchy import CREHierarchy; print([m for m in dir(CREHierarchy) if 'hub' in m.lower()])"`

- [ ] **Step 2: If hubs_raw doesn't exist, add it or adjust T0 to use the correct property**

The T0 script needs a dict mapping hub_id -> {name, path, parent} for `build_hub_records()`. Check what CREHierarchy exposes and adapt either the hierarchy class or the `build_hub_records()` function.

If the hierarchy exposes `hierarchy.hubs` as a dict of hub_id -> hub objects with `.name`, `.path`, `.parent` attributes, then modify `build_hub_records()` in `tract/crosswalk/populate.py` to accept the hierarchy directly:

```python
def build_hub_records_from_hierarchy(hierarchy: CREHierarchy) -> list[dict]:
    """Build hub records from CREHierarchy for insert_hubs()."""
    records = []
    for hub_id in sorted(hierarchy.hubs.keys()):
        hub = hierarchy.hubs[hub_id]
        parent = hierarchy.get_parent(hub_id)
        records.append({
            "id": hub_id,
            "name": hub.name,
            "path": getattr(hub, "path", ""),
            "parent_id": parent.hub_id if parent else None,
        })
    return records
```

- [ ] **Step 3: Update T0 script to use the correct function**

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_crosswalk_populate.py -v`
Expected: PASS

- [ ] **Step 5: Commit if changes were needed**

```bash
git add tract/crosswalk/populate.py tract/hierarchy.py
git commit -m "fix: adapt hub record builder to CREHierarchy API"
```

---

### Task 10: Integration Test — Full T0→T1→T2 Pipeline Smoke Run

This is the execution and verification step. No new code — just running the pipeline and verifying outputs.

- [ ] **Step 1: Run T0**

```bash
python -m scripts.phase1c.t0_extract_similarities 2>&1 | tee results/phase1c/t0_log.txt
```

Verify:
- 5 NPZ files exist in `results/phase1c/similarities/`
- crosswalk.db exists and has data in all 4 tables

- [ ] **Step 2: Run T1**

```bash
python -m scripts.phase1c.t1_calibrate_and_train 2>&1 | tee results/phase1c/t1_log.txt
```

Verify:
- `results/phase1c/calibration/t_lofo_result.json` exists
- `results/phase1c/deployment_model/model/model/` has adapter files
- `results/phase1c/holdout/` has calibration_links.json and canary_links.json

- [ ] **Step 3: Run T2**

```bash
python -m scripts.phase1c.t2_inference_and_calibrate 2>&1 | tee results/phase1c/t2_log.txt
```

Verify:
- All quality gates have results in `results/phase1c/calibration/`
- `results/phase1c/round_1/review.json` exists with correct item count
- Assignments inserted into crosswalk.db

- [ ] **Step 4: Report gate results**

```bash
python3 -c "
import json
gates = {}
for f in ['ece_gate', 'conformal', 'ood']:
    d = json.load(open(f'results/phase1c/calibration/{f}.json'))
    gates[f] = d.get('passed', d.get('gate_passed', '?'))
print('=== GATE 2: Calibration ===')
for k, v in gates.items():
    print(f'  {k}: {\"PASS\" if v else \"FAIL\"}')
all_pass = all(gates.values())
print(f'  Overall: {\"PASS\" if all_pass else \"FAIL\"}')
"
```

- [ ] **Step 5: Full test suite regression check**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All 324+ tests pass.

- [ ] **Step 6: Commit results and scripts**

```bash
git add results/phase1c/t0_log.txt results/phase1c/t1_log.txt results/phase1c/t2_log.txt
git commit -m "results: Phase 1C orchestration T0→T1→T2 pipeline run with gate results"
```

---

## Self-Review Checklist

| Spec Section | Plan Coverage |
|-------------|--------------|
| 3. Similarity Extraction (T0) | Task 5 (T0 script) |
| 4. Calibration Pipeline | Tasks 6 (T_lofo), 7 (T_deploy, ECE, conformal, OOD, threshold) |
| 5. Deployment Model | Tasks 2 (model I/O), 3 (training wrapper), 6 (training execution) |
| 5.4 Unmapped AI Controls | Task 1 (unmapped loader) |
| 6.1 Canary Design | Task 7 (canaries interleaved in review.json) |
| 6.2 Round Structure | Task 8 (T4 round processing) |
| 6.3 Stopping Criteria | Task 8 (uses evaluate_stopping_criteria) |
| 7.2 Population Strategy | Tasks 4 (helpers), 5 (T0 initial), 7 (T2 predictions) |
| 8. Execution Flow | Tasks 5-8 (T0/T1/T2/T4 scripts) |
| 9. Quality Gates | Task 7 (ECE, conformal, OOD gates), Task 10 (verification) |
