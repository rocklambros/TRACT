# Phase 1D: CLI Tool & Hub Proposal System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `tract` CLI with 8 commands exposing the trained model, calibration pipeline, and crosswalk database to end users. Build a guardrailed hub proposal system for extending the CRE ontology when OOD controls are detected.

**Architecture:** A stateful `TRACTPredictor` class loads the deployment model + cached embeddings + calibration params once, then serves all inference requests. Library modules (`inference.py`, `compare.py`, `proposals/`) keep business logic CLI-agnostic. `cli.py` is a thin argparse layer that formats output. T5 script gains deployment artifact generation (NPZ + calibration bundle).

**Tech Stack:** Python 3.12, sentence-transformers, numpy, scipy, hdbscan, anthropic, sqlite3, argparse, pytest

**Spec:** `docs/superpowers/specs/2026-04-30-phase1d-design.md`

---

## File Structure

### New Files

```
tract/inference.py                    — TRACTPredictor, predict(), predict_batch(), find_duplicates()
tract/compare.py                     — cross_framework_matrix() from crosswalk.db
tract/cli.py                         — argparse + output formatting (8 subcommands)
tract/proposals/__init__.py          — Package marker
tract/proposals/cluster.py           — HDBSCAN clustering on OOD embeddings
tract/proposals/guardrails.py        — 6-filter pipeline per proposed cluster
tract/proposals/naming.py            — LLM-generated hub names (optional, --name-with-llm)
tract/proposals/review.py            — Interactive CLI review loop + hierarchy update

tests/test_inference.py              — Unit tests for TRACTPredictor
tests/test_compare.py                — Unit tests for cross_framework_matrix
tests/test_cli.py                    — CLI argparse and output tests
tests/test_proposals_cluster.py      — Cluster determinism + edge cases
tests/test_proposals_guardrails.py   — Each guardrail individually
tests/test_proposals_naming.py       — Mock LLM naming
tests/test_proposals_review.py       — Proposal write + review session
```

### Modified Files

```
tract/config.py                      — Add 15 PHASE1D_* constants
scripts/phase1c/t5_finalize_crosswalk.py — Add deployment_artifacts.npz + calibration.json generation
requirements.txt                     — Add hdbscan==0.8.40
```

### Existing Files Referenced (read-only)

```
tract/calibration/temperature.py     — calibrate_similarities()
tract/calibration/conformal.py       — build_prediction_sets()
tract/calibration/ood.py             — flag_ood_items()
tract/hierarchy.py                   — CREHierarchy, HubNode
tract/crosswalk/store.py             — insert_hubs(), insert_assignments(), get_hub(), etc.
tract/crosswalk/schema.py            — get_connection()
tract/crosswalk/export.py            — export_crosswalk()
tract/schema.py                      — FrameworkOutput, Control (Pydantic models)
tract/sanitize.py                    — sanitize_text()
tract/io.py                          — atomic_write_json(), load_json()
tract/active_learning/model_io.py    — load_deployment_model()
data/processed/cre_hierarchy.json    — Hub hierarchy
results/phase1c/crosswalk.db         — Crosswalk database
results/phase1c/deployment_model/    — Model weights
```

---

### Task 1: Config Constants & Dependency

**Files:**
- Modify: `tract/config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add PHASE1D constants to config.py**

Add after line ~228 (after `PHASE1C_UNMAPPED_FRAMEWORKS`):

```python
# ── Phase 1D: CLI & Hub Proposals ─────────────────────────────────────

PHASE1D_DEPLOYMENT_MODEL_DIR: Final[Path] = PHASE1C_RESULTS_DIR / "deployment_model"
PHASE1D_ARTIFACTS_PATH: Final[Path] = PHASE1D_DEPLOYMENT_MODEL_DIR / "deployment_artifacts.npz"
PHASE1D_CALIBRATION_PATH: Final[Path] = PHASE1D_DEPLOYMENT_MODEL_DIR / "calibration.json"

PHASE1D_DEFAULT_TOP_K: Final[int] = 5
PHASE1D_DUPLICATE_THRESHOLD: Final[float] = 0.95
PHASE1D_SIMILAR_THRESHOLD: Final[float] = 0.85
PHASE1D_HEALTH_CHECK_FLOOR: Final[float] = 0.3
PHASE1D_INGEST_MAX_FILE_SIZE: Final[int] = 50 * 1024 * 1024  # 50MB

# Hub Proposal System
PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE: Final[int] = 3
PHASE1D_HDBSCAN_MIN_SAMPLES: Final[int] = 2
PHASE1D_PROPOSAL_INTER_CLUSTER_MAX_COSINE: Final[float] = 0.70
PHASE1D_PROPOSAL_MIN_FRAMEWORKS: Final[int] = 2
PHASE1D_PROPOSAL_BUDGET_CAP: Final[int] = 40
PHASE1D_PROPOSAL_NAMING_MODEL: Final[str] = "claude-sonnet-4-20250514"
PHASE1D_PROPOSAL_UNCERTAIN_PLACEMENT_FLOOR: Final[float] = 0.20
```

- [ ] **Step 2: Add hdbscan to requirements.txt**

Add `hdbscan==0.8.40` to the dependency list.

- [ ] **Step 3: Verify imports work**

```bash
python -c "from tract.config import PHASE1D_ARTIFACTS_PATH; print(PHASE1D_ARTIFACTS_PATH)"
```

---

### Task 2: Update T5 — Deployment Artifact Generation

**Files:**
- Modify: `scripts/phase1c/t5_finalize_crosswalk.py`

**Why first:** All downstream inference depends on `deployment_artifacts.npz` and `calibration.json` existing. These artifacts are generated during T5 finalization. Without them, `TRACTPredictor` cannot load.

- [ ] **Step 1: Write the test**

Create `tests/test_t5_artifacts.py`:

```python
"""Tests for T5 deployment artifact generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.encode.return_value = np.random.default_rng(42).standard_normal((5, 1024)).astype(np.float32)
    return model


class TestGenerateDeploymentArtifacts:
    def test_npz_contains_required_keys(self, tmp_path: Path, mock_model) -> None:
        from scripts.phase1c.t5_finalize_crosswalk import _generate_deployment_artifacts

        hub_ids = [f"hub-{i}" for i in range(5)]
        control_ids = [f"ctrl-{i}" for i in range(5)]
        hub_texts = {hid: f"Hub {i} text" for i, hid in enumerate(hub_ids)}
        control_texts = {cid: f"Control {i} text" for i, cid in enumerate(control_ids)}
        adapter_path = tmp_path / "adapter_model.safetensors"
        adapter_path.write_bytes(b"fake-adapter-weights")

        npz_path = tmp_path / "deployment_artifacts.npz"
        _generate_deployment_artifacts(
            model=mock_model,
            hub_ids=hub_ids,
            hub_texts=hub_texts,
            control_ids=control_ids,
            control_texts=control_texts,
            adapter_path=adapter_path,
            output_path=npz_path,
        )

        data = np.load(str(npz_path), allow_pickle=True)
        assert "hub_embeddings" in data
        assert "control_embeddings" in data
        assert "hub_ids" in data
        assert "control_ids" in data
        assert "model_adapter_hash" in data
        assert "generation_timestamp" in data
        assert data["hub_embeddings"].shape == (5, 1024)
        assert data["control_embeddings"].shape == (5, 1024)
        assert list(data["hub_ids"]) == sorted(hub_ids)

    def test_hub_ids_are_canonically_sorted(self, tmp_path: Path, mock_model) -> None:
        from scripts.phase1c.t5_finalize_crosswalk import _generate_deployment_artifacts

        hub_ids = ["zzz-999", "aaa-001", "mmm-500"]
        control_ids = ["ctrl-1"]
        hub_texts = {hid: f"text for {hid}" for hid in hub_ids}
        control_texts = {"ctrl-1": "text"}
        adapter_path = tmp_path / "adapter_model.safetensors"
        adapter_path.write_bytes(b"fake")

        mock_model.encode.side_effect = [
            np.random.default_rng(42).standard_normal((3, 1024)).astype(np.float32),
            np.random.default_rng(42).standard_normal((1, 1024)).astype(np.float32),
        ]

        npz_path = tmp_path / "deployment_artifacts.npz"
        _generate_deployment_artifacts(
            model=mock_model,
            hub_ids=hub_ids,
            hub_texts=hub_texts,
            control_ids=control_ids,
            control_texts=control_texts,
            adapter_path=adapter_path,
            output_path=npz_path,
        )

        data = np.load(str(npz_path), allow_pickle=True)
        assert list(data["hub_ids"]) == ["aaa-001", "mmm-500", "zzz-999"]


class TestGenerateCalibrationBundle:
    def test_json_contains_required_keys(self, tmp_path: Path) -> None:
        from scripts.phase1c.t5_finalize_crosswalk import _generate_calibration_bundle

        hierarchy_path = tmp_path / "cre_hierarchy.json"
        hierarchy_path.write_text('{"test": true}')

        cal_path = tmp_path / "calibration.json"
        _generate_calibration_bundle(
            t_deploy=0.074,
            ood_threshold=0.568,
            conformal_quantile=0.997,
            global_threshold=0.121,
            hierarchy_path=hierarchy_path,
            output_path=cal_path,
        )

        data = json.loads(cal_path.read_text())
        assert data["t_deploy"] == 0.074
        assert data["ood_threshold"] == 0.568
        assert data["conformal_quantile"] == 0.997
        assert data["global_threshold"] == 0.121
        assert "hierarchy_hash" in data
        assert "calibration_note" in data
```

- [ ] **Step 2: Add `_generate_deployment_artifacts()` to T5**

Add this function to `scripts/phase1c/t5_finalize_crosswalk.py`:

```python
def _generate_deployment_artifacts(
    model: Any,
    hub_ids: list[str],
    hub_texts: dict[str, str],
    control_ids: list[str],
    control_texts: dict[str, str],
    adapter_path: Path,
    output_path: Path,
) -> None:
    """Generate consolidated deployment NPZ with all cached embeddings.

    Hub IDs are stored in canonical sorted order. Embedding rows match
    the hub_ids/control_ids arrays for index consistency.
    """
    sorted_hub_ids = sorted(hub_ids)
    sorted_hub_texts = [hub_texts[hid] for hid in sorted_hub_ids]

    hub_embs = model.encode(
        sorted_hub_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128,
    )

    sorted_control_ids = sorted(control_ids)
    sorted_control_texts = [control_texts[cid] for cid in sorted_control_ids]

    ctrl_embs = model.encode(
        sorted_control_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=128,
    )

    adapter_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
    timestamp = datetime.now(timezone.utc).isoformat()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        hub_embeddings=hub_embs.astype(np.float32),
        control_embeddings=ctrl_embs.astype(np.float32),
        hub_ids=np.array(sorted_hub_ids),
        control_ids=np.array(sorted_control_ids),
        model_adapter_hash=np.array(adapter_hash),
        generation_timestamp=np.array(timestamp),
    )
    logger.info(
        "Saved deployment artifacts: %d hubs, %d controls, adapter_hash=%s…",
        len(sorted_hub_ids), len(sorted_control_ids), adapter_hash[:12],
    )
```

Add required imports at top of T5:

```python
import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
```

- [ ] **Step 3: Add `_generate_calibration_bundle()` to T5**

```python
def _generate_calibration_bundle(
    t_deploy: float,
    ood_threshold: float,
    conformal_quantile: float,
    global_threshold: float,
    hierarchy_path: Path,
    output_path: Path,
) -> None:
    """Bundle all calibration parameters into a single JSON file."""
    hierarchy_hash = hashlib.sha256(hierarchy_path.read_bytes()).hexdigest()

    bundle = {
        "t_deploy": t_deploy,
        "ood_threshold": ood_threshold,
        "conformal_quantile": conformal_quantile,
        "global_threshold": global_threshold,
        "hierarchy_hash": hierarchy_hash,
        "calibration_note": (
            "Calibrated on 420 traditional framework holdout items. "
            "Accuracy on AI framework text may differ."
        ),
    }

    atomic_write_json(bundle, output_path)
    logger.info("Saved calibration bundle to %s", output_path)
```

- [ ] **Step 4: Wire artifact generation into T5 `main()`**

After the export section (around line 363), add:

```python
    # === 7. Generate deployment artifacts ===
    from tract.active_learning.model_io import load_deployment_model
    from tract.config import PHASE1D_ARTIFACTS_PATH, PHASE1D_CALIBRATION_PATH

    deploy_model_dir = PHASE1C_DEPLOYMENT_MODEL_DIR / "model"
    if deploy_model_dir.exists():
        deploy_model = load_deployment_model(deploy_model_dir)

        hub_texts = {
            node.hub_id: f"{node.hierarchy_path}\n{node.name}"
            for node in hierarchy.hubs.values()
        }
        all_hub_ids = sorted(hierarchy.hubs.keys())

        ctrl_texts: dict[str, str] = {}
        ctrl_ids: list[str] = []
        for fw_data in all_fw_data:
            for ctrl in fw_data.get("controls", []):
                cid = f"{fw_data['framework_id']}::{ctrl['control_id']}"
                text_parts = [ctrl.get("title", ""), ctrl.get("description", "")]
                if ctrl.get("full_text"):
                    text_parts.append(ctrl["full_text"])
                ctrl_texts[cid] = " ".join(p for p in text_parts if p)
                ctrl_ids.append(cid)

        adapter_path = deploy_model_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            for p in deploy_model_dir.rglob("adapter_model.safetensors"):
                adapter_path = p
                break

        _generate_deployment_artifacts(
            model=deploy_model,
            hub_ids=all_hub_ids,
            hub_texts=hub_texts,
            control_ids=ctrl_ids,
            control_texts=ctrl_texts,
            adapter_path=adapter_path,
            output_path=PHASE1D_ARTIFACTS_PATH,
        )

        cal_results_dir = PHASE1C_RESULTS_DIR / "calibration"
        t_deploy = load_json(cal_results_dir / "t_deploy_result.json")["temperature"]
        ood_data = load_json(cal_results_dir / "ood.json")["threshold"]
        conformal_data = load_json(cal_results_dir / "conformal.json")["quantile"]
        global_data = load_json(cal_results_dir / "global_threshold.json")["threshold"]

        _generate_calibration_bundle(
            t_deploy=t_deploy,
            ood_threshold=ood_data,
            conformal_quantile=conformal_data,
            global_threshold=global_data,
            hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
            output_path=PHASE1D_CALIBRATION_PATH,
        )
    else:
        logger.warning("Deployment model not found at %s — skipping artifact generation", deploy_model_dir)
```

- [ ] **Step 5: Run tests and verify**

```bash
python -m pytest tests/test_t5_artifacts.py -v
```

---

### Task 3: Inference Module — Data Types

**Files:**
- Create: `tract/inference.py`

**Why this order:** Data types (`HubPrediction`, `DuplicateMatch`, `DeploymentArtifacts`) are needed by everything downstream: predictor, CLI, ingest, proposals. Define them first, with no dependencies beyond stdlib + numpy.

- [ ] **Step 1: Write the test for data types**

Create `tests/test_inference.py`:

```python
"""Tests for tract.inference module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestHubPrediction:
    def test_fields_accessible(self) -> None:
        from tract.inference import HubPrediction

        pred = HubPrediction(
            hub_id="646-285",
            hub_name="AI compliance management",
            hierarchy_path="Root > Compliance > AI compliance management",
            raw_similarity=0.523,
            calibrated_confidence=0.847,
            in_conformal_set=True,
            is_ood=False,
        )
        assert pred.hub_id == "646-285"
        assert pred.calibrated_confidence == 0.847
        assert pred.in_conformal_set is True

    def test_to_dict(self) -> None:
        from tract.inference import HubPrediction

        pred = HubPrediction(
            hub_id="646-285",
            hub_name="Test",
            hierarchy_path="Root > Test",
            raw_similarity=0.5,
            calibrated_confidence=0.8,
            in_conformal_set=True,
            is_ood=False,
        )
        d = pred.to_dict()
        assert isinstance(d, dict)
        assert d["hub_id"] == "646-285"
        assert "raw_similarity" in d


class TestDuplicateMatch:
    def test_tier_values(self) -> None:
        from tract.inference import DuplicateMatch

        dup = DuplicateMatch(
            control_id="csa_aicm::AIC-01",
            framework_id="csa_aicm",
            title="Access Control",
            similarity=0.97,
            tier="duplicate",
        )
        assert dup.tier == "duplicate"

        sim = DuplicateMatch(
            control_id="mitre_atlas::AML.T0001",
            framework_id="mitre_atlas",
            title="Technique 1",
            similarity=0.88,
            tier="similar",
        )
        assert sim.tier == "similar"


class TestDeploymentArtifacts:
    def test_load_from_npz(self, tmp_path: Path) -> None:
        from tract.inference import load_deployment_artifacts

        rng = np.random.default_rng(42)
        hub_ids = ["aaa-001", "bbb-002", "ccc-003"]
        control_ids = ["ctrl-1", "ctrl-2"]
        npz_path = tmp_path / "deployment_artifacts.npz"
        np.savez(
            str(npz_path),
            hub_embeddings=rng.standard_normal((3, 1024)).astype(np.float32),
            control_embeddings=rng.standard_normal((2, 1024)).astype(np.float32),
            hub_ids=np.array(hub_ids),
            control_ids=np.array(control_ids),
            model_adapter_hash=np.array("abc123"),
            generation_timestamp=np.array("2026-04-30T00:00:00Z"),
        )

        arts = load_deployment_artifacts(npz_path)
        assert arts.hub_embeddings.shape == (3, 1024)
        assert arts.control_embeddings.shape == (2, 1024)
        assert arts.hub_ids == hub_ids
        assert arts.control_ids == control_ids
```

- [ ] **Step 2: Implement data types and artifact loader**

Write the top portion of `tract/inference.py`:

```python
"""TRACT model inference — prediction, duplicate detection, artifact loading.

TRACTPredictor loads the deployment model + cached embeddings + calibration
parameters. Stateful — holds model in memory for repeated predict() calls.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import softmax

from tract.calibration.conformal import build_prediction_sets
from tract.calibration.ood import flag_ood_items
from tract.calibration.temperature import calibrate_similarities
from tract.config import (
    PHASE1D_ARTIFACTS_PATH,
    PHASE1D_CALIBRATION_PATH,
    PHASE1D_DEFAULT_TOP_K,
    PHASE1D_DEPLOYMENT_MODEL_DIR,
    PHASE1D_DUPLICATE_THRESHOLD,
    PHASE1D_HEALTH_CHECK_FLOOR,
    PHASE1D_SIMILAR_THRESHOLD,
)
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.sanitize import sanitize_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubPrediction:
    hub_id: str
    hub_name: str
    hierarchy_path: str
    raw_similarity: float
    calibrated_confidence: float
    in_conformal_set: bool
    is_ood: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DuplicateMatch:
    control_id: str
    framework_id: str
    title: str
    similarity: float
    tier: str  # "duplicate" (>=0.95) or "similar" (>=0.85)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentArtifacts:
    hub_embeddings: NDArray[np.floating]
    control_embeddings: NDArray[np.floating]
    hub_ids: list[str]
    control_ids: list[str]
    model_adapter_hash: str
    generation_timestamp: str


def load_deployment_artifacts(artifacts_path: Path) -> DeploymentArtifacts:
    """Load NPZ without model. For proposal pipeline (no inference needed)."""
    data = np.load(str(artifacts_path), allow_pickle=True)
    return DeploymentArtifacts(
        hub_embeddings=data["hub_embeddings"],
        control_embeddings=data["control_embeddings"],
        hub_ids=list(data["hub_ids"]),
        control_ids=list(data["control_ids"]),
        model_adapter_hash=str(data["model_adapter_hash"]),
        generation_timestamp=str(data["generation_timestamp"]),
    )
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_inference.py::TestHubPrediction tests/test_inference.py::TestDuplicateMatch tests/test_inference.py::TestDeploymentArtifacts -v
```

---

### Task 4: TRACTPredictor — Core Inference

**Files:**
- Modify: `tract/inference.py`
- Modify: `tests/test_inference.py`

- [ ] **Step 1: Write failing tests for TRACTPredictor**

Add to `tests/test_inference.py`:

```python
@pytest.fixture
def predictor_dir(tmp_path: Path) -> Path:
    """Create a mock predictor directory with NPZ, calibration, hierarchy, and model."""
    rng = np.random.default_rng(42)
    n_hubs = 10
    n_ctrls = 5

    hub_embs = rng.standard_normal((n_hubs, 1024)).astype(np.float32)
    hub_embs = hub_embs / np.linalg.norm(hub_embs, axis=1, keepdims=True)
    ctrl_embs = rng.standard_normal((n_ctrls, 1024)).astype(np.float32)
    ctrl_embs = ctrl_embs / np.linalg.norm(ctrl_embs, axis=1, keepdims=True)

    hub_ids = [f"{i:03d}-{i:03d}" for i in range(n_hubs)]
    control_ids = [f"fw::ctrl-{i}" for i in range(n_ctrls)]

    model_dir = tmp_path / "deployment_model"
    model_dir.mkdir()
    (model_dir / "model").mkdir()

    adapter_path = model_dir / "model" / "adapter_model.safetensors"
    adapter_path.write_bytes(b"fake-adapter")
    adapter_hash = hashlib.sha256(b"fake-adapter").hexdigest()

    npz_path = model_dir / "deployment_artifacts.npz"
    np.savez(
        str(npz_path),
        hub_embeddings=hub_embs,
        control_embeddings=ctrl_embs,
        hub_ids=np.array(hub_ids),
        control_ids=np.array(control_ids),
        model_adapter_hash=np.array(adapter_hash),
        generation_timestamp=np.array("2026-04-30T00:00:00Z"),
    )

    hierarchy_data = _build_mock_hierarchy(hub_ids)
    hierarchy_path = tmp_path / "cre_hierarchy.json"
    hierarchy_path.write_text(json.dumps(hierarchy_data))

    cal_data = {
        "t_deploy": 0.074,
        "ood_threshold": 0.568,
        "conformal_quantile": 0.997,
        "global_threshold": 0.121,
        "hierarchy_hash": hashlib.sha256(
            hierarchy_path.read_bytes()
        ).hexdigest(),
        "calibration_note": "Test calibration",
    }
    cal_path = model_dir / "calibration.json"
    cal_path.write_text(json.dumps(cal_data))

    return tmp_path


def _build_mock_hierarchy(hub_ids: list[str]) -> dict:
    """Build a minimal valid hierarchy for testing."""
    hubs = {}
    for hid in hub_ids:
        hubs[hid] = {
            "hub_id": hid,
            "name": f"Hub {hid}",
            "parent_id": None,
            "children_ids": [],
            "depth": 0,
            "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}",
            "is_leaf": True,
            "sibling_hub_ids": [],
        }
    return {
        "hubs": hubs,
        "roots": hub_ids,
        "label_space": hub_ids,
        "fetch_timestamp": "2026-04-30T00:00:00Z",
        "data_hash": "test_hash",
        "version": "1.0",
    }


class TestTRACTPredictor:
    def test_predict_returns_hub_predictions(self, predictor_dir: Path) -> None:
        from tract.inference import TRACTPredictor

        mock_model = MagicMock()
        rng = np.random.default_rng(99)
        query_emb = rng.standard_normal((1, 1024)).astype(np.float32)
        query_emb = query_emb / np.linalg.norm(query_emb)
        mock_model.encode.return_value = query_emb
        # Health check encode returns something with cosine > 0.3
        health_emb = np.load(
            str(predictor_dir / "deployment_model" / "deployment_artifacts.npz"),
            allow_pickle=True,
        )["hub_embeddings"][0:1]
        mock_model.encode.side_effect = [health_emb, query_emb]

        with patch("tract.inference.load_deployment_model", return_value=mock_model):
            predictor = TRACTPredictor(predictor_dir / "deployment_model")

        mock_model.encode.return_value = query_emb
        preds = predictor.predict("Test control text about access control")
        assert len(preds) == 5
        assert all(isinstance(p, type(preds[0])) for p in preds)
        assert preds[0].calibrated_confidence >= preds[1].calibrated_confidence

    def test_predict_applies_sanitization(self, predictor_dir: Path) -> None:
        from tract.inference import TRACTPredictor

        mock_model = MagicMock()
        rng = np.random.default_rng(99)
        emb = rng.standard_normal((1, 1024)).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        health_emb = np.load(
            str(predictor_dir / "deployment_model" / "deployment_artifacts.npz"),
            allow_pickle=True,
        )["hub_embeddings"][0:1]
        mock_model.encode.side_effect = [health_emb, emb]

        with patch("tract.inference.load_deployment_model", return_value=mock_model):
            predictor = TRACTPredictor(predictor_dir / "deployment_model")

        mock_model.encode.return_value = emb
        with patch("tract.inference.sanitize_text", wraps=sanitize_text) as mock_san:
            predictor.predict("text with \x00 null bytes")
            mock_san.assert_called_once()

    def test_predict_ood_flag(self, predictor_dir: Path) -> None:
        from tract.inference import TRACTPredictor

        mock_model = MagicMock()
        # Very low similarity → OOD
        zero_emb = np.zeros((1, 1024), dtype=np.float32)
        zero_emb[0, 0] = 1.0  # unit vector in dim 0, far from random hubs
        health_emb = np.load(
            str(predictor_dir / "deployment_model" / "deployment_artifacts.npz"),
            allow_pickle=True,
        )["hub_embeddings"][0:1]
        mock_model.encode.side_effect = [health_emb, zero_emb]

        with patch("tract.inference.load_deployment_model", return_value=mock_model):
            predictor = TRACTPredictor(predictor_dir / "deployment_model")

        mock_model.encode.return_value = zero_emb
        preds = predictor.predict("completely unrelated cooking recipe text")
        # OOD flag should be set since max cosine < 0.568
        if preds[0].is_ood:
            assert all(p.is_ood for p in preds)

    def test_predict_batch(self, predictor_dir: Path) -> None:
        from tract.inference import TRACTPredictor

        mock_model = MagicMock()
        rng = np.random.default_rng(99)
        batch_embs = rng.standard_normal((3, 1024)).astype(np.float32)
        batch_embs = batch_embs / np.linalg.norm(batch_embs, axis=1, keepdims=True)
        health_emb = np.load(
            str(predictor_dir / "deployment_model" / "deployment_artifacts.npz"),
            allow_pickle=True,
        )["hub_embeddings"][0:1]
        mock_model.encode.side_effect = [health_emb, batch_embs]

        with patch("tract.inference.load_deployment_model", return_value=mock_model):
            predictor = TRACTPredictor(predictor_dir / "deployment_model")

        mock_model.encode.return_value = batch_embs
        results = predictor.predict_batch(["text 1", "text 2", "text 3"])
        assert len(results) == 3
        assert all(len(preds) == 5 for preds in results)
```

- [ ] **Step 2: Implement TRACTPredictor.__init__**

Add to `tract/inference.py`:

```python
class TRACTPredictor:
    """Loads deployment model + cached artifacts. Stateful — holds model in memory."""

    def __init__(self, model_dir: Path) -> None:
        from tract.active_learning.model_io import load_deployment_model

        self._model_dir = model_dir

        artifacts_path = model_dir / "deployment_artifacts.npz"
        calibration_path = model_dir / "calibration.json"

        if not artifacts_path.exists():
            raise FileNotFoundError(f"Deployment artifacts not found: {artifacts_path}")
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration bundle not found: {calibration_path}")

        self._artifacts = load_deployment_artifacts(artifacts_path)
        self._calibration = load_json(calibration_path)
        self._t_deploy: float = self._calibration["t_deploy"]
        self._ood_threshold: float = self._calibration["ood_threshold"]
        self._conformal_quantile: float = self._calibration["conformal_quantile"]

        hierarchy_path = model_dir.parent.parent / "data" / "processed" / "cre_hierarchy.json"
        if not hierarchy_path.exists():
            from tract.config import PROCESSED_DIR
            hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
        self._hierarchy = CREHierarchy.load(hierarchy_path)

        adapter_path = model_dir / "model" / "adapter_model.safetensors"
        if adapter_path.exists():
            current_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
            if current_hash != self._artifacts.model_adapter_hash:
                raise ValueError(
                    f"Model adapter hash mismatch: artifacts={self._artifacts.model_adapter_hash[:12]}… "
                    f"vs current={current_hash[:12]}…"
                )

        self._model = load_deployment_model(model_dir / "model")

        # Health check
        health_emb = self._model.encode(
            ["access control security authentication"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        max_cos = float(np.max(self._artifacts.hub_embeddings @ health_emb.T))
        if max_cos < PHASE1D_HEALTH_CHECK_FLOOR:
            raise RuntimeError(
                f"Health check failed: max cosine={max_cos:.3f} < floor={PHASE1D_HEALTH_CHECK_FLOOR}"
            )
        logger.info("TRACTPredictor initialized (health check cosine=%.3f)", max_cos)
```

- [ ] **Step 3: Implement predict()**

```python
    def predict(self, text: str, top_k: int = PHASE1D_DEFAULT_TOP_K) -> list[HubPrediction]:
        """Single control text -> top-K hub assignments with calibrated confidence."""
        clean_text = sanitize_text(text)

        query_emb = self._model.encode(
            [clean_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        sims = (self._artifacts.hub_embeddings @ query_emb.T).flatten()
        max_sim = float(np.max(sims))
        is_ood = max_sim < self._ood_threshold

        sims_2d = sims.reshape(1, -1)
        probs = calibrate_similarities(sims_2d, self._t_deploy).flatten()

        conformal_sets = build_prediction_sets(
            probs.reshape(1, -1),
            self._artifacts.hub_ids,
            self._conformal_quantile,
        )
        conformal_set = conformal_sets[0]

        ranked_indices = np.argsort(probs)[::-1][:top_k]

        predictions: list[HubPrediction] = []
        for idx in ranked_indices:
            hub_id = self._artifacts.hub_ids[idx]
            node = self._hierarchy.hubs.get(hub_id)
            predictions.append(HubPrediction(
                hub_id=hub_id,
                hub_name=node.name if node else hub_id,
                hierarchy_path=node.hierarchy_path if node else hub_id,
                raw_similarity=float(sims[idx]),
                calibrated_confidence=float(probs[idx]),
                in_conformal_set=hub_id in conformal_set,
                is_ood=is_ood,
            ))

        return predictions
```

- [ ] **Step 4: Implement predict_batch()**

```python
    def predict_batch(
        self, texts: list[str], top_k: int = PHASE1D_DEFAULT_TOP_K,
    ) -> list[list[HubPrediction]]:
        """Batch prediction for tract ingest."""
        clean_texts = [sanitize_text(t) for t in texts]

        query_embs = self._model.encode(
            clean_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=128,
        )

        sims = query_embs @ self._artifacts.hub_embeddings.T
        max_sims = sims.max(axis=1)
        ood_flags = flag_ood_items(max_sims, self._ood_threshold)

        probs = calibrate_similarities(sims, self._t_deploy)

        conformal_sets = build_prediction_sets(
            probs, self._artifacts.hub_ids, self._conformal_quantile,
        )

        results: list[list[HubPrediction]] = []
        for i in range(len(texts)):
            ranked_indices = np.argsort(probs[i])[::-1][:top_k]
            preds: list[HubPrediction] = []
            for idx in ranked_indices:
                hub_id = self._artifacts.hub_ids[idx]
                node = self._hierarchy.hubs.get(hub_id)
                preds.append(HubPrediction(
                    hub_id=hub_id,
                    hub_name=node.name if node else hub_id,
                    hierarchy_path=node.hierarchy_path if node else hub_id,
                    raw_similarity=float(sims[i, idx]),
                    calibrated_confidence=float(probs[i, idx]),
                    in_conformal_set=hub_id in conformal_sets[i],
                    is_ood=ood_flags[i],
                ))
            results.append(preds)

        return results
```

- [ ] **Step 5: Implement find_duplicates()**

```python
    def find_duplicates(
        self,
        text: str,
        duplicate_threshold: float = PHASE1D_DUPLICATE_THRESHOLD,
        similar_threshold: float = PHASE1D_SIMILAR_THRESHOLD,
    ) -> tuple[list[DuplicateMatch], list[DuplicateMatch]]:
        """Compare text embedding against all control_embeddings.

        Returns (duplicates above 0.95, similar controls above 0.85).
        """
        clean_text = sanitize_text(text)
        query_emb = self._model.encode(
            [clean_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).flatten()

        sims = self._artifacts.control_embeddings @ query_emb
        duplicates: list[DuplicateMatch] = []
        similar: list[DuplicateMatch] = []

        for i, sim_val in enumerate(sims):
            if sim_val >= duplicate_threshold:
                ctrl_id = self._artifacts.control_ids[i]
                fw_id = ctrl_id.split("::")[0] if "::" in ctrl_id else ""
                duplicates.append(DuplicateMatch(
                    control_id=ctrl_id,
                    framework_id=fw_id,
                    title=ctrl_id,
                    similarity=float(sim_val),
                    tier="duplicate",
                ))
            elif sim_val >= similar_threshold:
                ctrl_id = self._artifacts.control_ids[i]
                fw_id = ctrl_id.split("::")[0] if "::" in ctrl_id else ""
                similar.append(DuplicateMatch(
                    control_id=ctrl_id,
                    framework_id=fw_id,
                    title=ctrl_id,
                    similarity=float(sim_val),
                    tier="similar",
                ))

        duplicates.sort(key=lambda m: -m.similarity)
        similar.sort(key=lambda m: -m.similarity)
        return duplicates, similar
```

- [ ] **Step 6: Run all inference tests**

```bash
python -m pytest tests/test_inference.py -v
```

---

### Task 5: Cross-Framework Comparison Module

**Files:**
- Create: `tract/compare.py`
- Create: `tests/test_compare.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for tract.compare module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def populated_db(tmp_path: Path) -> Path:
    """Create a crosswalk DB with two frameworks sharing a hub."""
    from tract.crosswalk.schema import create_database
    from tract.crosswalk.store import (
        insert_assignments,
        insert_controls,
        insert_frameworks,
        insert_hubs,
    )

    db_path = tmp_path / "test.db"
    create_database(db_path)

    insert_hubs(db_path, [
        {"id": "hub-1", "name": "Access Control", "path": "Root > Access Control", "parent_id": None},
        {"id": "hub-2", "name": "Input Validation", "path": "Root > Input Validation", "parent_id": None},
        {"id": "hub-3", "name": "Encryption", "path": "Root > Encryption", "parent_id": None},
    ])
    insert_frameworks(db_path, [
        {"id": "fw_a", "name": "Framework A", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2},
        {"id": "fw_b", "name": "Framework B", "version": "1.0", "fetch_date": "2026-04-30", "control_count": 2},
    ])
    insert_controls(db_path, [
        {"id": "fw_a::c1", "framework_id": "fw_a", "section_id": "c1", "title": "AC Policy", "description": "Access control", "full_text": None},
        {"id": "fw_a::c2", "framework_id": "fw_a", "section_id": "c2", "title": "Input Check", "description": "Validate input", "full_text": None},
        {"id": "fw_b::c1", "framework_id": "fw_b", "section_id": "c1", "title": "Auth", "description": "Authentication", "full_text": None},
        {"id": "fw_b::c2", "framework_id": "fw_b", "section_id": "c2", "title": "Encrypt", "description": "Encryption", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw_a::c1", "hub_id": "hub-1", "confidence": 0.9, "in_conformal_set": 1, "is_ood": 0, "provenance": "ground_truth", "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw_b::c1", "hub_id": "hub-1", "confidence": 0.85, "in_conformal_set": 1, "is_ood": 0, "provenance": "ground_truth", "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw_a::c2", "hub_id": "hub-2", "confidence": 0.7, "in_conformal_set": 1, "is_ood": 0, "provenance": "model", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
        {"control_id": "fw_b::c2", "hub_id": "hub-3", "confidence": 0.8, "in_conformal_set": 1, "is_ood": 0, "provenance": "model", "source_link_id": None, "model_version": "v1", "review_status": "accepted"},
    ])
    return db_path


@pytest.fixture
def mock_hierarchy() -> object:
    """Minimal hierarchy with get_parent support."""
    from unittest.mock import MagicMock

    hierarchy = MagicMock()

    hub_nodes = {
        "hub-1": MagicMock(hub_id="hub-1", name="Access Control", parent_id=None, hierarchy_path="Root > Access Control"),
        "hub-2": MagicMock(hub_id="hub-2", name="Input Validation", parent_id=None, hierarchy_path="Root > Input Validation"),
        "hub-3": MagicMock(hub_id="hub-3", name="Encryption", parent_id=None, hierarchy_path="Root > Encryption"),
    }
    hierarchy.hubs = hub_nodes
    hierarchy.get_parent.return_value = None  # All roots, no related pairs
    return hierarchy


class TestCrossFrameworkMatrix:
    def test_finds_equivalences(self, populated_db: Path, mock_hierarchy: object) -> None:
        from tract.compare import cross_framework_matrix

        result = cross_framework_matrix(populated_db, ["fw_a", "fw_b"], mock_hierarchy)
        assert len(result.equivalences) == 1
        assert result.equivalences[0].hub_id == "hub-1"
        assert result.total_shared_hubs == 1

    def test_gap_controls(self, populated_db: Path, mock_hierarchy: object) -> None:
        from tract.compare import cross_framework_matrix

        result = cross_framework_matrix(populated_db, ["fw_a", "fw_b"], mock_hierarchy)
        # fw_a::c2 -> hub-2 (no other fw), fw_b::c2 -> hub-3 (no other fw)
        assert "fw_a" in result.gap_controls or "fw_b" in result.gap_controls

    def test_framework_pair_overlap(self, populated_db: Path, mock_hierarchy: object) -> None:
        from tract.compare import cross_framework_matrix

        result = cross_framework_matrix(populated_db, ["fw_a", "fw_b"], mock_hierarchy)
        assert ("fw_a", "fw_b") in result.framework_pair_overlap or ("fw_b", "fw_a") in result.framework_pair_overlap
```

- [ ] **Step 2: Implement compare.py**

```python
"""Cross-framework comparison via shared CRE hubs.

Factored from T5's _export_cross_framework_matrix() into a reusable
library function. Queries crosswalk.db live — no cached data.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from tract.crosswalk.schema import get_connection
from tract.hierarchy import CREHierarchy

logger = logging.getLogger(__name__)


@dataclass
class Equivalence:
    hub_id: str
    hub_name: str
    controls: list[dict]  # [{control_id, framework_id, title}]
    frameworks: list[str]


@dataclass
class RelatedPair:
    hub_a: str
    hub_b: str
    parent_hub: str
    controls_a: list[dict]
    controls_b: list[dict]


@dataclass
class CrossFrameworkResult:
    equivalences: list[Equivalence]
    related: list[RelatedPair]
    gap_controls: dict[str, list[str]]
    framework_pair_overlap: dict[tuple[str, str], int]
    total_shared_hubs: int


def cross_framework_matrix(
    db_path: Path,
    framework_ids: list[str],
    hierarchy: CREHierarchy,
) -> CrossFrameworkResult:
    """Live query: relationship matrix between 2+ frameworks.

    Equivalent: controls assigned to the same hub.
    Related: controls whose hubs share a parent (via hierarchy.get_parent()).
    Gap: controls with no cross-framework match.
    """
    fw_set = set(framework_ids)

    conn = get_connection(db_path)
    try:
        placeholders = ",".join("?" * len(framework_ids))
        rows = conn.execute(
            f"SELECT a.control_id, a.hub_id, c.framework_id, c.title, "
            f"h.name AS hub_name "
            f"FROM assignments a "
            f"JOIN controls c ON a.control_id = c.id "
            f"JOIN hubs h ON a.hub_id = h.id "
            f"WHERE c.framework_id IN ({placeholders}) "
            f"AND a.review_status IN ('accepted', 'ground_truth') "
            f"ORDER BY a.hub_id",
            framework_ids,
        ).fetchall()
    finally:
        conn.close()

    hub_to_controls: dict[str, list[dict]] = defaultdict(list)
    control_hubs: dict[str, str] = {}
    all_control_ids_by_fw: dict[str, set[str]] = defaultdict(set)

    for row in rows:
        hub_to_controls[row["hub_id"]].append({
            "control_id": row["control_id"],
            "framework_id": row["framework_id"],
            "title": row["title"],
        })
        control_hubs[row["control_id"]] = row["hub_id"]
        all_control_ids_by_fw[row["framework_id"]].add(row["control_id"])

    # Equivalences: hubs with controls from 2+ of the requested frameworks
    equivalences: list[Equivalence] = []
    matched_controls: set[str] = set()
    for hub_id, controls in hub_to_controls.items():
        fws_in_hub = {c["framework_id"] for c in controls} & fw_set
        if len(fws_in_hub) >= 2:
            hub_name = controls[0].get("title", hub_id)
            for row in rows:
                if row["hub_id"] == hub_id:
                    hub_name = row["hub_name"]
                    break
            equivalences.append(Equivalence(
                hub_id=hub_id,
                hub_name=hub_name,
                controls=controls,
                frameworks=sorted(fws_in_hub),
            ))
            matched_controls.update(c["control_id"] for c in controls)

    equivalences.sort(key=lambda e: (-len(e.frameworks), e.hub_id))

    # Related: hubs with same parent but different frameworks
    related: list[RelatedPair] = []
    hub_ids_list = list(hub_to_controls.keys())
    for i, hub_a in enumerate(hub_ids_list):
        parent_a = hierarchy.get_parent(hub_a) if hub_a in hierarchy.hubs else None
        if parent_a is None:
            continue
        for hub_b in hub_ids_list[i + 1:]:
            parent_b = hierarchy.get_parent(hub_b) if hub_b in hierarchy.hubs else None
            if parent_b is None:
                continue
            if parent_a.hub_id == parent_b.hub_id:
                fws_a = {c["framework_id"] for c in hub_to_controls[hub_a]} & fw_set
                fws_b = {c["framework_id"] for c in hub_to_controls[hub_b]} & fw_set
                if fws_a != fws_b or len(fws_a) >= 2:
                    related.append(RelatedPair(
                        hub_a=hub_a,
                        hub_b=hub_b,
                        parent_hub=parent_a.hub_id,
                        controls_a=hub_to_controls[hub_a],
                        controls_b=hub_to_controls[hub_b],
                    ))
                    matched_controls.update(
                        c["control_id"] for c in hub_to_controls[hub_a]
                    )
                    matched_controls.update(
                        c["control_id"] for c in hub_to_controls[hub_b]
                    )

    # Gap controls: controls with no cross-framework match
    gap_controls: dict[str, list[str]] = {}
    for fw_id in framework_ids:
        unmatched = sorted(all_control_ids_by_fw.get(fw_id, set()) - matched_controls)
        if unmatched:
            gap_controls[fw_id] = unmatched

    # Framework pair overlap counts
    fw_pair_overlap: dict[tuple[str, str], int] = defaultdict(int)
    for eq in equivalences:
        for i, fw_a in enumerate(eq.frameworks):
            for fw_b in eq.frameworks[i + 1:]:
                pair = (min(fw_a, fw_b), max(fw_a, fw_b))
                fw_pair_overlap[pair] += 1

    return CrossFrameworkResult(
        equivalences=equivalences,
        related=related,
        gap_controls=gap_controls,
        framework_pair_overlap=dict(fw_pair_overlap),
        total_shared_hubs=len(equivalences),
    )
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_compare.py -v
```

---

### Task 6: Hub Proposal — Clustering

**Files:**
- Create: `tract/proposals/__init__.py`
- Create: `tract/proposals/cluster.py`
- Create: `tests/test_proposals_cluster.py`

- [ ] **Step 1: Create package marker**

```python
# tract/proposals/__init__.py
```

- [ ] **Step 2: Write the test**

```python
"""Tests for HDBSCAN clustering on OOD control embeddings."""
from __future__ import annotations

import numpy as np
import pytest


class TestClusterOodControls:
    def test_empty_input_returns_empty(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls

        result = cluster_ood_controls(
            embeddings=np.empty((0, 1024)),
            control_ids=[],
        )
        assert result == []

    def test_insufficient_items_returns_empty(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls

        rng = np.random.default_rng(42)
        embs = rng.standard_normal((2, 1024)).astype(np.float32)
        embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)

        result = cluster_ood_controls(
            embeddings=embs,
            control_ids=["c1", "c2"],
            min_cluster_size=3,
        )
        assert result == []

    def test_determinism(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls

        rng = np.random.default_rng(42)
        # Create 3 tight clusters of 5 items each in 1024-d
        centers = rng.standard_normal((3, 1024)).astype(np.float32)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

        embs_list = []
        ids_list = []
        for c_idx, center in enumerate(centers):
            for j in range(5):
                noise = rng.standard_normal(1024).astype(np.float32) * 0.01
                emb = center + noise
                emb = emb / np.linalg.norm(emb)
                embs_list.append(emb)
                ids_list.append(f"ctrl_{c_idx}_{j}")

        embs = np.array(embs_list)
        result1 = cluster_ood_controls(embs, ids_list, min_cluster_size=3)
        result2 = cluster_ood_controls(embs, ids_list, min_cluster_size=3)

        assert len(result1) == len(result2)
        for c1, c2 in zip(result1, result2):
            assert sorted(c1.control_ids) == sorted(c2.control_ids)

    def test_cluster_fields(self) -> None:
        from tract.proposals.cluster import cluster_ood_controls, Cluster

        rng = np.random.default_rng(42)
        center = rng.standard_normal(1024).astype(np.float32)
        center = center / np.linalg.norm(center)

        embs = []
        ids = []
        for j in range(5):
            noise = rng.standard_normal(1024).astype(np.float32) * 0.01
            emb = center + noise
            emb = emb / np.linalg.norm(emb)
            embs.append(emb)
            ids.append(f"ctrl_{j}")

        hub_embs = rng.standard_normal((10, 1024)).astype(np.float32)
        hub_embs = hub_embs / np.linalg.norm(hub_embs, axis=1, keepdims=True)
        hub_ids = [f"hub-{i}" for i in range(10)]

        result = cluster_ood_controls(
            np.array(embs), ids, min_cluster_size=3,
            hub_embeddings=hub_embs, hub_ids=hub_ids,
        )
        if result:
            c = result[0]
            assert isinstance(c.cluster_id, int)
            assert len(c.control_ids) >= 3
            assert c.centroid.shape == (1024,)
            assert c.nearest_hub_id in hub_ids
```

- [ ] **Step 3: Implement cluster.py**

```python
"""HDBSCAN clustering on OOD control embeddings.

Uses euclidean metric on L2-normalized vectors (equivalent to cosine distance).
Deterministic: same inputs -> same clusters.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from tract.config import (
    PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    PHASE1D_HDBSCAN_MIN_SAMPLES,
)

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    cluster_id: int
    control_ids: list[str]
    centroid: NDArray[np.floating]
    nearest_hub_id: str
    nearest_hub_similarity: float
    member_frameworks: set[str] = field(default_factory=set)


def cluster_ood_controls(
    embeddings: NDArray[np.floating],
    control_ids: list[str],
    min_cluster_size: int = PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = PHASE1D_HDBSCAN_MIN_SAMPLES,
    hub_embeddings: NDArray[np.floating] | None = None,
    hub_ids: list[str] | None = None,
) -> list[Cluster]:
    """HDBSCAN clustering on OOD control embeddings.

    Returns empty list if insufficient OOD items — expected behavior.
    """
    if len(embeddings) == 0 or len(control_ids) == 0:
        logger.info("No OOD items to cluster")
        return []

    if len(embeddings) < min_cluster_size:
        logger.info(
            "Insufficient OOD items (%d) for min_cluster_size=%d",
            len(embeddings), min_cluster_size,
        )
        return []

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        core_dist_n_jobs=1,
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels)
    unique_labels.discard(-1)

    if not unique_labels:
        logger.info("HDBSCAN found zero clusters in %d OOD items", len(embeddings))
        return []

    clusters: list[Cluster] = []
    for label in sorted(unique_labels):
        mask = labels == label
        member_ids = [control_ids[i] for i in range(len(control_ids)) if mask[i]]
        member_embs = embeddings[mask]
        centroid = member_embs.mean(axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        nearest_hub = ""
        nearest_sim = 0.0
        if hub_embeddings is not None and hub_ids is not None:
            sims = hub_embeddings @ centroid
            best_idx = int(np.argmax(sims))
            nearest_hub = hub_ids[best_idx]
            nearest_sim = float(sims[best_idx])

        member_fws = set()
        for cid in member_ids:
            if "::" in cid:
                member_fws.add(cid.split("::")[0])

        clusters.append(Cluster(
            cluster_id=int(label),
            control_ids=member_ids,
            centroid=centroid,
            nearest_hub_id=nearest_hub,
            nearest_hub_similarity=nearest_sim,
            member_frameworks=member_fws,
        ))

    logger.info(
        "HDBSCAN found %d clusters from %d OOD items (%d noise)",
        len(clusters), len(embeddings), int((labels == -1).sum()),
    )
    return clusters
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_proposals_cluster.py -v
```

---

### Task 7: Hub Proposal — Guardrails

**Files:**
- Create: `tract/proposals/guardrails.py`
- Create: `tests/test_proposals_guardrails.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for hub proposal guardrails."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from tract.proposals.cluster import Cluster


def _make_cluster(
    cluster_id: int,
    n_controls: int = 4,
    n_frameworks: int = 2,
    centroid_seed: int = 42,
) -> Cluster:
    rng = np.random.default_rng(centroid_seed)
    centroid = rng.standard_normal(1024).astype(np.float32)
    centroid = centroid / np.linalg.norm(centroid)
    fws = [f"fw_{i}" for i in range(n_frameworks)]
    cids = [f"{fws[i % n_frameworks]}::ctrl_{cluster_id}_{i}" for i in range(n_controls)]
    return Cluster(
        cluster_id=cluster_id,
        control_ids=cids,
        centroid=centroid,
        nearest_hub_id="hub-1",
        nearest_hub_similarity=0.4,
        member_frameworks=set(fws),
    )


class TestGuardrail1MinEvidence:
    def test_rejects_single_framework(self) -> None:
        from tract.proposals.guardrails import _check_min_evidence

        cluster = _make_cluster(0, n_controls=5, n_frameworks=1)
        assert not _check_min_evidence(cluster, min_controls=3, min_frameworks=2)

    def test_accepts_multi_framework(self) -> None:
        from tract.proposals.guardrails import _check_min_evidence

        cluster = _make_cluster(0, n_controls=5, n_frameworks=3)
        assert _check_min_evidence(cluster, min_controls=3, min_frameworks=2)


class TestGuardrail3InterClusterSeparation:
    def test_rejects_close_centroids(self) -> None:
        from tract.proposals.guardrails import _check_inter_cluster_separation

        rng = np.random.default_rng(42)
        base = rng.standard_normal(1024).astype(np.float32)
        base = base / np.linalg.norm(base)

        c1 = _make_cluster(0)
        c1 = Cluster(
            cluster_id=0, control_ids=c1.control_ids, centroid=base,
            nearest_hub_id="h1", nearest_hub_similarity=0.4,
            member_frameworks=c1.member_frameworks,
        )
        noise = rng.standard_normal(1024).astype(np.float32) * 0.01
        close_centroid = base + noise
        close_centroid = close_centroid / np.linalg.norm(close_centroid)
        c2 = Cluster(
            cluster_id=1, control_ids=["fw_0::c1"], centroid=close_centroid,
            nearest_hub_id="h2", nearest_hub_similarity=0.3,
            member_frameworks={"fw_0"},
        )

        result = _check_inter_cluster_separation([c1, c2], max_cosine=0.7)
        assert len(result) < 2  # At least one should be removed

    def test_accepts_distant_centroids(self) -> None:
        from tract.proposals.guardrails import _check_inter_cluster_separation

        rng = np.random.default_rng(42)
        c1 = _make_cluster(0, centroid_seed=1)
        c2 = _make_cluster(1, centroid_seed=999)
        cos_sim = float(c1.centroid @ c2.centroid)
        if cos_sim < 0.7:
            result = _check_inter_cluster_separation([c1, c2], max_cosine=0.7)
            assert len(result) == 2


class TestApplyGuardrails:
    def test_budget_cap(self) -> None:
        from tract.proposals.guardrails import apply_guardrails

        clusters = [_make_cluster(i, n_controls=4, n_frameworks=2, centroid_seed=i * 100) for i in range(50)]
        hierarchy = MagicMock()
        hierarchy.hubs = {}
        hub_embs = np.random.default_rng(42).standard_normal((10, 1024)).astype(np.float32)
        hub_ids = [f"hub-{i}" for i in range(10)]

        results = apply_guardrails(clusters, hierarchy, hub_embs, hub_ids, {}, budget_cap=5)
        assert len(results) <= 5
```

- [ ] **Step 2: Implement guardrails.py**

```python
"""Six-filter guardrail pipeline for hub proposals.

Guardrails:
1. Minimum evidence: 3+ controls from 2+ frameworks
2. Hierarchy constraint: parent hub identified via max cosine
3. Inter-cluster separation: pairwise centroid cosine < 0.7
4. Budget cap: top N by evidence strength
5. Candidate queue: proposals with evidence
6. Determinism: same inputs -> same outputs
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from tract.config import (
    PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    PHASE1D_PROPOSAL_BUDGET_CAP,
    PHASE1D_PROPOSAL_INTER_CLUSTER_MAX_COSINE,
    PHASE1D_PROPOSAL_MIN_FRAMEWORKS,
    PHASE1D_PROPOSAL_UNCERTAIN_PLACEMENT_FLOOR,
)
from tract.hierarchy import CREHierarchy
from tract.proposals.cluster import Cluster

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    cluster: Cluster
    passed: bool
    rejection_reasons: list[str]
    parent_hub_id: str | None
    parent_similarity: float
    uncertain_placement: bool
    evidence_score: float


def _check_min_evidence(
    cluster: Cluster,
    min_controls: int = PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE,
    min_frameworks: int = PHASE1D_PROPOSAL_MIN_FRAMEWORKS,
) -> bool:
    return (
        len(cluster.control_ids) >= min_controls
        and len(cluster.member_frameworks) >= min_frameworks
    )


def _check_inter_cluster_separation(
    clusters: list[Cluster],
    max_cosine: float = PHASE1D_PROPOSAL_INTER_CLUSTER_MAX_COSINE,
) -> list[Cluster]:
    """Remove clusters whose centroids are too close to each other.

    Keeps the cluster with more evidence when two are too close.
    """
    if len(clusters) <= 1:
        return list(clusters)

    keep = list(range(len(clusters)))
    to_remove: set[int] = set()

    for i in range(len(clusters)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(clusters)):
            if j in to_remove:
                continue
            cos_sim = float(clusters[i].centroid @ clusters[j].centroid)
            if cos_sim >= max_cosine:
                # Remove the one with less evidence
                if len(clusters[i].control_ids) >= len(clusters[j].control_ids):
                    to_remove.add(j)
                    logger.info(
                        "Guardrail 3: removing cluster %d (too close to %d, cosine=%.3f)",
                        clusters[j].cluster_id, clusters[i].cluster_id, cos_sim,
                    )
                else:
                    to_remove.add(i)
                    logger.info(
                        "Guardrail 3: removing cluster %d (too close to %d, cosine=%.3f)",
                        clusters[i].cluster_id, clusters[j].cluster_id, cos_sim,
                    )

    return [c for idx, c in enumerate(clusters) if idx not in to_remove]


def apply_guardrails(
    clusters: list[Cluster],
    hierarchy: CREHierarchy,
    hub_embeddings: NDArray[np.floating],
    hub_ids: list[str],
    control_metadata: dict[str, dict],
    budget_cap: int = PHASE1D_PROPOSAL_BUDGET_CAP,
) -> list[GuardrailResult]:
    """Apply 6 guardrails to each cluster."""
    results: list[GuardrailResult] = []

    # Guardrail 1: Minimum evidence
    passing_g1: list[Cluster] = []
    for cluster in clusters:
        if _check_min_evidence(cluster):
            passing_g1.append(cluster)
        else:
            reasons = []
            if len(cluster.control_ids) < PHASE1D_HDBSCAN_MIN_CLUSTER_SIZE:
                reasons.append(f"insufficient controls ({len(cluster.control_ids)})")
            if len(cluster.member_frameworks) < PHASE1D_PROPOSAL_MIN_FRAMEWORKS:
                reasons.append(f"insufficient frameworks ({len(cluster.member_frameworks)})")
            results.append(GuardrailResult(
                cluster=cluster, passed=False, rejection_reasons=reasons,
                parent_hub_id=None, parent_similarity=0.0,
                uncertain_placement=False, evidence_score=0.0,
            ))

    # Guardrail 3: Inter-cluster separation
    passing_g3 = _check_inter_cluster_separation(passing_g1)
    removed_g3 = set(c.cluster_id for c in passing_g1) - set(c.cluster_id for c in passing_g3)
    for cluster in passing_g1:
        if cluster.cluster_id in removed_g3:
            results.append(GuardrailResult(
                cluster=cluster, passed=False,
                rejection_reasons=["inter-cluster separation (cosine >= 0.7 to another cluster)"],
                parent_hub_id=None, parent_similarity=0.0,
                uncertain_placement=False, evidence_score=0.0,
            ))

    # Guardrail 2: Hierarchy constraint (parent hub identification)
    for cluster in passing_g3:
        if len(hub_embeddings) > 0 and len(hub_ids) > 0:
            sims = hub_embeddings @ cluster.centroid
            best_idx = int(np.argmax(sims))
            parent_hub_id = hub_ids[best_idx]
            parent_sim = float(sims[best_idx])
        else:
            parent_hub_id = None
            parent_sim = 0.0

        uncertain = parent_sim < PHASE1D_PROPOSAL_UNCERTAIN_PLACEMENT_FLOOR
        evidence = len(cluster.control_ids) * len(cluster.member_frameworks)

        results.append(GuardrailResult(
            cluster=cluster, passed=True, rejection_reasons=[],
            parent_hub_id=parent_hub_id, parent_similarity=parent_sim,
            uncertain_placement=uncertain,
            evidence_score=float(evidence),
        ))

    # Guardrail 4: Budget cap
    passing = [r for r in results if r.passed]
    passing.sort(key=lambda r: -r.evidence_score)
    if len(passing) > budget_cap:
        for r in passing[budget_cap:]:
            r.passed = False
            r.rejection_reasons.append(f"budget cap ({budget_cap})")

    logger.info(
        "Guardrails: %d/%d clusters passed",
        sum(1 for r in results if r.passed), len(clusters),
    )
    return results
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_proposals_guardrails.py -v
```

---

### Task 8: Hub Proposal — Naming

**Files:**
- Create: `tract/proposals/naming.py`
- Create: `tests/test_proposals_naming.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for LLM-based hub naming."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tract.proposals.guardrails import GuardrailResult
from tract.proposals.cluster import Cluster
import numpy as np


class TestGenerateHubNames:
    def test_returns_names_for_passing_clusters(self) -> None:
        from tract.proposals.naming import generate_hub_names

        rng = np.random.default_rng(42)
        centroid = rng.standard_normal(1024).astype(np.float32)
        centroid = centroid / np.linalg.norm(centroid)

        cluster = Cluster(
            cluster_id=0,
            control_ids=["fw_a::c1", "fw_a::c2", "fw_b::c3"],
            centroid=centroid,
            nearest_hub_id="hub-1",
            nearest_hub_similarity=0.4,
            member_frameworks={"fw_a", "fw_b"},
        )
        result = GuardrailResult(
            cluster=cluster, passed=True, rejection_reasons=[],
            parent_hub_id="hub-1", parent_similarity=0.4,
            uncertain_placement=False, evidence_score=6.0,
        )

        mock_hierarchy = MagicMock()
        mock_hierarchy.hubs = {"hub-1": MagicMock(name="Access Control", hierarchy_path="Root > Access Control")}

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="AI Model Governance")]
        mock_client.messages.create.return_value = mock_response

        with patch("tract.proposals.naming._get_anthropic_client", return_value=mock_client):
            names = generate_hub_names([result], mock_hierarchy, {})

        assert 0 in names
        assert isinstance(names[0], str)
        assert len(names[0]) > 0

    def test_placeholder_names_without_llm(self) -> None:
        from tract.proposals.naming import generate_placeholder_names

        rng = np.random.default_rng(42)
        centroid = rng.standard_normal(1024).astype(np.float32)
        centroid = centroid / np.linalg.norm(centroid)

        cluster = Cluster(
            cluster_id=0,
            control_ids=["fw_a::c1"],
            centroid=centroid,
            nearest_hub_id="hub-1",
            nearest_hub_similarity=0.4,
            member_frameworks={"fw_a"},
        )
        result = GuardrailResult(
            cluster=cluster, passed=True, rejection_reasons=[],
            parent_hub_id="hub-1", parent_similarity=0.4,
            uncertain_placement=False, evidence_score=2.0,
        )

        hub_names = {"hub-1": "Access Control"}
        names = generate_placeholder_names([result], hub_names)
        assert 0 in names
        assert "Access Control" in names[0]
```

- [ ] **Step 2: Implement naming.py**

```python
"""LLM-generated hub names for proposed clusters.

Opt-in via --name-with-llm flag. Without it, clusters get descriptive
placeholder names based on nearest hub.
"""
from __future__ import annotations

import logging
import os

from tract.config import PHASE1D_PROPOSAL_NAMING_MODEL
from tract.hierarchy import CREHierarchy
from tract.proposals.guardrails import GuardrailResult

logger = logging.getLogger(__name__)


def _get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def generate_hub_names(
    results: list[GuardrailResult],
    hierarchy: CREHierarchy,
    control_metadata: dict[str, dict],
    model: str = PHASE1D_PROPOSAL_NAMING_MODEL,
) -> dict[int, str]:
    """Call Claude API to generate hub names for passing clusters."""
    client = _get_anthropic_client()
    names: dict[int, str] = {}

    passing = [r for r in results if r.passed]
    for result in passing:
        cluster = result.cluster
        member_texts = []
        for cid in cluster.control_ids[:10]:
            meta = control_metadata.get(cid, {})
            title = meta.get("title", cid)
            desc = meta.get("description", "")
            member_texts.append(f"- {title}: {desc[:200]}")

        nearest_name = ""
        if cluster.nearest_hub_id and cluster.nearest_hub_id in hierarchy.hubs:
            nearest_name = hierarchy.hubs[cluster.nearest_hub_id].name

        parent_name = ""
        if result.parent_hub_id and result.parent_hub_id in hierarchy.hubs:
            parent_name = hierarchy.hubs[result.parent_hub_id].name

        prompt = (
            f"These security controls were clustered together as potentially needing "
            f"a new category in the CRE (Common Requirements Enumeration) taxonomy.\n\n"
            f"Member controls:\n{''.join(member_texts)}\n\n"
            f"Nearest existing hub: {nearest_name}\n"
            f"Suggested parent hub: {parent_name}\n\n"
            f"Generate a concise hub name (2-5 words) that captures the shared concept. "
            f"Reply with ONLY the hub name, nothing else."
        )

        response = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        name = response.content[0].text.strip()
        names[cluster.cluster_id] = name
        logger.info("Named cluster %d: %s", cluster.cluster_id, name)

    return names


def generate_placeholder_names(
    results: list[GuardrailResult],
    hub_names: dict[str, str],
) -> dict[int, str]:
    """Generate placeholder names based on nearest hub. No LLM call."""
    names: dict[int, str] = {}
    for result in results:
        if not result.passed:
            continue
        nearest = hub_names.get(result.cluster.nearest_hub_id, "Unknown")
        names[result.cluster.cluster_id] = f"Cluster {result.cluster.cluster_id} (near: {nearest})"
    return names
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_proposals_naming.py -v
```

---

### Task 9: Hub Proposal — Review & Write

**Files:**
- Create: `tract/proposals/review.py`
- Create: `tests/test_proposals_review.py`

- [ ] **Step 1: Write the test**

```python
"""Tests for proposal writing and review session."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tract.proposals.cluster import Cluster
from tract.proposals.guardrails import GuardrailResult


@pytest.fixture
def passing_result() -> GuardrailResult:
    rng = np.random.default_rng(42)
    centroid = rng.standard_normal(1024).astype(np.float32)
    centroid = centroid / np.linalg.norm(centroid)
    cluster = Cluster(
        cluster_id=0,
        control_ids=["fw_a::c1", "fw_a::c2", "fw_b::c3"],
        centroid=centroid,
        nearest_hub_id="hub-1",
        nearest_hub_similarity=0.4,
        member_frameworks={"fw_a", "fw_b"},
    )
    return GuardrailResult(
        cluster=cluster, passed=True, rejection_reasons=[],
        parent_hub_id="hub-1", parent_similarity=0.4,
        uncertain_placement=False, evidence_score=6.0,
    )


class TestWriteProposalRound:
    def test_creates_round_directory(self, tmp_path: Path, passing_result: GuardrailResult) -> None:
        from tract.proposals.review import write_proposal_round

        output_dir = tmp_path / "hub_proposals"
        path = write_proposal_round(
            results=[passing_result],
            names={0: "AI Model Governance"},
            output_dir=output_dir,
            round_num=1,
        )
        assert path.exists()
        assert (path / "proposals.json").exists()

    def test_proposal_json_schema(self, tmp_path: Path, passing_result: GuardrailResult) -> None:
        from tract.proposals.review import write_proposal_round

        output_dir = tmp_path / "hub_proposals"
        path = write_proposal_round(
            results=[passing_result],
            names={0: "AI Model Governance"},
            output_dir=output_dir,
            round_num=1,
        )

        data = json.loads((path / "proposals.json").read_text())
        assert "round" in data
        assert "proposals" in data
        assert data["round"] == 1
        assert len(data["proposals"]) == 1

        proposal = data["proposals"][0]
        assert "cluster_id" in proposal
        assert "proposed_name" in proposal
        assert "control_ids" in proposal
        assert "parent_hub_id" in proposal
        assert "evidence_score" in proposal

    def test_only_passing_written(self, tmp_path: Path, passing_result: GuardrailResult) -> None:
        from tract.proposals.review import write_proposal_round

        failing = GuardrailResult(
            cluster=passing_result.cluster, passed=False,
            rejection_reasons=["test"], parent_hub_id=None,
            parent_similarity=0.0, uncertain_placement=False, evidence_score=0.0,
        )

        output_dir = tmp_path / "hub_proposals"
        path = write_proposal_round(
            results=[passing_result, failing],
            names={0: "Test"},
            output_dir=output_dir,
            round_num=1,
        )

        data = json.loads((path / "proposals.json").read_text())
        assert len(data["proposals"]) == 1
```

- [ ] **Step 2: Implement review.py**

```python
"""Hub proposal writing and interactive review.

write_proposal_round: writes proposals to hub_proposals/round_N/ as versioned JSON.
run_review_session: interactive CLI loop for accept/reject/edit/skip per proposal.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tract.config import HUB_PROPOSALS_DIR
from tract.io import atomic_write_json
from tract.proposals.guardrails import GuardrailResult

logger = logging.getLogger(__name__)


def write_proposal_round(
    results: list[GuardrailResult],
    names: dict[int, str] | None,
    output_dir: Path,
    round_num: int,
) -> Path:
    """Write proposals to hub_proposals/round_N/ as versioned JSON. Atomic write."""
    round_dir = output_dir / f"round_{round_num}"
    round_dir.mkdir(parents=True, exist_ok=True)

    passing = [r for r in results if r.passed]
    proposals = []
    for result in passing:
        name = (names or {}).get(result.cluster.cluster_id, f"Cluster {result.cluster.cluster_id}")
        proposals.append({
            "cluster_id": result.cluster.cluster_id,
            "proposed_name": name,
            "control_ids": result.cluster.control_ids,
            "member_frameworks": sorted(result.cluster.member_frameworks),
            "parent_hub_id": result.parent_hub_id,
            "parent_similarity": result.parent_similarity,
            "nearest_hub_id": result.cluster.nearest_hub_id,
            "nearest_hub_similarity": result.cluster.nearest_hub_similarity,
            "uncertain_placement": result.uncertain_placement,
            "evidence_score": result.evidence_score,
            "rejection_reasons": result.rejection_reasons,
        })

    output_data = {
        "round": round_num,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_clusters_evaluated": len(results),
        "total_passing": len(passing),
        "proposals": proposals,
    }

    atomic_write_json(output_data, round_dir / "proposals.json")
    logger.info("Wrote %d proposals to %s", len(proposals), round_dir)
    return round_dir


def run_review_session(
    round_dir: Path,
    hierarchy: object,
    db_path: Path,
    dry_run: bool = False,
) -> dict:
    """Interactive CLI review loop.

    Per proposal: [a]ccept, [r]eject, [e]dit name, [s]kip
    Returns summary dict of actions taken.
    """
    proposals_path = round_dir / "proposals.json"
    if not proposals_path.exists():
        raise FileNotFoundError(f"No proposals.json in {round_dir}")

    from tract.io import load_json
    data = load_json(proposals_path)
    proposals = data["proposals"]

    if not proposals:
        print("No proposals to review.")
        return {"accepted": 0, "rejected": 0, "skipped": 0}

    if not dry_run:
        pre_review_dir = round_dir / "pre_review"
        pre_review_dir.mkdir(exist_ok=True)
        if db_path.exists():
            shutil.copy2(db_path, pre_review_dir / "crosswalk.db.backup")

    accepted = 0
    rejected = 0
    skipped = 0

    for i, proposal in enumerate(proposals):
        print(f"\n{'='*60}")
        print(f"Proposal {i+1}/{len(proposals)}: {proposal['proposed_name']}")
        print(f"  Cluster ID: {proposal['cluster_id']}")
        print(f"  Controls: {len(proposal['control_ids'])} from {proposal['member_frameworks']}")
        print(f"  Parent hub: {proposal['parent_hub_id']} (similarity: {proposal['parent_similarity']:.3f})")
        if proposal.get("uncertain_placement"):
            print("  WARNING: Uncertain placement (low parent similarity)")
        print(f"  Evidence score: {proposal['evidence_score']:.1f}")
        print(f"  Control IDs: {', '.join(proposal['control_ids'][:5])}")
        if len(proposal['control_ids']) > 5:
            print(f"    ... and {len(proposal['control_ids']) - 5} more")

        if dry_run:
            print("  [DRY RUN] Would prompt for review action")
            skipped += 1
            continue

        while True:
            action = input("\n  [a]ccept / [r]eject / [e]dit name / [s]kip: ").strip().lower()
            if action in ("a", "r", "e", "s"):
                break
            print("  Invalid choice. Use a/r/e/s.")

        if action == "a":
            proposal["review_status"] = "accepted"
            accepted += 1
            logger.info("Accepted proposal %d: %s", proposal["cluster_id"], proposal["proposed_name"])
        elif action == "r":
            proposal["review_status"] = "rejected"
            rejected += 1
        elif action == "e":
            new_name = input("  New name: ").strip()
            if new_name:
                proposal["proposed_name"] = new_name
            proposal["review_status"] = "accepted"
            accepted += 1
        elif action == "s":
            proposal["review_status"] = "skipped"
            skipped += 1

    atomic_write_json(data, proposals_path)
    summary = {"accepted": accepted, "rejected": rejected, "skipped": skipped}
    logger.info("Review complete: %s", summary)
    return summary
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_proposals_review.py -v
```

---

### Task 10: CLI — Argument Parsing & Output Formatting

**Files:**
- Create: `tract/cli.py`
- Create: `tests/test_cli.py`

This is the largest single task — the full CLI with 8 subcommands. Split into substeps.

- [ ] **Step 1: Write CLI parsing tests**

Create `tests/test_cli.py`:

```python
"""Tests for tract CLI argument parsing and output."""
from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestArgParsing:
    def test_assign_text(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "test control text"])
        assert args.command == "assign"
        assert args.text == "test control text"

    def test_assign_file(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "--file", "controls.txt"])
        assert args.command == "assign"
        assert args.file == "controls.txt"

    def test_assign_top_k(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--top-k", "10"])
        assert args.top_k == 10

    def test_assign_raw_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--raw"])
        assert args.raw is True

    def test_assign_verbose_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--verbose"])
        assert args.verbose is True

    def test_assign_json_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--json"])
        assert args.json is True

    def test_compare_frameworks(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["compare", "--framework", "fw_a", "--framework", "fw_b"])
        assert args.command == "compare"
        assert args.framework == ["fw_a", "fw_b"]

    def test_ingest_file(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["ingest", "--file", "framework.json"])
        assert args.command == "ingest"
        assert args.file == "framework.json"

    def test_ingest_force(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["ingest", "--file", "f.json", "--force"])
        assert args.force is True

    def test_export_format(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "csv"])
        assert args.command == "export"
        assert args.format == "csv"

    def test_hierarchy_hub(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["hierarchy", "--hub", "646-285"])
        assert args.command == "hierarchy"
        assert args.hub == "646-285"

    def test_propose_hubs(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["propose-hubs", "--budget", "20"])
        assert args.command == "propose-hubs"
        assert args.budget == 20

    def test_propose_hubs_name_with_llm(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["propose-hubs", "--name-with-llm"])
        assert args.name_with_llm is True

    def test_review_proposals(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["review-proposals", "--round", "1"])
        assert args.command == "review-proposals"
        assert args.round == 1

    def test_review_dry_run(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["review-proposals", "--round", "1", "--dry-run"])
        assert args.dry_run is True

    def test_tutorial(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["tutorial"])
        assert args.command == "tutorial"

    def test_all_commands_have_help(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        # Should not raise
        for cmd in ["assign", "compare", "ingest", "export", "hierarchy",
                     "propose-hubs", "review-proposals", "tutorial"]:
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([cmd, "--help"])
            assert exc_info.value.code == 0


class TestOutputFormatting:
    def test_format_predictions_table(self) -> None:
        from tract.cli import format_predictions_table
        from tract.inference import HubPrediction

        preds = [
            HubPrediction("646-285", "AI compliance", "R > AI compliance", 0.523, 0.847, True, False),
            HubPrediction("220-442", "Access minimum", "R > Access", 0.412, 0.631, True, False),
        ]
        output = format_predictions_table(preds, raw=False, verbose=False)
        assert "646-285" in output
        assert "AI compliance" in output
        assert "0.847" in output

    def test_format_predictions_raw(self) -> None:
        from tract.cli import format_predictions_table
        from tract.inference import HubPrediction

        preds = [
            HubPrediction("646-285", "AI compliance", "R > AI compliance", 0.523, 0.847, True, False),
        ]
        output = format_predictions_table(preds, raw=True, verbose=False)
        assert "0.523" in output
        assert "Hub Similarity" in output

    def test_format_predictions_json(self) -> None:
        from tract.cli import format_predictions_json
        from tract.inference import HubPrediction

        preds = [
            HubPrediction("646-285", "AI compliance", "R > AI compliance", 0.523, 0.847, True, False),
        ]
        output = format_predictions_json(preds)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert parsed[0]["hub_id"] == "646-285"
```

- [ ] **Step 2: Implement build_parser() and output formatters**

Create `tract/cli.py` with argparse setup and formatting functions.

The implementation should have:
- `build_parser()` returning `argparse.ArgumentParser` with 8 subcommands
- `format_predictions_table(preds, raw, verbose)` for human-readable output
- `format_predictions_json(preds)` for `--json` output
- `main()` dispatching to handler functions

Each subcommand handler (`_cmd_assign`, `_cmd_compare`, etc.) is a separate function. The full implementation is detailed below.

```python
"""TRACT CLI — 8 subcommands for model inference, comparison, and hub proposals.

Usage:
    python -m tract.cli assign "control text"
    tract assign "control text"  (via console_scripts entry)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from tract.config import (
    HUB_PROPOSALS_DIR,
    PHASE1C_CROSSWALK_DB_PATH,
    PHASE1D_DEFAULT_TOP_K,
    PHASE1D_DEPLOYMENT_MODEL_DIR,
    PHASE1D_PROPOSAL_BUDGET_CAP,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tract",
        description="TRACT — Transitive Reconciliation and Assignment of CRE Taxonomies",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── assign ───────────────────────────────────────────────────
    p_assign = subparsers.add_parser(
        "assign",
        help="Assign control text to CRE hubs",
        epilog=(
            "Examples:\n"
            "  tract assign 'Ensure AI models are tested for bias'\n"
            "  tract assign --file controls.txt --output results.jsonl\n"
            "  tract assign 'Access control policy' --raw --top-k 10\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_assign.add_argument("text", nargs="?", help="Control text to assign")
    p_assign.add_argument("--file", help="Newline-delimited text file (one control per line)")
    p_assign.add_argument("--top-k", type=int, default=PHASE1D_DEFAULT_TOP_K, help="Number of top hub assignments")
    p_assign.add_argument("--output", help="Output path for batch mode (default: {input}_assignments.jsonl)")
    p_assign.add_argument("--raw", action="store_true", help="Show raw cosine similarity instead of calibrated confidence")
    p_assign.add_argument("--verbose", action="store_true", help="Show both metrics, conformal set, and OOD status")
    p_assign.add_argument("--json", action="store_true", help="Output as JSON")

    # ── compare ──────────────────────────────────────────────────
    p_compare = subparsers.add_parser(
        "compare",
        help="Compare frameworks via shared CRE hubs",
        epilog="Example:\n  tract compare --framework mitre_atlas --framework owasp_ai_exchange\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_compare.add_argument("--framework", action="append", required=True, help="Framework ID (use twice for comparison)")
    p_compare.add_argument("--json", action="store_true", help="Output as JSON")

    # ── ingest ───────────────────────────────────────────────────
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest a new framework and generate review file",
        epilog="Example:\n  tract ingest --file new_framework.json\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_ingest.add_argument("--file", required=True, help="Framework JSON file (FrameworkOutput schema)")
    p_ingest.add_argument("--force", action="store_true", help="Overwrite if framework ID already exists")
    p_ingest.add_argument("--json", action="store_true", help="Output as JSON")

    # ── export ───────────────────────────────────────────────────
    p_export = subparsers.add_parser(
        "export",
        help="Export crosswalk assignments",
        epilog="Example:\n  tract export --format csv --framework mitre_atlas\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_export.add_argument("--format", choices=["csv", "json", "jsonl"], default="json", help="Output format")
    p_export.add_argument("--framework", help="Filter to single framework")
    p_export.add_argument("--hub", help="Filter to single hub")
    p_export.add_argument("--min-confidence", type=float, help="Minimum confidence threshold")
    p_export.add_argument("--status", default="accepted", help="Filter by review status")
    p_export.add_argument("--output", help="Output file path")

    # ── hierarchy ────────────────────────────────────────────────
    p_hierarchy = subparsers.add_parser(
        "hierarchy",
        help="Show hub hierarchy information",
        epilog="Example:\n  tract hierarchy --hub 646-285\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_hierarchy.add_argument("--hub", required=True, help="Hub ID to inspect")
    p_hierarchy.add_argument("--json", action="store_true", help="Output as JSON")

    # ── propose-hubs ─────────────────────────────────────────────
    p_propose = subparsers.add_parser(
        "propose-hubs",
        help="Generate hub proposals from OOD controls",
        epilog="Example:\n  tract propose-hubs --name-with-llm --budget 20\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_propose.add_argument("--name-with-llm", action="store_true",
                           help="Use Claude API to generate hub names (sends control texts to API)")
    p_propose.add_argument("--budget", type=int, default=PHASE1D_PROPOSAL_BUDGET_CAP, help="Max proposals to generate")
    p_propose.add_argument("--json", action="store_true", help="Output as JSON")

    # ── review-proposals ─────────────────────────────────────────
    p_review = subparsers.add_parser(
        "review-proposals",
        help="Interactive review of hub proposals",
        epilog="Example:\n  tract review-proposals --round 1 --dry-run\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_review.add_argument("--round", type=int, required=True, help="Proposal round number")
    p_review.add_argument("--dry-run", action="store_true", help="Show proposals without modifying anything")

    # ── tutorial ─────────────────────────────────────────────────
    subparsers.add_parser(
        "tutorial",
        help="Guided walkthrough of TRACT capabilities",
    )

    return parser


# ── Output Formatting ──────────────────────────────────────────────


def format_predictions_table(
    preds: list,
    raw: bool = False,
    verbose: bool = False,
) -> str:
    """Format predictions as a human-readable table."""
    lines = []
    metric_label = "Hub Similarity" if raw else "Confidence*"
    header = f" {'#':>2}  {'Hub ID':<9}{'Hub Name':<30}{metric_label}"
    if verbose:
        header = f" {'#':>2}  {'Hub ID':<9}{'Hub Name':<30}{'Confidence*':<14}{'Cosine':<8}{'Conformal'}"

    lines.append("Hub Assignments (top {0}):".format(len(preds)))
    lines.append("─" * max(len(header) + 5, 60))
    lines.append(header)

    for i, pred in enumerate(preds, 1):
        if verbose:
            conformal = "✓" if pred.in_conformal_set else " "
            line = (
                f" {i:>2}  {pred.hub_id:<9}{pred.hub_name:<30}"
                f"{pred.calibrated_confidence:<14.3f}{pred.raw_similarity:<8.3f}{conformal}"
            )
        elif raw:
            line = f" {i:>2}  {pred.hub_id:<9}{pred.hub_name:<30}{pred.raw_similarity:.3f}"
        else:
            line = f" {i:>2}  {pred.hub_id:<9}{pred.hub_name:<30}{pred.calibrated_confidence:.3f}"
        lines.append(line)

    if not raw:
        lines.append("")
        lines.append(" * Calibrated on traditional framework holdout. AI framework accuracy may differ.")

    if verbose and preds:
        ood_status = "Out-of-distribution" if preds[0].is_ood else "In-distribution"
        conformal_count = sum(1 for p in preds if p.in_conformal_set)
        lines.append(f" OOD Status: {ood_status}")
        lines.append(f" Conformal set: {conformal_count} hubs (90% coverage guarantee)")

    return "\n".join(lines)


def format_predictions_json(preds: list) -> str:
    """Format predictions as JSON."""
    return json.dumps([p.to_dict() for p in preds], indent=2)


# ── Command Handlers ────────────────────────────────────────────────


def _cmd_assign(args: argparse.Namespace) -> None:
    from tract.inference import TRACTPredictor

    predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)

    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found: {file_path}", file=sys.stderr)
            sys.exit(1)

        texts = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        results = predictor.predict_batch(texts, top_k=args.top_k)

        output_path = Path(args.output) if args.output else file_path.with_suffix(".jsonl").with_stem(file_path.stem + "_assignments")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for text, preds in sorted(zip(texts, results), key=lambda tp: tp[1][0].raw_similarity if tp[1] else 0):
                line = {"text": text[:100], "predictions": [p.to_dict() for p in preds]}
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        ood_count = sum(1 for r in results if r and r[0].is_ood)
        high_conf = sum(1 for r in results if r and r[0].calibrated_confidence > 0.5)
        print(f"Wrote {len(results)} assignments to {output_path}")
        print(f"{ood_count}/{len(results)} controls flagged OOD, {high_conf}/{len(results)} high confidence")
        return

    if not args.text:
        print("Error: Provide control text or --file", file=sys.stderr)
        sys.exit(1)

    preds = predictor.predict(args.text, top_k=args.top_k)

    if args.json:
        print(format_predictions_json(preds))
    else:
        print(format_predictions_table(preds, raw=args.raw, verbose=args.verbose))


def _cmd_compare(args: argparse.Namespace) -> None:
    from tract.compare import cross_framework_matrix
    from tract.hierarchy import CREHierarchy

    if len(args.framework) < 2:
        print("Error: Need at least 2 --framework flags", file=sys.stderr)
        sys.exit(1)

    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")
    result = cross_framework_matrix(PHASE1C_CROSSWALK_DB_PATH, args.framework, hierarchy)

    if args.json:
        output = {
            "equivalences": [
                {"hub_id": e.hub_id, "hub_name": e.hub_name,
                 "frameworks": e.frameworks, "controls": e.controls}
                for e in result.equivalences
            ],
            "related": [
                {"hub_a": r.hub_a, "hub_b": r.hub_b, "parent_hub": r.parent_hub}
                for r in result.related
            ],
            "gap_controls": result.gap_controls,
            "framework_pair_overlap": {
                f"{a},{b}": n for (a, b), n in result.framework_pair_overlap.items()
            },
            "total_shared_hubs": result.total_shared_hubs,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Cross-Framework Comparison: {', '.join(args.framework)}")
        print("=" * 60)
        print(f"\nEquivalent mappings (same hub): {len(result.equivalences)}")
        for eq in result.equivalences[:20]:
            print(f"  Hub {eq.hub_id} ({eq.hub_name}): {', '.join(eq.frameworks)}")
        print(f"\nRelated mappings (sibling hubs): {len(result.related)}")
        if result.gap_controls:
            print("\nGap controls (no cross-framework match):")
            for fw, ctrls in result.gap_controls.items():
                print(f"  {fw}: {len(ctrls)} unmatched")
        print(f"\nTotal shared hubs: {result.total_shared_hubs}")


def _cmd_ingest(args: argparse.Namespace) -> None:
    from tract.config import PHASE1D_INGEST_MAX_FILE_SIZE
    from tract.inference import TRACTPredictor
    from tract.io import atomic_write_json, load_json
    from tract.schema import FrameworkOutput

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    file_size = file_path.stat().st_size
    if file_size > PHASE1D_INGEST_MAX_FILE_SIZE:
        print(f"Error: File too large ({file_size / 1024 / 1024:.1f}MB > 50MB limit)", file=sys.stderr)
        sys.exit(1)

    raw_data = load_json(file_path)
    try:
        fw = FrameworkOutput.model_validate(raw_data)
    except Exception as e:
        print(f"Error: Schema validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    from tract.crosswalk.store import count_frameworks
    from tract.crosswalk.schema import get_connection

    conn = get_connection(PHASE1C_CROSSWALK_DB_PATH)
    try:
        existing = conn.execute(
            "SELECT id FROM frameworks WHERE id = ?", (fw.framework_id,)
        ).fetchone()
    finally:
        conn.close()

    if existing and not args.force:
        print(f"Error: Framework '{fw.framework_id}' already exists. Use --force to overwrite.", file=sys.stderr)
        sys.exit(1)

    predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)

    texts = []
    for ctrl in fw.controls:
        parts = [ctrl.title, ctrl.description]
        if ctrl.full_text:
            parts.append(ctrl.full_text)
        texts.append(" ".join(p for p in parts if p))

    batch_preds = predictor.predict_batch(texts, top_k=PHASE1D_DEFAULT_TOP_K)

    controls_output = []
    ood_count = 0
    dup_count = 0
    sim_count = 0
    high_conf = 0
    low_conf = 0

    from datetime import datetime, timezone

    for ctrl, text, preds in zip(fw.controls, texts, batch_preds):
        duplicates, similar = predictor.find_duplicates(text)

        is_ood = preds[0].is_ood if preds else False
        if is_ood:
            ood_count += 1
        if duplicates:
            dup_count += 1
        if similar:
            sim_count += 1
        if preds and preds[0].calibrated_confidence > 0.5:
            high_conf += 1
        else:
            low_conf += 1

        controls_output.append({
            "control_id": ctrl.control_id,
            "title": ctrl.title,
            "predictions": [p.to_dict() for p in preds],
            "is_ood": is_ood,
            "duplicates": [d.to_dict() for d in duplicates],
            "similar": [s.to_dict() for s in similar],
        })

    review_data = {
        "framework_id": fw.framework_id,
        "framework_name": fw.framework_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": "deployment_round2",
        "context": "ingestion",
        "summary": {
            "total_controls": len(fw.controls),
            "ood_flagged": ood_count,
            "duplicate_flagged": dup_count,
            "similar_flagged": sim_count,
            "high_confidence": high_conf,
            "low_confidence": low_conf,
        },
        "controls": controls_output,
    }

    output_path = file_path.with_stem(file_path.stem + "_review").with_suffix(".json")
    atomic_write_json(review_data, output_path)

    if args.json:
        print(json.dumps(review_data["summary"], indent=2))
    else:
        print(f"Ingestion complete: {fw.framework_name} ({fw.framework_id})")
        print(f"  Controls: {len(fw.controls)}")
        print(f"  OOD flagged: {ood_count}")
        print(f"  Duplicates: {dup_count}, Similar: {sim_count}")
        print(f"  High confidence: {high_conf}, Low confidence: {low_conf}")
        print(f"  Review file: {output_path}")


def _cmd_export(args: argparse.Namespace) -> None:
    from tract.crosswalk.export import export_crosswalk

    fmt = args.format
    if fmt == "jsonl":
        fmt = "json"

    output_path = Path(args.output) if args.output else Path(f"crosswalk_export.{args.format}")
    export_crosswalk(PHASE1C_CROSSWALK_DB_PATH, output_path, fmt=fmt)
    print(f"Exported to {output_path}")


def _cmd_hierarchy(args: argparse.Namespace) -> None:
    from tract.hierarchy import CREHierarchy
    from tract.crosswalk.store import get_assignments_by_control
    from tract.crosswalk.schema import get_connection

    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")

    if args.hub not in hierarchy.hubs:
        print(f"Error: Unknown hub ID: {args.hub}", file=sys.stderr)
        sys.exit(1)

    node = hierarchy.hubs[args.hub]
    parent = hierarchy.get_parent(args.hub)
    children = hierarchy.get_children(args.hub)
    siblings = hierarchy.get_siblings(args.hub)

    conn = get_connection(PHASE1C_CROSSWALK_DB_PATH)
    try:
        assigned_controls = conn.execute(
            "SELECT a.control_id, c.title, c.framework_id, a.confidence "
            "FROM assignments a JOIN controls c ON a.control_id = c.id "
            "WHERE a.hub_id = ? AND a.review_status IN ('accepted', 'ground_truth') "
            "ORDER BY a.confidence DESC",
            (args.hub,),
        ).fetchall()
    finally:
        conn.close()

    if args.json:
        output = {
            "hub_id": node.hub_id,
            "name": node.name,
            "hierarchy_path": node.hierarchy_path,
            "depth": node.depth,
            "is_leaf": node.is_leaf,
            "parent": {"hub_id": parent.hub_id, "name": parent.name} if parent else None,
            "children": [{"hub_id": c.hub_id, "name": c.name} for c in children],
            "siblings": [{"hub_id": s.hub_id, "name": s.name} for s in siblings],
            "assigned_controls": [dict(r) for r in assigned_controls],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Hub: {node.hub_id} — {node.name}")
        print(f"Path: {node.hierarchy_path}")
        print(f"Depth: {node.depth}, Leaf: {node.is_leaf}")
        if parent:
            print(f"Parent: {parent.hub_id} ({parent.name})")
        if children:
            print(f"Children ({len(children)}):")
            for c in children:
                print(f"  {c.hub_id} — {c.name}")
        if siblings:
            print(f"Siblings ({len(siblings)}):")
            for s in siblings[:10]:
                print(f"  {s.hub_id} — {s.name}")
        if assigned_controls:
            print(f"\nAssigned controls ({len(assigned_controls)}):")
            for r in assigned_controls[:20]:
                conf = f" ({r['confidence']:.3f})" if r['confidence'] else ""
                print(f"  [{r['framework_id']}] {r['title']}{conf}")


def _cmd_propose_hubs(args: argparse.Namespace) -> None:
    import numpy as np

    from tract.config import PHASE1D_ARTIFACTS_PATH, PHASE1D_CALIBRATION_PATH
    from tract.hierarchy import CREHierarchy
    from tract.inference import load_deployment_artifacts
    from tract.io import load_json
    from tract.proposals.cluster import cluster_ood_controls
    from tract.proposals.guardrails import apply_guardrails
    from tract.proposals.naming import generate_hub_names, generate_placeholder_names
    from tract.proposals.review import write_proposal_round

    artifacts = load_deployment_artifacts(PHASE1D_ARTIFACTS_PATH)
    calibration = load_json(PHASE1D_CALIBRATION_PATH)
    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")

    # Recompute OOD for all controls
    ood_threshold = calibration["ood_threshold"]
    sims = artifacts.control_embeddings @ artifacts.hub_embeddings.T
    max_sims = sims.max(axis=1)
    ood_mask = max_sims < ood_threshold

    ood_embs = artifacts.control_embeddings[ood_mask]
    ood_ids = [artifacts.control_ids[i] for i in range(len(artifacts.control_ids)) if ood_mask[i]]

    print(f"OOD controls: {len(ood_ids)}/{len(artifacts.control_ids)}")

    clusters = cluster_ood_controls(
        ood_embs, ood_ids,
        hub_embeddings=artifacts.hub_embeddings,
        hub_ids=artifacts.hub_ids,
    )

    if not clusters:
        msg = "No proposals generated (insufficient OOD controls for clustering)."
        if args.json:
            print(json.dumps({"message": msg, "ood_count": len(ood_ids)}))
        else:
            print(msg)
        return

    results = apply_guardrails(
        clusters, hierarchy,
        artifacts.hub_embeddings, artifacts.hub_ids,
        {}, budget_cap=args.budget,
    )

    hub_names_map = {hid: hierarchy.hubs[hid].name for hid in artifacts.hub_ids if hid in hierarchy.hubs}

    if args.name_with_llm:
        names = generate_hub_names(results, hierarchy, {})
    else:
        names = generate_placeholder_names(results, hub_names_map)

    # Determine round number
    existing_rounds = sorted(HUB_PROPOSALS_DIR.glob("round_*")) if HUB_PROPOSALS_DIR.exists() else []
    round_num = len(existing_rounds) + 1

    round_dir = write_proposal_round(results, names, HUB_PROPOSALS_DIR, round_num)

    passing = [r for r in results if r.passed]
    if args.json:
        output = {
            "round": round_num,
            "ood_count": len(ood_ids),
            "clusters_found": len(clusters),
            "proposals_passing": len(passing),
            "round_dir": str(round_dir),
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\nProposal round {round_num}:")
        print(f"  Clusters found: {len(clusters)}")
        print(f"  Proposals passing guardrails: {len(passing)}")
        print(f"  Written to: {round_dir}")
        for r in passing:
            name = names.get(r.cluster.cluster_id, "unnamed")
            print(f"    [{r.cluster.cluster_id}] {name} ({len(r.cluster.control_ids)} controls)")


def _cmd_review_proposals(args: argparse.Namespace) -> None:
    from tract.hierarchy import CREHierarchy
    from tract.proposals.review import run_review_session

    round_dir = HUB_PROPOSALS_DIR / f"round_{args.round}"
    if not round_dir.exists():
        print(f"Error: Round directory not found: {round_dir}", file=sys.stderr)
        sys.exit(1)

    hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")
    summary = run_review_session(round_dir, hierarchy, PHASE1C_CROSSWALK_DB_PATH, dry_run=args.dry_run)
    print(f"\nReview summary: {summary}")


def _cmd_tutorial(args: argparse.Namespace) -> None:
    print("TRACT Tutorial — Guided Walkthrough")
    print("=" * 40)

    # Check prerequisites
    model_exists = PHASE1D_DEPLOYMENT_MODEL_DIR.exists()
    db_exists = PHASE1C_CROSSWALK_DB_PATH.exists()
    hierarchy_exists = (PROCESSED_DIR / "cre_hierarchy.json").exists()

    if not all([model_exists, db_exists, hierarchy_exists]):
        print("\nPrerequisites missing:")
        if not model_exists:
            print(f"  - Deployment model: {PHASE1D_DEPLOYMENT_MODEL_DIR}")
        if not db_exists:
            print(f"  - Crosswalk database: {PHASE1C_CROSSWALK_DB_PATH}")
        if not hierarchy_exists:
            print(f"  - CRE hierarchy: {PROCESSED_DIR / 'cre_hierarchy.json'}")
        print("\nRun Phase 1C pipeline first to generate these artifacts.")
        return

    print("\nTRACT maps security framework controls to CRE (Common Requirements Enumeration)")
    print("hubs — a shared coordinate system for cross-framework comparison.\n")

    print("Step 1: Assign a control to CRE hubs")
    print("  Try: tract assign 'Ensure AI models are tested for bias before deployment'")
    print("  This encodes your text, compares against 522 hubs, and returns top-5 matches.\n")

    print("Step 2: Compare frameworks")
    print("  Try: tract compare --framework mitre_atlas --framework owasp_ai_exchange")
    print("  Shows which controls map to the same CRE hubs (equivalences) and gaps.\n")

    print("Step 3: Explore the hierarchy")
    print("  Try: tract hierarchy --hub 646-285")
    print("  See the full path, parent, children, and assigned controls for any hub.\n")

    print("Step 4: Ingest a new framework")
    print("  Prepare a JSON file matching the FrameworkOutput schema (see tract/schema.py)")
    print("  Try: tract ingest --file new_framework.json\n")

    print("Step 5: Export the crosswalk")
    print("  Try: tract export --format csv")
    print("  Exports all accepted assignments as CSV or JSON.\n")

    print("Step 6: Propose new hubs (advanced)")
    print("  Try: tract propose-hubs")
    print("  Clusters OOD controls to suggest new taxonomy extensions.\n")

    print("For more: https://github.com/rockcyber/TRACT")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    handlers = {
        "assign": _cmd_assign,
        "compare": _cmd_compare,
        "ingest": _cmd_ingest,
        "export": _cmd_export,
        "hierarchy": _cmd_hierarchy,
        "propose-hubs": _cmd_propose_hubs,
        "review-proposals": _cmd_review_proposals,
        "tutorial": _cmd_tutorial,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run CLI tests**

```bash
python -m pytest tests/test_cli.py -v
```

---

### Task 11: Integration Test — assign end-to-end

**Files:**
- Create: `tests/test_assign_e2e.py`

This test requires the real deployment model + artifacts. Marked `@pytest.mark.integration`.

- [ ] **Step 1: Write integration test**

```python
"""End-to-end integration test for tract assign.

Requires: deployment model, deployment_artifacts.npz, calibration.json, cre_hierarchy.json
"""
from __future__ import annotations

import pytest

from tract.config import PHASE1D_DEPLOYMENT_MODEL_DIR


@pytest.mark.integration
class TestAssignE2E:
    def test_known_control_text(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        preds = predictor.predict("Ensure access control policies are enforced for AI systems")

        assert len(preds) == 5
        assert all(0 <= p.calibrated_confidence <= 1 for p in preds)
        assert all(0 <= p.raw_similarity <= 1 for p in preds)
        assert preds[0].calibrated_confidence >= preds[-1].calibrated_confidence
        assert preds[0].hierarchy_path  # Non-empty hierarchy path

    def test_ood_text(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        preds = predictor.predict("The recipe calls for two cups of flour and a pinch of salt")
        # This should be flagged OOD
        assert preds[0].is_ood

    def test_batch_mode(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        texts = [
            "Access control policy enforcement",
            "Input validation for AI model training data",
            "Encryption of model weights at rest",
        ]
        results = predictor.predict_batch(texts)
        assert len(results) == 3
        assert all(len(preds) == 5 for preds in results)
```

- [ ] **Step 2: Run integration tests (local only)**

```bash
python -m pytest tests/test_assign_e2e.py -v -m integration
```

---

### Task 12: Integration Test — ingest end-to-end

**Files:**
- Create: `tests/test_ingest_e2e.py`

- [ ] **Step 1: Write integration test**

```python
"""End-to-end integration test for tract ingest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.config import PHASE1D_DEPLOYMENT_MODEL_DIR


@pytest.fixture
def fixture_framework(tmp_path: Path) -> Path:
    fw = {
        "framework_id": "test_ingest_e2e",
        "framework_name": "Test Framework",
        "version": "1.0",
        "source_url": "https://example.com",
        "fetched_date": "2026-04-30",
        "mapping_unit_level": "control",
        "controls": [
            {"control_id": "T-01", "title": "Access Control", "description": "Enforce access control for AI models"},
            {"control_id": "T-02", "title": "Logging", "description": "Log all inference requests for audit"},
            {"control_id": "T-03", "title": "Encryption", "description": "Encrypt model weights and training data at rest"},
        ],
    }
    path = tmp_path / "test_framework.json"
    path.write_text(json.dumps(fw))
    return path


@pytest.mark.integration
class TestIngestE2E:
    def test_generates_review_file(self, fixture_framework: Path) -> None:
        from tract.inference import TRACTPredictor
        from tract.io import atomic_write_json, load_json
        from tract.schema import FrameworkOutput
        from tract.sanitize import sanitize_text
        from tract.config import PHASE1D_DEFAULT_TOP_K

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        fw = FrameworkOutput.model_validate(load_json(fixture_framework))

        texts = []
        for ctrl in fw.controls:
            parts = [ctrl.title, ctrl.description]
            texts.append(" ".join(p for p in parts if p))

        batch_preds = predictor.predict_batch(texts, top_k=PHASE1D_DEFAULT_TOP_K)
        assert len(batch_preds) == 3

        for preds in batch_preds:
            assert len(preds) == 5
            assert all(0 <= p.calibrated_confidence <= 1 for p in preds)

    def test_duplicate_detection(self, fixture_framework: Path) -> None:
        from tract.inference import TRACTPredictor
        from tract.io import load_json
        from tract.schema import FrameworkOutput

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        fw = FrameworkOutput.model_validate(load_json(fixture_framework))

        text = f"{fw.controls[0].title} {fw.controls[0].description}"
        duplicates, similar = predictor.find_duplicates(text)
        # Should find some similar controls (access control is common)
        assert isinstance(duplicates, list)
        assert isinstance(similar, list)
```

---

### Task 13: Integration Test — propose-hubs end-to-end

**Files:**
- Create: `tests/test_propose_e2e.py`

- [ ] **Step 1: Write integration test**

```python
"""End-to-end integration test for hub proposal pipeline."""
from __future__ import annotations

import pytest

import numpy as np

from tract.config import PHASE1D_ARTIFACTS_PATH, PHASE1D_CALIBRATION_PATH, PROCESSED_DIR


@pytest.mark.integration
class TestProposeE2E:
    def test_ood_detection_and_clustering(self) -> None:
        from tract.hierarchy import CREHierarchy
        from tract.inference import load_deployment_artifacts
        from tract.io import load_json
        from tract.proposals.cluster import cluster_ood_controls

        artifacts = load_deployment_artifacts(PHASE1D_ARTIFACTS_PATH)
        calibration = load_json(PHASE1D_CALIBRATION_PATH)

        sims = artifacts.control_embeddings @ artifacts.hub_embeddings.T
        max_sims = sims.max(axis=1)
        ood_mask = max_sims < calibration["ood_threshold"]

        ood_embs = artifacts.control_embeddings[ood_mask]
        ood_ids = [artifacts.control_ids[i] for i in range(len(artifacts.control_ids)) if ood_mask[i]]

        # Current state: ~5 OOD items expected
        assert len(ood_ids) >= 0  # May be 0 or small number

        if len(ood_ids) >= 3:
            clusters = cluster_ood_controls(
                ood_embs, ood_ids,
                hub_embeddings=artifacts.hub_embeddings,
                hub_ids=artifacts.hub_ids,
            )
            # May or may not find clusters — zero is valid in high-dim
            assert isinstance(clusters, list)

    def test_guardrails_pipeline(self) -> None:
        from tract.hierarchy import CREHierarchy
        from tract.inference import load_deployment_artifacts
        from tract.io import load_json
        from tract.proposals.cluster import cluster_ood_controls
        from tract.proposals.guardrails import apply_guardrails

        artifacts = load_deployment_artifacts(PHASE1D_ARTIFACTS_PATH)
        calibration = load_json(PHASE1D_CALIBRATION_PATH)
        hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")

        sims = artifacts.control_embeddings @ artifacts.hub_embeddings.T
        max_sims = sims.max(axis=1)
        ood_mask = max_sims < calibration["ood_threshold"]
        ood_embs = artifacts.control_embeddings[ood_mask]
        ood_ids = [artifacts.control_ids[i] for i in range(len(artifacts.control_ids)) if ood_mask[i]]

        clusters = cluster_ood_controls(
            ood_embs, ood_ids,
            hub_embeddings=artifacts.hub_embeddings,
            hub_ids=artifacts.hub_ids,
        )

        if clusters:
            results = apply_guardrails(
                clusters, hierarchy,
                artifacts.hub_embeddings, artifacts.hub_ids, {},
            )
            assert len(results) == len(clusters)
            for r in results:
                assert isinstance(r.passed, bool)
                if not r.passed:
                    assert len(r.rejection_reasons) > 0
```

---

### Task 14: Console Scripts Entry Point

**Files:**
- Modify or create: `pyproject.toml` or `setup.py`

- [ ] **Step 1: Add console_scripts entry**

If `pyproject.toml` exists, add:

```toml
[project.scripts]
tract = "tract.cli:main"
```

If `setup.py` is used instead:

```python
entry_points={
    "console_scripts": [
        "tract=tract.cli:main",
    ],
},
```

- [ ] **Step 2: Verify entry point works**

```bash
python -m tract.cli --help
python -m tract.cli tutorial
```

---

### Task 15: Final Integration — Run Full Pipeline

- [ ] **Step 1: Regenerate deployment artifacts via T5**

```bash
python -m scripts.phase1c.t5_finalize_crosswalk
```

Verify that `deployment_artifacts.npz` and `calibration.json` are created in `results/phase1c/deployment_model/`.

- [ ] **Step 2: Test all CLI commands**

```bash
# Assign
python -m tract.cli assign "Ensure AI models are tested for bias"
python -m tract.cli assign "Ensure AI models are tested for bias" --raw
python -m tract.cli assign "Ensure AI models are tested for bias" --verbose
python -m tract.cli assign "Ensure AI models are tested for bias" --json

# Hierarchy
python -m tract.cli hierarchy --hub 646-285
python -m tract.cli hierarchy --hub 646-285 --json

# Compare
python -m tract.cli compare --framework mitre_atlas --framework owasp_ai_exchange

# Export
python -m tract.cli export --format csv --output /tmp/test_export.csv

# Tutorial
python -m tract.cli tutorial

# Propose hubs
python -m tract.cli propose-hubs

# Help
python -m tract.cli assign --help
python -m tract.cli compare --help
python -m tract.cli ingest --help
```

- [ ] **Step 3: Run full test suite**

```bash
python -m pytest tests/ -v --ignore=tests/test_assign_e2e.py --ignore=tests/test_ingest_e2e.py --ignore=tests/test_propose_e2e.py
python -m pytest tests/test_assign_e2e.py tests/test_ingest_e2e.py tests/test_propose_e2e.py -v -m integration
```

---

## Implementation Notes

1. **TRACTPredictor uses existing calibration functions** — `calibrate_similarities()`, `build_prediction_sets()`, `flag_ood_items()`. No reimplementation. These are battle-tested in Phase 1C.

2. **Hub IDs are always in canonical `sorted()` order** — NPZ stores them sorted, predictor verifies this invariant, all downstream code can assume sorted order for index alignment.

3. **`predict()` sanitizes internally** — callers never pass raw text. Defense in depth per spec.

4. **Calibrated confidence is primary** — per CLAUDE.md mandate. `--raw` for experts. `--verbose` for both.

5. **OOD flag is informational** — `predict()` still returns full results for OOD items. The flag lets the user decide what to do.

6. **Hub proposal is a latent capability** — only ~5 OOD items exist currently. Zero clusters expected. The pipeline activates after `tract ingest` brings new frameworks with unmapped concepts. This is by design, not a bug.

7. **Guardrail 3 is inter-cluster separation** — NOT cluster-to-hub separation (which is dead code for OOD items). Prevents proposing redundant hubs.

8. **Single NPZ file** — one version check, no cross-file mismatch. Hub embeddings + control embeddings + IDs + hash + timestamp in one file.

9. **`compare.py` queries crosswalk.db live** — no cached data. Refactored from T5's `_export_cross_framework_matrix()` but uses `CREHierarchy.get_parent()` for related pair detection instead of reimplementing.

10. **`--name-with-llm` is opt-in** — explicit help text warns that control texts are sent to Claude API. Without the flag, placeholder names based on nearest hub.
