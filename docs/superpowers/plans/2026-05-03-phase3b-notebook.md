# Phase 3B: Experimental Narrative Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a publication-quality Jupyter notebook (~168 cells, ~35 figures, 5 Plotly interactives) that tells the complete TRACT story — from zero-shot baselines through model training, human review, and a hands-on CLI tutorial — in the voice of a practitioner educating peers.

**Architecture:** One notebook (`notebooks/tract_experimental_narrative.ipynb`) built programmatically via `nbformat`, plus a shared helper module (`notebooks/nb_helpers.py`). All data loaded from canonical project paths — no copies, no hardcoded values. Helper module defines `PROJECT_ROOT` via `Path(__file__)` since `__file__` is undefined in Jupyter cells. Pre-computed base embeddings script generates the "before" panel for Figure 5.2.

**Tech Stack:** Python 3.12, nbformat 5.10.4, matplotlib 3.10.8, seaborn 0.13.2, plotly 5.24.1, kaleido (for Plotly PNG fallbacks), sklearn (t-SNE), numpy, json

**Spec:** `docs/superpowers/specs/2026-05-03-phase3b-notebook-design.md`

---

## File Structure

### New Files

```
notebooks/nb_helpers.py                            — Shared utilities: paths, palette, FigureCounter, axis styling, Plotly fallback, data loaders
notebooks/tract_experimental_narrative.ipynb        — The notebook (13 sections + 2 appendices, ~168 cells)
scripts/precompute_base_embeddings.py              — One-time script: load base BGE-large-v1.5, encode ~3,300 texts, save to results/phase1b/base_bge_embeddings.npz
tests/test_nb_helpers.py                           — Tests for nb_helpers module
```

### Modified Files

```
.gitignore                             — Add notebooks/.ipynb_checkpoints/
```

### Existing Files Referenced (read-only)

```
results/phase0/exp1_embedding_baseline_bge.json    — BGE zero-shot metrics (unfirewalled)
results/phase0/exp1_embedding_baseline_gte.json    — GTE zero-shot metrics
results/phase0/exp1_embedding_baseline_deberta.json — DeBERTa zero-shot metrics
results/phase0/exp2_llm_probe.json                 — Opus zero-shot metrics
results/phase0/exp3_hierarchy_paths_bge.json       — Hierarchy path ablation (BGE)
results/phase0/exp4_hub_descriptions.json          — Description ablation
results/phase0/exp5_knn_baseline.json              — kNN baseline
results/phase0/exp6_fewshot_sonnet.json            — Few-shot Sonnet
results/phase0/exp7_e5_mistral_7b.json             — E5-Mistral results
results/phase0/exp7_sfr_embedding_2.json           — SFR results
results/phase1b/zero_shot_firewalled_baseline/     — Firewalled zero-shot per-fold metrics
results/phase1b/phase1b_textaware/fold_*/          — Fine-tuned per-fold metrics + trainer_state.json
results/phase1c/calibration/                       — Temperature, ECE, OOD data
results/phase1c/deployment_model/deployment_artifacts.npz — Hub/control embeddings (522×1024, 2802×1024)
data/processed/cre_hierarchy.json                  — Hub hierarchy (522 hubs, 400 label space)
results/review/review_metrics.json                 — Review outcomes (copied from worktree)
results/review/review_export.json                  — Review predictions + calibration items
build/dataset/crosswalk_v1.0.jsonl                 — Published crosswalk
build/dataset/framework_metadata.json              — Per-framework stats
```

---

## Task 0: Prerequisites

- [ ] Copy Phase 3 review files from worktree to main
- [ ] Copy Phase 3 dataset staging files from worktree to main
- [ ] Install kaleido for Plotly PNG fallbacks
- [ ] Add `.ipynb_checkpoints/` to `.gitignore`
- [ ] Create `notebooks/` directory

### Implementation

```bash
# 1. Copy review files
mkdir -p results/review
cp .worktrees/phase3/results/review/review_metrics.json results/review/
cp .worktrees/phase3/results/review/review_export.json results/review/
cp .worktrees/phase3/results/review/hub_reference.json results/review/
cp .worktrees/phase3/results/review/reviewer_guide.md results/review/

# 2. Copy dataset staging files
mkdir -p build/dataset
cp .worktrees/phase3/build/dataset/crosswalk_v1.0.jsonl build/dataset/
cp .worktrees/phase3/build/dataset/framework_metadata.json build/dataset/
cp .worktrees/phase3/build/dataset/review_metrics.json build/dataset/

# 3. Install kaleido
pip install kaleido

# 4. Create notebooks directory (as Python package for test imports)
mkdir -p notebooks
touch notebooks/__init__.py
```

Add to `.gitignore`:
```
notebooks/.ipynb_checkpoints/
```

### Tests

Verify all prerequisite files exist:
```bash
test -f results/review/review_metrics.json && echo "review_metrics OK"
test -f results/review/review_export.json && echo "review_export OK"
test -f results/review/hub_reference.json && echo "hub_reference OK"
test -f build/dataset/crosswalk_v1.0.jsonl && echo "crosswalk OK"
test -f build/dataset/framework_metadata.json && echo "framework_metadata OK"
python -c "import kaleido; print(f'kaleido {kaleido.__version__} OK')"
```

### Commit

`feat: add Phase 3B prerequisites — copy review/dataset files, install kaleido`

---

## Task 1: nb_helpers.py + Tests

- [ ] Create `notebooks/nb_helpers.py` with all shared utilities
- [ ] Create `tests/test_nb_helpers.py` with full test coverage
- [ ] Run tests green

### Implementation

**File: `notebooks/nb_helpers.py`**

```python
"""Shared utilities for the TRACT experimental narrative notebook.

Defines PROJECT_ROOT via Path(__file__) — the notebook imports path constants
from here since __file__ is undefined in Jupyter cells.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
PHASE0_DIR = RESULTS_DIR / "phase0"
PHASE1B_DIR = RESULTS_DIR / "phase1b"
PHASE1C_DIR = RESULTS_DIR / "phase1c"
REVIEW_DIR = RESULTS_DIR / "review"
BRIDGE_DIR = RESULTS_DIR / "bridge"
DATASET_DIR = PROJECT_ROOT / "build" / "dataset"

# ─── Palette (Okabe-Ito, colorblind-safe) ────────────────────────────
OKABE_ITO = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
    "#999999",  # grey
]
SEQUENTIAL_BLUE = "Blues"
SEQUENTIAL_ORANGE = "Oranges"
DIVERGING = "RdBu_r"


# ─── Figure Counter ──────────────────────────────────────────────────
class FigureCounter:
    """Track figure numbers per section. Instantiate ONCE in the notebook setup cell."""

    def __init__(self) -> None:
        self._counts: dict[int, int] = {}

    def next(self, section: int) -> str:
        self._counts[section] = self._counts.get(section, 0) + 1
        return f"Figure {section}.{self._counts[section]}"

    def current(self, section: int) -> str:
        count = self._counts.get(section, 0)
        if count == 0:
            raise ValueError(f"No figures yet in section {section}")
        return f"Figure {section}.{count}"

    def reset(self) -> None:
        self._counts = {}


# ─── Axis Styling ────────────────────────────────────────────────────
def style_axes(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    fig_num: str,
) -> None:
    """Apply consistent styling to a matplotlib axes."""
    ax.set_title(f"{fig_num}: {title}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


# ─── Plotly Static Fallback ──────────────────────────────────────────
def plotly_with_fallback(
    fig: Any,
    fig_num: str,
    title: str,
    width: int = 900,
    height: int = 600,
) -> None:
    """Display a Plotly figure with a static PNG fallback via kaleido.

    Shows the interactive figure, then embeds a static PNG as an IPython
    Image so the notebook renders meaningfully in HTML/PDF exports.
    """
    fig.update_layout(
        title=f"{fig_num}: {title}",
        width=width,
        height=height,
    )
    fig.show()

    try:
        from IPython.display import Image, display
        png_bytes = fig.to_image(format="png", width=width, height=height)
        display(Image(data=png_bytes))
    except Exception:
        logger.warning("kaleido not available — static PNG fallback skipped for %s", fig_num)


# ─── Data Loaders ────────────────────────────────────────────────────
def _load_json(path: Path) -> Any:
    """Load a JSON file with utf-8 encoding."""
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_phase0_experiment(experiment: str) -> dict:
    """Load a Phase 0 experiment result file.

    Args:
        experiment: Filename without extension, e.g. 'exp1_embedding_baseline_bge'
    """
    path = PHASE0_DIR / f"{experiment}.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 0 experiment not found: {path}")
    return _load_json(path)


def load_firewalled_baseline() -> dict:
    """Load Phase 1B firewalled zero-shot baseline aggregate metrics."""
    path = PHASE1B_DIR / "zero_shot_firewalled_baseline" / "aggregate_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Firewalled baseline not found: {path}")
    return _load_json(path)


def load_fold_metrics(run_name: str, fold: str) -> dict:
    """Load metrics.json for a specific fold of a training run.

    Args:
        run_name: e.g. 'phase1b_textaware'
        fold: e.g. 'fold_MITRE_ATLAS'
    """
    path = PHASE1B_DIR / run_name / fold / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Fold metrics not found: {path}")
    return _load_json(path)


def load_training_logs(run_name: str, fold: str) -> list[dict]:
    """Load trainer_state.json log_history for a specific fold.

    Returns the list of log entries (epoch, loss, grad_norm, learning_rate, step).
    """
    fold_dir = PHASE1B_DIR / run_name / fold
    checkpoint_dirs = sorted(fold_dir.glob("checkpoint-*"))
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint directories found in {fold_dir}")
    trainer_state = checkpoint_dirs[-1] / "trainer_state.json"
    if not trainer_state.exists():
        raise FileNotFoundError(f"trainer_state.json not found: {trainer_state}")
    data = _load_json(trainer_state)
    return data["log_history"]


def load_calibration_data() -> dict:
    """Load all calibration results: temperature, ECE, OOD."""
    cal_dir = PHASE1C_DIR / "calibration"
    return {
        "temperature": _load_json(cal_dir / "t_deploy_result.json"),
        "ece": _load_json(cal_dir / "ece_gate.json"),
        "ood": _load_json(cal_dir / "ood.json"),
    }


def load_deployment_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load pre-computed deployment embeddings.

    Returns:
        (hub_embeddings, control_embeddings, hub_ids, control_ids)
    """
    path = PHASE1C_DIR / "deployment_model" / "deployment_artifacts.npz"
    if not path.exists():
        raise FileNotFoundError(f"Deployment artifacts not found: {path}")
    data = np.load(str(path), allow_pickle=False)
    return (
        data["hub_embeddings"],
        data["control_embeddings"],
        data["hub_ids"],
        data["control_ids"],
    )


def load_review_metrics() -> dict:
    """Load Phase 3 expert review metrics."""
    path = REVIEW_DIR / "review_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"Review metrics not found: {path}")
    return _load_json(path)


def load_review_export() -> dict:
    """Load Phase 3 review export (predictions + calibration items)."""
    path = REVIEW_DIR / "review_export.json"
    if not path.exists():
        raise FileNotFoundError(f"Review export not found: {path}")
    return _load_json(path)


def load_cre_hierarchy() -> dict:
    """Load CRE hierarchy (hubs, roots, label_space)."""
    path = DATA_DIR / "processed" / "cre_hierarchy.json"
    if not path.exists():
        raise FileNotFoundError(f"CRE hierarchy not found: {path}")
    return _load_json(path)


def load_crosswalk() -> list[dict]:
    """Load published crosswalk JSONL as a list of dicts."""
    path = DATASET_DIR / "crosswalk_v1.0.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Crosswalk not found: {path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_framework_metadata() -> list[dict]:
    """Load framework metadata from dataset staging."""
    path = DATASET_DIR / "framework_metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Framework metadata not found: {path}")
    return _load_json(path)


# ─── Prerequisite Checker ────────────────────────────────────────────
PREREQUISITE_PATHS = [
    PHASE0_DIR / "exp1_embedding_baseline_bge.json",
    PHASE0_DIR / "exp2_llm_probe.json",
    PHASE0_DIR / "exp5_knn_baseline.json",
    PHASE0_DIR / "exp6_fewshot_sonnet.json",
    PHASE0_DIR / "exp3_hierarchy_paths_bge.json",
    PHASE0_DIR / "exp4_hub_descriptions.json",
    PHASE1B_DIR / "zero_shot_firewalled_baseline" / "aggregate_metrics.json",
    PHASE1C_DIR / "calibration" / "t_deploy_result.json",
    PHASE1C_DIR / "calibration" / "ece_gate.json",
    PHASE1C_DIR / "calibration" / "ood.json",
    PHASE1C_DIR / "deployment_model" / "deployment_artifacts.npz",
    DATA_DIR / "processed" / "cre_hierarchy.json",
    REVIEW_DIR / "review_metrics.json",
    REVIEW_DIR / "review_export.json",
    DATASET_DIR / "crosswalk_v1.0.jsonl",
    DATASET_DIR / "framework_metadata.json",
]


def check_prerequisites() -> list[str]:
    """Check all prerequisite files exist. Returns list of missing paths."""
    missing = []
    for path in PREREQUISITE_PATHS:
        if not path.exists():
            missing.append(str(path.relative_to(PROJECT_ROOT)))
    return missing
```

**File: `tests/test_nb_helpers.py`**

```python
"""Tests for nb_helpers module."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _patch_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch PROJECT_ROOT and all derived paths to use tmp_path."""
    import notebooks.nb_helpers as helpers

    monkeypatch.setattr(helpers, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(helpers, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(helpers, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(helpers, "PHASE0_DIR", tmp_path / "results" / "phase0")
    monkeypatch.setattr(helpers, "PHASE1B_DIR", tmp_path / "results" / "phase1b")
    monkeypatch.setattr(helpers, "PHASE1C_DIR", tmp_path / "results" / "phase1c")
    monkeypatch.setattr(helpers, "REVIEW_DIR", tmp_path / "results" / "review")
    monkeypatch.setattr(helpers, "BRIDGE_DIR", tmp_path / "results" / "bridge")
    monkeypatch.setattr(helpers, "DATASET_DIR", tmp_path / "build" / "dataset")


class TestFigureCounter:
    def test_first_figure_in_section(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        assert fc.next(1) == "Figure 1.1"

    def test_sequential_figures(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(3)
        assert fc.next(3) == "Figure 3.2"

    def test_multiple_sections(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(1)
        fc.next(2)
        fc.next(1)
        assert fc.next(2) == "Figure 2.2"

    def test_current_returns_last(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(5)
        fc.next(5)
        assert fc.current(5) == "Figure 5.2"

    def test_current_raises_if_no_figures(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        with pytest.raises(ValueError, match="No figures yet"):
            fc.current(1)

    def test_reset(self) -> None:
        from notebooks.nb_helpers import FigureCounter

        fc = FigureCounter()
        fc.next(1)
        fc.next(1)
        fc.reset()
        assert fc.next(1) == "Figure 1.1"


class TestStyleAxes:
    def test_applies_title_and_labels(self) -> None:
        import matplotlib.pyplot as plt

        from notebooks.nb_helpers import style_axes

        fig, ax = plt.subplots()
        style_axes(ax, "Test Title", "X", "Y", "Figure 1.1")
        assert "Figure 1.1" in ax.get_title()
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Y"
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        plt.close(fig)


class TestPalette:
    def test_okabe_ito_has_8_colors(self) -> None:
        from notebooks.nb_helpers import OKABE_ITO

        assert len(OKABE_ITO) == 8

    def test_all_hex(self) -> None:
        from notebooks.nb_helpers import OKABE_ITO

        for color in OKABE_ITO:
            assert color.startswith("#")
            assert len(color) == 7


class TestPaths:
    def test_project_root_is_directory(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import PROJECT_ROOT

        assert PROJECT_ROOT == tmp_path

    def test_results_dir(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import RESULTS_DIR

        assert RESULTS_DIR == tmp_path / "results"


class TestLoadPhase0Experiment:
    def test_loads_json(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_phase0_experiment

        p0 = tmp_path / "results" / "phase0"
        p0.mkdir(parents=True)
        (p0 / "exp1_test.json").write_text(json.dumps({"key": "val"}), encoding="utf-8")
        result = load_phase0_experiment("exp1_test")
        assert result == {"key": "val"}

    def test_raises_on_missing(self) -> None:
        from notebooks.nb_helpers import load_phase0_experiment

        with pytest.raises(FileNotFoundError):
            load_phase0_experiment("nonexistent")


class TestLoadFirewalledBaseline:
    def test_loads_aggregate(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_firewalled_baseline

        d = tmp_path / "results" / "phase1b" / "zero_shot_firewalled_baseline"
        d.mkdir(parents=True)
        (d / "aggregate_metrics.json").write_text(
            json.dumps({"aggregate_hit1": {"mean": 0.399}}), encoding="utf-8"
        )
        result = load_firewalled_baseline()
        assert result["aggregate_hit1"]["mean"] == 0.399


class TestLoadFoldMetrics:
    def test_loads_fold(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_fold_metrics

        d = tmp_path / "results" / "phase1b" / "run1" / "fold_A"
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps({"hit_at_1": 0.5}), encoding="utf-8")
        result = load_fold_metrics("run1", "fold_A")
        assert result["hit_at_1"] == 0.5


class TestLoadTrainingLogs:
    def test_loads_from_last_checkpoint(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_training_logs

        fold_dir = tmp_path / "results" / "phase1b" / "run1" / "fold_A"
        ckpt = fold_dir / "checkpoint-100"
        ckpt.mkdir(parents=True)
        (ckpt / "trainer_state.json").write_text(
            json.dumps({"log_history": [{"epoch": 1, "loss": 0.5}]}), encoding="utf-8"
        )
        result = load_training_logs("run1", "fold_A")
        assert result == [{"epoch": 1, "loss": 0.5}]

    def test_raises_on_no_checkpoints(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_training_logs

        fold_dir = tmp_path / "results" / "phase1b" / "run1" / "fold_B"
        fold_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No checkpoint"):
            load_training_logs("run1", "fold_B")


class TestLoadCalibrationData:
    def test_loads_all_three(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_calibration_data

        cal = tmp_path / "results" / "phase1c" / "calibration"
        cal.mkdir(parents=True)
        (cal / "t_deploy_result.json").write_text(json.dumps({"temperature": 0.074}), encoding="utf-8")
        (cal / "ece_gate.json").write_text(json.dumps({"ece": 0.079}), encoding="utf-8")
        (cal / "ood.json").write_text(json.dumps({"threshold": 0.568}), encoding="utf-8")
        result = load_calibration_data()
        assert result["temperature"]["temperature"] == 0.074
        assert result["ece"]["ece"] == 0.079
        assert result["ood"]["threshold"] == 0.568


class TestLoadDeploymentEmbeddings:
    def test_loads_arrays(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_deployment_embeddings

        d = tmp_path / "results" / "phase1c" / "deployment_model"
        d.mkdir(parents=True)
        np.savez(
            str(d / "deployment_artifacts.npz"),
            hub_embeddings=np.zeros((5, 10)),
            control_embeddings=np.ones((3, 10)),
            hub_ids=np.array(["h1", "h2", "h3", "h4", "h5"]),
            control_ids=np.array(["c1", "c2", "c3"]),
        )
        hubs, controls, hub_ids, ctrl_ids = load_deployment_embeddings()
        assert hubs.shape == (5, 10)
        assert controls.shape == (3, 10)
        assert len(hub_ids) == 5
        assert len(ctrl_ids) == 3


class TestLoadReviewMetrics:
    def test_loads(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_review_metrics

        d = tmp_path / "results" / "review"
        d.mkdir(parents=True)
        (d / "review_metrics.json").write_text(
            json.dumps({"overall": {"accepted": 680}}), encoding="utf-8"
        )
        result = load_review_metrics()
        assert result["overall"]["accepted"] == 680


class TestLoadCREHierarchy:
    def test_loads(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_cre_hierarchy

        d = tmp_path / "data" / "processed"
        d.mkdir(parents=True)
        (d / "cre_hierarchy.json").write_text(
            json.dumps({"hubs": {}, "roots": [3], "label_space": []}), encoding="utf-8"
        )
        result = load_cre_hierarchy()
        assert result["roots"] == [3]


class TestLoadCrosswalk:
    def test_loads_jsonl(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import load_crosswalk

        d = tmp_path / "build" / "dataset"
        d.mkdir(parents=True)
        lines = [json.dumps({"control_id": f"c{i}"}) for i in range(3)]
        (d / "crosswalk_v1.0.jsonl").write_text("\n".join(lines), encoding="utf-8")
        result = load_crosswalk()
        assert len(result) == 3
        assert result[0]["control_id"] == "c0"


class TestCheckPrerequisites:
    def test_all_missing_returns_all(self) -> None:
        from notebooks.nb_helpers import check_prerequisites

        missing = check_prerequisites()
        assert len(missing) > 0

    def test_all_present_returns_empty(self, tmp_path: Path) -> None:
        from notebooks.nb_helpers import PREREQUISITE_PATHS, check_prerequisites

        for path in PREREQUISITE_PATHS:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npz":
                np.savez(str(path), dummy=np.array([0]))
            else:
                path.write_text("{}", encoding="utf-8")
        assert check_prerequisites() == []
```

### Commit

`feat: add nb_helpers.py shared utilities with full test suite`

---

## Task 2: Pre-compute Base BGE Embeddings Script

- [ ] Create `scripts/precompute_base_embeddings.py`
- [ ] Script loads base BGE-large-v1.5, encodes all control + hub texts, saves to `results/phase1b/base_bge_embeddings.npz`

### Implementation

**File: `scripts/precompute_base_embeddings.py`**

```python
"""Pre-compute base BGE-large-v1.5 embeddings (before fine-tuning) for Figure 5.2.

Loads the same texts used in deployment_artifacts.npz, encodes them with the
un-fine-tuned base model, and saves the results. This is a one-time operation
(~2 min on Tegra ARM64, requires ~1.3 GB model download on first run).

Usage:
    python scripts/precompute_base_embeddings.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tract.config import PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROJECT_ROOT / "results" / "phase1b" / "base_bge_embeddings.npz"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64


def load_texts() -> tuple[list[str], list[str], list[str], list[str]]:
    """Load hub and control texts matching deployment_artifacts.npz order.

    Returns:
        (hub_texts, control_texts, hub_ids, control_ids)
    """
    artifacts = np.load(
        str(PROJECT_ROOT / "results" / "phase1c" / "deployment_model" / "deployment_artifacts.npz"),
        allow_pickle=False,
    )
    hub_ids = list(artifacts["hub_ids"])
    control_ids = list(artifacts["control_ids"])

    hierarchy = json.loads(
        (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
    )
    hubs = hierarchy["hubs"]

    hub_texts = []
    for hid in hub_ids:
        hub = hubs.get(str(hid), hubs.get(hid, {}))
        path = hub.get("hierarchy_path", "")
        name = hub.get("name", "")
        hub_texts.append(f"{path} {name}".strip())

    crosswalk_path = PROJECT_ROOT / "build" / "dataset" / "crosswalk_v1.0.jsonl"
    control_map: dict[str, str] = {}
    if crosswalk_path.exists():
        with crosswalk_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    row = json.loads(line)
                    cid = row.get("control_id", "")
                    title = row.get("control_title", "")
                    if cid and cid not in control_map:
                        control_map[cid] = title

    control_texts = [control_map.get(cid, str(cid)) for cid in control_ids]

    return hub_texts, control_texts, hub_ids, control_ids


def main() -> None:
    if OUTPUT_PATH.exists():
        logger.info("Output already exists: %s — skipping", OUTPUT_PATH)
        return

    logger.info("Loading texts from deployment artifacts...")
    hub_texts, control_texts, hub_ids, control_ids = load_texts()
    logger.info("Hub texts: %d, Control texts: %d", len(hub_texts), len(control_texts))

    logger.info("Loading base model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    logger.info("Encoding hub texts...")
    hub_embeddings = model.encode(
        hub_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    logger.info("Encoding control texts...")
    control_embeddings = model.encode(
        control_texts, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True
    )

    logger.info("Saving to %s", OUTPUT_PATH)
    np.savez(
        str(OUTPUT_PATH),
        hub_embeddings=hub_embeddings,
        control_embeddings=control_embeddings,
        hub_ids=np.array(hub_ids),
        control_ids=np.array(control_ids),
    )
    logger.info(
        "Done. hub_embeddings=%s, control_embeddings=%s",
        hub_embeddings.shape,
        control_embeddings.shape,
    )


if __name__ == "__main__":
    main()
```

### Tests

Run the script and verify output:
```bash
python scripts/precompute_base_embeddings.py
python -c "
import numpy as np
d = np.load('results/phase1b/base_bge_embeddings.npz', allow_pickle=False)
print(f'hub_embeddings: {d[\"hub_embeddings\"].shape}')
print(f'control_embeddings: {d[\"control_embeddings\"].shape}')
assert d['hub_embeddings'].shape[1] == 1024
assert d['control_embeddings'].shape[1] == 1024
print('OK')
"
```

### Commit

`feat: add pre-compute script for base BGE embeddings (Figure 5.2)`

---

## Task 3: Notebook Skeleton + Setup Cells

- [ ] Create `notebooks/tract_experimental_narrative.ipynb` with title cell and setup code cell
- [ ] Setup cell imports, seeds, checks prerequisites, initializes FigureCounter
- [ ] Build using `nbformat` programmatically

### Implementation

Create the notebook skeleton using a Python script that writes it via nbformat:

**Script logic (run once, then the notebook file persists):**

```python
import nbformat

nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

# Cell 0: Title (markdown)
nb.cells.append(nbformat.v4.new_markdown_cell(
    "# TRACT: From Zero-Shot to Expert-Reviewed Security Crosswalk\n\n"
    "**A practitioner's guide to mapping AI security frameworks using machine learning**\n\n"
    "This notebook tells the story of how we built a system that reads any security control "
    "and tells you which part of a universal security taxonomy it belongs to. We'll walk through "
    "every experiment, every failure, and every hard-won insight — in plain language.\n\n"
    "---"
))

# Cell 1: Setup (code)
nb.cells.append(nbformat.v4.new_code_cell(
    '''"""Setup: imports, seeds, prerequisite checks."""
import sys
import json
import random
import warnings
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import TSNE

# Ensure nb_helpers is importable from notebooks/
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent))

from nb_helpers import (
    PROJECT_ROOT, RESULTS_DIR, DATA_DIR,
    PHASE0_DIR, PHASE1B_DIR, PHASE1C_DIR,
    REVIEW_DIR, BRIDGE_DIR, DATASET_DIR,
    OKABE_ITO, SEQUENTIAL_BLUE, SEQUENTIAL_ORANGE, DIVERGING,
    FigureCounter, style_axes, plotly_with_fallback,
    load_phase0_experiment, load_firewalled_baseline,
    load_fold_metrics, load_training_logs,
    load_calibration_data, load_deployment_embeddings,
    load_review_metrics, load_review_export,
    load_cre_hierarchy, load_crosswalk, load_framework_metadata,
    check_prerequisites,
)

# Reproducibility
random.seed(42)
np.random.seed(42)

# Matplotlib defaults
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 120,
    "font.size": 11,
    "axes.titlesize": 13,
    "savefig.bbox": "tight",
})
sns.set_style("whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)

# Figure counter — instantiate once
fig_counter = FigureCounter()

# Prerequisite check
missing = check_prerequisites()
if missing:
    print("⚠️  Missing prerequisite files:")
    for m in missing:
        print(f"   - {m}")
    print("\\nSee the notebook README for setup instructions.")
else:
    print("✓ All prerequisite files found")

# Verify CWD
cwd = Path.cwd()
if cwd.name != "notebooks":
    print(f"⚠️  Expected CWD = notebooks/, got {cwd.name}/. Shell cells may not work correctly.")
else:
    print(f"✓ CWD: {cwd}")

print(f"✓ PROJECT_ROOT: {PROJECT_ROOT}")
print(f"✓ NumPy {np.__version__}, Matplotlib {matplotlib.__version__}")'''
))

nbformat.write(nb, "notebooks/tract_experimental_narrative.ipynb")
```

The notebook will be built incrementally — each subsequent task adds cells to the existing file using `nbformat.read()` → append cells → `nbformat.write()`.

### Tests

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/tract_experimental_narrative.ipynb', as_version=4)
assert len(nb.cells) == 2
assert nb.cells[0].cell_type == 'markdown'
assert nb.cells[1].cell_type == 'code'
assert 'FigureCounter' in nb.cells[1].source
assert 'check_prerequisites' in nb.cells[1].source
print(f'Notebook has {len(nb.cells)} cells — OK')
"
```

### Commit

`feat: create notebook skeleton with setup cells and prerequisite checks`

---

## Task 4: Section 1 — Introduction & Motivation (10 cells, 2 figures)

- [ ] Add narrative markdown: the N² problem, CRE as GPS, assignment paradigm
- [ ] Add Figure 1.1: CRE hierarchy sunburst (Plotly interactive)
- [ ] Add Figure 1.2: N² vs assignment diagram (matplotlib)
- [ ] Add Plain English blockquote

### Implementation

Append to existing notebook via nbformat:

**Cells to add:**

1. **Markdown** — Section header + "You're a security architect" opener
2. **Markdown** — CRE as a shared coordinate system explanation
3. **Code** — Load CRE hierarchy data
4. **Code** — Figure 1.1: Plotly sunburst of CRE hierarchy (522 hubs, collapsible)
   ```python
   hierarchy = load_cre_hierarchy()
   hubs = hierarchy["hubs"]
   
   ids, names, parents = [], [], []
   for hub_id, hub in sorted(hubs.items()):
       ids.append(hub_id)
       names.append(hub["name"])
       parents.append(hub.get("parent_id") or "")
   
   fig = go.Figure(go.Sunburst(
       ids=ids,
       labels=names,
       parents=parents,
       maxdepth=3,
   ))
   fig_num = fig_counter.next(1)
   plotly_with_fallback(fig, fig_num, "CRE Hub Taxonomy — 522 Hubs")
   ```
5. **Markdown** — Interpret the sunburst
6. **Markdown** — The assignment paradigm: `g(control_text) → CRE_hub`
7. **Code** — Figure 1.2: N² vs assignment diagram (matplotlib, simple schematic)
   ```python
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
   
   # Left: N² pairwise
   n_fw = 6
   for i in range(n_fw):
       for j in range(i+1, n_fw):
           ax1.plot([i, j], [0, 0], 'o-', color=OKABE_ITO[4], alpha=0.3, markersize=8)
   ax1.set_xlim(-0.5, n_fw-0.5)
   ax1.set_title(f"Pairwise: {n_fw*(n_fw-1)//2} comparisons", fontsize=12)
   ax1.set_yticks([])
   
   # Right: Assignment paradigm
   for i in range(n_fw):
       ax2.annotate("", xy=(3, 2-i*0.7), xytext=(i, -2),
                     arrowprops=dict(arrowstyle="->", color=OKABE_ITO[i % len(OKABE_ITO)]))
   ax2.set_title(f"Assignment: {n_fw} mappings", fontsize=12)
   ax2.set_yticks([])
   
   fig_num = fig_counter.next(1)
   fig.suptitle(f"{fig_num}: Why Assignment Scales", fontsize=13, fontweight="bold")
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
8. **Markdown** — Why assignment matters for practitioners
9. **Markdown** — What this notebook covers (roadmap)
10. **Markdown** — Plain English blockquote:
    > **Plain English:** We built a system that reads any security control and tells you which part of a universal security taxonomy it belongs to. This lets you instantly compare any two frameworks — without reading thousands of controls manually.

### Commit

`feat: add Section 1 — Introduction & Motivation (10 cells, 2 figures)`

---

## Task 5: Section 2 — Data Landscape (12 cells, 3 figures)

- [ ] Add narrative markdown: training data, long tail, multi-hub mappings
- [ ] Add Figure 2.1: Framework→hub Sankey (Plotly interactive)
- [ ] Add Figure 2.2: Hub link distribution histogram
- [ ] Add Figure 2.3: Framework size comparison bar chart
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — Section header + training data explanation
2. **Code** — Load crosswalk + framework metadata + hierarchy
3. **Code** — Figure 2.1: Sankey (Plotly) — framework→hub cluster flows
   ```python
   crosswalk = load_crosswalk()
   fw_metadata = load_framework_metadata()
   
   # Group controls by framework and top-level hub cluster
   hierarchy = load_cre_hierarchy()
   hubs = hierarchy["hubs"]
   
   def get_root_hub(hub_id: str) -> str:
       """Walk up the hierarchy to find the root ancestor."""
       current = hub_id
       while current in hubs and (hubs[current].get("parent_id") or None):
           current = hubs[current]["parent_id"]
       return current
   
   # Count flows: framework_name → root_hub_name
   from collections import Counter
   flows = Counter()
   for row in crosswalk:
       root = get_root_hub(row["hub_id"])
       root_name = hubs[root]["name"] if root in hubs else root
       flows[(row["framework_name"], root_name)] += 1
   
   # Top 5 root hubs by total flow (avoid visual clutter)
   root_totals = Counter()
   for (fw, root), count in flows.items():
       root_totals[root] += count
   top_roots = {r for r, _ in root_totals.most_common(5)}
   
   # Build Sankey node/link arrays
   fw_names = sorted({fw for fw, root in flows if root in top_roots})
   root_names = sorted(top_roots)
   labels = fw_names + root_names
   label_idx = {name: i for i, name in enumerate(labels)}
   
   sources, targets, values = [], [], []
   for (fw, root), count in flows.items():
       if root in top_roots and fw in label_idx:
           sources.append(label_idx[fw])
           targets.append(label_idx[root])
           values.append(count)
   
   colors = [OKABE_ITO[i % len(OKABE_ITO)] for i in range(len(labels))]
   
   fig = go.Figure(go.Sankey(
       node=dict(label=labels, color=colors, pad=15),
       link=dict(source=sources, target=targets, value=values),
   ))
   fig_num = fig_counter.next(2)
   plotly_with_fallback(fig, fig_num, "Framework-to-Hub Cluster Mappings")
   ```
4. **Markdown** — Interpret the Sankey
5. **Code** — Figure 2.2: Hub link distribution (matplotlib)
   ```python
   hub_link_counts = Counter(row["hub_id"] for row in crosswalk)
   counts = sorted(hub_link_counts.values(), reverse=True)
   
   fig, ax = plt.subplots(figsize=(12, 5))
   ax.bar(range(len(counts)), counts, color=OKABE_ITO[4], alpha=0.8)
   # Annotate 80/20 point
   cumsum = np.cumsum(counts) / sum(counts)
   p80 = np.searchsorted(cumsum, 0.8)
   ax.axvline(p80, color=OKABE_ITO[5], linestyle="--", label=f"80% of links in top {p80} hubs")
   ax.legend()
   fig_num = fig_counter.next(2)
   style_axes(ax, "Hub Link Distribution — The Long Tail", "Hubs (sorted by link count)", "Number of links", fig_num)
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
6. **Markdown** — Long tail interpretation
7. **Code** — Figure 2.3: Framework size comparison (horizontal bar)
   ```python
   fw_names = [fw["framework_name"] for fw in fw_metadata]
   fw_counts = [fw["total_controls"] for fw in fw_metadata]
   fw_types = [fw["coverage_type"] for fw in fw_metadata]
   colors = [OKABE_ITO[2] if t == "ground_truth" else OKABE_ITO[0] for t in fw_types]
   
   fig, ax = plt.subplots(figsize=(10, 8))
   y_pos = range(len(fw_names))
   ax.barh(y_pos, fw_counts, color=colors)
   ax.set_yticks(y_pos)
   ax.set_yticklabels(fw_names, fontsize=9)
   fig_num = fig_counter.next(2)
   style_axes(ax, "Controls per Framework", "Number of controls", "", fig_num)
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
8. **Markdown** — Multi-hub mappings: "35% of controls map to more than one hub"
9. **Code** — Compute multi-hub stats from crosswalk
10. **Markdown** — What multi-hub means for evaluation
11. **Markdown** — Data provenance: OpenCRE + expert links
12. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 2 — Data Landscape (12 cells, 3 figures)`

---

## Task 6: Section 3 — Phase 0 Baselines (14 cells, 3 figures)

- [ ] Add narrative markdown: DeBERTa disaster, BGE wins, hierarchy paths, Opus ceiling
- [ ] Add Figure 3.1: Model comparison bar chart
- [ ] Add Figure 3.2: Per-framework radar chart
- [ ] Add Figure 3.3: Hierarchy path impact paired bar chart
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "Can a pre-trained model do this?"
2. **Code** — Load all Phase 0 experiment results
   ```python
   bge = load_phase0_experiment("exp1_embedding_baseline_bge")
   gte = load_phase0_experiment("exp1_embedding_baseline_gte")
   deberta = load_phase0_experiment("exp1_embedding_baseline_deberta")
   opus = load_phase0_experiment("exp2_llm_probe")
   knn = load_phase0_experiment("exp5_knn_baseline")
   fewshot = load_phase0_experiment("exp6_fewshot_sonnet")
   
   # Extract hit@1 for each model
   # NOTE: Opus all_198 gives 0.553 (unfirewalled, n=197) but summary.json reports 0.465.
   # Use summary.json value for Gate B context; use all_198 for direct comparison.
   models = {
       "DeBERTa-v3-NLI": deberta["models"]["deberta-v3-nli"]["all_198"]["hit_at_1"]["mean"],
       "kNN (k=5)": knn["k_values"]["k5"]["all"]["hit_at_1"]["mean"],
       "GTE-large": gte["models"]["gte-large-v1.5"]["all_198"]["hit_at_1"]["mean"],
       "BGE-large-v1.5": bge["models"]["bge-large-v1.5"]["all_198"]["hit_at_1"]["mean"],
       "Few-shot Sonnet (desc)": fewshot["variants"]["sonnet-desc"]["all"]["hit_at_1"]["mean"],
       "Few-shot Sonnet (no desc)": fewshot["variants"]["sonnet-nodesc"]["all"]["hit_at_1"]["mean"],
       "Claude Opus": opus["all_198"]["hit_at_1"]["mean"],
   }
   ```
3. **Code** — Figure 3.1: Model comparison bar chart (sorted by performance)
4. **Markdown** — "The DeBERTa disaster" — NLI ≠ semantic similarity
5. **Markdown** — "BGE-large wins zero-shot" — hit@1=0.348 unfirewalled
6. **Code** — Figure 3.2: Per-framework radar chart
   ```python
   # Extract per-fold data from BGE (list structure: [{framework, metrics, n_items}])
   bge_folds = bge["models"]["bge-large-v1.5"]["per_fold"]
   # ... build radar chart for multiple models × frameworks
   ```
7. **Markdown** — The Opus ceiling (hit@1=0.553 unfirewalled at $0.60/control; summary.json reports 0.465 from partial evaluation of 99/197 controls)
8. **Code** — Load hierarchy path ablation data
   ```python
   paths_bge = load_phase0_experiment("exp3_hierarchy_paths_bge")
   descs = load_phase0_experiment("exp4_hub_descriptions")
   ```
9. **Code** — Figure 3.3: Hierarchy path impact (paired bar: with/without)
10. **Markdown** — Hierarchy paths: +7.6%, descriptions: -4.8%
11. **Markdown** — "Structure > prose for this task"
12. **Markdown** — Key insight: BGE feasible but needs fine-tuning
13. **Markdown** — Small-fold caveats
14. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 3 — Phase 0 Baselines (14 cells, 3 figures)`

---

## Task 7: Section 4 — Base Model Selection (10 cells, 2 figures)

- [ ] Add narrative: per-fold complementarity, selection rationale
- [ ] Add Figure 4.1: Performance matrix heatmap (models × frameworks)
- [ ] Add Figure 4.2: Embedding space t-SNE (Plotly interactive, ~500 points)
- [ ] Add t-SNE caveat, Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — Section header + "Per-fold complementarity"
2. **Code** — Build performance matrix from Phase 0 per-fold data
3. **Code** — Figure 4.1: Heatmap (seaborn/matplotlib, models × frameworks, colored by hit@1)
4. **Markdown** — BGE-large selection rationale (highest aggregate, most consistent)
5. **Code** — Load BASE (pre-fine-tuning) embeddings for t-SNE
   ```python
   # Use BASE BGE embeddings, not deployment (fine-tuned) embeddings.
   # Section 4 is about model SELECTION, before fine-tuning happened.
   base_path = RESULTS_DIR / "phase1b" / "base_bge_embeddings.npz"
   base_data = np.load(str(base_path), allow_pickle=False)
   ctrl_emb = base_data["control_embeddings"]
   ctrl_ids = base_data["control_ids"]
   
   # Subsample to ~500 controls for Plotly performance
   rng = np.random.RandomState(42)
   n_sample = min(500, len(ctrl_emb))
   idx = rng.choice(len(ctrl_emb), n_sample, replace=False)
   sample_emb = ctrl_emb[idx]
   sample_ids = ctrl_ids[idx]
   
   # t-SNE
   tsne = TSNE(n_components=2, perplexity=30, random_state=42)
   coords = tsne.fit_transform(sample_emb)
   ```
6. **Code** — Figure 4.2: Plotly scatter with hover showing control text
7. **Markdown** — t-SNE caveat: "distances between clusters are not meaningful"
8. **Markdown** — What the clusters reveal
9. **Markdown** — Selection decision summary
10. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 4 — Base Model Selection (10 cells, 2 figures)`

---

## Task 8: Section 5 — Contrastive Fine-Tuning (12 cells, 3 figures)

- [ ] Add narrative: MNRL, LoRA, text-aware batching
- [ ] Add Figure 5.1: Training loss curves across folds
- [ ] Add Figure 5.2: Before/after t-SNE (base BGE vs fine-tuned)
- [ ] Add Figure 5.3: Negative sampling distribution (computed)
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "Teaching the Model" + contrastive training explanation
2. **Markdown** — MNRL, LoRA rank 16, text-aware batching (plain-language explanations)
3. **Code** — Load training logs for all folds
   ```python
   FOLDS = [
       "fold_MITRE_ATLAS", "fold_NIST_AI_100-2", "fold_OWASP_AI_Exchange",
       "fold_OWASP_Top10_for_LLM", "fold_OWASP_Top10_for_ML",
   ]
   fold_logs = {}
   for fold in FOLDS:
       fold_logs[fold] = load_training_logs("phase1b_textaware", fold)
   ```
4. **Code** — Figure 5.1: Loss curves (matplotlib, one line per fold)
   ```python
   fig, ax = plt.subplots(figsize=(10, 6))
   for i, (fold, logs) in enumerate(fold_logs.items()):
       steps = [l["step"] for l in logs if "loss" in l]
       losses = [l["loss"] for l in logs if "loss" in l]
       label = fold.replace("fold_", "").replace("_", " ")
       ax.plot(steps, losses, color=OKABE_ITO[i], label=label, alpha=0.8)
   ax.legend(fontsize=9)
   fig_num = fig_counter.next(5)
   style_axes(ax, "Training Loss by Fold", "Step", "Loss", fig_num)
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
5. **Markdown** — Training dynamics interpretation (convergence, overfitting signs)
6. **Code** — Load base BGE embeddings + deployment embeddings for before/after
   ```python
   base_data = np.load(str(PHASE1B_DIR / "base_bge_embeddings.npz"), allow_pickle=False)
   base_ctrl_emb = base_data["control_embeddings"]
   
   hub_emb, fine_ctrl_emb, hub_ids, ctrl_ids = load_deployment_embeddings()
   
   # Subsample same indices for both
   rng = np.random.RandomState(42)
   n_sample = min(500, len(base_ctrl_emb))
   idx = rng.choice(len(base_ctrl_emb), n_sample, replace=False)
   ```
7. **Code** — Figure 5.2: Side-by-side t-SNE (before/after fine-tuning)
   ```python
   tsne_before = TSNE(n_components=2, perplexity=30, random_state=42)
   coords_before = tsne_before.fit_transform(base_ctrl_emb[idx])
   
   tsne_after = TSNE(n_components=2, perplexity=30, random_state=42)
   coords_after = tsne_after.fit_transform(fine_ctrl_emb[idx])
   
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
   # Color by framework prefix from control_ids
   # ...
   fig_num = fig_counter.next(5)
   fig.suptitle(f"{fig_num}: Embedding Space — Before vs After Fine-Tuning", fontsize=13, fontweight="bold")
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
8. **Markdown** — What the before/after shows
9. **Code** — Figure 5.3: Negative sampling distribution (computed from batch size)
   ```python
   # MNRL uses in-batch negatives: batch_size - 1 negatives per positive
   batch_size = 64
   n_negatives = batch_size - 1
   
   fig, ax = plt.subplots(figsize=(8, 5))
   # ... bar chart showing effective negative count by batch position
   fig_num = fig_counter.next(5)
   style_axes(ax, "In-Batch Negatives per Positive Pair", "Batch size", "Effective negatives", fig_num)
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
10. **Markdown** — Why batch composition matters
11. **Markdown** — Key technical decisions summary
12. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 5 — Contrastive Fine-Tuning (12 cells, 3 figures)`

---

## Task 9: Section 6 — Ablation Analysis (10 cells, 2 figures)

- [ ] Add narrative: ablation approach (zero-shot scope), hierarchy paths +7.6%, descriptions -2.1%
- [ ] Add Figure 6.1: Forest plot with 95% CIs
- [ ] Add Figure 6.2: Interaction heatmap
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "What actually mattered" + ablation methodology
2. **Markdown** — Important scope note: ablation on zero-shot model, not fine-tuned
3. **Code** — Load ablation data from Phase 0
   ```python
   paths_bge = load_phase0_experiment("exp3_hierarchy_paths_bge")
   descs = load_phase0_experiment("exp4_hub_descriptions")
   
   # exp3 stores PRE-COMPUTED deltas (not absolute values)
   # Key structure: models["bge-large-v1.5"]["deltas_all_198"]["hit_at_1"]
   # with fields: delta_mean, ci_low, ci_high (NOT "mean")
   paths_delta = paths_bge["models"]["bge-large-v1.5"]["deltas_all_198"]["hit_at_1"]
   delta_paths = paths_delta["delta_mean"]      # +0.076
   ci_low_paths = paths_delta["ci_low"]          # 0.015
   ci_high_paths = paths_delta["ci_high"]        # 0.136
   
   # exp4 uses "deltas_subset" (evaluated on described-hub subset only)
   descs_delta = descs["models"]["bge-large-v1.5"]["deltas_subset"]["hit_at_1"]
   delta_descs = descs_delta["delta_mean"]       # -0.048
   ci_low_descs = descs_delta["ci_low"]           # -0.124
   ci_high_descs = descs_delta["ci_high"]         # 0.028
   ```
4. **Code** — Figure 6.1: Ablation impact chart (matplotlib, delta ± CI per factor)
   ```python
   fig, ax = plt.subplots(figsize=(10, 5))
   # Each row: factor name, delta, ci_low, ci_high
   factors = [
       ("Hierarchy paths", delta_paths, ci_low_paths, ci_high_paths),
       ("Hub descriptions", delta_descs, ci_low_descs, ci_high_descs),
   ]
   for i, (name, delta, lo, hi) in enumerate(factors):
       ax.errorbar(delta, i, xerr=[[delta-lo], [hi-delta]], fmt='o',
                    color=OKABE_ITO[0] if delta > 0 else OKABE_ITO[5],
                    capsize=5, markersize=8)
       ax.text(delta + 0.01, i, f"{delta:+.3f}", va="center")
   ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
   ax.set_yticks(range(len(factors)))
   ax.set_yticklabels([f[0] for f in factors])
   fig_num = fig_counter.next(6)
   style_axes(ax, "Zero-Shot Ablation: Feature Impact on Hit@1", "Δ Hit@1", "", fig_num)
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
5. **Markdown** — Interpretation: "Structure > prose"
6. **Code** — Figure 6.2: Interaction heatmap (hierarchy paths × descriptions × model)
7. **Markdown** — "Wide CI means the true effect could be as small as 1.5%"
8. **Markdown** — Implications for fine-tuning design
9. **Markdown** — Dead ends: what we tried that didn't work
10. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 6 — Ablation Analysis (10 cells, 2 figures)`

---

## Task 10: Section 7 — Hub Firewall (10 cells, 2 figures)

- [ ] Add narrative: LOFO, firewall mechanism, what happens without it
- [ ] Add Figure 7.1: Firewalled vs unfirewalled per-framework comparison
- [ ] Add Figure 7.2: Leakage magnitude per hub
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "The Hub Firewall — Honest Evaluation"
2. **Markdown** — Why LOFO matters (information leakage explanation)
3. **Code** — Load firewalled and unfirewalled baselines
   ```python
   firewalled = load_firewalled_baseline()
   unfirewalled_bge = load_phase0_experiment("exp1_embedding_baseline_bge")
   
   # Firewalled per-fold from aggregate_metrics.json: per_fold dict
   fw_perfold = firewalled["per_fold"]
   
   # Unfirewalled per-fold from exp1: per_fold list
   unfw_perfold = {
       item["framework"]: item["metrics"]["hit_at_1"]
       for item in unfirewalled_bge["models"]["bge-large-v1.5"]["per_fold"]
   }
   ```
4. **Code** — Figure 7.1: Grouped bar chart (firewalled vs unfirewalled)
5. **Markdown** — "Without the firewall, ATLAS hit@1 jumps from X to Y"
6. **Markdown** — The firewall mechanism explained
7. **Code** — Figure 7.2: Leakage magnitude per hub (derived: unfirewalled – firewalled)
8. **Markdown** — Why this matters for trust
9. **Markdown** — How we ensure the firewall in code
10. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 7 — Hub Firewall (10 cells, 2 figures)`

---

## Task 11: Section 8 — Final Results (14 cells, 4 figures)

- [ ] Add narrative: end-to-end and controlled comparison, per-fold deep dive, small-fold caveats
- [ ] Add Figure 8.1: Baseline vs fine-tuned grouped bar with CIs
- [ ] Add Figure 8.2: Delta waterfall
- [ ] Add Figure 8.3: Full metrics table (hit@1, hit@5, MRR, NDCG@10)
- [ ] Add Figure 8.4: Bootstrap CI comparison (fine-tuned vs zero-shot vs Opus)
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "The Honest Picture" + two ways to measure improvement
2. **Code** — Load all per-fold metrics (firewalled baseline + fine-tuned)
   ```python
   finetuned_metrics = {}
   for fold in FOLDS:
       finetuned_metrics[fold] = load_fold_metrics("phase1b_textaware", fold)
   
   baseline_metrics = firewalled["per_fold"]
   ```
3. **Code** — Figure 8.1: Grouped bar chart (baseline vs fine-tuned per fold, with CIs)
4. **Markdown** — Per-fold deep dive: NIST (+0.322), OWASP AI Exchange (+0.143), ATLAS (+0.006)
5. **Code** — Figure 8.2: Waterfall chart showing per-fold deltas
6. **Markdown** — Small-fold caveat: LLM Top 10 (n=6), ML Top 10 (n=7)
7. **Code** — Build full metrics table
   ```python
   # Columns: fold, hit@1, hit@5, MRR, NDCG@10 for both baseline and finetuned
   ```
8. **Code** — Figure 8.3: Heatmap or styled table of all metrics
9. **Markdown** — Multi-hub evaluation note
10. **Markdown** — Opus comparison: "The fine-tuned model (hit@1≈0.537, firewalled LOFO) achieves comparable performance to Opus zero-shot (0.553, unfirewalled) at 1/1000th the cost — and under a stricter evaluation protocol where the hub firewall prevents information leakage. These numbers are not directly comparable: firewalling makes the task harder, so the fine-tuned model's honest 0.537 may actually represent stronger performance than Opus's 0.553."
11. **Code** — Figure 8.4: Forest plot comparing fine-tuned vs zero-shot vs Opus CIs
12. **Markdown** — Bootstrap methodology note
13. **Markdown** — The headline: +52% relative improvement end-to-end
14. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 8 — Final Results (14 cells, 4 figures)`

---

## Task 12: Section 9 — Error Analysis (12 cells, 3 figures)

- [ ] Add narrative: ATLAS deep dive, error taxonomy, attractor hubs, per-control examples
- [ ] Add Figure 9.1: Error analysis scatter (Plotly interactive, ~500 points)
- [ ] Add Figure 9.2: Hub confusion matrix heatmap
- [ ] Add Figure 9.3: Similarity distribution (correct vs errors)
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "Where the Model Gets It Wrong" + ATLAS deep dive
2. **Markdown** — Error taxonomy definition (same-parent / same-grandparent / unrelated-subtree)
3. **Code** — Load review export + hierarchy for error classification
   ```python
   review = load_review_export()
   hierarchy = load_cre_hierarchy()
   hubs = hierarchy["hubs"]
   
   predictions = review["predictions"]
   # Filter: real predictions only (id >= 0 excludes 20 calibration items)
   # review_export fields: assigned_hub_id (model prediction), reviewer_hub_id (expert correction),
   # decision ("accepted"/"reassigned"), raw_similarity, confidence, control_text, framework_id
   errors = [p for p in predictions if p.get("id", 0) >= 0 and p.get("decision") == "reassigned"]
   correct = [p for p in predictions if p.get("id", 0) >= 0 and p.get("decision") == "accepted"]
   
   print(f"Accepted: {len(correct)}, Reassigned: {len(errors)}")
   ```
4. **Code** — Classify errors by taxonomy (same-parent / same-grandparent / unrelated-subtree)
   ```python
   def classify_error(error: dict, hubs: dict) -> str:
       """Classify reassignment by hub relationship in CRE hierarchy."""
       model_hub = error["assigned_hub_id"]
       expert_hub = error["reviewer_hub_id"]
       
       def get_ancestors(hub_id: str) -> list[str]:
           path = []
           current = hub_id
           while current and current in hubs:
               path.append(current)
               current = hubs[current].get("parent_id") or None
           return path
       
       model_ancestors = get_ancestors(model_hub)
       expert_ancestors = get_ancestors(expert_hub)
       
       # Same parent = siblings in the tree
       model_parent = model_ancestors[1] if len(model_ancestors) > 1 else None
       expert_parent = expert_ancestors[1] if len(expert_ancestors) > 1 else None
       if model_parent and model_parent == expert_parent:
           return "same-parent"
       
       # Same grandparent = cousins
       model_gp = model_ancestors[2] if len(model_ancestors) > 2 else None
       expert_gp = expert_ancestors[2] if len(expert_ancestors) > 2 else None
       if model_gp and model_gp == expert_gp:
           return "same-grandparent"
       
       return "unrelated-subtree"
   
   error_types = [classify_error(e, hubs) for e in errors]
   from collections import Counter
   taxonomy = Counter(error_types)
   print(f"Error taxonomy: {dict(taxonomy)}")
   ```
5. **Code** — Figure 9.1: Plotly scatter (t-SNE of errors + subsampled correct, hover shows details)
   ```python
   # Subsample: keep ALL errors (~196), subsample correct to ~300
   # Total ~500 points for Plotly budget
   rng = np.random.RandomState(42)
   n_correct_sample = min(300, len(correct))
   correct_sample = [correct[i] for i in rng.choice(len(correct), n_correct_sample, replace=False)]
   all_items = errors + correct_sample
   labels = ["error"] * len(errors) + ["correct"] * len(correct_sample)
   
   # Load deployment embeddings for t-SNE
   hub_emb, ctrl_emb, hub_ids, ctrl_ids = load_deployment_embeddings()
   # ... match control_ids to items, compute t-SNE, plot
   ```
6. **Markdown** — ATLAS: "77% unrelated-subtree errors — confused at top level"
7. **Code** — Figure 9.2: Hub confusion matrix (top confused pairs, seaborn heatmap)
   ```python
   # Build confusion pairs from reassigned items
   confusion_pairs = [(e["assigned_hub_id"], e["reviewer_hub_id"]) for e in errors]
   pair_counts = Counter(confusion_pairs)
   # Show top 10 most confused hub pairs
   ```
8. **Markdown** — Attractor hubs explanation
9. **Code** — Figure 9.3: Similarity distribution (KDE: correct vs errors)
   ```python
   correct_sims = [p["raw_similarity"] for p in correct]
   error_sims = [p["raw_similarity"] for p in errors]
   # KDE plot comparing distributions
   ```
10. **Markdown** — Per-control examples (ATLAS: "Validate AI Model" case study)
11. **Markdown** — What this means for improvement
12. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 9 — Error Analysis (12 cells, 3 figures)`

---

## Task 13: Section 10 — Calibration (10 cells, 3 figures)

- [ ] Add narrative: raw scores ≠ probabilities, temperature scaling, ECE, OOD detection
- [ ] Add Figure 10.1: Reliability diagram (before/after)
- [ ] Add Figure 10.2: Confidence histogram
- [ ] Add Figure 10.3: OOD separation
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "The model outputs scores that aren't probabilities"
2. **Code** — Load calibration data
   ```python
   cal = load_calibration_data()
   temperature = cal["temperature"]["temperature"]  # 0.0738
   ece = cal["ece"]["ece"]                           # 0.0793
   ece_ci = cal["ece"]["ece_ci"]                     # {ci_high: 0.111, ci_low: 0.049}
   ood = cal["ood"]                                   # {threshold: 0.568, separation_rate: 0.967}
   ```
3. **Code** — Figure 10.1: Reliability diagram (matplotlib, diagonal = perfect)
4. **Markdown** — Temperature = 0.074, "like recalibrating a thermometer"
5. **Code** — Figure 10.2: Confidence histogram
6. **Markdown** — ECE = 0.079 (95% CI: 0.049–0.111), "upper CI crosses 0.10 — marginal"
7. **Code** — Figure 10.3: OOD separation (two distributions + threshold line)
8. **Markdown** — OOD: threshold = 0.568, 96.7% of true OOD caught
9. **Markdown** — What this means: "a '70% confident' prediction is actually right ~70% of the time"
10. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 10 — Calibration (10 cells, 3 figures)`

---

## Task 14: Section 11 — Human Review (12 cells, 3 figures)

- [ ] Add narrative: review process, results, per-framework breakdown, calibration quality, single-reviewer limitation
- [ ] Add Figure 11.1: Per-framework acceptance rate bar chart
- [ ] Add Figure 11.2: Reassigned items flow (Sankey or alluvial)
- [ ] Add Figure 11.3: Calibration item agreement breakdown
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — "What the Expert Found" + review process description
2. **Code** — Load review metrics
   ```python
   review = load_review_metrics()
   overall = review["overall"]  # {accepted: 680, accepted_rate: 77.4, reassigned: 196, rejected: 2}
   per_fw = review["per_framework"]
   quality = review["reviewer_quality"]  # {agreed: 13, disagreements: [...], quality_score: 0.65}
   ```
3. **Markdown** — Headline results: 680 accepted (77.4%), 196 reassigned (22.3%), 2 rejected (0.2%)
4. **Code** — Figure 11.1: Per-framework acceptance rate (horizontal bar, colored by tier)
5. **Markdown** — Per-framework story: CSA AICM 99%, EU AI Act 100%, AIUC-1 29%, CoSAI 45%
6. **Code** — Figure 11.2: Reassigned items — from predicted hub to expert-chosen hub
7. **Markdown** — Calibration quality: 13/20 agreed, 7/20 reassigned, 0 rejected
8. **Code** — Compute Cohen's κ and Figure 11.3: Agreement breakdown
   ```python
   # Cohen's κ against naive baseline (77.4% overall acceptance)
   # Expected: ~negative, because reviewer was MORE critical of calibration items
   n_cal = 20
   agreed = 13
   base_rate = overall["accepted_rate"] / 100  # 0.774
   
   # κ = (p_o - p_e) / (1 - p_e)
   p_o = agreed / n_cal  # 0.65
   p_e = base_rate  # 0.774 (expected agreement if independent)
   kappa = (p_o - p_e) / (1 - p_e)  # negative
   ```
9. **Markdown** — Cohen's κ ≈ -0.55: "κ is a poor metric for multi-hub tasks" + nuance
10. **Markdown** — Single-reviewer limitation
11. **Markdown** — Trust levels by framework category
12. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 11 — Human Review (12 cells, 3 figures)`

---

## Task 15: Section 12 — CLI Tutorial (18 cells, 0 figures)

- [ ] Add three workflows with real CLI commands
- [ ] Workflow A: assign + hierarchy + compare
- [ ] Workflow B: prepare + validate + ingest
- [ ] Workflow C: export + HuggingFace load_dataset (markdown only)
- [ ] Commands that modify state shown as markdown code blocks, not executed

### Implementation

**Key cells:**

1. **Markdown** — "Enough theory. Let's use the tool." + transition
2. **Markdown** — Workflow A header: "I have a control, what hub does it map to?"
3. **Code (shell)** — `!python -m tract.cli assign "Implement multi-factor authentication for all privileged accounts"`
4. **Markdown** — Interpret the assign output
5. **Code (shell)** — `!python -m tract.cli hierarchy --hub <hub_id>` (use a realistic hub ID)
6. **Markdown** — Interpret the hierarchy output
7. **Code (shell)** — `!python -m tract.cli compare --framework nist_800_53 --framework iso_27001`
8. **Markdown** — Interpret comparison results
9. **Markdown** — Workflow B header: "I have a new framework to onboard"
10. **Code (shell)** — `!python -m tract.cli prepare --file examples/sample_framework.csv --framework-id sample_fw --name "Sample Framework" --format csv --output /tmp/demo.json`
11. **Markdown** — Interpret prepare output
12. **Code (shell)** — `!python -m tract.cli validate --file /tmp/demo.json`
13. **Markdown** — Interpret validation
14. **Code (shell)** — `!python -m tract.cli ingest --file /tmp/demo.json`
15. **Markdown** — Interpret ingest
16. **Markdown** — Workflow C header: "I want to explore the published crosswalk"
17. **Code (shell)** — `!python -m tract.cli export --format jsonl --framework mitre_atlas`
18. **Markdown** — HuggingFace `load_dataset()` shown as markdown code block (not executed — network call) + Plain English blockquote

Note: Commands that modify state (`bridge --commit`, `publish-hf`, `publish-dataset`, `import-ground-truth`, `review-import`) are documented as markdown code blocks, not executed.

### Commit

`feat: add Section 12 — CLI Tutorial (18 cells, 3 workflows)`

---

## Task 16: Section 13 — What We Built and What We Learned (16 cells, 3 figures)

- [ ] Add narrative: journey recap, master results table, trust levels, limitations, future work
- [ ] Add Figure 13.1: Journey timeline (Plotly interactive)
- [ ] Add Figure 13.2: Master comparison table (all approaches × all metrics)
- [ ] Add Figure 13.3: Trust level guide heatmap
- [ ] Add Plain English blockquote

### Implementation

**Key cells:**

1. **Markdown** — Journey recap (1 page, each step in 1-2 sentences)
2. **Code** — Build master results table from data files (no hardcoded values)
   ```python
   # Load all results programmatically
   deberta = load_phase0_experiment("exp1_embedding_baseline_deberta")
   gte = load_phase0_experiment("exp1_embedding_baseline_gte")
   bge = load_phase0_experiment("exp1_embedding_baseline_bge")
   knn = load_phase0_experiment("exp5_knn_baseline")
   fewshot = load_phase0_experiment("exp6_fewshot_sonnet")
   opus = load_phase0_experiment("exp2_llm_probe")
   firewalled = load_firewalled_baseline()
   
   master_table = [
       {"approach": "DeBERTa-v3-NLI", "hit1": deberta["models"]["deberta-v3-nli"]["all_198"]["hit_at_1"]["mean"], "lesson": "NLI ≠ semantic similarity"},
       {"approach": "kNN (k=5)", "hit1": knn["k_values"]["k5"]["all"]["hit_at_1"]["mean"], "lesson": "Neighborhood helps but insufficient"},
       {"approach": "GTE-large", "hit1": gte["models"]["gte-large-v1.5"]["all_198"]["hit_at_1"]["mean"], "lesson": "Decent but inconsistent across folds"},
       {"approach": "BGE-large-v1.5 (zero-shot)", "hit1": bge["models"]["bge-large-v1.5"]["all_198"]["hit_at_1"]["mean"], "lesson": "Best off-the-shelf embedding model"},
       {"approach": "Few-shot Sonnet (desc)", "hit1": fewshot["variants"]["sonnet-desc"]["all"]["hit_at_1"]["mean"], "lesson": "In-context learning works but expensive"},
       {"approach": "BGE firewalled baseline", "hit1": firewalled["aggregate_hit1"]["mean"], "lesson": "Firewall actually HELPS aggregate"},
       {"approach": "Claude Opus (zero-shot)", "hit1": opus["all_198"]["hit_at_1"]["mean"], "lesson": "LLM ceiling — $0.60/control"},
       {"approach": "BGE fine-tuned (TRACT)", "hit1": _load_json(PHASE1B_DIR / "phase1b_textaware" / "corrected_metrics.json")["aggregate_hit1"]["mean"], "lesson": "Contrastive training + LOFO = honest gains"},
   ]
   ```
3. **Code** — Figure 13.2: Master comparison table (matplotlib table or heatmap)
4. **Markdown** — "Should I trust this crosswalk?" (nuanced answer by framework category)
5. **Code** — Figure 13.3: Trust level guide (framework × trust tier heatmap)
6. **Markdown** — Known limitations (4 items, concrete)
7. **Markdown** — What would make this better (4 items)
8. **Code** — Build journey milestones for Plotly timeline
   ```python
   milestones = [
       {"date": "2026-04-27", "event": "Project start — PRD written", "outcome": "success"},
       {"date": "2026-04-27", "event": "Data preparation — 9 parsers", "outcome": "success"},
       {"date": "2026-04-28", "event": "DeBERTa-v3-NLI: hit@1=0.000", "outcome": "failure"},
       {"date": "2026-04-28", "event": "BGE-large zero-shot: hit@1=0.348", "outcome": "success"},
       {"date": "2026-04-28", "event": "Opus ceiling: hit@1=0.553 (unfirewalled)", "outcome": "mixed"},
       {"date": "2026-04-28", "event": "Hierarchy paths: +7.6%", "outcome": "success"},
       {"date": "2026-04-29", "event": "Contrastive fine-tuning launched", "outcome": "success"},
       {"date": "2026-04-29", "event": "LOFO evaluation: hit@1=0.537 (firewalled)", "outcome": "success"},
       {"date": "2026-04-30", "event": "Calibration: T=0.074, ECE=0.079", "outcome": "success"},
       {"date": "2026-04-30", "event": "OOD detection: 96.7% separation", "outcome": "success"},
       {"date": "2026-05-01", "event": "Framework prep pipeline", "outcome": "success"},
       {"date": "2026-05-02", "event": "HuggingFace model published", "outcome": "success"},
       {"date": "2026-05-02", "event": "Bridge analysis: 46/63 accepted", "outcome": "mixed"},
       {"date": "2026-05-03", "event": "Expert review: 77.4% accepted", "outcome": "success"},
       {"date": "2026-05-03", "event": "Dataset published to HuggingFace", "outcome": "success"},
   ]
   ```
9. **Code** — Figure 13.1: Plotly timeline (interactive, color-coded by outcome)
10. **Markdown** — The real lesson: "assignment paradigm + honest evaluation = trustworthy crosswalk"
11. **Markdown** — For practitioners: what to do next
12. **Markdown** — For researchers: open questions
13. **Markdown** — Acknowledgments
14. **Markdown** — Citation (BibTeX)
15. **Markdown** — License (CC-BY-SA-4.0 for the notebook itself)
16. **Markdown** — Plain English blockquote

### Commit

`feat: add Section 13 — What We Built and What We Learned (16 cells, 3 figures)`

---

## Task 17: Appendices A & B (8 cells, 2 figures)

- [ ] Add Appendix A: Experiment Log (4 cells, 1 figure)
- [ ] Add Appendix B: Visual Style Guide (4 cells, 1 figure)

### Implementation

**Appendix A cells:**

1. **Markdown** — "Appendix A: Experiment Log" header
2. **Code** — Build experiment log table from data files (load dynamically, do not hardcode)
   ```python
   # Load all experiment results programmatically
   bge = load_phase0_experiment("exp1_embedding_baseline_bge")
   gte = load_phase0_experiment("exp1_embedding_baseline_gte")
   deberta = load_phase0_experiment("exp1_embedding_baseline_deberta")
   opus = load_phase0_experiment("exp2_llm_probe")
   knn = load_phase0_experiment("exp5_knn_baseline")
   fewshot = load_phase0_experiment("exp6_fewshot_sonnet")
   
   experiments = [
       {"run": "exp1_bge", "model": "BGE-large-v1.5",
        "hit1": bge["models"]["bge-large-v1.5"]["all_198"]["hit_at_1"]["mean"],
        "params": "1.3B", "notes": "Zero-shot"},
       {"run": "exp1_gte", "model": "GTE-large-v1.5",
        "hit1": gte["models"]["gte-large-v1.5"]["all_198"]["hit_at_1"]["mean"],
        "params": "335M", "notes": "Zero-shot"},
       {"run": "exp1_deberta", "model": "DeBERTa-v3-NLI",
        "hit1": deberta["models"]["deberta-v3-nli"]["all_198"]["hit_at_1"]["mean"],
        "params": "304M", "notes": "NLI — total failure"},
       {"run": "exp2_opus", "model": "Claude Opus",
        "hit1": opus["all_198"]["hit_at_1"]["mean"],
        "params": "N/A", "notes": "Zero-shot (unfirewalled)"},
       {"run": "exp5_knn", "model": "kNN (k=5)",
        "hit1": knn["k_values"]["k5"]["all"]["hit_at_1"]["mean"],
        "params": "N/A", "notes": "Zero-shot"},
       {"run": "exp6_sonnet_desc", "model": "Few-shot Sonnet (desc)",
        "hit1": fewshot["variants"]["sonnet-desc"]["all"]["hit_at_1"]["mean"],
        "params": "N/A", "notes": "3-shot with descriptions"},
       # ... Phase 1B
       {"run": "phase1b_textaware", "model": "BGE-large + LoRA",
        "hit1": _load_json(PHASE1B_DIR / "phase1b_textaware" / "corrected_metrics.json")["aggregate_hit1"]["mean"],
        "params": "+4.2M", "notes": "Final deployment (firewalled LOFO)"},
   ]
   ```
3. **Code** — Figure A.1: Full experiment comparison table (matplotlib)
4. **Markdown** — Notes on reproducibility (seeds, git SHAs)

**Appendix B cells:**

1. **Markdown** — "Appendix B: Visual Style Guide" header
2. **Code** — Palette swatches + accessibility notes
   ```python
   fig, ax = plt.subplots(figsize=(10, 2))
   for i, (color, name) in enumerate(zip(OKABE_ITO, [
       "Orange", "Sky Blue", "Bluish Green", "Yellow",
       "Blue", "Vermilion", "Reddish Purple", "Grey",
   ])):
       ax.barh(0, 1, left=i, color=color, edgecolor="white", linewidth=2)
       ax.text(i + 0.5, 0, f"{name}\n{color}", ha="center", va="center", fontsize=8)
   ax.set_xlim(0, 8)
   ax.set_ylim(-0.5, 0.5)
   ax.axis("off")
   fig_num = "Figure B.1"
   ax.set_title(f"{fig_num}: Okabe-Ito Palette (colorblind-safe)", fontsize=12, fontweight="bold")
   plt.tight_layout()
   plt.show()
   plt.close(fig)
   ```
3. **Markdown** — Font choices, axis styling conventions
4. **Markdown** — Citation: Okabe & Ito (2008), "Color Universal Design"

### Commit

`feat: add Appendices A & B — Experiment Log and Visual Style Guide`

---

## Task 18: Final Validation

- [ ] Run all tests (`pytest tests/test_nb_helpers.py`)
- [ ] Verify notebook cell count ≥ 128 (target ~168)
- [ ] Verify figure count ≥ 24 (target ~35)
- [ ] Verify markdown-to-code ratio ≥ 1.5:1
- [ ] Verify notebook file size < 5 MB
- [ ] Run notebook top-to-bottom (kernel restart + run all)
- [ ] Check no absolute system paths in output cells
- [ ] Verify all 5 Plotly figures have PNG fallbacks
- [ ] Verify Okabe-Ito palette used consistently (no rainbow/jet)
- [ ] Verify every section has Plain English blockquote
- [ ] Verify `plt.close(fig)` after every matplotlib figure

### Validation Script

```bash
# Test suite
python -m pytest tests/test_nb_helpers.py -q

# Cell count
python -c "
import nbformat
nb = nbformat.read('notebooks/tract_experimental_narrative.ipynb', as_version=4)
md = sum(1 for c in nb.cells if c.cell_type == 'markdown')
code = sum(1 for c in nb.cells if c.cell_type == 'code')
total = len(nb.cells)
ratio = md / max(code, 1)
print(f'Total cells: {total} (target ≥ 128)')
print(f'Markdown: {md}, Code: {code}, Ratio: {ratio:.1f}:1 (target ≥ 1.5:1)')
assert total >= 128, f'Too few cells: {total}'
assert ratio >= 1.5, f'Ratio too low: {ratio}'
print('✓ Cell count and ratio OK')
"

# File size
python -c "
from pathlib import Path
size_mb = Path('notebooks/tract_experimental_narrative.ipynb').stat().st_size / 1e6
print(f'Notebook size: {size_mb:.1f} MB (target < 5 MB)')
assert size_mb < 5, f'Too large: {size_mb:.1f} MB'
print('✓ File size OK')
"

# No absolute paths in output
python -c "
import nbformat, json
nb = nbformat.read('notebooks/tract_experimental_narrative.ipynb', as_version=4)
for i, cell in enumerate(nb.cells):
    for output in cell.get('outputs', []):
        text = json.dumps(output)
        assert '/home/rock/' not in text, f'Cell {i} has absolute path in output'
print('✓ No absolute paths in output')
"

# Run notebook (after all sections added)
jupyter nbconvert --to notebook --execute notebooks/tract_experimental_narrative.ipynb --output /tmp/tract_test_run.ipynb --ExecutePreprocessor.timeout=600
```

### Commit

`chore: validate Phase 3B notebook — all checks passing`

---

## Implementation Notes

### nbformat Workflow

Each task (4–17) appends cells to the existing notebook file:

```python
import nbformat

nb = nbformat.read("notebooks/tract_experimental_narrative.ipynb", as_version=4)

# Add cells
nb.cells.append(nbformat.v4.new_markdown_cell("## Section N: Title"))
nb.cells.append(nbformat.v4.new_code_cell("# code here"))

nbformat.write(nb, "notebooks/tract_experimental_narrative.ipynb")
```

### Data Structure Reference

Frequently-accessed data structures (verified empirically during adversarial review):

- **exp1 per_fold**: `list[{framework: str, metrics: {hit_at_1: float, ...}, n_items: int}]`
- **exp1 model keys**: `"bge-large-v1.5"`, `"gte-large-v1.5"`, `"deberta-v3-nli"` (NOT `"deberta-v3-large-nli"` or `"gte-large"`)
- **exp2 (Opus)**: `all_198.hit_at_1 = {ci_high, ci_low, mean}` where mean=0.553 (unfirewalled, n=197). NOTE: summary.json reports 0.465 from partial evaluation (99/197 controls)
- **exp3 ablation**: `models["bge-large-v1.5"]["deltas_all_198"]["hit_at_1"]` = `{ci_high, ci_low, delta_mean}` — note `delta_mean` NOT `mean`
- **exp4 descriptions**: `models["bge-large-v1.5"]["deltas_subset"]["hit_at_1"]` = `{ci_high, ci_low, delta_mean}` (delta_mean = -0.048)
- **exp6 few-shot**: variant keys are `"sonnet-desc"` and `"sonnet-nodesc"` (NOT `"3shot_with_descriptions"`)
- **firewalled baseline per_fold**: `dict[str, {hit_at_1: float, ...}]` (framework name keys)
- **fold metrics**: `{hit_at_1: float, hit_at_5: float, mrr: float, ndcg_at_10: float}`
- **fold predictions.json**: `list[{control_text, framework, ground_truth_hub_id, predicted_top10}]` — NOT `{correct, predicted_hub, true_hub}`
- **trainer_state log_history**: `list[{epoch: float, loss: float, grad_norm: float, learning_rate: float, step: int}]`
- **review_export.json**: field is `"decision"` NOT `"review_decision"`, values: `"accepted"`, `"reassigned"`, `"rejected"`. Each prediction has: `assigned_hub_id` (model's choice), `reviewer_hub_id` (expert's correction, when reassigned), `raw_similarity`, `confidence`, `control_text`, `control_title`, `framework_id`, `framework_name`, `alternative_hubs` (list), `is_ood`, `in_conformal_set`
- **corrected_metrics.json** (Phase 1B): `aggregate_hit1 = {ci_high, ci_low, mean, n_resamples, n_total}` where mean=0.5374 (NOT 0.531 — the 0.531 was from a different experiment's single fold)
- **calibration items**: `review_export.predictions` entries where `id < 0` (20 items)
- **review_metrics.overall**: `{accepted: 680, accepted_rate: 77.4, reassigned: 196, rejected: 2}`
- **deployment_artifacts.npz**: `hub_embeddings (522,1024), control_embeddings (2802,1024), hub_ids (522,), control_ids (2802,)`
- **cre_hierarchy.json**: root hubs have `parent_id: null` (Python None) — use `hub.get("parent_id") or ""` for Plotly
- **control_id separators**: crosswalk JSONL uses single colon (`aiuc_1:A001.1`), deployment NPZ uses double colon (`aiuc_1::A001.1`) — beware when joining

### CLI Command Syntax (Verified)

```bash
python -m tract.cli assign "control text"
python -m tract.cli hierarchy --hub <hub_id>
python -m tract.cli compare --framework <fw1> --framework <fw2>
python -m tract.cli prepare --file <path> --framework-id <id> --name "Name" --format csv --output <path>
python -m tract.cli validate --file <path>
python -m tract.cli ingest --file <path>
python -m tract.cli export --format jsonl --framework <fw_id>
```

### t-SNE Budget

4 runs × ~14s each = ~55s total on Tegra ARM64:
1. Section 4 (Figure 4.2): base embeddings, ~500 points
2. Section 5 (Figure 5.2 left): base BGE, ~500 points
3. Section 5 (Figure 5.2 right): fine-tuned, ~500 points
4. Section 9 (Figure 9.1): error analysis, ~500 points

### Plotly Data Budget

5 interactive figures, total JSON target < 1.5 MB:
1. Section 1: Sunburst — 522 nodes (small)
2. Section 2: Sankey — framework→hub cluster flows (small)
3. Section 4: t-SNE scatter — ~500 points (moderate)
4. Section 9: Error scatter — ~500 points (moderate)
5. Section 13: Timeline — ~15 milestones (tiny)
