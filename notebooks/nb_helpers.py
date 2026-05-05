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
    """Display a Plotly figure.

    Uses fig.show() which embeds the interactive widget and Plotly JSON
    in the cell output. GitHub, nbviewer, and JupyterLab all render the
    Plotly JSON natively — no separate static PNG needed.
    """
    fig.update_layout(
        title=f"{fig_num}: {title}",
        width=width,
        height=height,
    )
    fig.show()


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
    """Load OpenCRE hierarchy (hubs, roots, label_space)."""
    path = DATA_DIR / "processed" / "cre_hierarchy.json"
    if not path.exists():
        raise FileNotFoundError(f"OpenCRE hierarchy not found: {path}")
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


CANONICAL_EXPORT_DIR = PROJECT_ROOT / "canonical_export"


def load_canonical_snapshot(framework_id: str) -> dict:
    """Load a canonical export snapshot for a specific framework."""
    path = CANONICAL_EXPORT_DIR / framework_id / "snapshot.json"
    if not path.exists():
        raise FileNotFoundError(f"Canonical snapshot not found: {path}")
    return _load_json(path)


def load_canonical_changeset(framework_id: str) -> dict:
    """Load a canonical export changeset for a specific framework."""
    path = CANONICAL_EXPORT_DIR / framework_id / "changeset.json"
    if not path.exists():
        raise FileNotFoundError(f"Canonical changeset not found: {path}")
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
