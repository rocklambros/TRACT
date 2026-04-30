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
