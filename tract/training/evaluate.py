"""LOFO evaluation harness with statistical testing.

Implements:
- Fold-stratified micro-average bootstrap CIs (10K resamples, vectorized)
- Paired bootstrap for method comparisons (vectorized)
- Benjamini-Hochberg FDR control at q=0.10
- Per-framework soft floor checks
- Covered/uncovered hub split metrics
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.phase0.common import (
    ndcg_at_k,
    reciprocal_rank,
    score_predictions,
)
from tract.config import (
    PHASE1B_BOOTSTRAP_CI_LEVEL,
    PHASE1B_BOOTSTRAP_N_RESAMPLES,
    PHASE1B_BOOTSTRAP_SEED,
    PHASE1B_SOFT_FLOOR_LARGE,
    PHASE1B_SOFT_FLOOR_NIST,
)

logger = logging.getLogger(__name__)

LARGE_FOLDS: dict[str, float] = {
    "MITRE ATLAS": PHASE1B_SOFT_FLOOR_LARGE,
    "OWASP AI Exchange": PHASE1B_SOFT_FLOOR_LARGE,
    "NIST AI 100-2": PHASE1B_SOFT_FLOOR_NIST,
}


def rank_hubs_by_similarity(
    query_emb: NDArray[np.floating[Any]],
    hub_embs: NDArray[np.floating[Any]],
    hub_ids: list[str],
) -> list[str]:
    """Rank hub IDs by cosine similarity to query embedding.

    Assumes both query_emb and hub_embs are L2-normalized.
    """
    if query_emb.ndim != 1:
        raise ValueError(f"query_emb must be 1-D, got shape {query_emb.shape}")
    if hub_embs.ndim != 2:
        raise ValueError(f"hub_embs must be 2-D, got shape {hub_embs.shape}")
    if hub_embs.shape[0] != len(hub_ids):
        raise ValueError(
            f"hub_embs rows ({hub_embs.shape[0]}) != hub_ids length ({len(hub_ids)})"
        )

    sims = hub_embs @ query_emb
    ranked_indices = np.argsort(sims)[::-1]
    return [hub_ids[i] for i in ranked_indices]


def _build_fold_index_matrix(
    fold_sizes: list[int],
    n_resamples: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.intp], list[NDArray[np.intp]]]:
    """Pre-generate all bootstrap resampling indices in one shot.

    Returns:
        full_indices: (n_resamples, total_n) index array into concatenated fold values
        per_fold_indices: list of (n_resamples, fold_size) arrays (for paired bootstrap)
    """
    per_fold_indices: list[NDArray[np.intp]] = []
    offset = 0
    for fold_size in fold_sizes:
        fold_idx = rng.integers(0, fold_size, size=(n_resamples, fold_size))
        per_fold_indices.append(fold_idx + offset)
        offset += fold_size

    full_indices = np.concatenate(per_fold_indices, axis=1)
    return full_indices, per_fold_indices


def fold_stratified_bootstrap_ci(
    fold_values: list[NDArray[np.floating[Any]]],
    n_resamples: int = PHASE1B_BOOTSTRAP_N_RESAMPLES,
    ci_level: float = PHASE1B_BOOTSTRAP_CI_LEVEL,
    seed: int = PHASE1B_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Fold-stratified micro-average bootstrap CI (vectorized).

    For each of n_resamples bootstrap replicates:
    1. Resample items with replacement WITHIN each fold (preserving fold sizes)
    2. Concatenate resampled folds
    3. Compute aggregate mean

    All resampling is done via pre-generated index matrices — no Python loops
    over resamples.
    """
    if not fold_values:
        raise ValueError("fold_values must be non-empty")
    for i, fv in enumerate(fold_values):
        if len(fv) == 0:
            raise ValueError(f"Fold {i} is empty")

    rng = np.random.default_rng(seed)
    all_values = np.concatenate(fold_values)
    total_n = len(all_values)
    fold_sizes = [len(fv) for fv in fold_values]

    full_indices, _ = _build_fold_index_matrix(fold_sizes, n_resamples, rng)

    resampled = all_values[full_indices]
    boot_means = resampled.mean(axis=1)

    alpha = (1 - ci_level) / 2

    return {
        "mean": float(np.mean(all_values)),
        "ci_low": float(np.percentile(boot_means, 100 * alpha)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha))),
        "n_total": total_n,
        "n_resamples": n_resamples,
    }


def paired_bootstrap_delta(
    fold_values_a: list[NDArray[np.floating[Any]]],
    fold_values_b: list[NDArray[np.floating[Any]]],
    n_resamples: int = PHASE1B_BOOTSTRAP_N_RESAMPLES,
    ci_level: float = PHASE1B_BOOTSTRAP_CI_LEVEL,
    seed: int = PHASE1B_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Paired bootstrap CI for difference (B - A), vectorized.

    Per-item deltas within each fold, fold-stratified resampling.
    Pairing cancels item-level difficulty for reduced variance.
    """
    if len(fold_values_a) != len(fold_values_b):
        raise ValueError(
            f"Fold count mismatch: {len(fold_values_a)} vs {len(fold_values_b)}"
        )
    for i, (va, vb) in enumerate(zip(fold_values_a, fold_values_b)):
        if len(va) != len(vb):
            raise ValueError(f"Fold {i} size mismatch: {len(va)} vs {len(vb)}")

    rng = np.random.default_rng(seed)

    fold_deltas = [vb - va for va, vb in zip(fold_values_a, fold_values_b)]
    all_deltas = np.concatenate(fold_deltas)
    fold_sizes = [len(d) for d in fold_deltas]

    full_indices, _ = _build_fold_index_matrix(fold_sizes, n_resamples, rng)

    resampled = all_deltas[full_indices]
    boot_delta_means = resampled.mean(axis=1)

    alpha = (1 - ci_level) / 2
    p_value = float(np.mean(boot_delta_means <= 0))

    return {
        "delta_mean": float(np.mean(all_deltas)),
        "ci_low": float(np.percentile(boot_delta_means, 100 * alpha)),
        "ci_high": float(np.percentile(boot_delta_means, 100 * (1 - alpha))),
        "p_value": p_value,
    }


def benjamini_hochberg(
    p_values: list[float],
    q: float = 0.10,
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR control.

    Returns (rejected, adjusted_p_values) where rejected[i] is True if
    hypothesis i is rejected at FDR level q, and adjusted_p_values[i]
    is the BH-adjusted p-value for hypothesis i.
    """
    m = len(p_values)
    if m == 0:
        return [], []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.empty(m)
    adjusted[m - 1] = sorted_p[m - 1]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * m / (i + 1))

    adjusted = np.minimum(adjusted, 1.0)

    result_adjusted = np.empty(m)
    result_adjusted[sorted_indices] = adjusted

    rejected = [bool(p <= q) for p in result_adjusted]
    return rejected, result_adjusted.tolist()


def check_soft_floors(
    per_fold_deltas: dict[str, dict[str, float]],
) -> dict[str, str]:
    """Check per-framework soft floor constraints.

    Only enforced for large folds (ATLAS n=65, OWASP-X n=64, NIST n=45).
    Returns dict of framework -> violation message for any that fail.
    """
    violations: dict[str, str] = {}

    for framework, floor in LARGE_FOLDS.items():
        if framework not in per_fold_deltas:
            continue
        delta_info = per_fold_deltas[framework]
        ci_low = delta_info.get("ci_low", 0.0)
        if ci_low < floor:
            violations[framework] = (
                f"Soft floor violation: CI low={ci_low:.3f} < floor={floor:.3f}"
            )
            logger.warning(
                "SOFT FLOOR VIOLATION: %s — %s", framework, violations[framework]
            )

    return violations


def compute_covered_uncovered_split(
    eval_items: list[Any],
    predictions: list[list[str]],
    training_hub_ids: set[str],
) -> dict[str, dict[str, float]]:
    """Split metrics by covered (hub seen in training) vs uncovered hubs.

    Covered hubs had at least one training example from non-held-out frameworks.
    Uncovered hubs (~385/458) had zero training examples — this is the true
    generalization test.
    """
    covered_preds: list[list[str]] = []
    covered_gt: list[str] = []
    uncovered_preds: list[list[str]] = []
    uncovered_gt: list[str] = []

    for item, pred in zip(eval_items, predictions):
        if item.ground_truth_hub_id in training_hub_ids:
            covered_preds.append(pred)
            covered_gt.append(item.ground_truth_hub_id)
        else:
            uncovered_preds.append(pred)
            uncovered_gt.append(item.ground_truth_hub_id)

    result: dict[str, dict[str, float]] = {}
    if covered_preds:
        result["covered"] = score_predictions(covered_preds, covered_gt)
        result["covered"]["n"] = float(len(covered_preds))
    if uncovered_preds:
        result["uncovered"] = score_predictions(uncovered_preds, uncovered_gt)
        result["uncovered"]["n"] = float(len(uncovered_preds))

    return result


def evaluate_on_fold(
    model: Any,
    eval_items: list[Any],
    hub_ids: list[str],
    hub_texts: dict[str, str],
) -> tuple[dict[str, float], list[list[str]], NDArray[np.floating[Any]]]:
    """Evaluate a model on one LOFO fold.

    Args:
        model: SentenceTransformer (or anything with .encode()).
        eval_items: List of EvalItem with .control_text and .ground_truth_hub_id.
        hub_ids: Ordered list of hub IDs to rank over.
        hub_texts: hub_id -> text representation for encoding.

    Returns:
        (metrics_dict, per_item_ranked_predictions, per_item_hit1_indicators)
    """
    hub_texts_ordered = [hub_texts[hid] for hid in hub_ids]
    hub_embs = model.encode(
        hub_texts_ordered,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=128,
    )

    control_texts = [item.control_text for item in eval_items]
    query_embs = model.encode(
        control_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=128,
    )

    predictions: list[list[str]] = []
    for q_emb in query_embs:
        ranked = rank_hubs_by_similarity(q_emb, hub_embs, hub_ids)
        predictions.append(ranked)

    ground_truth = [item.ground_truth_hub_id for item in eval_items]
    valid_hub_sets = [
        item.valid_hub_ids if hasattr(item, "valid_hub_ids") else frozenset({item.ground_truth_hub_id})
        for item in eval_items
    ]
    metrics = score_predictions(predictions, ground_truth, valid_hub_sets)

    hit1_indicators = np.array(
        [
            1.0 if pred and pred[0] in vs else 0.0
            for pred, vs in zip(predictions, valid_hub_sets)
        ]
    )

    return metrics, predictions, hit1_indicators
