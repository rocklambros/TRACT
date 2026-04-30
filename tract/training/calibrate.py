"""Temperature scaling calibration for cosine similarities.

Calibration is performed INSIDE the LOFO loop using the 10% AI
validation split (same split used for early stopping).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


def calibrate_similarities(
    similarities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Convert cosine similarities to calibrated probabilities.

    P(hub_i | control) = exp(sim_i / T) / sum(exp(sim_j / T))
    """
    scaled = similarities / temperature
    return softmax(scaled, axis=1)


def _negative_log_likelihood(
    similarities: np.ndarray,
    ground_truth_indices: np.ndarray,
    temperature: float,
) -> float:
    """Compute NLL for a given temperature."""
    probs = calibrate_similarities(similarities, temperature)
    log_probs = np.log(probs[np.arange(len(ground_truth_indices)), ground_truth_indices] + 1e-10)
    return -float(np.mean(log_probs))


def find_optimal_temperature(
    similarities: np.ndarray,
    ground_truth_indices: np.ndarray,
    t_min: float = 0.01,
    t_max: float = 5.0,
    n_grid: int = 200,
) -> float:
    """Find temperature T that minimizes NLL on validation set via grid search."""
    temperatures = np.linspace(t_min, t_max, n_grid)
    best_t = 1.0
    best_nll = float("inf")

    for t in temperatures:
        nll = _negative_log_likelihood(similarities, ground_truth_indices, t)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    logger.info("Optimal temperature: %.4f (NLL=%.4f, searched %d values in [%.2f, %.2f])",
                best_t, best_nll, n_grid, t_min, t_max)
    return best_t


def find_global_threshold(
    similarities: np.ndarray,
    ground_truth_indices: np.ndarray,
    temperature: float,
    n_thresholds: int = 200,
) -> float:
    """Find global threshold at max-F1 for multi-label assignment."""
    probs = calibrate_similarities(similarities, temperature)

    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5

    for t in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(ground_truth_indices)):
            gt_idx = ground_truth_indices[i]
            for j in range(probs.shape[1]):
                predicted = probs[i, j] >= t
                is_true = j == gt_idx
                if predicted and is_true:
                    tp += 1
                elif predicted and not is_true:
                    fp += 1
                elif not predicted and is_true:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    logger.info("Global threshold: %.4f (F1=%.4f)", best_threshold, best_f1)
    return best_threshold
