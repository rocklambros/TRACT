"""Result shapes for the calibration fits written into calibration.json.

TRACTPredictor reads these values back at inference time (t_deploy,
ood_threshold, conformal_quantile), so the fits are a contract between the
calibration stage and the served model, not scratch dicts.
"""
from __future__ import annotations

from typing import TypedDict


class TemperatureFit(TypedDict):
    """Grid-search result for a single temperature fit."""

    temperature: float
    nll: float
    grid_min_t: float
    grid_max_t: float


class LofoTemperatureFit(TemperatureFit):
    """Pooled LOFO fit, with the per-fold NLL and weights behind the pooling."""

    per_fold_nll: dict[str, float]
    fold_weights: dict[str, float]


class ThresholdFit(TypedDict):
    """Max-F1 global threshold for multi-label assignment."""

    threshold: float
    f1: float


class OODValidation(TypedDict):
    """Separation check for the OOD threshold against synthetic non-security text."""

    separation_rate: float
    n_below: int
    n_total: int
    threshold: float
    gate_passed: bool


class ECEResult(TypedDict):
    """Expected calibration error with a bootstrap interval."""

    ece: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


class KSTestResult(TypedDict):
    """Kolmogorov-Smirnov comparison of traditional vs AI similarity spreads."""

    ks_statistic: float
    p_value: float
    n_traditional: int
    n_ai: int
