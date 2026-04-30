"""Tests for OOD detection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


class TestOODThreshold:
    def test_threshold_is_5th_percentile(self) -> None:
        from tract.calibration.ood import compute_ood_threshold

        max_sims = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.85, 0.75, 0.65,
                             0.55, 0.45, 0.35, 0.42, 0.52, 0.62, 0.72, 0.82, 0.88, 0.92])

        threshold = compute_ood_threshold(max_sims)
        expected = float(np.percentile(max_sims, 5))
        assert threshold == pytest.approx(expected)


class TestOODValidation:
    def test_validates_against_synthetic(self) -> None:
        from tract.calibration.ood import validate_ood_threshold

        ood_max_sims = np.array([0.05, 0.03, 0.08, 0.02, 0.04] * 6)
        threshold = 0.20

        result = validate_ood_threshold(ood_max_sims, threshold)
        assert result["separation_rate"] >= 0.9
        assert result["n_below"] == 30
        assert result["n_total"] == 30

    def test_fails_if_ood_above_threshold(self) -> None:
        from tract.calibration.ood import validate_ood_threshold

        ood_max_sims = np.array([0.5, 0.6, 0.7, 0.8, 0.9] * 6)
        threshold = 0.20

        result = validate_ood_threshold(ood_max_sims, threshold)
        assert result["separation_rate"] < 0.9


class TestFlagOOD:
    def test_flags_low_similarity_items(self) -> None:
        from tract.calibration.ood import flag_ood_items

        max_sims = np.array([0.1, 0.5, 0.05, 0.9, 0.02])
        threshold = 0.15

        flags = flag_ood_items(max_sims, threshold)
        assert flags == [True, False, True, False, True]


class TestOODFixture:
    def test_fixture_has_30_texts(self) -> None:
        fixture = Path(__file__).parent / "fixtures" / "ood_synthetic_texts.json"
        texts = json.loads(fixture.read_text(encoding="utf-8"))
        assert len(texts) == 30
        assert all(isinstance(t, str) for t in texts)
