"""Tests for review metrics computation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.review.metrics import compute_review_metrics


def _make_prediction(
    id: int,
    framework_id: str = "fw_alpha",
    framework_name: str = "Alpha Framework",
    status: str = "accepted",
    confidence: float = 0.85,
    is_ood: bool = False,
    assigned_hub_id: str = "hub-1",
    reviewer_hub_id: str | None = None,
) -> dict:
    return {
        "id": id,
        "framework_id": framework_id,
        "framework_name": framework_name,
        "status": status,
        "confidence": confidence,
        "is_ood": is_ood,
        "assigned_hub_id": assigned_hub_id,
        "reviewer_hub_id": reviewer_hub_id,
        "reviewer_notes": None,
    }


def _make_review_data(predictions: list[dict]) -> dict:
    return {
        "metadata": {"total_predictions": len(predictions)},
        "predictions": predictions,
    }


class TestOverallRates:
    def test_rates_computation(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, status="accepted"),
            _make_prediction(2, status="accepted"),
            _make_prediction(3, status="rejected"),
            _make_prediction(4, status="reassigned"),
            _make_prediction(5, status="accepted"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        assert result["overall"]["accepted"] == 3
        assert result["overall"]["rejected"] == 1
        assert result["overall"]["reassigned"] == 1
        assert result["overall"]["accepted_rate"] == 60.0
        assert result["overall"]["rejected_rate"] == 20.0
        assert result["overall"]["reassigned_rate"] == 20.0


class TestPerFrameworkBreakdown:
    def test_two_frameworks(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, framework_id="fw_alpha", framework_name="Alpha", status="accepted"),
            _make_prediction(2, framework_id="fw_alpha", framework_name="Alpha", status="rejected"),
            _make_prediction(3, framework_id="fw_beta", framework_name="Beta", status="accepted"),
            _make_prediction(4, framework_id="fw_beta", framework_name="Beta", status="reassigned"),
            _make_prediction(5, framework_id="fw_beta", framework_name="Beta", status="accepted"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        per_fw = result["per_framework"]
        assert "fw_alpha" in per_fw
        assert per_fw["fw_alpha"]["accepted"] == 1
        assert per_fw["fw_alpha"]["rejected"] == 1
        assert "fw_beta" in per_fw
        assert per_fw["fw_beta"]["accepted"] == 2
        assert per_fw["fw_beta"]["reassigned"] == 1


class TestCalibrationQualityScore:
    def test_all_agree(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, status="accepted"),
            _make_prediction(-1, status="accepted", assigned_hub_id="cal-hub-1"),
            _make_prediction(-2, status="accepted", assigned_hub_id="cal-hub-2"),
            _make_prediction(-3, status="accepted", assigned_hub_id="cal-hub-3"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        rq = result["reviewer_quality"]
        assert rq["quality_score"] == 1.0
        assert rq["agreed"] == 3
        assert rq["disagreements"] == []

    def test_some_disagree(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, status="accepted"),
            _make_prediction(-1, status="accepted", assigned_hub_id="cal-hub-1"),
            _make_prediction(-2, status="reassigned", assigned_hub_id="cal-hub-2", reviewer_hub_id="alt-hub"),
            _make_prediction(-3, status="rejected", assigned_hub_id="cal-hub-3"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        rq = result["reviewer_quality"]
        assert rq["quality_score"] == pytest.approx(1 / 3, abs=0.01)
        assert rq["agreed"] == 1
        assert len(rq["disagreements"]) == 2


class TestCalibrationDisagreements:
    def test_disagreement_details(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(-1, status="reassigned", assigned_hub_id="cal-hub-1", reviewer_hub_id="alt-hub"),
            _make_prediction(-2, status="rejected", assigned_hub_id="cal-hub-2"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        disag = result["reviewer_quality"]["disagreements"]
        assert len(disag) == 2
        ids = {d["id"] for d in disag}
        assert ids == {-1, -2}
        reassigned_item = next(d for d in disag if d["id"] == -1)
        assert reassigned_item["status"] == "reassigned"
        assert reassigned_item["assigned_hub_id"] == "cal-hub-1"
        assert reassigned_item["reviewer_hub_id"] == "alt-hub"


class TestPartialReview:
    def test_pending_items(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, status="accepted"),
            _make_prediction(2, status="accepted"),
            _make_prediction(3, status="pending"),
            _make_prediction(4, status="pending"),
            _make_prediction(5, status="rejected"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        cov = result["coverage"]
        assert cov["total_predictions"] == 5
        assert cov["reviewed"] == 3
        assert cov["pending"] == 2
        assert cov["completion_pct"] == 60.0


class TestImportRoundIncrements:
    def test_first_round(self, tmp_path: Path) -> None:
        preds = [_make_prediction(1, status="accepted")]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)
        assert result["import_round"] == 1

    def test_second_round(self, tmp_path: Path) -> None:
        preds = [_make_prediction(1, status="accepted")]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"

        compute_review_metrics(tmp_path / "db", review_data, output)
        result = compute_review_metrics(tmp_path / "db", review_data, output)
        assert result["import_round"] == 2

    def test_third_round(self, tmp_path: Path) -> None:
        preds = [_make_prediction(1, status="accepted")]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"

        compute_review_metrics(tmp_path / "db", review_data, output)
        compute_review_metrics(tmp_path / "db", review_data, output)
        result = compute_review_metrics(tmp_path / "db", review_data, output)
        assert result["import_round"] == 3


class TestConfidenceAnalysis:
    def test_high_and_low_confidence(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, status="accepted", confidence=0.9),
            _make_prediction(2, status="accepted", confidence=0.8),
            _make_prediction(3, status="rejected", confidence=0.3),
            _make_prediction(4, status="accepted", confidence=0.2),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        ca = result["confidence_analysis"]
        assert ca["high_confidence"]["total"] == 2
        assert ca["high_confidence"]["accepted"] == 2
        assert ca["high_confidence"]["acceptance_rate"] == 100.0
        assert ca["low_confidence"]["total"] == 2
        assert ca["low_confidence"]["accepted"] == 1

    def test_ood_items(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, status="accepted", is_ood=True, confidence=0.4),
            _make_prediction(2, status="rejected", is_ood=True, confidence=0.3),
            _make_prediction(3, status="accepted", is_ood=False, confidence=0.9),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(tmp_path / "db", review_data, output)

        ca = result["confidence_analysis"]
        assert ca["ood_items"]["total"] == 2
        assert ca["ood_items"]["accepted"] == 1
        assert ca["ood_items"]["acceptance_rate"] == 50.0


class TestOutputFile:
    def test_output_file_written(self, tmp_path: Path) -> None:
        preds = [_make_prediction(1, status="accepted")]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        compute_review_metrics(tmp_path / "db", review_data, output)

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "coverage" in data
        assert "overall" in data
        assert "per_framework" in data
        assert "reviewer_quality" in data
        assert "confidence_analysis" in data
        assert "import_round" in data
