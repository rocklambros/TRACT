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


def _db(tmp_path: Path, calibration_ids: tuple[int, ...] = ()) -> Path:
    """A real crosswalk DB where the given assignment ids are calibration rows.

    `db_path` used to be documented as reserved for future use and the tests
    passed a path with nothing behind it. It is load-bearing now: calibration
    membership is read from the store rather than from a negative id in the
    reviewed file, because the negative id was visible to the reviewer those
    items exist to measure (F19).
    """
    from tract.crosswalk.schema import create_database, get_connection

    db_path = tmp_path / "metrics.db"
    if db_path.exists():
        return db_path
    create_database(db_path)
    if not calibration_ids:
        return db_path

    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
            "VALUES ('fw_alpha', 'Alpha Framework', '1.0', '2026-05-01', 0)",
        )
        conn.execute(
            "INSERT INTO hubs (id, name, path, parent_id) "
            "VALUES ('cal-hub-1', 'Cal Hub', '/cal', NULL)",
        )
        for assignment_id in calibration_ids:
            control_id = f"fw_alpha:cal-{assignment_id}"
            conn.execute(
                "INSERT INTO controls (id, framework_id, section_id, title) "
                "VALUES (?, 'fw_alpha', ?, 'Calibration control')",
                (control_id, f"CAL-{assignment_id}"),
            )
            conn.execute(
                "INSERT INTO assignments (id, control_id, hub_id, confidence, "
                "in_conformal_set, is_ood, provenance, model_version) "
                "VALUES (?, ?, 'cal-hub-1', 1.0, 1, 0, 'ground_truth_T1-AI', 'v1')",
                (assignment_id, control_id),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


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
        result = compute_review_metrics(_db(tmp_path), review_data, output)

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
        result = compute_review_metrics(_db(tmp_path), review_data, output)

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
            _make_prediction(901, status="accepted", assigned_hub_id="cal-hub-1"),
            _make_prediction(902, status="accepted", assigned_hub_id="cal-hub-2"),
            _make_prediction(903, status="accepted", assigned_hub_id="cal-hub-3"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(
            _db(tmp_path, (901, 902, 903)), review_data, output,
        )

        rq = result["reviewer_quality"]
        assert rq["quality_score"] == 1.0
        assert rq["agreed"] == 3
        assert rq["disagreements"] == []

    def test_some_disagree(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(1, status="accepted"),
            _make_prediction(901, status="accepted", assigned_hub_id="cal-hub-1"),
            _make_prediction(902, status="reassigned", assigned_hub_id="cal-hub-2", reviewer_hub_id="alt-hub"),
            _make_prediction(903, status="rejected", assigned_hub_id="cal-hub-3"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(
            _db(tmp_path, (901, 902, 903)), review_data, output,
        )

        rq = result["reviewer_quality"]
        assert rq["quality_score"] == pytest.approx(1 / 3, abs=0.01)
        assert rq["agreed"] == 1
        assert len(rq["disagreements"]) == 2


class TestCalibrationDisagreements:
    def test_disagreement_details(self, tmp_path: Path) -> None:
        preds = [
            _make_prediction(901, status="reassigned", assigned_hub_id="cal-hub-1", reviewer_hub_id="alt-hub"),
            _make_prediction(902, status="rejected", assigned_hub_id="cal-hub-2"),
        ]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        result = compute_review_metrics(
            _db(tmp_path, (901, 902)), review_data, output,
        )

        disag = result["reviewer_quality"]["disagreements"]
        assert len(disag) == 2
        ids = {d["id"] for d in disag}
        assert ids == {901, 902}
        reassigned_item = next(d for d in disag if d["id"] == 901)
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
        result = compute_review_metrics(_db(tmp_path), review_data, output)

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
        result = compute_review_metrics(_db(tmp_path), review_data, output)
        assert result["import_round"] == 1

    def test_second_round(self, tmp_path: Path) -> None:
        preds = [_make_prediction(1, status="accepted")]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"

        compute_review_metrics(_db(tmp_path), review_data, output)
        result = compute_review_metrics(_db(tmp_path), review_data, output)
        assert result["import_round"] == 2

    def test_third_round(self, tmp_path: Path) -> None:
        preds = [_make_prediction(1, status="accepted")]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"

        compute_review_metrics(_db(tmp_path), review_data, output)
        compute_review_metrics(_db(tmp_path), review_data, output)
        result = compute_review_metrics(_db(tmp_path), review_data, output)
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
        result = compute_review_metrics(_db(tmp_path), review_data, output)

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
        result = compute_review_metrics(_db(tmp_path), review_data, output)

        ca = result["confidence_analysis"]
        assert ca["ood_items"]["total"] == 2
        assert ca["ood_items"]["accepted"] == 1
        assert ca["ood_items"]["acceptance_rate"] == 50.0


class TestOutputFile:
    def test_output_file_written(self, tmp_path: Path) -> None:
        preds = [_make_prediction(1, status="accepted")]
        review_data = _make_review_data(preds)
        output = tmp_path / "metrics.json"
        compute_review_metrics(_db(tmp_path), review_data, output)

        assert output.exists()
        data = json.loads(output.read_text(encoding="utf-8"))
        assert "coverage" in data
        assert "overall" in data
        assert "per_framework" in data
        assert "reviewer_quality" in data
        assert "confidence_analysis" in data
        assert "import_round" in data
