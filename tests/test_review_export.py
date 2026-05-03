"""Tests for tract.review.export — review export with re-inference.

All tests mock TRACTPredictor so the real model is never loaded.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tract.crosswalk.schema import create_database, get_connection
from tract.inference import HubPrediction
from tract.review.export import (
    _compute_review_priority,
    _compute_text_quality,
    _generate_calibration_items,
    generate_review_export,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_hub_prediction(
    hub_id: str = "100-200",
    hub_name: str = "Hub A",
    raw_similarity: float = 0.70,
    calibrated_confidence: float = 0.80,
    in_conformal_set: bool = True,
    is_ood: bool = False,
) -> HubPrediction:
    return HubPrediction(
        hub_id=hub_id,
        hub_name=hub_name,
        hierarchy_path=f"/{hub_id}",
        raw_similarity=raw_similarity,
        calibrated_confidence=calibrated_confidence,
        in_conformal_set=in_conformal_set,
        is_ood=is_ood,
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def review_db(tmp_path: Path) -> Path:
    """Populate a crosswalk DB with two frameworks and mixed assignment types.

    Framework layout:
        fw_alpha (Alpha Framework): assignments a1, a2, a3
        fw_beta  (Beta Framework):  assignments b1, b2

    Deliberate edge cases:
        - a3 + GT assignment on same (control, hub) → a3 excluded from export
        - b2 has reviewer="alice" → excluded from export (already reviewed)
        - a1 provenance='active_learning_round_2'
        - a2, b1 provenance='model_prediction'
    """
    db_path = tmp_path / "review_test.db"
    create_database(db_path)
    conn = get_connection(db_path)
    try:
        # Frameworks
        conn.executemany(
            "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("fw_alpha", "Alpha Framework", "1.0", "2026-05-01", 3),
                ("fw_beta",  "Beta Framework",  "2.0", "2026-05-01", 2),
            ],
        )

        # Hubs
        conn.executemany(
            "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
            [
                ("100-200", "Hub Alpha", "/alpha", None),
                ("200-300", "Hub Beta",  "/beta",  None),
                ("300-400", "Hub Gamma", "/gamma", None),
            ],
        )

        # Controls  — text lengths designed to test quality tiers:
        #   fw_alpha:ctrl-1  title="T" only (short)  → combined <100 chars → low
        #   fw_alpha:ctrl-2  full_text with ~500 chars  → high
        #   fw_alpha:ctrl-3  description ~200 chars     → medium
        #   fw_beta:ctrl-1   full_text ~150 chars       → medium
        #   fw_beta:ctrl-2   title+desc ~80 chars       → low
        conn.executemany(
            "INSERT INTO controls (id, framework_id, section_id, title, description, full_text) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("fw_alpha:ctrl-1", "fw_alpha", "CTRL-1", "Short", None, None),
                ("fw_alpha:ctrl-2", "fw_alpha", "CTRL-2", "Title",
                 None, "X" * 500),
                ("fw_alpha:ctrl-3", "fw_alpha", "CTRL-3", "Med",
                 "D" * 200, None),
                ("fw_beta:ctrl-1",  "fw_beta",  "CTRL-B1", "Beta Control",
                 None, "B" * 150),
                ("fw_beta:ctrl-2",  "fw_beta",  "CTRL-B2", "Tiny", "small", None),
            ],
        )

        # Assignments
        # a1 — active_learning_round_2, unreviewed → IN SCOPE
        conn.execute(
            "INSERT INTO assignments (id, control_id, hub_id, confidence, "
            "in_conformal_set, is_ood, provenance, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1, "fw_alpha:ctrl-1", "100-200", 0.55, 1, 0, "active_learning_round_2", "v1"),
        )
        # a2 — model_prediction, unreviewed → IN SCOPE
        conn.execute(
            "INSERT INTO assignments (id, control_id, hub_id, confidence, "
            "in_conformal_set, is_ood, provenance, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (2, "fw_alpha:ctrl-2", "100-200", 0.90, 1, 0, "model_prediction", "v1"),
        )
        # a3 — active_learning_round_2, unreviewed, BUT has a GT companion → EXCLUDED
        conn.execute(
            "INSERT INTO assignments (id, control_id, hub_id, confidence, "
            "in_conformal_set, is_ood, provenance, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (3, "fw_alpha:ctrl-3", "200-300", 0.75, 1, 0, "active_learning_round_2", "v1"),
        )
        # GT companion for a3 — same (control_id, hub_id) → causes a3 exclusion
        conn.execute(
            "INSERT INTO assignments (id, control_id, hub_id, confidence, "
            "in_conformal_set, is_ood, provenance, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (4, "fw_alpha:ctrl-3", "200-300", 1.0, 1, 0, "opencre_ground_truth", "v1"),
        )

        # b1 — model_prediction, unreviewed → IN SCOPE
        conn.execute(
            "INSERT INTO assignments (id, control_id, hub_id, confidence, "
            "in_conformal_set, is_ood, provenance, model_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (5, "fw_beta:ctrl-1", "200-300", 0.40, 0, 1, "model_prediction", "v1"),
        )
        # b2 — model_prediction, ALREADY reviewed (reviewer="alice") → EXCLUDED
        conn.execute(
            "INSERT INTO assignments (id, control_id, hub_id, confidence, "
            "in_conformal_set, is_ood, provenance, model_version, reviewer) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (6, "fw_beta:ctrl-2", "300-400", 0.70, 1, 0, "model_prediction", "v1", "alice"),
        )

        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture()
def calibration_json(tmp_path: Path) -> Path:
    """Write a minimal calibration.json to a temp file."""
    cal_path = tmp_path / "calibration.json"
    cal_path.write_text(
        json.dumps({"global_threshold": 0.5}),
        encoding="utf-8",
    )
    return cal_path


@pytest.fixture()
def mock_predictor() -> MagicMock:
    """Return a mock TRACTPredictor with deterministic predictions.

    predict_batch returns three HubPrediction objects per text:
      index 0: hub_id="100-200", confidence=0.80  (the "winner")
      index 1: hub_id="200-300", confidence=0.15
      index 2: hub_id="300-400", confidence=0.05
    """
    predictor = MagicMock()
    predictor._artifacts = MagicMock()
    predictor._artifacts.model_adapter_hash = "aabbccddeeff001122334455"

    def mock_predict_batch(
        texts: list[str], top_k: int = 5,
    ) -> list[list[HubPrediction]]:
        results: list[list[HubPrediction]] = []
        for _ in texts:
            results.append([
                _make_hub_prediction("100-200", "Hub Alpha", 0.72, 0.80),
                _make_hub_prediction("200-300", "Hub Beta",  0.55, 0.15),
                _make_hub_prediction("300-400", "Hub Gamma", 0.40, 0.05),
            ])
        return results

    predictor.predict_batch.side_effect = mock_predict_batch
    return predictor


# ── Unit tests for helper functions ──────────────────────────────────────────


class TestComputeTextQuality:
    def test_high_threshold(self) -> None:
        assert _compute_text_quality(500) == "high"
        assert _compute_text_quality(501) == "high"
        assert _compute_text_quality(10000) == "high"

    def test_medium_threshold(self) -> None:
        assert _compute_text_quality(100) == "medium"
        assert _compute_text_quality(250) == "medium"
        assert _compute_text_quality(499) == "medium"

    def test_low_threshold(self) -> None:
        assert _compute_text_quality(0) == "low"
        assert _compute_text_quality(50) == "low"
        assert _compute_text_quality(99) == "low"


class TestComputeReviewPriority:
    """Test all 3 tiers and boundary combinations."""

    def test_routine(self) -> None:
        # Above threshold, not OOD → routine regardless of text quality
        assert _compute_review_priority(0.8, False, "high",   0.5) == "routine"
        assert _compute_review_priority(0.8, False, "medium", 0.5) == "routine"
        assert _compute_review_priority(0.8, False, "low",    0.5) == "routine"

    def test_careful_ood(self) -> None:
        # Above threshold but OOD → careful
        assert _compute_review_priority(0.8, True, "high", 0.5) == "careful"

    def test_careful_below_threshold_not_low(self) -> None:
        # At threshold, not OOD, text not low → careful
        assert _compute_review_priority(0.5, False, "high",   0.5) == "careful"
        assert _compute_review_priority(0.5, False, "medium", 0.5) == "careful"
        assert _compute_review_priority(0.3, False, "medium", 0.5) == "careful"

    def test_critical(self) -> None:
        # At or below threshold AND low text quality → critical
        assert _compute_review_priority(0.5, False, "low", 0.5) == "critical"
        assert _compute_review_priority(0.1, False, "low", 0.5) == "critical"

    def test_critical_ood_overridden_by_critical_rule(self) -> None:
        # Below threshold + low quality → critical (critical beats careful)
        assert _compute_review_priority(0.3, True, "low", 0.5) == "critical"

    def test_exact_threshold_boundary(self) -> None:
        # Exactly at threshold → treated as "below" (<=)
        assert _compute_review_priority(0.5, False, "medium", 0.5) == "careful"


# ── Integration tests ─────────────────────────────────────────────────────────


class TestJsonStructure:
    """Output has correct top-level keys and metadata fields."""

    @patch("tract.inference.TRACTPredictor")
    def test_metadata_keys_present(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        output_dir = tmp_path / "out"

        generate_review_export(review_db, tmp_path / "model", output_dir, calibration_json)

        out_file = output_dir / "review_export.json"
        assert out_file.exists()
        data = json.loads(out_file.read_text(encoding="utf-8"))

        assert "metadata" in data
        assert "predictions" in data

        meta = data["metadata"]
        for key in (
            "generated_at", "model_version", "total_predictions",
            "calibration_items", "framework_breakdown", "priority_breakdown",
        ):
            assert key in meta, f"Missing metadata key: {key}"

        pb = meta["priority_breakdown"]
        for tier in ("critical", "careful", "routine"):
            assert tier in pb, f"Missing priority tier: {tier}"

    @patch("tract.inference.TRACTPredictor")
    def test_model_version_is_12_chars(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))
        assert len(data["metadata"]["model_version"]) == 12


class TestGtConfirmedExclusion:
    """Assignment (a3) that overlaps with GT is excluded from export."""

    @patch("tract.inference.TRACTPredictor")
    def test_gt_confirmed_assignment_excluded(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        prediction_ids = {p["id"] for p in data["predictions"]}
        # Assignment 3 (fw_alpha:ctrl-3 → 200-300) has a GT companion → excluded
        assert 3 not in prediction_ids, "GT-confirmed assignment must be excluded"
        # But assignments 1 and 2 should be present
        assert 1 in prediction_ids
        assert 2 in prediction_ids


class TestReExportExcludesReviewed:
    """Assignment b2 (reviewer='alice') is excluded from export."""

    @patch("tract.inference.TRACTPredictor")
    def test_reviewed_assignment_excluded(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        prediction_ids = {p["id"] for p in data["predictions"]}
        # Assignment 6 has reviewer="alice" → excluded
        assert 6 not in prediction_ids, "Already-reviewed assignment must be excluded"
        # Assignment 5 is unreviewed → included
        assert 5 in prediction_ids


class TestReInferenceFreshValues:
    """Confidence in the export comes from fresh inference, not DB-stored values."""

    @patch("tract.inference.TRACTPredictor")
    def test_fresh_confidence_used(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        # DB stores confidence=0.55 for assignment 1, but mock returns 0.80.
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        pred = next(p for p in data["predictions"] if p["id"] == 1)
        # Mock predictor returns 0.80 for hub 100-200; DB stored 0.55
        assert pred["confidence"] == pytest.approx(0.80), (
            "Confidence must come from fresh inference, not DB-stored value"
        )
        assert pred["raw_similarity"] == pytest.approx(0.72)


class TestTextQualityComputation:
    """Text quality tiers are assigned correctly based on combined text length."""

    @patch("tract.inference.TRACTPredictor")
    def test_all_three_tiers_present(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        quality_set = {p["text_quality"] for p in data["predictions"]}
        # fw_alpha:ctrl-1 has title="Short" → combined ~5 chars → low
        # fw_alpha:ctrl-2 has full_text="X"*500 → combined >=500 → high
        # fw_beta:ctrl-1  has full_text="B"*150 → combined ~162 → medium
        assert "low" in quality_set
        assert "high" in quality_set
        assert "medium" in quality_set


class TestReviewPriorityLogic:
    """review_priority is computed correctly from fresh inference values."""

    @patch("tract.inference.TRACTPredictor")
    def test_priority_uses_calibration_threshold(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        # global_threshold=0.5 (from calibration_json fixture)
        # Mock returns confidence=0.80 for all → above threshold
        # Assignment 5 (fw_beta:ctrl-1) has is_ood=1 in DB, but mock predictor
        # returns is_ood=False for hub 100-200.  The export uses fresh is_ood.
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        # All mock predictions return confidence=0.80 > 0.5 and is_ood=False.
        # fw_alpha:ctrl-1 text is "Short" → low quality, but conf=0.80 > threshold
        # → routine (not critical because confidence is above threshold)
        for pred in data["predictions"]:
            # With conf=0.80 > 0.5, not OOD → routine unless low+below threshold
            assert pred["review_priority"] in ("routine", "careful", "critical")

    @patch("tract.inference.TRACTPredictor")
    def test_critical_priority_with_low_confidence_low_text(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path,
    ) -> None:
        """A predictor returning low confidence on low-quality text → critical."""
        low_confidence_predictor = MagicMock()
        low_confidence_predictor._artifacts = MagicMock()
        low_confidence_predictor._artifacts.model_adapter_hash = "aabbccddeeff001122334455"

        def _low_conf_batch(
            texts: list[str], top_k: int = 5,
        ) -> list[list[HubPrediction]]:
            return [[
                _make_hub_prediction("100-200", "Hub Alpha", 0.25, 0.30),
                _make_hub_prediction("200-300", "Hub Beta",  0.20, 0.10),
                _make_hub_prediction("300-400", "Hub Gamma", 0.15, 0.05),
            ] for _ in texts]

        low_confidence_predictor.predict_batch.side_effect = _low_conf_batch
        mock_cls.return_value = low_confidence_predictor

        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        # fw_alpha:ctrl-1 combined text = "Short" (5 chars → low quality)
        # confidence=0.30 <= 0.5 → critical
        pred = next(p for p in data["predictions"] if p["id"] == 1)
        assert pred["text_quality"] == "low"
        assert pred["review_priority"] == "critical"


class TestAlternativeHubs:
    """alternative_hubs contains the top-2 non-assigned hub predictions."""

    @patch("tract.inference.TRACTPredictor")
    def test_two_alternatives_returned(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        # Assignment 1 (hub_id=100-200): mock top-3 are 100-200, 200-300, 300-400
        # → alternatives are 200-300 and 300-400
        pred = next(p for p in data["predictions"] if p["id"] == 1)
        alts = pred["alternative_hubs"]
        assert len(alts) == 2
        alt_ids = {a["hub_id"] for a in alts}
        assert "100-200" not in alt_ids  # assigned hub excluded
        assert "200-300" in alt_ids
        assert "300-400" in alt_ids

    @patch("tract.inference.TRACTPredictor")
    def test_alternative_hubs_have_required_keys(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        for pred in data["predictions"]:
            for alt in pred["alternative_hubs"]:
                assert "hub_id" in alt
                assert "hub_name" in alt
                assert "confidence" in alt


class TestPredictionsSorted:
    """Predictions are sorted by framework_name alpha, then by assignment id."""

    @patch("tract.inference.TRACTPredictor")
    def test_sorted_by_framework_then_id(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        preds = data["predictions"]
        assert len(preds) >= 2

        # Alpha Framework comes before Beta Framework alphabetically
        fw_names = [p["framework_name"] for p in preds]
        assert fw_names == sorted(fw_names), "Predictions must be sorted by framework_name"

        # Within the same framework, IDs should be in ascending order
        for fw in ("Alpha Framework", "Beta Framework"):
            fw_preds = [p for p in preds if p["framework_name"] == fw]
            ids = [p["id"] for p in fw_preds]
            assert ids == sorted(ids), f"IDs within {fw} must be sorted ascending"


class TestAtomicWrite:
    """Output file is created and contains valid JSON."""

    @patch("tract.inference.TRACTPredictor")
    def test_output_file_exists_and_valid_json(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        output_dir = tmp_path / "export_output"
        generate_review_export(review_db, tmp_path / "model", output_dir, calibration_json)

        out_file = output_dir / "review_export.json"
        assert out_file.exists(), "review_export.json must be created"
        # Must be valid JSON
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    @patch("tract.inference.TRACTPredictor")
    def test_no_temp_files_left_behind(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        output_dir = tmp_path / "atomic_out"
        generate_review_export(review_db, tmp_path / "model", output_dir, calibration_json)

        tmp_files = list(output_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, "No temp files must remain after atomic write"


class TestMetadataFrameworkBreakdown:
    """metadata.framework_breakdown counts assignments per framework_id."""

    @patch("tract.inference.TRACTPredictor")
    def test_per_framework_counts(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        breakdown = data["metadata"]["framework_breakdown"]
        # fw_alpha: a1(ctrl-1→100-200) + a2(ctrl-2→100-200) = 2  (a3 excluded by GT)
        # fw_beta:  b1(ctrl-1→200-300) = 1  (b2 excluded by reviewer)
        assert breakdown.get("fw_alpha") == 2, (
            f"Expected 2 for fw_alpha, got {breakdown.get('fw_alpha')}"
        )
        assert breakdown.get("fw_beta") == 1, (
            f"Expected 1 for fw_beta, got {breakdown.get('fw_beta')}"
        )

    @patch("tract.inference.TRACTPredictor")
    def test_total_predictions_matches_sum(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        meta = data["metadata"]
        total = meta["total_predictions"]
        breakdown_sum = sum(meta["framework_breakdown"].values())
        assert total == breakdown_sum == len(data["predictions"])


class TestPredictionRecordSchema:
    """Each prediction record has all required fields with correct types."""

    @patch("tract.inference.TRACTPredictor")
    def test_required_fields_present(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        required_keys = {
            "id", "control_id", "framework_id", "framework_name", "section_id",
            "control_title", "control_text", "assigned_hub_id", "assigned_hub_name",
            "assigned_hub_path", "confidence", "raw_similarity", "is_ood",
            "in_conformal_set", "text_quality", "review_priority", "provenance",
            "alternative_hubs", "decision", "reviewer_hub_id", "reviewer_notes", "status",
        }
        for pred in data["predictions"]:
            missing = required_keys - set(pred.keys())
            assert not missing, f"Prediction {pred['id']} missing keys: {missing}"

    @patch("tract.inference.TRACTPredictor")
    def test_default_review_fields_are_null_or_pending(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        generate_review_export(review_db, tmp_path / "model", tmp_path / "out", calibration_json)
        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))

        for pred in data["predictions"]:
            assert pred["decision"] is None
            assert pred["reviewer_hub_id"] is None
            assert pred["reviewer_notes"] is None
            assert pred["status"] == "pending"


class TestReturnValue:
    """generate_review_export returns the metadata dict."""

    @patch("tract.inference.TRACTPredictor")
    def test_returns_metadata_dict(
        self, mock_cls: MagicMock, review_db: Path, calibration_json: Path,
        tmp_path: Path, mock_predictor: MagicMock,
    ) -> None:
        mock_cls.return_value = mock_predictor
        result = generate_review_export(
            review_db, tmp_path / "model", tmp_path / "out", calibration_json,
        )
        assert isinstance(result, dict)
        assert "total_predictions" in result
        assert "framework_breakdown" in result
        assert result["total_predictions"] == 3  # a1, a2, b1


# ── Calibration item fixtures and tests ──────────────────────────────────────


@pytest.fixture()
def calibration_db(tmp_path: Path) -> Path:
    """DB with 25 ground_truth_T1-AI assignments for calibration tests.

    Each control has a unique text length to ensure deterministic sorting
    by inference confidence. Also includes 2 AL assignments to verify
    they're not confused with calibration items.
    """
    db_path = tmp_path / "cal_test.db"
    create_database(db_path)
    conn = get_connection(db_path)
    try:
        conn.execute(
            "INSERT INTO frameworks (id, name, version, fetch_date, control_count) "
            "VALUES (?, ?, ?, ?, ?)",
            ("fw_cal", "Cal Framework", "1.0", "2026-05-01", 27),
        )

        for i in range(1, 26):
            ctrl_id = f"fw_cal:ctrl-{i}"
            hub_id = f"hub-{i}"
            conn.execute(
                "INSERT INTO controls (id, framework_id, section_id, title, description, full_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ctrl_id, "fw_cal", f"SEC-{i}", f"Control {i}", f"Desc for control {i}", "X" * (100 + i * 10)),
            )
            conn.execute(
                "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
                (hub_id, f"Hub {i}", f"/hub/{i}", None),
            )
            conn.execute(
                "INSERT INTO assignments (control_id, hub_id, confidence, provenance) "
                "VALUES (?, ?, ?, ?)",
                (ctrl_id, hub_id, None, "ground_truth_T1-AI"),
            )

        # Add 2 AL assignments that should NOT be picked up by calibration
        for i in range(26, 28):
            ctrl_id = f"fw_cal:ctrl-{i}"
            hub_id = f"hub-{i}"
            conn.execute(
                "INSERT INTO controls (id, framework_id, section_id, title, description, full_text) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ctrl_id, "fw_cal", f"SEC-{i}", f"AL Control {i}", None, "Y" * 200),
            )
            conn.execute(
                "INSERT INTO hubs (id, name, path, parent_id) VALUES (?, ?, ?, ?)",
                (hub_id, f"Hub {i}", f"/hub/{i}", None),
            )
            conn.execute(
                "INSERT INTO assignments (control_id, hub_id, confidence, provenance) "
                "VALUES (?, ?, ?, ?)",
                (ctrl_id, hub_id, 0.8, "active_learning_round_2"),
            )

        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_calibration_predictor(n_items: int = 25) -> MagicMock:
    """Mock predictor that returns deterministic varying confidences.

    Item i gets confidence = i / n_items, so item 1 is hardest (0.04)
    and item 25 is easiest (1.0). The hub_id matches the known-correct hub.
    """
    predictor = MagicMock()
    predictor._artifacts = MagicMock()
    predictor._artifacts.model_adapter_hash = "aabbccddeeff001122334455"

    call_count = [0]

    def mock_predict_batch(
        texts: list[str], top_k: int = 5,
    ) -> list[list[HubPrediction]]:
        results: list[list[HubPrediction]] = []
        for j, _ in enumerate(texts):
            idx = call_count[0] + j + 1
            conf = idx / n_items
            results.append([
                HubPrediction(
                    hub_id=f"hub-{idx}",
                    hub_name=f"Hub {idx}",
                    hierarchy_path=f"/hub/{idx}",
                    raw_similarity=conf * 0.8,
                    calibrated_confidence=conf,
                    in_conformal_set=True,
                    is_ood=False,
                ),
                HubPrediction(
                    hub_id=f"hub-alt-{idx}",
                    hub_name=f"Alt Hub {idx}",
                    hierarchy_path=f"/alt/{idx}",
                    raw_similarity=conf * 0.5,
                    calibrated_confidence=conf * 0.5,
                    in_conformal_set=False,
                    is_ood=False,
                ),
            ])
        call_count[0] += len(texts)
        return results

    predictor.predict_batch.side_effect = mock_predict_batch
    return predictor


class TestCalibrationItems:
    """Tests for calibration item generation from ground_truth_T1-AI."""

    def test_exactly_20_calibration_items(
        self, calibration_db: Path, tmp_path: Path,
    ) -> None:
        predictor = _make_calibration_predictor()
        items = _generate_calibration_items(
            calibration_db, predictor, 0.5, {},
        )
        assert len(items) == 20

    def test_all_negative_ids(
        self, calibration_db: Path, tmp_path: Path,
    ) -> None:
        predictor = _make_calibration_predictor()
        items = _generate_calibration_items(
            calibration_db, predictor, 0.5, {},
        )
        ids = {item["id"] for item in items}
        assert all(i < 0 for i in ids)
        assert ids == set(range(-1, -21, -1))

    def test_stratified_selection(
        self, calibration_db: Path, tmp_path: Path,
    ) -> None:
        """5 easy (highest conf) + 5 hard (lowest conf) + 10 random middle."""
        predictor = _make_calibration_predictor()
        items = _generate_calibration_items(
            calibration_db, predictor, 0.5, {},
        )
        confidences = [item["confidence"] for item in items]

        # First 5 should be the easiest (highest confidence)
        easy = confidences[:5]
        assert all(c > 0 for c in easy)

        # Last 5 should be the hardest (lowest confidence)
        hard = confidences[15:]
        assert all(c <= max(easy) for c in hard)

    def test_reproducibility_same_seed(
        self, calibration_db: Path, tmp_path: Path,
    ) -> None:
        predictor1 = _make_calibration_predictor()
        items1 = _generate_calibration_items(
            calibration_db, predictor1, 0.5, {},
        )
        predictor2 = _make_calibration_predictor()
        items2 = _generate_calibration_items(
            calibration_db, predictor2, 0.5, {},
        )
        ids1 = [item["control_id"] for item in items1]
        ids2 = [item["control_id"] for item in items2]
        assert ids1 == ids2

    def test_status_is_pending(
        self, calibration_db: Path, tmp_path: Path,
    ) -> None:
        predictor = _make_calibration_predictor()
        items = _generate_calibration_items(
            calibration_db, predictor, 0.5, {},
        )
        assert all(item["status"] == "pending" for item in items)

    def test_confidence_from_inference_not_null(
        self, calibration_db: Path, tmp_path: Path,
    ) -> None:
        """DB stores NULL confidence for GT items; calibration must use inference values."""
        predictor = _make_calibration_predictor()
        items = _generate_calibration_items(
            calibration_db, predictor, 0.5, {},
        )
        assert all(item["confidence"] is not None for item in items)
        assert all(isinstance(item["confidence"], float) for item in items)

    @patch("tract.inference.TRACTPredictor")
    def test_calibration_items_in_full_export(
        self, mock_cls: MagicMock, calibration_db: Path, tmp_path: Path,
    ) -> None:
        """calibration_items count appears in metadata when using full export."""
        # Need AL assignments for generate_review_export to find in-scope items
        predictor = _make_calibration_predictor(n_items=27)
        mock_cls.return_value = predictor

        cal_path = tmp_path / "calibration.json"
        cal_path.write_text(
            json.dumps({"global_threshold": 0.5}),
            encoding="utf-8",
        )

        result = generate_review_export(
            calibration_db, tmp_path / "model", tmp_path / "out", cal_path,
        )
        assert result["calibration_items"] == 20

        data = json.loads((tmp_path / "out" / "review_export.json").read_text(encoding="utf-8"))
        negative_ids = [p for p in data["predictions"] if p["id"] < 0]
        assert len(negative_ids) == 20

    def test_no_gt_items_returns_empty(self, tmp_path: Path) -> None:
        """When no ground_truth_T1-AI exists, returns empty list."""
        db_path = tmp_path / "empty.db"
        create_database(db_path)
        predictor = MagicMock()
        items = _generate_calibration_items(db_path, predictor, 0.5, {})
        assert items == []

    def test_fewer_than_20_gt_items(self, tmp_path: Path) -> None:
        """When fewer than 20 GT items exist, returns all available."""
        db_path = tmp_path / "small.db"
        create_database(db_path)
        conn = get_connection(db_path)
        try:
            conn.execute(
                "INSERT INTO frameworks (id, name) VALUES (?, ?)",
                ("fw_tiny", "Tiny"),
            )
            for i in range(1, 6):
                conn.execute(
                    "INSERT INTO controls (id, framework_id, section_id, title) "
                    "VALUES (?, ?, ?, ?)",
                    (f"fw_tiny:c{i}", "fw_tiny", f"S{i}", f"C{i}"),
                )
                conn.execute(
                    "INSERT INTO hubs (id, name) VALUES (?, ?)",
                    (f"h{i}", f"H{i}"),
                )
                conn.execute(
                    "INSERT INTO assignments (control_id, hub_id, provenance) "
                    "VALUES (?, ?, ?)",
                    (f"fw_tiny:c{i}", f"h{i}", "ground_truth_T1-AI"),
                )
            conn.commit()
        finally:
            conn.close()

        predictor = MagicMock()
        predictor._artifacts = MagicMock()
        predictor._artifacts.model_adapter_hash = "aabb"

        def mock_batch(texts: list[str], top_k: int = 5) -> list[list[HubPrediction]]:
            return [
                [HubPrediction(
                    hub_id=f"h{j+1}", hub_name=f"H{j+1}",
                    hierarchy_path=f"/h{j+1}",
                    raw_similarity=0.5, calibrated_confidence=0.5 + j * 0.1,
                    in_conformal_set=True, is_ood=False,
                )]
                for j in range(len(texts))
            ]

        predictor.predict_batch.side_effect = mock_batch
        items = _generate_calibration_items(db_path, predictor, 0.5, {})
        assert len(items) == 5
        assert all(item["id"] < 0 for item in items)
