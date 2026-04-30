"""End-to-end integration test for tract ingest."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.config import PHASE1D_DEPLOYMENT_MODEL_DIR


@pytest.fixture
def fixture_framework(tmp_path: Path) -> Path:
    fw = {
        "framework_id": "test_ingest_e2e",
        "framework_name": "Test Framework",
        "version": "1.0",
        "source_url": "https://example.com",
        "fetched_date": "2026-04-30",
        "mapping_unit_level": "control",
        "controls": [
            {"control_id": "T-01", "title": "Access Control", "description": "Enforce access control for AI models"},
            {"control_id": "T-02", "title": "Logging", "description": "Log all inference requests for audit"},
            {"control_id": "T-03", "title": "Encryption", "description": "Encrypt model weights and training data at rest"},
        ],
    }
    path = tmp_path / "test_framework.json"
    path.write_text(json.dumps(fw))
    return path


@pytest.mark.integration
class TestIngestE2E:
    def test_generates_review_file(self, fixture_framework: Path) -> None:
        from tract.inference import TRACTPredictor
        from tract.io import load_json
        from tract.schema import FrameworkOutput
        from tract.config import PHASE1D_DEFAULT_TOP_K

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        fw = FrameworkOutput.model_validate(load_json(fixture_framework))

        texts = []
        for ctrl in fw.controls:
            parts = [ctrl.title, ctrl.description]
            texts.append(" ".join(p for p in parts if p))

        batch_preds = predictor.predict_batch(texts, top_k=PHASE1D_DEFAULT_TOP_K)
        assert len(batch_preds) == 3

        for preds in batch_preds:
            assert len(preds) == 5
            assert all(0 <= p.calibrated_confidence <= 1 for p in preds)

    def test_duplicate_detection(self, fixture_framework: Path) -> None:
        from tract.inference import TRACTPredictor
        from tract.io import load_json
        from tract.schema import FrameworkOutput

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        fw = FrameworkOutput.model_validate(load_json(fixture_framework))

        text = f"{fw.controls[0].title} {fw.controls[0].description}"
        duplicates, similar = predictor.find_duplicates(text)
        assert isinstance(duplicates, list)
        assert isinstance(similar, list)
