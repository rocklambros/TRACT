"""End-to-end integration test for tract assign.

Requires: deployment model, deployment_artifacts.npz, calibration.json, cre_hierarchy.json
"""
from __future__ import annotations

import pytest

from tract.config import PHASE1D_DEPLOYMENT_MODEL_DIR


@pytest.mark.integration
class TestAssignE2E:
    def test_known_control_text(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        preds = predictor.predict("Ensure access control policies are enforced for AI systems")

        assert len(preds) == 5
        assert all(0 <= p.calibrated_confidence <= 1 for p in preds)
        assert all(0 <= p.raw_similarity <= 1 for p in preds)
        assert preds[0].calibrated_confidence >= preds[-1].calibrated_confidence
        assert preds[0].hierarchy_path  # Non-empty hierarchy path

    def test_ood_text(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        preds = predictor.predict("The recipe calls for two cups of flour and a pinch of salt")
        assert preds[0].is_ood

    def test_batch_mode(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        texts = [
            "Access control policy enforcement",
            "Input validation for AI model training data",
            "Encryption of model weights at rest",
        ]
        results = predictor.predict_batch(texts)
        assert len(results) == 3
        assert all(len(preds) == 5 for preds in results)
