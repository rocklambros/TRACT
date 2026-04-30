"""Tests for tract.inference module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tract.sanitize import sanitize_text


class TestHubPrediction:
    def test_fields_accessible(self) -> None:
        from tract.inference import HubPrediction

        pred = HubPrediction(
            hub_id="646-285",
            hub_name="AI compliance management",
            hierarchy_path="Root > Compliance > AI compliance management",
            raw_similarity=0.523,
            calibrated_confidence=0.847,
            in_conformal_set=True,
            is_ood=False,
        )
        assert pred.hub_id == "646-285"
        assert pred.calibrated_confidence == 0.847
        assert pred.in_conformal_set is True

    def test_to_dict(self) -> None:
        from tract.inference import HubPrediction

        pred = HubPrediction(
            hub_id="646-285",
            hub_name="Test",
            hierarchy_path="Root > Test",
            raw_similarity=0.5,
            calibrated_confidence=0.8,
            in_conformal_set=True,
            is_ood=False,
        )
        d = pred.to_dict()
        assert isinstance(d, dict)
        assert d["hub_id"] == "646-285"
        assert "raw_similarity" in d


class TestDuplicateMatch:
    def test_tier_values(self) -> None:
        from tract.inference import DuplicateMatch

        dup = DuplicateMatch(
            control_id="csa_aicm::AIC-01",
            framework_id="csa_aicm",
            title="Access Control",
            similarity=0.97,
            tier="duplicate",
        )
        assert dup.tier == "duplicate"

        sim = DuplicateMatch(
            control_id="mitre_atlas::AML.T0001",
            framework_id="mitre_atlas",
            title="Technique 1",
            similarity=0.88,
            tier="similar",
        )
        assert sim.tier == "similar"


class TestDeploymentArtifacts:
    def test_load_from_npz(self, tmp_path: Path) -> None:
        from tract.inference import load_deployment_artifacts

        rng = np.random.default_rng(42)
        hub_ids = ["aaa-001", "bbb-002", "ccc-003"]
        control_ids = ["ctrl-1", "ctrl-2"]
        npz_path = tmp_path / "deployment_artifacts.npz"
        np.savez(
            str(npz_path),
            hub_embeddings=rng.standard_normal((3, 1024)).astype(np.float32),
            control_embeddings=rng.standard_normal((2, 1024)).astype(np.float32),
            hub_ids=np.array(hub_ids),
            control_ids=np.array(control_ids),
            model_adapter_hash=np.array("abc123"),
            generation_timestamp=np.array("2026-04-30T00:00:00Z"),
        )

        arts = load_deployment_artifacts(npz_path)
        assert arts.hub_embeddings.shape == (3, 1024)
        assert arts.control_embeddings.shape == (2, 1024)
        assert arts.hub_ids == hub_ids
        assert arts.control_ids == control_ids


def _build_mock_hierarchy(hub_ids: list[str]) -> dict:
    """Build a minimal valid hierarchy for testing."""
    hubs = {}
    for hid in hub_ids:
        hubs[hid] = {
            "hub_id": hid,
            "name": f"Hub {hid}",
            "parent_id": None,
            "children_ids": [],
            "depth": 0,
            "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}",
            "is_leaf": True,
            "sibling_hub_ids": [],
        }
    return {
        "hubs": hubs,
        "roots": hub_ids,
        "label_space": hub_ids,
        "fetch_timestamp": "2026-04-30T00:00:00Z",
        "data_hash": "test_hash",
        "version": "1.0",
    }


@pytest.fixture
def predictor_dir(tmp_path: Path) -> Path:
    """Create a mock predictor directory with NPZ, calibration, hierarchy, and model."""
    rng = np.random.default_rng(42)
    n_hubs = 10
    n_ctrls = 5

    hub_embs = rng.standard_normal((n_hubs, 1024)).astype(np.float32)
    hub_embs = hub_embs / np.linalg.norm(hub_embs, axis=1, keepdims=True)
    ctrl_embs = rng.standard_normal((n_ctrls, 1024)).astype(np.float32)
    ctrl_embs = ctrl_embs / np.linalg.norm(ctrl_embs, axis=1, keepdims=True)

    hub_ids = [f"{i:03d}-{i:03d}" for i in range(n_hubs)]
    control_ids = [f"fw::ctrl-{i}" for i in range(n_ctrls)]

    model_dir = tmp_path / "deployment_model"
    model_dir.mkdir()
    (model_dir / "model").mkdir()

    adapter_path = model_dir / "model" / "adapter_model.safetensors"
    adapter_path.write_bytes(b"fake-adapter")
    adapter_hash = hashlib.sha256(b"fake-adapter").hexdigest()

    npz_path = model_dir / "deployment_artifacts.npz"
    np.savez(
        str(npz_path),
        hub_embeddings=hub_embs,
        control_embeddings=ctrl_embs,
        hub_ids=np.array(hub_ids),
        control_ids=np.array(control_ids),
        model_adapter_hash=np.array(adapter_hash),
        generation_timestamp=np.array("2026-04-30T00:00:00Z"),
    )

    hierarchy_data = _build_mock_hierarchy(hub_ids)
    hierarchy_path = tmp_path / "cre_hierarchy.json"
    hierarchy_path.write_text(json.dumps(hierarchy_data))

    cal_data = {
        "t_deploy": 0.074,
        "ood_threshold": 0.568,
        "conformal_quantile": 0.997,
        "global_threshold": 0.121,
        "hierarchy_hash": hashlib.sha256(
            hierarchy_path.read_bytes()
        ).hexdigest(),
        "calibration_note": "Test calibration",
    }
    cal_path = model_dir / "calibration.json"
    cal_path.write_text(json.dumps(cal_data))

    return tmp_path


class TestTRACTPredictor:
    def _make_predictor(self, predictor_dir: Path, mock_model: MagicMock) -> object:
        """Helper to create a TRACTPredictor with patched model and PROCESSED_DIR."""
        from tract.inference import TRACTPredictor

        health_emb = np.load(
            str(predictor_dir / "deployment_model" / "deployment_artifacts.npz"),
            allow_pickle=True,
        )["hub_embeddings"][0:1]
        mock_model.encode.side_effect = [health_emb]

        with patch("tract.active_learning.model_io.load_deployment_model", return_value=mock_model), \
             patch("tract.inference.PROCESSED_DIR", predictor_dir):
            predictor = TRACTPredictor(predictor_dir / "deployment_model")

        mock_model.encode.side_effect = None
        return predictor

    def test_predict_returns_hub_predictions(self, predictor_dir: Path) -> None:
        from tract.inference import TRACTPredictor

        mock_model = MagicMock()
        rng = np.random.default_rng(99)
        query_emb = rng.standard_normal((1, 1024)).astype(np.float32)
        query_emb = query_emb / np.linalg.norm(query_emb)

        predictor = self._make_predictor(predictor_dir, mock_model)

        mock_model.encode.return_value = query_emb
        preds = predictor.predict("Test control text about access control")
        assert len(preds) == 5
        assert all(isinstance(p, type(preds[0])) for p in preds)
        assert preds[0].calibrated_confidence >= preds[1].calibrated_confidence

    def test_predict_applies_sanitization(self, predictor_dir: Path) -> None:
        mock_model = MagicMock()
        rng = np.random.default_rng(99)
        emb = rng.standard_normal((1, 1024)).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        predictor = self._make_predictor(predictor_dir, mock_model)

        mock_model.encode.return_value = emb
        with patch("tract.inference.sanitize_text", wraps=sanitize_text) as mock_san:
            predictor.predict("text with \x00 null bytes")
            mock_san.assert_called_once()

    def test_predict_ood_flag(self, predictor_dir: Path) -> None:
        mock_model = MagicMock()
        zero_emb = np.zeros((1, 1024), dtype=np.float32)
        zero_emb[0, 0] = 1.0

        predictor = self._make_predictor(predictor_dir, mock_model)

        mock_model.encode.return_value = zero_emb
        preds = predictor.predict("completely unrelated cooking recipe text")
        if preds[0].is_ood:
            assert all(p.is_ood for p in preds)

    def test_predict_batch(self, predictor_dir: Path) -> None:
        mock_model = MagicMock()
        rng = np.random.default_rng(99)
        batch_embs = rng.standard_normal((3, 1024)).astype(np.float32)
        batch_embs = batch_embs / np.linalg.norm(batch_embs, axis=1, keepdims=True)

        predictor = self._make_predictor(predictor_dir, mock_model)

        mock_model.encode.return_value = batch_embs
        results = predictor.predict_batch(["text 1", "text 2", "text 3"])
        assert len(results) == 3
        assert all(len(preds) == 5 for preds in results)
