"""Tests for T5 deployment artifact generation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.encode.return_value = np.random.default_rng(42).standard_normal((5, 1024)).astype(np.float32)
    return model


class TestGenerateDeploymentArtifacts:
    def test_npz_contains_required_keys(self, tmp_path: Path, mock_model) -> None:
        from scripts.phase1c.t5_finalize_crosswalk import _generate_deployment_artifacts

        hub_ids = [f"hub-{i}" for i in range(5)]
        control_ids = [f"ctrl-{i}" for i in range(5)]
        hub_texts = {hid: f"Hub {i} text" for i, hid in enumerate(hub_ids)}
        control_texts = {cid: f"Control {i} text" for i, cid in enumerate(control_ids)}
        adapter_path = tmp_path / "adapter_model.safetensors"
        adapter_path.write_bytes(b"fake-adapter-weights")

        npz_path = tmp_path / "deployment_artifacts.npz"
        _generate_deployment_artifacts(
            model=mock_model,
            hub_ids=hub_ids,
            hub_texts=hub_texts,
            control_ids=control_ids,
            control_texts=control_texts,
            adapter_path=adapter_path,
            output_path=npz_path,
        )

        data = np.load(str(npz_path), allow_pickle=True)
        assert "hub_embeddings" in data
        assert "control_embeddings" in data
        assert "hub_ids" in data
        assert "control_ids" in data
        assert "model_adapter_hash" in data
        assert "generation_timestamp" in data
        assert data["hub_embeddings"].shape == (5, 1024)
        assert data["control_embeddings"].shape == (5, 1024)
        assert list(data["hub_ids"]) == sorted(hub_ids)

    def test_hub_ids_are_canonically_sorted(self, tmp_path: Path, mock_model) -> None:
        from scripts.phase1c.t5_finalize_crosswalk import _generate_deployment_artifacts

        hub_ids = ["zzz-999", "aaa-001", "mmm-500"]
        control_ids = ["ctrl-1"]
        hub_texts = {hid: f"text for {hid}" for hid in hub_ids}
        control_texts = {"ctrl-1": "text"}
        adapter_path = tmp_path / "adapter_model.safetensors"
        adapter_path.write_bytes(b"fake")

        mock_model.encode.side_effect = [
            np.random.default_rng(42).standard_normal((3, 1024)).astype(np.float32),
            np.random.default_rng(42).standard_normal((1, 1024)).astype(np.float32),
        ]

        npz_path = tmp_path / "deployment_artifacts.npz"
        _generate_deployment_artifacts(
            model=mock_model,
            hub_ids=hub_ids,
            hub_texts=hub_texts,
            control_ids=control_ids,
            control_texts=control_texts,
            adapter_path=adapter_path,
            output_path=npz_path,
        )

        data = np.load(str(npz_path), allow_pickle=True)
        assert list(data["hub_ids"]) == ["aaa-001", "mmm-500", "zzz-999"]


class TestGenerateCalibrationBundle:
    def test_json_contains_required_keys(self, tmp_path: Path) -> None:
        from scripts.phase1c.t5_finalize_crosswalk import _generate_calibration_bundle

        hierarchy_path = tmp_path / "cre_hierarchy.json"
        hierarchy_path.write_text('{"test": true}')

        cal_path = tmp_path / "calibration.json"
        _generate_calibration_bundle(
            t_deploy=0.074,
            ood_threshold=0.568,
            conformal_quantile=0.997,
            global_threshold=0.121,
            hierarchy_path=hierarchy_path,
            output_path=cal_path,
        )

        data = json.loads(cal_path.read_text())
        assert data["t_deploy"] == 0.074
        assert data["ood_threshold"] == 0.568
        assert data["conformal_quantile"] == 0.997
        assert data["global_threshold"] == 0.121
        assert "hierarchy_hash" in data
        assert "calibration_note" in data
