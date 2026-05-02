"""Tests for tract.publish.bundle — inference data bundling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _setup_source_files(src: Path) -> dict[str, Path]:
    """Create minimal source files for bundling."""
    paths = {}

    desc_path = src / "hub_descriptions_reviewed.json"
    desc_path.write_text(json.dumps({"hub-1": "A hub"}, indent=2))
    paths["hub_descriptions"] = desc_path

    hier_data = {
        "hubs": {"hub-1": {"hub_id": "hub-1", "name": "Hub 1", "related_hub_ids": []}},
        "roots": ["hub-1"], "label_space": ["hub-1"],
        "fetch_timestamp": "2026-01-01", "data_hash": "test", "version": "1.1",
    }
    hier_path = src / "cre_hierarchy.json"
    hier_path.write_text(json.dumps(hier_data, indent=2))
    paths["hierarchy"] = hier_path

    cal_path = src / "calibration.json"
    cal_path.write_text(json.dumps({"t_deploy": 0.074, "ood_threshold": 0.568}))
    paths["calibration"] = cal_path

    artifacts_path = src / "deployment_artifacts.npz"
    hub_ids = np.array(["hub-1"])
    hub_emb = np.ones((1, 1024), dtype=np.float32)
    hub_emb /= np.linalg.norm(hub_emb)
    np.savez(str(artifacts_path), hub_embeddings=hub_emb, hub_ids=hub_ids,
             control_embeddings=np.zeros((1, 1024)), control_ids=np.array(["c-1"]))
    paths["artifacts"] = artifacts_path

    report_path = src / "bridge_report.json"
    report_path.write_text(json.dumps({"counts": {"accepted": 0}}))
    paths["bridge_report"] = report_path

    return paths


class TestBundleInferenceData:

    def test_all_files_copied(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        staging = tmp_path / "staging"
        staging.mkdir()

        bundle_inference_data(staging, **paths)

        assert (staging / "hub_descriptions.json").exists()
        assert (staging / "cre_hierarchy.json").exists()
        assert (staging / "calibration.json").exists()
        assert (staging / "hub_ids.json").exists()
        assert (staging / "bridge_report.json").exists()

    def test_hub_ids_extracted_from_npz(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        staging = tmp_path / "staging"
        staging.mkdir()

        bundle_inference_data(staging, **paths)

        hub_ids = json.loads((staging / "hub_ids.json").read_text())
        assert hub_ids == ["hub-1"]

    def test_hub_embeddings_bundled(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        staging = tmp_path / "staging"
        staging.mkdir()

        bundle_inference_data(staging, **paths)
        assert (staging / "hub_embeddings.npy").exists()

    def test_missing_file_raises(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        paths["hub_descriptions"].unlink()
        staging = tmp_path / "staging"
        staging.mkdir()

        with pytest.raises(FileNotFoundError):
            bundle_inference_data(staging, **paths)
