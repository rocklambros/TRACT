"""Tests for tract.inference module."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


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
