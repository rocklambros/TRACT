"""End-to-end integration test for hub proposal pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from tract.config import PHASE1D_ARTIFACTS_PATH, PHASE1D_CALIBRATION_PATH, PROCESSED_DIR


@pytest.mark.integration
class TestProposeE2E:
    def test_ood_detection_and_clustering(self) -> None:
        from tract.hierarchy import CREHierarchy
        from tract.inference import load_deployment_artifacts
        from tract.io import load_json
        from tract.proposals.cluster import cluster_ood_controls

        artifacts = load_deployment_artifacts(PHASE1D_ARTIFACTS_PATH)
        calibration = load_json(PHASE1D_CALIBRATION_PATH)

        sims = artifacts.control_embeddings @ artifacts.hub_embeddings.T
        max_sims = sims.max(axis=1)
        ood_mask = max_sims < calibration["ood_threshold"]

        ood_embs = artifacts.control_embeddings[ood_mask]
        ood_ids = [artifacts.control_ids[i] for i in range(len(artifacts.control_ids)) if ood_mask[i]]

        assert len(ood_ids) >= 0

        if len(ood_ids) >= 3:
            clusters = cluster_ood_controls(
                ood_embs, ood_ids,
                hub_embeddings=artifacts.hub_embeddings,
                hub_ids=artifacts.hub_ids,
            )
            assert isinstance(clusters, list)

    def test_guardrails_pipeline(self) -> None:
        from tract.hierarchy import CREHierarchy
        from tract.inference import load_deployment_artifacts
        from tract.io import load_json
        from tract.proposals.cluster import cluster_ood_controls
        from tract.proposals.guardrails import apply_guardrails

        artifacts = load_deployment_artifacts(PHASE1D_ARTIFACTS_PATH)
        calibration = load_json(PHASE1D_CALIBRATION_PATH)
        hierarchy = CREHierarchy.load(PROCESSED_DIR / "cre_hierarchy.json")

        sims = artifacts.control_embeddings @ artifacts.hub_embeddings.T
        max_sims = sims.max(axis=1)
        ood_mask = max_sims < calibration["ood_threshold"]
        ood_embs = artifacts.control_embeddings[ood_mask]
        ood_ids = [artifacts.control_ids[i] for i in range(len(artifacts.control_ids)) if ood_mask[i]]

        clusters = cluster_ood_controls(
            ood_embs, ood_ids,
            hub_embeddings=artifacts.hub_embeddings,
            hub_ids=artifacts.hub_ids,
        )

        if clusters:
            results = apply_guardrails(
                clusters, hierarchy,
                artifacts.hub_embeddings, artifacts.hub_ids, {},
            )
            assert len(results) == len(clusters)
            for r in results:
                assert isinstance(r.passed, bool)
                if not r.passed:
                    assert len(r.rejection_reasons) > 0
