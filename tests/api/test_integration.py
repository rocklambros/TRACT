"""Integration tests — skipped unless real deployment artifacts exist locally.

These exercise the FULL app: real lifespan, real TRACTPredictor, real BGE
embeddings. They take seconds to set up (model load) and may use GPU. Run
manually with `pytest tests/api/test_integration.py -m integration`.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tract.api import create_app
from tract.config import PHASE1D_DEPLOYMENT_MODEL_DIR


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (PHASE1D_DEPLOYMENT_MODEL_DIR / "deployment_artifacts.npz").exists(),
        reason="Deployment artifacts not present — integration test skipped",
    ),
]


@pytest.fixture(scope="module")
def real_client() -> TestClient:
    """One app shared across this module — model load is expensive."""
    app = create_app()
    with TestClient(app) as client:
        # `with` triggers the lifespan, which loads the real TRACTPredictor.
        yield client


def test_real_assign_returns_valid_response(real_client: TestClient) -> None:
    """A real prediction on real text should return a well-formed response."""
    resp = real_client.post(
        "/v1/assign",
        json={"text": "Implement input validation for user-supplied data", "top_k": 3},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["assignments"]) == 3
    for a in body["assignments"]:
        assert isinstance(a["hub_id"], str) and a["hub_id"]
        assert 0.0 <= a["calibrated_confidence"] <= 1.0
        assert -1.0 <= a["raw_similarity"] <= 1.0


def test_real_health_endpoint(real_client: TestClient) -> None:
    """Health probe should succeed and report a deterministic adapter hash.

    Accepts either a 64-char SHA-256 (separate LoRA adapter) or any non-empty
    sentinel string (e.g. "no_adapter" when the model is a merged checkpoint
    with no standalone adapter file).
    """
    resp = real_client.get("/v1/health")
    assert resp.status_code == 200
    hash_val = resp.json()["model_adapter_hash"]
    assert isinstance(hash_val, str) and len(hash_val) > 0


def test_real_gibberish_triggers_ood(real_client: TestClient) -> None:
    """Random tokens should fall below the OOD similarity threshold."""
    resp = real_client.post(
        "/v1/assign",
        json={"text": "xyzzy plugh wibble fnord blorft 12345 lorem ipsum", "top_k": 3},
    )
    assert resp.json()["ood_flag"] is True
