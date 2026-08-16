"""Tests for /v1/health and /v1/version."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_predictor_state(client: TestClient) -> None:
    resp = client.get("/v1/health")
    assert resp.status_code == 200

    body = resp.json()
    assert body["status"] == "ok"
    assert body["t_deploy"] == 0.074
    assert body["ood_threshold"] == 0.568
    assert len(body["model_adapter_hash"]) == 64   # SHA-256 hex length


def test_version_returns_all_required_fields(client: TestClient) -> None:
    resp = client.get("/v1/version")
    assert resp.status_code == 200

    body = resp.json()
    assert set(body.keys()) == {
        "tract_version",
        "model_version",
        "model_adapter_hash",
        "deployment_artifact_timestamp",
    }
    assert all(isinstance(v, str) and v for v in body.values())
    assert body["model_version"] == "test-model-v0"
