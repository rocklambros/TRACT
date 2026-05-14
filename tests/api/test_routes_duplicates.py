"""Tests for /v1/duplicates."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_duplicates_happy_path(client: TestClient) -> None:
    resp = client.post(
        "/v1/duplicates",
        json={"text": "Manage user accounts", "duplicate_threshold": 0.95, "similar_threshold": 0.85},
    )
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["duplicates"]) == 1
    assert len(body["similar"]) == 1
    assert body["duplicates"][0]["tier"] == "duplicate"
    assert body["similar"][0]["tier"] == "similar"
    assert body["duplicates"][0]["control_id"] == "nist_800_53::AC-2"


def test_duplicates_response_includes_required_fields(client: TestClient) -> None:
    resp = client.post(
        "/v1/duplicates",
        json={"text": "Anything"},
    )
    match = resp.json()["duplicates"][0]
    assert set(match.keys()) == {"control_id", "framework_id", "similarity", "tier"}


def test_duplicates_rejects_inverted_thresholds(client: TestClient) -> None:
    """The cross-field validator should reject duplicate < similar."""
    resp = client.post(
        "/v1/duplicates",
        json={"text": "x", "duplicate_threshold": 0.5, "similar_threshold": 0.9},
    )
    assert resp.status_code == 422
    assert "duplicate_threshold" in resp.text


def test_duplicates_uses_default_thresholds(client: TestClient) -> None:
    """When thresholds are omitted the request is still valid."""
    resp = client.post("/v1/duplicates", json={"text": "Anything"})
    assert resp.status_code == 200
