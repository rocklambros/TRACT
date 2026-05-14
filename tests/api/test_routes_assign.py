"""Tests for /v1/assign and /v1/assign/batch."""
from __future__ import annotations

from fastapi.testclient import TestClient


# ── /v1/assign — happy path & response shape ───────────────────────────────


def test_assign_happy_path(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"text": "Implement input validation", "top_k": 3})
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["assignments"]) == 3
    assert body["ood_flag"] is False
    assert body["model_version"] == "test-model-v0"


def test_assign_rank_is_1_indexed_and_monotonic(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"text": "Anything", "top_k": 3})
    ranks = [a["rank"] for a in resp.json()["assignments"]]
    assert ranks == [1, 2, 3]


def test_assign_assignments_include_all_hub_fields(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"text": "Anything", "top_k": 1})
    first = resp.json()["assignments"][0]

    expected_fields = {
        "hub_id", "hub_name", "hierarchy_path", "raw_similarity",
        "calibrated_confidence", "in_conformal_set", "rank",
    }
    assert expected_fields.issubset(first.keys())
    # is_ood must NOT leak per-prediction — it's at the envelope level.
    assert "is_ood" not in first


def test_assign_top_k_respected(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"text": "Anything", "top_k": 1})
    assert len(resp.json()["assignments"]) == 1


# ── /v1/assign — pydantic-level validation (422 errors) ────────────────────


def test_assign_rejects_empty_text(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"text": "", "top_k": 3})
    assert resp.status_code == 422


def test_assign_rejects_missing_text(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"top_k": 3})
    assert resp.status_code == 422


def test_assign_rejects_top_k_zero(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"text": "x", "top_k": 0})
    assert resp.status_code == 422


def test_assign_rejects_top_k_too_large(client: TestClient) -> None:
    resp = client.post("/v1/assign", json={"text": "x", "top_k": 100})
    assert resp.status_code == 422


def test_assign_rejects_oversized_text(client: TestClient) -> None:
    huge = "x" * 9000
    resp = client.post("/v1/assign", json={"text": huge, "top_k": 3})
    assert resp.status_code == 422


# ── /v1/assign/batch ───────────────────────────────────────────────────────


def test_assign_batch_happy_path(client: TestClient) -> None:
    payload = {
        "controls": [
            {"id": "ctrl-1", "text": "Implement input validation"},
            {"id": "ctrl-2", "text": "Encrypt data at rest"},
        ],
        "top_k": 2,
    }
    resp = client.post("/v1/assign/batch", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert len(body["results"]) == 2
    assert body["results"][0]["control_id"] == "ctrl-1"
    assert body["results"][1]["control_id"] == "ctrl-2"
    assert len(body["results"][0]["assignments"]) == 2
    assert body["model_version"] == "test-model-v0"


def test_assign_batch_rejects_empty_list(client: TestClient) -> None:
    resp = client.post("/v1/assign/batch", json={"controls": [], "top_k": 3})
    assert resp.status_code == 422


def test_assign_batch_rejects_oversized_list(client: TestClient) -> None:
    payload = {
        "controls": [{"id": f"c-{i}", "text": "x"} for i in range(257)],
        "top_k": 3,
    }
    resp = client.post("/v1/assign/batch", json=payload)
    assert resp.status_code == 422
