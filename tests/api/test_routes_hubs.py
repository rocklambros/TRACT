"""Tests for /v1/hubs (list) and /v1/hubs/{hub_id} (detail)."""
from __future__ import annotations

from fastapi.testclient import TestClient


#/v1/hubs (list, paginated)


def test_list_hubs_default_pagination(client: TestClient) -> None:
    resp = client.get("/v1/hubs")
    assert resp.status_code == 200

    body = resp.json()
    assert body["page"] == 1
    assert body["page_size"] == 100
    assert body["total"] == 3   # ROOT, IAM, INPUT from the stub hierarchy
    assert len(body["hubs"]) == 3


def test_list_hubs_first_hub_has_required_summary_fields(client: TestClient) -> None:
    resp = client.get("/v1/hubs")
    first = resp.json()["hubs"][0]
    assert set(first.keys()) == {"hub_id", "name", "hierarchy_path"}


def test_list_hubs_pagination_slice(client: TestClient) -> None:
    """page=2 page_size=2 against 3 total hubs should yield 1 hub."""
    resp = client.get("/v1/hubs?page=2&page_size=2")
    body = resp.json()
    assert body["page"] == 2
    assert body["page_size"] == 2
    assert len(body["hubs"]) == 1 
    assert body["total"] == 3


def test_list_hubs_past_end_returns_empty(client: TestClient) -> None:
    """A page beyond the data must return [], not 404."""
    resp = client.get("/v1/hubs?page=99&page_size=100")
    assert resp.status_code == 200
    assert resp.json()["hubs"] == []


def test_list_hubs_rejects_page_zero(client: TestClient) -> None:
    resp = client.get("/v1/hubs?page=0")
    assert resp.status_code == 422


def test_list_hubs_rejects_page_size_too_large(client: TestClient) -> None:
    resp = client.get("/v1/hubs?page_size=999")
    assert resp.status_code == 422


# /v1/hubs/{hub_id}


def test_get_hub_found(client: TestClient) -> None:
    resp = client.get("/v1/hubs/IAM")
    assert resp.status_code == 200

    body = resp.json()
    assert body["hub_id"] == "IAM"
    assert body["name"] == "Identity & Access Management"
    assert body["parent_id"] == "ROOT"
    assert body["children_ids"] == []


def test_get_hub_root_has_children(client: TestClient) -> None:
    resp = client.get("/v1/hubs/ROOT")
    body = resp.json()
    assert body["parent_id"] is None
    assert set(body["children_ids"]) == {"IAM", "INPUT"}


def test_get_hub_not_found_returns_404(client: TestClient) -> None:
    resp = client.get("/v1/hubs/NONEXISTENT")
    assert resp.status_code == 404
    assert "not found" in resp.text.lower()
