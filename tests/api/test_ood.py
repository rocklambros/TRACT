"""OOD-specific behavior tests."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from tests.api.conftest import StubPredictor
from tract.api.routes import get_predictor, router
from tract.api.settings import ApiSettings, get_settings


def _build_app_with_ood_stub(ood: bool) -> TestClient:
    """Helper: build a fresh app whose stub returns is_ood=<ood> for every prediction."""
    predictor = StubPredictor(ood=ood)
    settings = ApiSettings(model_version="test-model-v0")
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"])
    app.include_router(router)
    app.dependency_overrides[get_predictor] = lambda: predictor
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


def test_ood_flag_is_false_for_in_distribution_input(client: TestClient) -> None:
    """The default stub returns is_ood=False on every prediction."""
    resp = client.post("/v1/assign", json={"text": "Normal control text", "top_k": 3})
    assert resp.json()["ood_flag"] is False


def test_ood_flag_is_true_when_predictor_says_so() -> None:
    """If the predictor flags input as OOD, the envelope should reflect it."""
    ood_client = _build_app_with_ood_stub(ood=True)
    resp = ood_client.post("/v1/assign", json={"text": "gibberish xyzzy", "top_k": 3})
    assert resp.json()["ood_flag"] is True


def test_ood_score_equals_top_raw_similarity(client: TestClient) -> None:
    """ood_score is the max raw_similarity — i.e., the top prediction's similarity."""
    resp = client.post("/v1/assign", json={"text": "Anything", "top_k": 3})
    body = resp.json()
    assert body["ood_score"] == body["assignments"][0]["raw_similarity"]


def test_batch_ood_flag_is_per_control() -> None:
    """In batch mode, each result entry has its own ood_flag."""
    ood_client = _build_app_with_ood_stub(ood=True)
    payload = {"controls": [{"id": "c-1", "text": "x"}, {"id": "c-2", "text": "y"}], "top_k": 2}
    resp = ood_client.post("/v1/assign/batch", json=payload)
    results = resp.json()["results"]
    assert all(r["ood_flag"] is True for r in results)
