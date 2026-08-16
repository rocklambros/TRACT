"""Shared fixtures for tract.api tests.

A `StubPredictor` mimics `TRACTPredictor` without loading the real model.
Each test gets a fresh FastAPI app with the stub injected via FastAPI's
`app.dependency_overrides` mechanism — the lifespan (which would load the
real model) is never triggered.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from tract.api.routes import get_predictor, router
from tract.api.settings import ApiSettings, get_settings
from tract.hierarchy import HubNode
from tract.inference import DuplicateMatch, HubPrediction



def _hub_node(
    hub_id: str,
    name: str,
    parent_id: str | None = None,
    children_ids: list[str] | None = None,
) -> HubNode:
    """Build a minimal valid HubNode for tests."""
    return HubNode(
        hub_id=hub_id,
        name=name,
        parent_id=parent_id,
        children_ids=children_ids or [],
        depth=0 if parent_id is None else 1,
        branch_root_id=hub_id if parent_id is None else parent_id,
        hierarchy_path=name if parent_id is None else f"Root > {name}",
        is_leaf=not children_ids,
    )


def _make_hierarchy() -> dict[str, HubNode]:
    """Three-node hierarchy: one root with two children."""
    return {
        "ROOT": _hub_node("ROOT", "Security", children_ids=["IAM", "INPUT"]),
        "IAM": _hub_node("IAM", "Identity & Access Management", parent_id="ROOT"),
        "INPUT": _hub_node("INPUT", "Input Validation", parent_id="ROOT"),
    }


def _canned_predictions(is_ood: bool = False) -> list[HubPrediction]:
    """Three predictions, ranked by descending similarity."""
    return [
        HubPrediction(
            hub_id="IAM",
            hub_name="Identity & Access Management",
            hierarchy_path="Root > Identity & Access Management",
            raw_similarity=0.92 if not is_ood else 0.30,
            calibrated_confidence=0.81 if not is_ood else 0.05,
            in_conformal_set=True,
            is_ood=is_ood,
        ),
        HubPrediction(
            hub_id="INPUT",
            hub_name="Input Validation",
            hierarchy_path="Root > Input Validation",
            raw_similarity=0.74 if not is_ood else 0.25,
            calibrated_confidence=0.12,
            in_conformal_set=True,
            is_ood=is_ood,
        ),
        HubPrediction(
            hub_id="ROOT",
            hub_name="Security",
            hierarchy_path="Security",
            raw_similarity=0.55 if not is_ood else 0.20,
            calibrated_confidence=0.07,
            in_conformal_set=False,
            is_ood=is_ood,
        ),
    ]




@dataclass
class _StubHierarchy:
    """Mimics CREHierarchy's `.hubs` attribute — the only thing routes use."""
    hubs: dict[str, HubNode]


class StubPredictor:
    """In-memory fake of TRACTPredictor. Same public surface, no model loaded.

    Toggle `ood` to make every prediction come back flagged as out-of-distribution.
    """

    def __init__(self, ood: bool = False) -> None:
        self.ood = ood
        self._hierarchy = _StubHierarchy(hubs=_make_hierarchy())

    # methods routes.py calls

    def predict(self, text: str, top_k: int = 5) -> list[HubPrediction]:
        return _canned_predictions(is_ood=self.ood)[:top_k]

    def predict_batch(self, texts: list[str], top_k: int = 5) -> list[list[HubPrediction]]:
        return [_canned_predictions(is_ood=self.ood)[:top_k] for _ in texts]

    def find_duplicates(
        self,
        text: str,
        duplicate_threshold: float = 0.95,
        similar_threshold: float = 0.85,
    ) -> tuple[list[DuplicateMatch], list[DuplicateMatch]]:
        dups = [
            DuplicateMatch(
                control_id="nist_800_53::AC-2",
                framework_id="nist_800_53",
                title="Account Management",
                similarity=0.97,
                tier="duplicate",
            ),
        ]
        sim = [
            DuplicateMatch(
                control_id="iso_27001::A.9.2",
                framework_id="iso_27001",
                title="User access management",
                similarity=0.88,
                tier="similar",
            ),
        ]
        return dups, sim

    # properties routes.py reads

    @property
    def hierarchy(self) -> _StubHierarchy:
        return self._hierarchy

    @property
    def model_adapter_hash(self) -> str:
        return "stubhash" * 8  # 64 chars like a real SHA-256

    @property
    def t_deploy(self) -> float:
        return 0.074

    @property
    def ood_threshold(self) -> float:
        return 0.568

    @property
    def deployment_timestamp(self) -> str:
        return "2026-01-01T00:00:00Z"


#fixtures


@pytest.fixture
def stub_predictor() -> StubPredictor:
    """Fresh stub per test — tests can flip .ood without leaking to neighbors."""
    return StubPredictor()


@pytest.fixture
def test_settings() -> ApiSettings:
    """Settings instance for tests. model_dir is not used because lifespan is skipped."""
    return ApiSettings(
        host="127.0.0.1",
        port=8000,
        workers=1,
        model_version="test-model-v0",
    )


@pytest.fixture
def app(stub_predictor: StubPredictor, test_settings: ApiSettings) -> FastAPI:
    """Build a test FastAPI app with the stub predictor wired in.

    Critically: we DO NOT use `create_app()` because that attaches the lifespan
    handler, which would try to load the real TRACTPredictor on first request.
    Instead we hand-assemble the same shape, then override `get_predictor` and
    `get_settings` via FastAPI's dependency-override mechanism.
    """
    fastapi_app = FastAPI(title="TRACT REST API (test)", version="1.0-test")
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    fastapi_app.include_router(router)
    fastapi_app.dependency_overrides[get_predictor] = lambda: stub_predictor
    fastapi_app.dependency_overrides[get_settings] = lambda: test_settings
    return fastapi_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Synchronous HTTP client wired to the in-process app — no real network."""
    return TestClient(app)
