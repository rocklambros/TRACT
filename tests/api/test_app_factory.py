"""Smoke tests for the app factory and route registration."""
from __future__ import annotations

from fastapi import FastAPI

from tract.api import create_app
from tract.api.settings import ApiSettings


def test_create_app_returns_fastapi_instance() -> None:
    """create_app() should build a FastAPI app without touching the network."""
    settings = ApiSettings(model_version="test-v0")
    app = create_app(settings)
    assert isinstance(app, FastAPI)
    assert app.title == "TRACT REST API"


def test_all_seven_endpoints_are_registered(app: FastAPI) -> None:
    """All v1 endpoints from the spec must be on the router."""
    routes = {(tuple(sorted(r.methods or [])), r.path) for r in app.routes if hasattr(r, "methods")}

    expected = {
        (("POST",), "/v1/assign"),
        (("POST",), "/v1/assign/batch"),
        (("POST",), "/v1/duplicates"),
        (("GET",), "/v1/hubs"),
        (("GET",), "/v1/hubs/{hub_id}"),
        (("GET",), "/v1/health"),
        (("GET",), "/v1/version"),
    }
    assert expected.issubset(routes)
