from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tract.api.lifecycle import build_lifespan
from tract.api.routes import router
from tract.api.settings import ApiSettings, get_settings


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional API settings. If not provided, default settings will be used.
    """
    settings = get_settings() if settings is None else settings
    app = FastAPI(
        title="TRACT REST API",
        version="1.0",
        lifespan=build_lifespan(settings),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app
