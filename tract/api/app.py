from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from tract.api.lifecycle import build_lifespan
from tract.api.middleware import MaxBodySizeMiddleware
from tract.api.routes import router
from tract.api.settings import ApiSettings, get_settings
import logging
from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

def _general_exception_handler(debug: bool):
    async def handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception("Unhandled exception (request_id=%s)", request_id)
        payload = {"error": "internal", "request_id": request_id}
        if debug:
            payload["exception_type"] = type(exc).__name__
        return JSONResponse(status_code=500, content=payload)
    return handler

def create_app() -> FastAPI:
    """Build the FastAPI application.

    Args:
        settings: Optional API settings. If not provided, default settings will be used.
    """
    settings = get_settings()
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
    app.add_exception_handler(Exception, _general_exception_handler(debug=settings.debug))
    #  CUDA OOM is special — fragmented GPU memory means the worker can't recover
    try:
        import torch
        if torch.cuda.is_available():
            import os
            async def cuda_oom_handler(request: Request, exc: torch.cuda.OutOfMemoryError):
                logger.error("CUDA OOM — exiting worker for clean restart", exc_info=True)
                os._exit(1)  # bypass cleanup, let uvicorn respawn fresh
            app.add_exception_handler(torch.cuda.OutOfMemoryError, cuda_oom_handler)
    except ImportError:
        pass  # CPU-only deploy

    app.add_middleware(MaxBodySizeMiddleware, max_body_bytes=settings.max_body_bytes)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(router)
    return app
