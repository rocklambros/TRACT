from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Rejects requests whose declared Content-Length exceeds a limit.

    Checks the Content-Length header only — a chunked-encoded request
    without that header bypasses this check and is caught downstream by
    per-field validation (max_batch_size, max_text_length) instead.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int) -> None:
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": "request body too large", "max_bytes": self.max_body_bytes},
            )
        return await call_next(request)
