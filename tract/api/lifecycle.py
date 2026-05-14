from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI

from tract.api.settings import ApiSettings
from tract.inference import TRACTPredictor


logger = logging.getLogger(__name__)


def build_lifespan(
    settings: ApiSettings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Factory: returns a lifespan async-cm bound to these settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Load the predictor on startup; let it die with the process on shutdown."""
        logger.info("Loading TRACTPredictor from %s", settings.model_dir)
        app.state.predictor = TRACTPredictor(settings.model_dir)
        yield
        logger.info("Shutting down TRACT API")

    return lifespan
