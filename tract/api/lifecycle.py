from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from tract.api.settings import ApiSettings, get_settings
from tract.inference import TRACTPredictor


logger = logging.getLogger(__name__)


def build_lifespan(
    settings: ApiSettings,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Factory: returns a lifespan async-cm bound to these settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        try:
            logger.info("Loading TRACTPredictor")  # don't log the full model_dir path — see note below
            app.state.predictor = TRACTPredictor(settings.model_dir)
            logger.info(
                "Predictor loaded — adapter_hash=%s, t_deploy=%.4f, ood_threshold=%.4f",
                app.state.predictor.model_adapter_hash[:12],
                app.state.predictor.t_deploy,
                app.state.predictor.ood_threshold,
            )
        except Exception as e:
            logger.error("Predictor load failed: %s", e, exc_info=True)
            # Fast-fail the worker. Under uvicorn, SystemExit propagates to the master,
            # which will exit non-zero rather than tight-looping (uvicorn >= 0.23).
            raise SystemExit(1) from e

        yield

        # cleanup on shutdown — important if you ever add GPU resources
        logger.info("Shutting down predictor")

    return lifespan
