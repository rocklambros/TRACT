"""TRACT model inference — prediction, duplicate detection, artifact loading.

TRACTPredictor loads the deployment model + cached embeddings + calibration
parameters. Stateful — holds model in memory for repeated predict() calls.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tract.config import (
    PHASE1D_ARTIFACTS_PATH,
    PHASE1D_CALIBRATION_PATH,
    PHASE1D_DEFAULT_TOP_K,
    PHASE1D_DEPLOYMENT_MODEL_DIR,
    PHASE1D_DUPLICATE_THRESHOLD,
    PHASE1D_HEALTH_CHECK_FLOOR,
    PHASE1D_SIMILAR_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubPrediction:
    hub_id: str
    hub_name: str
    hierarchy_path: str
    raw_similarity: float
    calibrated_confidence: float
    in_conformal_set: bool
    is_ood: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DuplicateMatch:
    control_id: str
    framework_id: str
    title: str
    similarity: float
    tier: str  # "duplicate" (>=0.95) or "similar" (>=0.85)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentArtifacts:
    hub_embeddings: NDArray[np.floating]
    control_embeddings: NDArray[np.floating]
    hub_ids: list[str]
    control_ids: list[str]
    model_adapter_hash: str
    generation_timestamp: str


def load_deployment_artifacts(artifacts_path: Path) -> DeploymentArtifacts:
    """Load NPZ without model. For proposal pipeline (no inference needed)."""
    data = np.load(str(artifacts_path), allow_pickle=True)
    return DeploymentArtifacts(
        hub_embeddings=data["hub_embeddings"],
        control_embeddings=data["control_embeddings"],
        hub_ids=list(data["hub_ids"]),
        control_ids=list(data["control_ids"]),
        model_adapter_hash=str(data["model_adapter_hash"]),
        generation_timestamp=str(data["generation_timestamp"]),
    )
