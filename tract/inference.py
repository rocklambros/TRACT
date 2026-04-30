"""TRACT model inference — prediction, duplicate detection, artifact loading.

TRACTPredictor loads the deployment model + cached embeddings + calibration
parameters. Stateful — holds model in memory for repeated predict() calls.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tract.calibration.conformal import build_prediction_sets
from tract.calibration.ood import flag_ood_items
from tract.calibration.temperature import calibrate_similarities
from tract.config import (
    PROCESSED_DIR,
    PHASE1D_ARTIFACTS_PATH,
    PHASE1D_CALIBRATION_PATH,
    PHASE1D_DEFAULT_TOP_K,
    PHASE1D_DEPLOYMENT_MODEL_DIR,
    PHASE1D_DUPLICATE_THRESHOLD,
    PHASE1D_HEALTH_CHECK_FLOOR,
    PHASE1D_SIMILAR_THRESHOLD,
)
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.sanitize import sanitize_text

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


class TRACTPredictor:
    """Loads deployment model + cached artifacts. Stateful — holds model in memory."""

    def __init__(self, model_dir: Path) -> None:
        from tract.active_learning.model_io import load_deployment_model

        self._model_dir = model_dir

        artifacts_path = model_dir / "deployment_artifacts.npz"
        calibration_path = model_dir / "calibration.json"

        if not artifacts_path.exists():
            raise FileNotFoundError(f"Deployment artifacts not found: {artifacts_path}")
        if not calibration_path.exists():
            raise FileNotFoundError(f"Calibration bundle not found: {calibration_path}")

        self._artifacts = load_deployment_artifacts(artifacts_path)
        self._calibration = load_json(calibration_path)
        self._t_deploy: float = self._calibration["t_deploy"]
        self._ood_threshold: float = self._calibration["ood_threshold"]
        self._conformal_quantile: float = self._calibration["conformal_quantile"]

        hierarchy_path = model_dir.parent.parent / "data" / "processed" / "cre_hierarchy.json"
        if not hierarchy_path.exists():
            hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
        self._hierarchy = CREHierarchy.load(hierarchy_path)

        st_model_dir = model_dir / "model"
        if (st_model_dir / "model").exists():
            st_model_dir = st_model_dir / "model"

        adapter_path = st_model_dir / "adapter_model.safetensors"
        if not adapter_path.exists():
            for p in st_model_dir.rglob("adapter_model.safetensors"):
                adapter_path = p
                break
        if adapter_path.exists():
            current_hash = hashlib.sha256(adapter_path.read_bytes()).hexdigest()
            if current_hash != self._artifacts.model_adapter_hash:
                raise ValueError(
                    f"Model adapter hash mismatch: artifacts={self._artifacts.model_adapter_hash[:12]}… "
                    f"vs current={current_hash[:12]}…"
                )

        self._model = load_deployment_model(st_model_dir)

        health_emb = self._model.encode(
            ["access control security authentication"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        max_cos = float(np.max(self._artifacts.hub_embeddings @ health_emb.T))
        if max_cos < PHASE1D_HEALTH_CHECK_FLOOR:
            raise RuntimeError(
                f"Health check failed: max cosine={max_cos:.3f} < floor={PHASE1D_HEALTH_CHECK_FLOOR}"
            )
        logger.info("TRACTPredictor initialized (health check cosine=%.3f)", max_cos)

    def predict(self, text: str, top_k: int = PHASE1D_DEFAULT_TOP_K) -> list[HubPrediction]:
        """Single control text -> top-K hub assignments with calibrated confidence."""
        clean_text = sanitize_text(text)

        query_emb = self._model.encode(
            [clean_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        sims = (self._artifacts.hub_embeddings @ query_emb.T).flatten()
        max_sim = float(np.max(sims))
        is_ood = max_sim < self._ood_threshold

        sims_2d = sims.reshape(1, -1)
        probs = calibrate_similarities(sims_2d, self._t_deploy).flatten()

        conformal_sets = build_prediction_sets(
            probs.reshape(1, -1),
            self._artifacts.hub_ids,
            self._conformal_quantile,
        )
        conformal_set = conformal_sets[0]

        ranked_indices = np.argsort(probs)[::-1][:top_k]

        predictions: list[HubPrediction] = []
        for idx in ranked_indices:
            hub_id = self._artifacts.hub_ids[idx]
            node = self._hierarchy.hubs.get(hub_id)
            predictions.append(HubPrediction(
                hub_id=hub_id,
                hub_name=node.name if node else hub_id,
                hierarchy_path=node.hierarchy_path if node else hub_id,
                raw_similarity=float(sims[idx]),
                calibrated_confidence=float(probs[idx]),
                in_conformal_set=hub_id in conformal_set,
                is_ood=is_ood,
            ))

        return predictions

    def predict_batch(
        self, texts: list[str], top_k: int = PHASE1D_DEFAULT_TOP_K,
    ) -> list[list[HubPrediction]]:
        """Batch prediction for tract ingest."""
        clean_texts = [sanitize_text(t) for t in texts]

        query_embs = self._model.encode(
            clean_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=128,
        )

        sims = query_embs @ self._artifacts.hub_embeddings.T
        max_sims = sims.max(axis=1)
        ood_flags = flag_ood_items(max_sims, self._ood_threshold)

        probs = calibrate_similarities(sims, self._t_deploy)

        conformal_sets = build_prediction_sets(
            probs, self._artifacts.hub_ids, self._conformal_quantile,
        )

        results: list[list[HubPrediction]] = []
        for i in range(len(texts)):
            ranked_indices = np.argsort(probs[i])[::-1][:top_k]
            preds: list[HubPrediction] = []
            for idx in ranked_indices:
                hub_id = self._artifacts.hub_ids[idx]
                node = self._hierarchy.hubs.get(hub_id)
                preds.append(HubPrediction(
                    hub_id=hub_id,
                    hub_name=node.name if node else hub_id,
                    hierarchy_path=node.hierarchy_path if node else hub_id,
                    raw_similarity=float(sims[i, idx]),
                    calibrated_confidence=float(probs[i, idx]),
                    in_conformal_set=hub_id in conformal_sets[i],
                    is_ood=ood_flags[i],
                ))
            results.append(preds)

        return results

    def find_duplicates(
        self,
        text: str,
        duplicate_threshold: float = PHASE1D_DUPLICATE_THRESHOLD,
        similar_threshold: float = PHASE1D_SIMILAR_THRESHOLD,
    ) -> tuple[list[DuplicateMatch], list[DuplicateMatch]]:
        """Compare text embedding against all control_embeddings.

        Returns (duplicates above 0.95, similar controls above 0.85).
        """
        clean_text = sanitize_text(text)
        query_emb = self._model.encode(
            [clean_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).flatten()

        sims = self._artifacts.control_embeddings @ query_emb
        duplicates: list[DuplicateMatch] = []
        similar: list[DuplicateMatch] = []

        for i, sim_val in enumerate(sims):
            if sim_val >= duplicate_threshold:
                ctrl_id = self._artifacts.control_ids[i]
                fw_id = ctrl_id.split("::")[0] if "::" in ctrl_id else ""
                duplicates.append(DuplicateMatch(
                    control_id=ctrl_id,
                    framework_id=fw_id,
                    title=ctrl_id,
                    similarity=float(sim_val),
                    tier="duplicate",
                ))
            elif sim_val >= similar_threshold:
                ctrl_id = self._artifacts.control_ids[i]
                fw_id = ctrl_id.split("::")[0] if "::" in ctrl_id else ""
                similar.append(DuplicateMatch(
                    control_id=ctrl_id,
                    framework_id=fw_id,
                    title=ctrl_id,
                    similarity=float(sim_val),
                    tier="similar",
                ))

        duplicates.sort(key=lambda m: -m.similarity)
        similar.sort(key=lambda m: -m.similarity)
        return duplicates, similar
