"""Model loading utilities for Phase 1C orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EXPECTED_DIM = 1024


def load_fold_model(fold_path: Path) -> SentenceTransformer:
    """Load a saved LOFO fold model with LoRA adapters.

    Args:
        fold_path: Path to fold directory (e.g., results/.../fold_MITRE_ATLAS).
                   Expects model/model/ subdirectory with adapter files.
    """
    model_dir = fold_path / "model" / "model"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = SentenceTransformer(str(model_dir))
    model.max_seq_length = 512

    emb = model.encode(["smoke test"], normalize_embeddings=True, show_progress_bar=False)
    if emb.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected dim={EXPECTED_DIM}, got {emb.shape[1]}")

    logger.info("Loaded fold model from %s (dim=%d)", fold_path.name, emb.shape[1])
    return model


def load_deployment_model(model_dir: Path) -> SentenceTransformer:
    """Load a saved deployment model.

    Args:
        model_dir: Path containing the saved model (with adapter files or full weights).
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model = SentenceTransformer(str(model_dir))
    model.max_seq_length = 512

    emb = model.encode(["smoke test"], normalize_embeddings=True, show_progress_bar=False)
    if emb.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected dim={EXPECTED_DIM}, got {emb.shape[1]}")

    logger.info("Loaded deployment model from %s (dim=%d)", model_dir, emb.shape[1])
    return model
