"""Model loading utilities for Phase 1C orchestration."""
from __future__ import annotations

import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer

from tract.training.checkpoint import assert_checkpoint_is_inert

logger = logging.getLogger(__name__)

EXPECTED_DIM = 1024


def load_fold_model(fold_path: Path) -> SentenceTransformer:
    """Load a saved LOFO fold model with LoRA adapters.

    Fold checkpoints come back off rented third-party GPU pods, so the
    directory is vetted before sentence-transformers is allowed to look at it:
    by the time it has read modules.json, whatever that file named has already
    been imported and run.

    Args:
        fold_path: Path to fold directory (e.g., results/.../fold_MITRE_ATLAS).
                   Expects model/model/ subdirectory with adapter files.

    Raises:
        FileNotFoundError: If the model/model/ subdirectory does not exist.
        ValueError: If the checkpoint fails assert_checkpoint_is_inert, or if
            the smoke-test embedding has the wrong dimension.
    """
    model_dir = fold_path / "model" / "model"
    assert_checkpoint_is_inert(model_dir)

    model = SentenceTransformer(str(model_dir), trust_remote_code=False)
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

    Raises:
        FileNotFoundError: If model_dir does not exist.
        ValueError: If the checkpoint fails assert_checkpoint_is_inert, or if
            the smoke-test embedding has the wrong dimension.
    """
    assert_checkpoint_is_inert(model_dir)

    model = SentenceTransformer(str(model_dir), trust_remote_code=False)
    model.max_seq_length = 512

    emb = model.encode(["smoke test"], normalize_embeddings=True, show_progress_bar=False)
    if emb.shape[1] != EXPECTED_DIM:
        raise ValueError(f"Expected dim={EXPECTED_DIM}, got {emb.shape[1]}")

    logger.info("Loaded deployment model from %s (dim=%d)", model_dir, emb.shape[1])
    return model
