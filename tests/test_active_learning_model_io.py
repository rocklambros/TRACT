"""Tests for model loading utilities."""
from __future__ import annotations

import pytest
from pathlib import Path


class TestLoadFoldModel:
    def test_loads_from_valid_path(self) -> None:
        from tract.active_learning.model_io import load_fold_model

        fold_path = Path("results/phase1b/phase1b_textaware/fold_MITRE_ATLAS")
        if not (fold_path / "model" / "model").exists():
            pytest.skip("Fold model not available")
        model = load_fold_model(fold_path)
        emb = model.encode(["test input"], normalize_embeddings=True)
        assert emb.shape == (1, 1024)

    def test_raises_on_missing_path(self) -> None:
        from tract.active_learning.model_io import load_fold_model

        with pytest.raises(FileNotFoundError):
            load_fold_model(Path("/nonexistent/path"))

    def test_smoke_test_embedding(self) -> None:
        from tract.active_learning.model_io import load_fold_model
        import numpy as np

        fold_path = Path("results/phase1b/phase1b_textaware/fold_MITRE_ATLAS")
        if not (fold_path / "model" / "model").exists():
            pytest.skip("Fold model not available")
        model = load_fold_model(fold_path)
        emb = model.encode(["Implement encryption for data at rest"], normalize_embeddings=True)
        assert emb.shape == (1, 1024)
        assert abs(float(np.linalg.norm(emb[0])) - 1.0) < 1e-5


class TestLoadDeploymentModel:
    def test_raises_on_missing_path(self) -> None:
        from tract.active_learning.model_io import load_deployment_model

        with pytest.raises(FileNotFoundError):
            load_deployment_model(Path("/nonexistent/model"))
