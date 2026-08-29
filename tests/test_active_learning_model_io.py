"""Tests for model loading utilities."""
from __future__ import annotations

import json
import pytest

# These need the optional `phase0` extra (and matplotlib for the notebook
# helpers), which the default CI test job does not install. Skip visibly
# rather than failing collection; run them with
# `pip install -e '.[phase0]' matplotlib`.
pytest.importorskip("sentence_transformers", reason="needs the phase0 extra")
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

    def test_load_deployment_model_rejects_auto_map(self, tmp_path: Path) -> None:
        from tract.active_learning.model_io import load_deployment_model

        (tmp_path / "config.json").write_text(
            json.dumps({"auto_map": {"AutoModel": "evil--repo.modeling.Evil"}}),
            encoding="utf-8")
        with pytest.raises(ValueError, match="custom code"):
            load_deployment_model(tmp_path)


class TestLoadersValidateBeforeConstructing:
    """Wiring only. The guard's own behaviour is covered by
    TestAssertCheckpointIsInert in tests/test_training_checkpoint.py, which
    needs no ML stack and therefore actually runs in CI. What is proved here is
    that both loaders reach the guard BEFORE they construct a
    SentenceTransformer -- if the call were moved after construction, the
    payload would have run by the time the ValueError arrived.
    """

    def _hostile_dir(self, model_dir: Path, marker: Path) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "modules.json").write_text(
            json.dumps([{"idx": 0, "name": "0", "path": "",
                         "type": "evil_module.EvilTransformer"}]),
            encoding="utf-8",
        )
        (model_dir / "evil_module.py").write_text(
            "import pathlib\n"
            f"pathlib.Path({str(marker)!r}).write_text('pwned')\n"
            "class EvilTransformer:\n    pass\n",
            encoding="utf-8",
        )

    def test_load_fold_model_refuses_a_hostile_checkpoint(self, tmp_path: Path) -> None:
        from tract.active_learning.model_io import load_fold_model

        marker = tmp_path / "executed"
        self._hostile_dir(tmp_path / "fold" / "model" / "model", marker)

        with pytest.raises(ValueError, match="custom code"):
            load_fold_model(tmp_path / "fold")
        assert not marker.exists(), "payload ran despite validation"

    def test_load_deployment_model_refuses_a_hostile_checkpoint(
        self, tmp_path: Path
    ) -> None:
        from tract.active_learning.model_io import load_deployment_model

        marker = tmp_path / "executed"
        self._hostile_dir(tmp_path / "model", marker)

        with pytest.raises(ValueError, match="custom code"):
            load_deployment_model(tmp_path / "model")
        assert not marker.exists(), "payload ran despite validation"
