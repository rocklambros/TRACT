"""End-to-end publish pipeline integration test (dry-run, no real model)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# These need the optional `phase0` extra (and matplotlib for the notebook
# helpers), which the default CI test job does not install. Skip visibly
# rather than failing collection; run them with
# `pip install -e '.[phase0]' matplotlib`.
pytest.importorskip("sentence_transformers", reason="needs the phase0 extra")


def _setup_publish_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create all files needed for a dry-run publish."""
    ws = {}

    bridge_report = {
        "counts": {"total": 2, "accepted": 1, "rejected": 1},
        # Required: the model card refuses to publish a hub-classification
        # table it did not measure. See tests/test_publish_model_card.py
        # TestBridgeSectionIsMeasuredNotFabricated.
        "hub_classification": {
            "ai_only": 78, "trad_only": 380,
            "naturally_bridged": 0, "unlinked": 64,
        },
        "candidates": [
            {"ai_hub_id": "AI-1", "trad_hub_id": "T-1", "status": "accepted",
             "cosine_similarity": 0.7, "reviewer_notes": ""},
            {"ai_hub_id": "AI-2", "trad_hub_id": "T-2", "status": "rejected",
             "cosine_similarity": 0.3, "reviewer_notes": "too weak"},
        ],
        "similarity_stats": {"mean": 0.5},
    }
    ws["bridge_report"] = tmp_path / "bridge_report.json"
    ws["bridge_report"].write_text(json.dumps(bridge_report))

    hier = {
        "hubs": {
            "AI-1": {"hub_id": "AI-1", "name": "AI Hub 1", "parent_id": None,
                     "children_ids": [], "depth": 0, "branch_root_id": "AI-1",
                     "hierarchy_path": "AI Hub 1", "is_leaf": True,
                     "sibling_hub_ids": [], "related_hub_ids": ["T-1"]},
            "T-1": {"hub_id": "T-1", "name": "Trad Hub 1", "parent_id": None,
                    "children_ids": [], "depth": 0, "branch_root_id": "T-1",
                    "hierarchy_path": "Trad Hub 1", "is_leaf": True,
                    "sibling_hub_ids": [], "related_hub_ids": ["AI-1"]},
        },
        "roots": ["AI-1", "T-1"], "label_space": ["AI-1", "T-1"],
        "fetch_timestamp": "2026-01-01", "data_hash": "test", "version": "1.1",
    }
    ws["hierarchy"] = tmp_path / "cre_hierarchy.json"
    ws["hierarchy"].write_text(json.dumps(hier, sort_keys=True, indent=2))

    ws["hub_descriptions"] = tmp_path / "hub_descriptions.json"
    ws["hub_descriptions"].write_text(json.dumps({"AI-1": "AI hub desc"}))

    ws["calibration"] = tmp_path / "calibration.json"
    ws["calibration"].write_text(json.dumps({
        "t_deploy": 0.074, "ood_threshold": 0.568, "conformal_quantile": 0.997,
    }))

    ws["ece_gate"] = tmp_path / "ece_gate.json"
    ws["ece_gate"].write_text(json.dumps({
        "ece": 0.079, "ece_ci": {"ci_low": 0.049, "ci_high": 0.111},
    }))

    hub_emb = np.ones((2, 1024), dtype=np.float32)
    hub_emb /= np.linalg.norm(hub_emb, axis=1, keepdims=True)
    ws["artifacts"] = tmp_path / "artifacts.npz"
    np.savez(str(ws["artifacts"]),
             hub_embeddings=hub_emb, hub_ids=np.array(["AI-1", "T-1"]),
             control_embeddings=np.zeros((1, 1024)), control_ids=np.array(["c-1"]))

    ws["model_dir"] = tmp_path / "model"
    ws["model_dir"].mkdir()

    ws["staging_dir"] = tmp_path / "staging"

    return ws


def _fake_merge(model_dir: Path, output_dir: Path) -> None:
    """Mock merge that creates expected model files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "0_Transformer").mkdir()
    (output_dir / "0_Transformer" / "model.safetensors").write_bytes(b"fake")
    (output_dir / "modules.json").write_text("[]")


class TestPublishDryRun:

    def test_dry_run_creates_staging(self, tmp_path) -> None:
        from tract.publish import publish_to_huggingface

        ws = _setup_publish_workspace(tmp_path)

        # hit1_indicators is required, not decorative: the card computes the
        # aggregate CI by bootstrapping these per-item values. It used to accept
        # a fold without them and print a fixed-width band around the point
        # estimate instead. Five hits in ten items matches hit1=0.5.
        fold_results = [
            {
                "fold": "Test Fold", "hit1": 0.5, "zs_hit1": 0.3, "n": 10,
                "hit_any": 0.6,
                "hit1_indicators": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            },
        ]

        with patch("tract.publish.merge.merge_lora_adapters", side_effect=_fake_merge):
            publish_to_huggingface(
                repo_id="test/repo",
                staging_dir=ws["staging_dir"],
                model_dir=ws["model_dir"],
                artifacts_path=ws["artifacts"],
                hierarchy_path=ws["hierarchy"],
                hub_descriptions_path=ws["hub_descriptions"],
                calibration_path=ws["calibration"],
                ece_gate_path=ws["ece_gate"],
                bridge_report_path=ws["bridge_report"],
                fold_results=fold_results,
                gpu_hours=1.0,
                dry_run=True,
            )

        assert ws["staging_dir"].exists()
        assert (ws["staging_dir"] / "README.md").exists()
        assert (ws["staging_dir"] / "predict.py").exists()
        assert (ws["staging_dir"] / "train.py").exists()
        assert (ws["staging_dir"] / "hub_ids.json").exists()
        assert (ws["staging_dir"] / "calibration.json").exists()

        # The licence record ships with the weights. The card's license_link is
        # a relative reference to NOTICE, so a staging directory without it
        # publishes a link to a file that is not there.
        from tract.licensing import LICENSE_TEXTS_DIR, NOTICE_FILENAME

        assert (ws["staging_dir"] / "LICENSE").is_file()
        notice = ws["staging_dir"] / NOTICE_FILENAME
        assert notice.is_file(), (
            f"{NOTICE_FILENAME} is not in the model staging directory, so the "
            f"published weights carry no per-framework terms and no "
            f"modification statement."
        )
        assert "Modifications to framework text" in notice.read_text(
            encoding="utf-8"
        )
        shipped_texts = ws["staging_dir"] / LICENSE_TEXTS_DIR.name
        assert shipped_texts.is_dir()
        assert {p.name for p in shipped_texts.iterdir()} == {
            p.name for p in LICENSE_TEXTS_DIR.iterdir()
        }

    def test_gate_blocks_without_report(self, tmp_path) -> None:
        from tract.publish import publish_to_huggingface

        ws = _setup_publish_workspace(tmp_path)
        ws["bridge_report"].unlink()

        with pytest.raises(ValueError, match="bridge_report.json"):
            publish_to_huggingface(
                repo_id="test/repo",
                staging_dir=ws["staging_dir"],
                model_dir=ws["model_dir"],
                artifacts_path=ws["artifacts"],
                hierarchy_path=ws["hierarchy"],
                hub_descriptions_path=ws["hub_descriptions"],
                calibration_path=ws["calibration"],
                ece_gate_path=ws["ece_gate"],
                bridge_report_path=ws["bridge_report"],
                fold_results=[],
                gpu_hours=0,
                dry_run=True,
            )

    def test_dry_run_does_not_execute_the_aibom_validator(self, tmp_path) -> None:
        """Step 5 sat above the dry-run return, so `publish-hf --dry-run` cloned
        and ran an external repository's HEAD on the publishing host. A dry run
        must not execute foreign code.
        """
        import tract.publish as publish_mod

        ws = _setup_publish_workspace(tmp_path)
        fold_results = [
            {
                "fold": "Test Fold", "hit1": 0.5, "zs_hit1": 0.3, "n": 10,
                "hit_any": 0.6,
                "hit1_indicators": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            },
        ]

        def _explode(*a, **kw) -> None:
            raise AssertionError("AIBOM validator ran without --validate-aibom")

        with patch.object(publish_mod, "_validate_aibom", side_effect=_explode), \
                patch("tract.publish.merge.merge_lora_adapters", side_effect=_fake_merge):
            publish_mod.publish_to_huggingface(
                repo_id="test/repo",
                staging_dir=ws["staging_dir"],
                model_dir=ws["model_dir"],
                artifacts_path=ws["artifacts"],
                hierarchy_path=ws["hierarchy"],
                hub_descriptions_path=ws["hub_descriptions"],
                calibration_path=ws["calibration"],
                ece_gate_path=ws["ece_gate"],
                bridge_report_path=ws["bridge_report"],
                fold_results=fold_results,
                gpu_hours=1.0,
                dry_run=True,
            )

        assert (ws["staging_dir"] / "README.md").exists()

    def test_unpinned_validate_aibom_fails_before_the_merge(self, tmp_path) -> None:
        """The pin used to be checked inside step 5, after step 1 had already
        merged — and merging loads the weights. A missing --aibom-commit must
        cost nothing.
        """
        from tract.publish import publish_to_huggingface

        ws = _setup_publish_workspace(tmp_path)

        def _explode_merge(*a, **kw) -> None:
            raise AssertionError("adapters were merged before the pin was checked")

        with patch("tract.publish.merge.merge_lora_adapters", side_effect=_explode_merge):
            with pytest.raises(ValueError, match="40-character commit SHA"):
                publish_to_huggingface(
                    repo_id="test/repo",
                    staging_dir=ws["staging_dir"],
                    model_dir=ws["model_dir"],
                    artifacts_path=ws["artifacts"],
                    hierarchy_path=ws["hierarchy"],
                    hub_descriptions_path=ws["hub_descriptions"],
                    calibration_path=ws["calibration"],
                    ece_gate_path=ws["ece_gate"],
                    bridge_report_path=ws["bridge_report"],
                    fold_results=[],
                    gpu_hours=0,
                    dry_run=True,
                    validate_aibom=True,
                )
