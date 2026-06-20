"""End-to-end integration test for tract assign.

Requires: deployment model, deployment_artifacts.npz, calibration.json, cre_hierarchy.json
"""
from __future__ import annotations

import pytest

from tract.config import PHASE1D_DEPLOYMENT_MODEL_DIR


@pytest.mark.integration
class TestAssignE2E:
    def test_known_control_text(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        preds = predictor.predict("Ensure access control policies are enforced for AI systems")

        assert len(preds) == 5
        assert all(0 <= p.calibrated_confidence <= 1 for p in preds)
        assert all(0 <= p.raw_similarity <= 1 for p in preds)
        assert preds[0].calibrated_confidence >= preds[-1].calibrated_confidence
        assert preds[0].hierarchy_path  # Non-empty hierarchy path

    def test_ood_text(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        preds = predictor.predict("The recipe calls for two cups of flour and a pinch of salt")
        assert preds[0].is_ood

    def test_batch_mode(self) -> None:
        from tract.inference import TRACTPredictor

        predictor = TRACTPredictor(PHASE1D_DEPLOYMENT_MODEL_DIR)
        texts = [
            "Access control policy enforcement",
            "Input validation for AI model training data",
            "Encryption of model weights at rest",
        ]
        results = predictor.predict_batch(texts)
        assert len(results) == 3
        assert all(len(preds) == 5 for preds in results)


# ---------------------------------------------------------------------------
# Unit tests — no model artefacts required
# ---------------------------------------------------------------------------
import json
import argparse
import tract.cli as cli  # noqa: E402 — after pytest integration block


class _FakePred:
    """raw_similarity descending != input order, to catch a reorder."""

    def __init__(self, sim: float) -> None:
        self._sim = sim

    @property
    def raw_similarity(self) -> float:
        return self._sim

    def to_dict(self) -> dict:
        return {"raw_similarity": self._sim}


class _FakePredictor:
    def __init__(self, *a: object, **k: object) -> None:
        pass

    def predict_batch(self, texts: list[str], top_k: int = 5) -> list[list[_FakePred]]:
        # lower similarity for earlier lines, so a similarity-sort would reverse them
        return [[_FakePred(0.1 * (i + 1))] for i, _ in enumerate(texts)]


def _run_assign_file(
    tmp_path: "Path",
    lines: list[str],
    monkeypatch: "pytest.MonkeyPatch",
) -> list[dict]:
    from pathlib import Path

    infile: Path = tmp_path / "controls.txt"
    infile.write_text("\n".join(lines), encoding="utf-8")
    outfile: Path = tmp_path / "out.jsonl"
    monkeypatch.setattr(cli, "TRACTPredictor", _FakePredictor, raising=False)
    monkeypatch.setattr("tract.inference.TRACTPredictor", _FakePredictor)
    monkeypatch.setattr(
        cli,
        "ensure_deployment_model",
        lambda: type("R", (), {"path": tmp_path, "source": "local"})(),
        raising=False,
    )
    args = argparse.Namespace(
        file=str(infile),
        output=str(outfile),
        text=None,
        top_k=5,
        json=False,
        raw=False,
        verbose=False,
    )
    cli._cmd_assign(args)
    return [json.loads(line) for line in outfile.read_text(encoding="utf-8").splitlines()]


def test_assign_file_preserves_order_full_text_and_index(
    tmp_path: "Path",
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    # Trailing whitespace is stripped (expected); the key invariant is no [:100] truncation.
    long_line = ("access control " * 20).strip()  # > 100 chars, no trailing space
    recs = _run_assign_file(
        tmp_path, [long_line, "encryption", "", "audit logging"], monkeypatch
    )
    assert [r["input_index"] for r in recs] == [1, 2, 4]   # gap where blank line 3 was
    assert recs[0]["text"] == long_line                     # not truncated to 100
    assert len(recs[0]["text"]) > 100                       # confirm it actually is long
    assert [r["text"] for r in recs] == [
        long_line,
        "encryption",
        "audit logging",
    ]  # input order
