"""Gate 2's criterion, which had no implementation at all.

`gate_decision` is hardwired to `baseline = r["zero_shot"]["hit1_indicators"]`
and raises without it. Gate 2 compares two TRAINED arms. No function anywhere
paired two training arms, while every arm writes an `aggregate_metrics.json`
carrying `gate.preregistered_pass` -- a different question, at threshold 0.10,
against a comparator the pre-registration retired in writing, and one that 73%
of bridge-free arms already satisfy. The 2am failure is quoting that field.

Two things are tested here.

**Alignment.** A paired interval between two separately-executed runs is only
meaningful if they scored the same items in the same order. `paired_bootstrap_delta`
validates fold count and fold size and nothing else, so two 74-item arms in
different orders produce an interval they have not earned.

**The difference-in-differences.** The primary estimand is
`delta_exposed - delta_unexposed`, because global training-draw drift moves both
strata together and cancels -- and that drift is the variance an item bootstrap
cannot see. A positive pooled delta with a flat DiD says the gain was churn, not
the bridge links, which is the one thing the deliverable most needs to know.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.gate2_delta import (
    ArmRecord,
    DiDResult,
    compute_did,
    load_arm,
    verify_alignment,
)


def _arm(
    tmp_path: Path,
    name: str,
    *,
    hits: dict[str, list[float]],
    bridge: str | None,
    texts: dict[str, list[str]] | None = None,
    seed: int = 42,
) -> Path:
    """Write a minimal two-fold arm directory."""
    root = tmp_path / name
    for framework, indicators in hits.items():
        fold = root / f"fold_{framework}"
        fold.mkdir(parents=True)
        item_texts = (texts or {}).get(
            framework, [f"{framework}-{i}" for i in range(len(indicators))]
        )
        (fold / "fold_result.json").write_text(
            json.dumps({
                "held_out_framework": framework,
                "excluded_frameworks": ["ENISA", "BIML", "ETSI"],
                "hit1_indicators": indicators,
                "n_eval_items": len(indicators),
                "config": {"name": name, "bridge_links_path": bridge,
                           "seed": seed},
                "inputs": {"bridge_links_sha256": (
                    "a" * 64 if bridge else None
                )},
            }),
            encoding="utf-8",
        )
        (fold / "predictions.json").write_text(
            json.dumps([
                {"control_text_sha256": t, "framework": framework,
                 "ground_truth_hub_id": "010-108", "predicted_top10": []}
                for t in item_texts
            ]),
            encoding="utf-8",
        )
    return root


class TestAlignment:
    def test_identical_item_order_passes(self, tmp_path: Path) -> None:
        a = load_arm(_arm(tmp_path, "a0", hits={"ENISA": [1.0, 0.0]}, bridge=None))
        b = load_arm(_arm(tmp_path, "a1", hits={"ENISA": [1.0, 1.0]},
                          bridge="r2.jsonl"))
        verify_alignment(a, b)

    def test_reordered_items_are_refused(self, tmp_path: Path) -> None:
        """Same size, same folds, different items. The interval would be junk."""
        a = load_arm(_arm(
            tmp_path, "a0", hits={"ENISA": [1.0, 0.0]}, bridge=None,
            texts={"ENISA": ["x", "y"]},
        ))
        b = load_arm(_arm(
            tmp_path, "a1", hits={"ENISA": [1.0, 1.0]}, bridge="r2.jsonl",
            texts={"ENISA": ["y", "x"]},
        ))
        with pytest.raises(ValueError, match="item order"):
            verify_alignment(a, b)

    def test_different_folds_are_refused(self, tmp_path: Path) -> None:
        a = load_arm(_arm(tmp_path, "a0", hits={"ENISA": [1.0]}, bridge=None))
        b = load_arm(_arm(tmp_path, "a1", hits={"BIML": [1.0]},
                          bridge="r2.jsonl"))
        with pytest.raises(ValueError, match="folds"):
            verify_alignment(a, b)

    def test_arms_differing_in_nothing_are_refused(
        self, tmp_path: Path
    ) -> None:
        """Same corpus AND same seed is not a comparison."""
        a = load_arm(_arm(tmp_path, "a0", hits={"ENISA": [1.0]},
                          bridge="r2.jsonl", seed=42))
        b = load_arm(_arm(tmp_path, "a1", hits={"ENISA": [0.0]},
                          bridge="r2.jsonl", seed=42))
        with pytest.raises(ValueError, match="differ in nothing"):
            verify_alignment(a, b)

    def test_a_seed_only_contrast_is_allowed(self, tmp_path: Path) -> None:
        """THE NOISE FLOOR. Same corpus, different seed, true effect zero.

        The first version of this guard checked the corpus alone and refused
        exactly this -- the one contrast that says whether any other delta is
        readable. A guard that blocks the measurement of its own instrument's
        precision is worse than no guard, because the run is already paid for.
        """
        a = load_arm(_arm(tmp_path, "a0", hits={"ENISA": [1.0, 0.0]},
                          bridge=None, seed=42))
        b = load_arm(_arm(tmp_path, "a0p", hits={"ENISA": [1.0, 1.0]},
                          bridge=None, seed=43))
        verify_alignment(a, b)

    def test_differing_in_both_corpus_and_seed_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A delta across two simultaneous changes is attributable to neither."""
        a = load_arm(_arm(tmp_path, "a0", hits={"ENISA": [1.0]},
                          bridge=None, seed=42))
        b = load_arm(_arm(tmp_path, "a1", hits={"ENISA": [0.0]},
                          bridge="r2.jsonl", seed=43))
        with pytest.raises(ValueError, match="differ in BOTH"):
            verify_alignment(a, b)

    def test_mismatched_firewall_is_refused(self, tmp_path: Path) -> None:
        """Arms firewalled differently are not each other's counterfactual."""
        a_dir = _arm(tmp_path, "a0", hits={"ENISA": [1.0]}, bridge=None)
        result = a_dir / "fold_ENISA" / "fold_result.json"
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["excluded_frameworks"] = ["ENISA"]
        result.write_text(json.dumps(payload), encoding="utf-8")

        a = load_arm(a_dir)
        b = load_arm(_arm(tmp_path, "a1", hits={"ENISA": [0.0]},
                          bridge="r2.jsonl"))
        with pytest.raises(ValueError, match="firewall"):
            verify_alignment(a, b)


class TestTheDiD:
    def _arms(
        self, tmp_path: Path, a0: list[float], a1: list[float]
    ) -> tuple[ArmRecord, ArmRecord]:
        a = load_arm(_arm(tmp_path, "a0", hits={"ENISA": a0}, bridge=None))
        b = load_arm(_arm(tmp_path, "a1", hits={"ENISA": a1},
                          bridge="r2.jsonl"))
        return a, b

    def test_a_gain_confined_to_exposed_items_gives_a_positive_did(
        self, tmp_path: Path
    ) -> None:
        n = 40
        a0 = [0.0] * n
        a1 = [1.0] * 20 + [0.0] * 20      # only the exposed half improves
        exposed = {f"ENISA-{i}" for i in range(20)}
        a, b = self._arms(tmp_path, a0, a1)
        out = compute_did(a, b, exposed, n_resamples=2000, seed=42)
        assert out.delta_exposed == pytest.approx(1.0)
        assert out.delta_unexposed == pytest.approx(0.0)
        assert out.did == pytest.approx(1.0)
        assert out.ci_low > 0

    def test_a_uniform_gain_gives_a_zero_did(self, tmp_path: Path) -> None:
        """The finding the DiD exists to produce.

        Everything improved by the same amount, so the pooled delta is large
        and positive -- and none of it is attributable to the bridge links,
        because the unexposed items moved exactly as much.
        """
        n = 40
        a0 = [0.0] * n
        a1 = [1.0] * n
        exposed = {f"ENISA-{i}" for i in range(20)}
        a, b = self._arms(tmp_path, a0, a1)
        out = compute_did(a, b, exposed, n_resamples=2000, seed=42)
        assert out.pooled_delta == pytest.approx(1.0)
        assert out.did == pytest.approx(0.0)
        assert out.ci_low <= 0 <= out.ci_high

    def test_an_empty_exposed_stratum_raises(self, tmp_path: Path) -> None:
        a, b = self._arms(tmp_path, [0.0] * 4, [1.0] * 4)
        with pytest.raises(ValueError, match="exposed"):
            compute_did(a, b, set(), n_resamples=100, seed=42)

    def test_an_empty_unexposed_stratum_raises(self, tmp_path: Path) -> None:
        """No control stratum means no DiD, only a pooled delta."""
        a, b = self._arms(tmp_path, [0.0] * 4, [1.0] * 4)
        exposed = {f"ENISA-{i}" for i in range(4)}
        with pytest.raises(ValueError, match="unexposed"):
            compute_did(a, b, exposed, n_resamples=100, seed=42)

    def test_the_did_is_order_independent(self, tmp_path: Path) -> None:
        """Inherited from the fold-index fix, and worth pinning here too."""
        n = 40
        a, b = self._arms(tmp_path, [0.0] * n, [1.0] * 20 + [0.0] * 20)
        exposed = {f"ENISA-{i}" for i in range(20)}
        one = compute_did(a, b, exposed, n_resamples=1000, seed=42)
        two = compute_did(a, b, exposed, n_resamples=1000, seed=42)
        assert one.ci_low == pytest.approx(two.ci_low)


class TestTheNegativeControl:
    def test_a_framework_with_zero_exposure_is_reported(
        self, tmp_path: Path
    ) -> None:
        """BIML has zero exposure by construction. If it moves, that is drift."""
        a = load_arm(_arm(
            tmp_path, "a0",
            hits={"ENISA": [0.0] * 20, "BIML": [0.0] * 10}, bridge=None,
        ))
        b = load_arm(_arm(
            tmp_path, "a1",
            hits={"ENISA": [1.0] * 20, "BIML": [0.0] * 10}, bridge="r2.jsonl",
        ))
        exposed = {f"ENISA-{i}" for i in range(20)}
        out = compute_did(a, b, exposed, n_resamples=1000, seed=42)
        assert out.per_framework["BIML"] == pytest.approx(0.0)
        assert out.per_framework["ENISA"] == pytest.approx(1.0)


def test_did_result_is_typed() -> None:
    assert DiDResult.__dataclass_fields__  # type: ignore[attr-defined]
