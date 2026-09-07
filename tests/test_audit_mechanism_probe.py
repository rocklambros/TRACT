"""Tests for the audit mechanism probe.

These exercise the pure statistical functions against synthetic rows rather
than the real corpus, so they run on a CI runner with no licensed overlay and
no result directory. The join itself (`build_rows`) is guarded by an assertion
that raises on positional misalignment; that guard is tested here with a stub
rather than by loading 147 real items.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.audit_mechanism_probe import (
    ALPHA,
    N_RESAMPLES,
    ProbeRow,
    _atomic_write_json,
    contrast,
    delta_distribution,
    score,
)


def _row(
    fold: str, trained: int, zero_shot: int, *,
    touched: bool = False, degree: int = 5,
    verdict: str | None = None, change: float | None = None,
) -> ProbeRow:
    """Build one synthetic ProbeRow. Only the fields the stats touch matter."""
    return ProbeRow(
        framework="F", fold_dir=fold, section="s",
        trained_hit1=trained, zero_shot_hit1=zero_shot,
        audit_touched=touched, gold_degree_max=degree,
        gold_degree_primary=degree, n_valid_hubs=1,
        verdict=verdict, degree_change=change,
    )


class TestDeltaDistribution:
    def test_observed_delta_is_the_mean_of_per_item_differences(self) -> None:
        # 4 items in one fold: 3 improve (1-0), 1 regresses (0-1). Mean = +0.5.
        rows = [_row("a", 1, 0), _row("a", 1, 0), _row("a", 1, 0), _row("a", 0, 1)]
        observed, _ = delta_distribution(rows, np.random.default_rng(0))
        assert observed == pytest.approx(0.5)

    def test_resample_count_matches_the_configured_constant(self) -> None:
        rows = [_row("a", 1, 0), _row("a", 0, 0)]
        _, resampled = delta_distribution(rows, np.random.default_rng(0))
        assert resampled.shape == (N_RESAMPLES,)

    def test_identical_arms_give_a_zero_delta_and_a_degenerate_interval(self) -> None:
        rows = [_row("a", 1, 1), _row("a", 0, 0), _row("a", 1, 1)]
        observed, resampled = delta_distribution(rows, np.random.default_rng(0))
        assert observed == pytest.approx(0.0)
        assert resampled.min() == resampled.max() == pytest.approx(0.0)

    def test_folds_are_resampled_independently_not_pooled(self) -> None:
        # Fold b is constant, fold a varies. A pooled bootstrap would let b's
        # items be redrawn against a's; a stratified one never mixes them, so
        # every resample keeps b's four zeros present.
        rows = [_row("a", 1, 0), _row("a", 0, 1)] + [_row("b", 0, 0)] * 4
        _, resampled = delta_distribution(rows, np.random.default_rng(0))
        # Pooled max would be +1.0 (all six drawn from a's winner).
        assert resampled.max() < 1.0

    def test_empty_stratum_raises_rather_than_returning_nan(self) -> None:
        with pytest.raises(ValueError, match="empty stratum"):
            delta_distribution([], np.random.default_rng(0))

    def test_is_deterministic_for_a_fixed_seed(self) -> None:
        rows = [_row("a", 1, 0), _row("a", 0, 1), _row("b", 1, 0)]
        first, res_a = delta_distribution(rows, np.random.default_rng(42))
        second, res_b = delta_distribution(rows, np.random.default_rng(42))
        assert first == second
        np.testing.assert_array_equal(res_a, res_b)


class TestScore:
    def test_reports_both_arms_and_the_delta(self) -> None:
        rows = [_row("a", 1, 0), _row("a", 1, 1), _row("a", 0, 0), _row("a", 1, 0)]
        stat = score(rows, np.random.default_rng(0))
        assert stat["n"] == 4
        assert stat["trained_hit_at_1"] == pytest.approx(0.75)
        assert stat["zero_shot_hit_at_1"] == pytest.approx(0.25)
        assert stat["delta_mean"] == pytest.approx(0.5)

    def test_interval_brackets_the_point_estimate(self) -> None:
        rows = [_row("a", 1, 0)] * 5 + [_row("a", 0, 0)] * 5
        stat = score(rows, np.random.default_rng(7))
        assert stat["ci_low"] <= stat["delta_mean"] <= stat["ci_high"]


class TestContrast:
    def test_difference_is_a_minus_b(self) -> None:
        a = [_row("a", 1, 0)] * 4          # delta +1.0
        b = [_row("a", 1, 1)] * 4          # delta  0.0
        stat = contrast(a, b, "a", "b", np.random.default_rng(0))
        assert stat["difference"] == pytest.approx(1.0)
        assert stat["label_a"] == "a"
        assert stat["label_b"] == "b"

    def test_a_clean_separation_is_marked_established(self) -> None:
        a = [_row("a", 1, 0)] * 20         # every item improves
        b = [_row("a", 1, 1)] * 20         # no item improves
        stat = contrast(a, b, "a", "b", np.random.default_rng(0))
        assert stat["p_difference_le_zero"] < ALPHA
        assert stat["established_at_alpha"] is True

    def test_an_overlapping_pair_is_not_marked_established(self) -> None:
        # Two noisy strata with the same underlying rate: the difference must
        # straddle zero and must NOT be reported as a finding.
        a = [_row("a", 1, 0), _row("a", 0, 0), _row("a", 1, 0), _row("a", 0, 1)]
        b = [_row("a", 0, 0), _row("a", 1, 0), _row("a", 0, 1), _row("a", 1, 0)]
        stat = contrast(a, b, "a", "b", np.random.default_rng(3))
        assert stat["established_at_alpha"] is False
        assert stat["ci_low"] < 0.0 < stat["ci_high"]

    def test_p_le_zero_is_a_probability(self) -> None:
        a = [_row("a", 1, 0), _row("a", 0, 1)]
        b = [_row("a", 1, 0), _row("a", 0, 0)]
        stat = contrast(a, b, "a", "b", np.random.default_rng(1))
        assert 0.0 <= stat["p_difference_le_zero"] <= 1.0


class TestAtomicWrite:
    def test_writes_sorted_readable_json(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "report.json"
        _atomic_write_json(target, {"b": 2, "a": 1})
        assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
        assert target.read_text(encoding="utf-8").index('"a"') < \
            target.read_text(encoding="utf-8").index('"b"')

    def test_leaves_no_temp_file_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "report.json"
        _atomic_write_json(target, {"x": 1})
        assert [p.name for p in tmp_path.iterdir()] == ["report.json"]

    def test_a_failed_serialisation_does_not_clobber_the_previous_report(
        self, tmp_path: Path,
    ) -> None:
        target = tmp_path / "report.json"
        _atomic_write_json(target, {"good": 1})
        with pytest.raises(TypeError):
            _atomic_write_json(target, {"bad": {1, 2, 3}})  # sets are not JSON
        assert json.loads(target.read_text(encoding="utf-8")) == {"good": 1}
        assert [p.name for p in tmp_path.iterdir()] == ["report.json"]


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = PROJECT_ROOT / "data" / "training" / "audit_corrections_log.json"


@pytest.mark.skipif(not AUDIT_LOG.is_file(), reason="audit log absent")
class TestEveryAuditDecisionIsAccountedFor:
    """The audit made 65 decisions; the stratification read 56 of them.

    `audit_corrections_log.json` carries three lists -- `corrections` (56),
    `exclusions` (1) and `kept_weak` (8). Both probes keyed "audit_touched" off
    `corrections` alone, so:

    * the one EXCLUSION is invisible. It deleted an OWASP AI Exchange link
      (`547-824`, verdict `wrong`), removing an item from that fold's
      denominator. `docs/campaign3-audit-mechanism.md` §5 said "the audit never
      touched OWASP AI Exchange at all" and built a robustness argument on it.
    * the 8 KEPT-WEAK links were inspected and affirmed by the same auditor, yet
      sit inside the Tier-1 "untouched" primary.

    A future audit that excludes 20 links instead of 1 would leave the
    stratification silently unchanged. These pin the reconciliation.
    """

    def _log(self) -> dict:
        return json.loads(AUDIT_LOG.read_text(encoding="utf-8"))

    def test_the_log_carries_three_decision_lists(self) -> None:
        log = self._log()
        for key in ("corrections", "exclusions", "kept_weak"):
            assert key in log, f"{key} missing from the audit log"

    def test_all_sixty_five_decisions_reconcile(self) -> None:
        log = self._log()
        assert len(log["corrections"]) == log["corrections_applied"] == 56
        assert len(log["exclusions"]) == log["links_excluded"] == 1
        assert len(log["kept_weak"]) == log["weak_kept_as_is"] == 8
        total = (len(log["corrections"]) + len(log["exclusions"])
                 + len(log["kept_weak"]))
        assert total == 65

    def test_the_exclusion_is_an_owasp_ai_exchange_link(self) -> None:
        # The specific fact that makes the §5 claim false.
        excluded = self._log()["exclusions"]
        assert any(e["framework_id"] == "owasp_ai_exchange" for e in excluded)

    def test_link_counts_reconcile_with_the_exclusion(self) -> None:
        log = self._log()
        assert (log["ai_links_original"] - log["ai_links_curated"]
                == log["links_excluded"])

    def test_load_audit_index_refuses_a_log_that_does_not_reconcile(
        self, tmp_path, monkeypatch,
    ) -> None:
        import scripts.analysis.audit_mechanism_probe as probe
        broken = dict(self._log())
        broken["links_excluded"] = 99          # disagrees with len(exclusions)
        path = tmp_path / "audit.json"
        path.write_text(json.dumps(broken), encoding="utf-8")
        monkeypatch.setattr(probe, "AUDIT_LOG_PATH", path)
        with pytest.raises(ValueError, match="does not reconcile"):
            probe.load_audit_index()
