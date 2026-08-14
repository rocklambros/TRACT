"""Tests for per-fold result persistence and micro-aggregation.

The RunPod path wrote a per-fold summary that kept only the metric dict and
dropped the per-item hit@1 indicators, and nothing aggregated the five summaries
at all. Averaging them by hand is a MACRO average: it weights a 6-item fold the
same as a 60-item one. TRACT reports a MICRO average, which needs the raw
indicators from every fold.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# orchestrate imports tract.training.data, which imports torch and datasets.
# The default CI test job installs requirements.txt only, and it runs pytest
# with -x, so an unguarded import here does not skip this file -- it aborts the
# whole job at collection. Guard on what is actually imported, not on numpy.
pytest.importorskip("numpy")
pytest.importorskip("torch", reason="needs the phase0 extra")
pytest.importorskip("datasets", reason="needs the phase0 extra")

from tract.training.orchestrate import (
    FOLD_RESULT_FILENAME,
    aggregate_fold_results,
    load_fold_results,
)


def _write_fold(
    root: Path,
    framework: str,
    indicators: list[int],
    git_sha: str = "abc1234",
    zero_shot: list[int] | None = None,
    inputs: dict[str, str | None] | None = None,
) -> None:
    fold_dir = root / f"fold_{framework.replace(' ', '_')}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "held_out_framework": framework,
        "metrics": {"hit_at_1": sum(indicators) / len(indicators)},
        "hit1_indicators": indicators,
        "n_eval_items": len(indicators),
        "n_training_pairs": 100,
        "elapsed_s": 1.0,
        "git_sha": git_sha,
    }
    if inputs is not None:
        record["inputs"] = inputs
    if zero_shot is not None:
        record["zero_shot"] = {"hit1_indicators": zero_shot}
    (fold_dir / FOLD_RESULT_FILENAME).write_text(json.dumps(record), encoding="utf-8")


class TestMicroVersusMacro:

    def test_micro_average_weights_items_not_folds(self) -> None:
        """The whole point of the fix, stated as an assertion.

        One small fold scoring perfectly and one large fold scoring poorly. The
        macro average is the mean of the two rates; the micro average is the
        mean over items. They must not be the same number.
        """
        small = [1] * 6                 # n=6,  hit@1 = 1.0
        large = [1] * 20 + [0] * 40     # n=60, hit@1 = 0.3333

        result = aggregate_fold_results([
            {"held_out_framework": "Tiny", "hit1_indicators": small, "n_eval_items": 6},
            {"held_out_framework": "Big", "hit1_indicators": large, "n_eval_items": 60},
        ])

        expected_micro = (6 + 20) / 66
        expected_macro = (1.0 + 20 / 60) / 2

        assert result["mean"] == pytest.approx(expected_micro, abs=1e-9)
        assert result["macro_mean"] == pytest.approx(expected_macro, abs=1e-9)
        assert result["n_total"] == 66
        # 0.394 vs 0.667 -- averaging fold summaries would have overstated the
        # headline by 27 points here.
        assert abs(result["mean"] - result["macro_mean"]) > 0.25

    def test_fold_sizes_are_recorded(self) -> None:
        result = aggregate_fold_results([
            {"held_out_framework": "A", "hit1_indicators": [1, 0], "n_eval_items": 2},
            {"held_out_framework": "B", "hit1_indicators": [1, 1, 1], "n_eval_items": 3},
        ])
        assert result["fold_sizes"] == {"A": 2, "B": 3}
        assert result["n_folds"] == 2

    def test_confidence_interval_brackets_the_estimate(self) -> None:
        result = aggregate_fold_results([
            {"held_out_framework": "A", "hit1_indicators": [1, 0] * 25, "n_eval_items": 50},
            {"held_out_framework": "B", "hit1_indicators": [1, 1, 0] * 20, "n_eval_items": 60},
        ])
        assert result["ci_low"] < result["mean"] < result["ci_high"]


class TestLoadFoldResults:

    def test_loads_and_sorts_by_framework(self, tmp_path: Path) -> None:
        _write_fold(tmp_path, "OWASP Top10 for LLM", [1, 0, 1])
        _write_fold(tmp_path, "MITRE ATLAS", [1, 1])

        records = load_fold_results(
            tmp_path, expected_frameworks={"MITRE ATLAS", "OWASP Top10 for LLM"},
        )
        assert [r["held_out_framework"] for r in records] == [
            "MITRE ATLAS", "OWASP Top10 for LLM",
        ]

    def test_refuses_a_partial_cross_validation(self, tmp_path: Path) -> None:
        """Four folds of five is a different experiment, not a smaller one.

        A fold that OOMs leaves an empty directory that rsyncs cleanly, so the
        aggregate would otherwise publish as LOFO with the pod that held the
        missing fold already terminated.
        """
        _write_fold(tmp_path, "MITRE ATLAS", [1, 1])
        _write_fold(tmp_path, "NIST AI 100-2", [1, 0])

        with pytest.raises(ValueError, match="Fold set mismatch"):
            load_fold_results(
                tmp_path,
                expected_frameworks={"MITRE ATLAS", "NIST AI 100-2", "OWASP AI Exchange"},
            )

    def test_refuses_folds_from_different_commits(self, tmp_path: Path) -> None:
        """A stale fold_result.json from an earlier run must not aggregate.

        collect() rsyncs without --delete into a directory that is never
        cleaned, so a re-run of four folds leaves the fifth behind. Mixed
        git_sha is that situation's fingerprint.
        """
        _write_fold(tmp_path, "MITRE ATLAS", [1, 1], git_sha="deadbee")
        _write_fold(tmp_path, "NIST AI 100-2", [1, 1, 1, 1], git_sha="0ldsha1")

        with pytest.raises(ValueError, match="produced by different code"):
            load_fold_results(
                tmp_path, expected_frameworks={"MITRE ATLAS", "NIST AI 100-2"},
            )

    def test_rejects_a_misaligned_zero_shot_baseline(self, tmp_path: Path) -> None:
        """Equal length is what makes the delta paired.

        A baseline of the wrong length cannot be aligned item-for-item, and a
        mis-paired delta would be reported with a paired interval it has not
        earned.
        """
        _write_fold(tmp_path, "MITRE ATLAS", [1, 1, 0], zero_shot=[1, 0])

        with pytest.raises(ValueError, match="not paired"):
            load_fold_results(tmp_path, expected_frameworks={"MITRE ATLAS"})

    def test_rejects_a_record_missing_indicators(self, tmp_path: Path) -> None:
        """This is the shape the old RunPod summary wrote."""
        fold_dir = tmp_path / "fold_MITRE_ATLAS"
        fold_dir.mkdir(parents=True)
        (fold_dir / FOLD_RESULT_FILENAME).write_text(
            json.dumps({
                "held_out_framework": "MITRE ATLAS",
                "metrics": {"hit_at_1": 0.5},
                "n_eval_items": 10,
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="hit1_indicators"):
            load_fold_results(tmp_path, expected_frameworks={"MITRE ATLAS"})

    def test_rejects_inconsistent_indicator_count(self, tmp_path: Path) -> None:
        fold_dir = tmp_path / "fold_MITRE_ATLAS"
        fold_dir.mkdir(parents=True)
        (fold_dir / FOLD_RESULT_FILENAME).write_text(
            json.dumps({
                "held_out_framework": "MITRE ATLAS",
                "metrics": {},
                "hit1_indicators": [1, 0],
                "n_eval_items": 10,
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="internally inconsistent"):
            load_fold_results(tmp_path, expected_frameworks={"MITRE ATLAS"})

    def test_refuses_to_aggregate_nothing(self, tmp_path: Path) -> None:
        """An empty results directory must not quietly produce a number."""
        with pytest.raises(ValueError, match="Nothing to aggregate"):
            load_fold_results(tmp_path, expected_frameworks={"MITRE ATLAS"})


class TestInputProvenance:
    """A fold's git SHA pins its code. Nothing pinned its data."""

    _FRAMEWORKS = {"A", "B"}

    def test_folds_from_one_snapshot_aggregate(self, tmp_path: Path) -> None:
        same = {"curated_links_sha256": "aa", "all_controls_sha256": "bb",
                "stopwords_sha256": None}
        _write_fold(tmp_path, "A", [1, 0], inputs=same)
        _write_fold(tmp_path, "B", [1, 1], inputs=same)

        records = load_fold_results(tmp_path, expected_frameworks=self._FRAMEWORKS)

        assert len(records) == 2
        assert records[0]["inputs"] == same

    def test_a_reparsed_corpus_mid_run_is_refused(self, tmp_path: Path) -> None:
        """Re-running a parser changes the anchors without moving the SHA.

        A fold trained before the fix and one trained after are two
        experiments wearing the same commit, and averaging them describes
        neither.
        """
        _write_fold(tmp_path, "A", [1, 0], inputs={
            "curated_links_sha256": "aa", "all_controls_sha256": "OLD",
            "stopwords_sha256": None,
        })
        _write_fold(tmp_path, "B", [1, 1], inputs={
            "curated_links_sha256": "aa", "all_controls_sha256": "NEW",
            "stopwords_sha256": None,
        })

        with pytest.raises(ValueError, match="different input data"):
            load_fold_results(tmp_path, expected_frameworks=self._FRAMEWORKS)

    def test_the_error_names_the_artifact_that_moved(self, tmp_path: Path) -> None:
        _write_fold(tmp_path, "A", [1, 0], inputs={
            "curated_links_sha256": "same", "stopwords_sha256": "OLD",
        })
        _write_fold(tmp_path, "B", [1, 1], inputs={
            "curated_links_sha256": "same", "stopwords_sha256": "NEW",
        })

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(tmp_path, expected_frameworks=self._FRAMEWORKS)

        message = str(excinfo.value)
        assert "stopwords_sha256" in message
        # The artifact that did NOT move must not be blamed.
        assert "curated_links_sha256" not in message

    def test_a_stopword_arm_cannot_absorb_a_plain_fold(self, tmp_path: Path) -> None:
        """Same corpus, one fold filtered and one not, is still two arms."""
        _write_fold(tmp_path, "A", [1, 0], inputs={
            "curated_links_sha256": "aa", "stopwords_sha256": "ff",
        })
        _write_fold(tmp_path, "B", [1, 1], inputs={
            "curated_links_sha256": "aa", "stopwords_sha256": None,
        })

        with pytest.raises(ValueError, match="different input data"):
            load_fold_results(tmp_path, expected_frameworks=self._FRAMEWORKS)

    def test_records_without_provenance_still_load_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Older records predate the inputs block; refusing them helps nobody."""
        _write_fold(tmp_path, "A", [1, 0])
        _write_fold(tmp_path, "B", [1, 1])

        with caplog.at_level("WARNING"):
            records = load_fold_results(tmp_path, expected_frameworks=self._FRAMEWORKS)

        assert len(records) == 2
        assert "data snapshot" in caplog.text


class TestArmLabelling:
    """One arm must not appear under two names in the WandB UI.

    run_fold labels from a TrainingConfig before the run; runpod_parallel
    labels from the persisted config block after it. They are separate
    functions in separate modules, so agreement is asserted, not assumed.
    """

    def test_both_labellers_agree_on_every_arm(self) -> None:
        from scripts.phase1b.run_fold import _arm_label
        from scripts.phase1b.runpod_parallel import _arm_from_config
        from tract.training.config import TrainingConfig

        seen: set[str] = set()
        for use_prose in (True, False):
            for stopwords in (True, False):
                for desc_only in (True, False):
                    config = TrainingConfig(
                        name="t",
                        use_prose=use_prose,
                        use_stopword_filter=stopwords,
                        use_description_only=desc_only,
                    )
                    before = _arm_label(config)
                    after = _arm_from_config(config.to_dict())
                    assert before == after, (use_prose, stopwords, desc_only)
                    seen.add(before)

        # The four arms of this campaign, plus the title-only baseline's
        # collapse: with use_prose off the other flags do not apply.
        assert "title-only" in seen
        assert "prose" in seen
        assert "prose-stopwords" in seen
        assert "prose-desconly" in seen
