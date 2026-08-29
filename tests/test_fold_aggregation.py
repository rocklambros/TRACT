"""Tests for per-fold result persistence and micro-aggregation.

The RunPod path wrote a per-fold summary that kept only the metric dict and
dropped the per-item hit@1 indicators, and nothing aggregated the five summaries
at all. Averaging them by hand is a MACRO average: it weights a 6-item fold the
same as a 60-item one. TRACT reports a MICRO average, which needs the raw
indicators from every fold.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

# orchestrate imports tract.training.data, which imports torch and datasets.
# The default CI test job installs requirements.txt only, and it runs pytest
# with -x, so an unguarded import here does not skip this file -- it aborts the
# whole job at collection. Guard on what is actually imported, not on numpy.
pytest.importorskip("numpy")
pytest.importorskip("torch", reason="needs the phase0 extra")
pytest.importorskip("datasets", reason="needs the phase0 extra")

from tract import staleness
from tract.config import FOLD_RESULT_FILENAME, PROJECT_ROOT
from tract.staleness import TRACKED_INPUTS, check_result
from tract.text_selection import merged_corpus_path, merged_corpus_sha256
from tract.training.data_quality import fold_input_digests
from tract.training.orchestrate import (
    aggregate_fold_results,
    load_fold_results,
)


def _write_fold(
    root: Path,
    framework: str,
    indicators: list[float],
    git_sha: str = "abc1234",
    zero_shot: list[float] | None = None,
    inputs: dict[str, str | None] | None = None,
    config: dict[str, object] | None = None,
    metrics: dict[str, float] | None = None,
    zero_shot_metrics: dict[str, float] | None = None,
) -> None:
    """Write one fold_result.json.

    metrics defaults to the summary the indicators actually imply, which is
    what a real producer writes. Passing it explicitly is how a test states
    the case where a record's two accounts of its own hit@1 disagree.
    """
    fold_dir = root / f"fold_{framework.replace(' ', '_')}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "held_out_framework": framework,
        "metrics": (
            {"hit_at_1": sum(indicators) / len(indicators)}
            if metrics is None else metrics
        ),
        "hit1_indicators": indicators,
        "n_eval_items": len(indicators),
        "n_training_pairs": 100,
        "elapsed_s": 1.0,
        "git_sha": git_sha,
    }
    if config is not None:
        record["config"] = config
    if inputs is not None:
        record["inputs"] = inputs
    if zero_shot is not None:
        block: dict[str, object] = {"hit1_indicators": zero_shot}
        if zero_shot_metrics is not None:
            block["metrics"] = zero_shot_metrics
        record["zero_shot"] = block
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
                    for fwid in (True, False):
                        config = TrainingConfig(
                            name="t",
                            use_prose=use_prose,
                            use_stopword_filter=stopwords,
                            use_description_only=desc_only,
                            use_framework_identity_filter=fwid,
                        )
                        before = _arm_label(config)
                        after = _arm_from_config(config.to_dict())
                        assert before == after, (
                            use_prose, stopwords, desc_only, fwid,
                        )
                        seen.add(before)

        # The arms of this campaign, plus the title-only baseline's collapse:
        # with use_prose off the other flags do not apply.
        assert "title-only" in seen
        assert "prose" in seen
        assert "prose-stopwords" in seen
        assert "prose-desconly" in seen
        assert "prose-fwid" in seen
        assert "prose-stopwords-fwid" in seen


class TestArmSeparation:
    """Every flag that defines an arm must separate folds.

    The guard listed use_prose and use_stopword_filter while the campaign runs
    four arms, so prose and prose-desconly both hashed to (True, False) and
    would have aggregated into one number describing neither.
    """

    _FRAMEWORKS = {"A", "B"}

    def _config(self, **flags: object) -> dict[str, object]:
        base: dict[str, object] = {
            "use_prose": True,
            "use_stopword_filter": False,
            "use_description_only": False,
        }
        base.update(flags)
        return base

    def test_one_arm_aggregates(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fold(root, "A", [1, 0], config=self._config())
            _write_fold(root, "B", [1, 1], config=self._config())
            assert len(load_fold_results(root, expected_frameworks=self._FRAMEWORKS)) == 2

    def test_prose_and_description_only_do_not_merge(self) -> None:
        """The exact pair the old guard could not tell apart."""
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fold(root, "A", [1, 0], config=self._config())
            _write_fold(root, "B", [1, 1],
                        config=self._config(use_description_only=True))
            with pytest.raises(ValueError, match="different arms"):
                load_fold_results(root, expected_frameworks=self._FRAMEWORKS)

    def test_prose_and_stopwords_do_not_merge(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fold(root, "A", [1, 0], config=self._config())
            _write_fold(root, "B", [1, 1],
                        config=self._config(use_stopword_filter=True))
            with pytest.raises(ValueError, match="different arms"):
                load_fold_results(root, expected_frameworks=self._FRAMEWORKS)

    def test_title_only_and_prose_do_not_merge(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_fold(root, "A", [1, 0], config=self._config(use_prose=False))
            _write_fold(root, "B", [1, 1], config=self._config())
            with pytest.raises(ValueError, match="different arms"):
                load_fold_results(root, expected_frameworks=self._FRAMEWORKS)

    def test_the_guard_covers_every_arm_flag_the_config_defines(self) -> None:
        """Adding a fifth arm flag without extending the guard is a bug.

        Asserted structurally so the omission shows up here rather than as two
        arms quietly averaging together.
        """
        from tract.training.config import TrainingConfig
        from tract.training.orchestrate import ARM_DEFINING_KEYS

        serialized = TrainingConfig(name="t").to_dict()
        for key in ARM_DEFINING_KEYS:
            assert key in serialized, key
        arm_flags = {
            k for k in serialized
            if k.startswith("use_") and isinstance(serialized[k], bool)
        }
        assert arm_flags <= set(ARM_DEFINING_KEYS), (
            f"TrainingConfig has boolean use_* flags {sorted(arm_flags)} not "
            f"covered by the guard {sorted(ARM_DEFINING_KEYS)}"
        )
        # Everything that changes what a run IS, beyond the anchor arms.
        for key in ("branch_balance_temperature", "base_model", "max_seq_length"):
            assert key in ARM_DEFINING_KEYS, (
                f"{key} changes the experiment but does not separate folds"
            )


class TestConfigurationSeparation:
    """Every dimension the campaign varies must keep folds apart."""

    _FRAMEWORKS = {"A", "B"}

    def _config(self, **flags: object) -> dict[str, object]:
        base: dict[str, object] = {
            "use_prose": True, "use_stopword_filter": False,
            "use_description_only": False, "branch_balance_temperature": 0.0,
            "base_model": "BAAI/bge-large-en-v1.5", "max_seq_length": 512,
        }
        base.update(flags)
        return base

    def _refuses(self, tmp_path: Path, **differing: object) -> None:
        _write_fold(tmp_path, "A", [1, 0], config=self._config())
        _write_fold(tmp_path, "B", [1, 1], config=self._config(**differing))
        with pytest.raises(ValueError, match="different arms"):
            load_fold_results(tmp_path, expected_frameworks=self._FRAMEWORKS)

    def test_branch_balance_separates(self, tmp_path: Path) -> None:
        """A rebalanced fold cannot average with an unbalanced one."""
        self._refuses(tmp_path, branch_balance_temperature=3.0)

    def test_encoder_separates(self, tmp_path: Path) -> None:
        self._refuses(tmp_path, base_model="Alibaba-NLP/gte-modernbert-base")

    def test_sequence_length_separates(self, tmp_path: Path) -> None:
        """Same encoder, different token budget, is a different experiment."""
        self._refuses(tmp_path, max_seq_length=8192)

    def test_an_identical_configuration_still_aggregates(
        self, tmp_path: Path
    ) -> None:
        _write_fold(tmp_path, "A", [1, 0], config=self._config(
            branch_balance_temperature=3.0, max_seq_length=8192))
        _write_fold(tmp_path, "B", [1, 1], config=self._config(
            branch_balance_temperature=3.0, max_seq_length=8192))
        assert len(load_fold_results(
            tmp_path, expected_frameworks=self._FRAMEWORKS)) == 2


class TestIndicatorDomain:
    """Every guard above this class checks SHAPE. None of them checked VALUE.

    A red-team pass wrote five fold_result.json files whose hit1_indicators
    were all 7.0, ran them through the real load_fold_results ->
    aggregate_fold_results -> gate_decision path, and got "AGGREGATE hit@1
    (micro): 7.0000" with point_estimate_pass, ci_low_pass and verdicts_agree
    all true. Presence, length, pairing, arm, inputs and git SHA were every
    one of them satisfied: nothing between the file and the headline number
    ever asked what the numbers meant.
    """

    # The five folds of the Campaign 2 arms, so the replay is the attack as it
    # was actually run rather than a two-fold miniature of it.
    _CAMPAIGN_FOLDS = ("ASVS", "CAPEC", "CWE", "ISO_27001", "NIST_800-53_v5")

    def test_the_fabricated_seven_campaign_is_refused(self, tmp_path: Path) -> None:
        for framework in self._CAMPAIGN_FOLDS:
            _write_fold(tmp_path, framework, [7] * 4, zero_shot=[1, 0, 0, 0])

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(
                tmp_path, expected_frameworks=set(self._CAMPAIGN_FOLDS),
            )

        message = str(excinfo.value)
        # The file, so the reader knows which fold to go and look at.
        assert "fold_ASVS" in message
        assert FOLD_RESULT_FILENAME in message
        # The field and the value, so the reader knows what they are looking at.
        assert "hit1_indicators" in message
        assert "7" in message

    def test_the_fabricated_campaign_never_reaches_an_aggregate(
        self, tmp_path: Path
    ) -> None:
        """The attack's payoff was a number, so assert the number never exists."""
        for framework in self._CAMPAIGN_FOLDS:
            _write_fold(tmp_path, framework, [7] * 4, zero_shot=[1, 0, 0, 0])

        with pytest.raises(ValueError):
            aggregate_fold_results(load_fold_results(
                tmp_path, expected_frameworks=set(self._CAMPAIGN_FOLDS),
            ))

    def test_a_probability_is_not_an_indicator(self, tmp_path: Path) -> None:
        """The producer-side version of the same defect.

        hit1_indicators and the raw cosine similarities are both per-item float
        arrays of the same length. Writing the wrong one is a one-line mistake
        that every shape check in this file passes.
        """
        _write_fold(tmp_path, "A", [1, 0, 0.5, 1])

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(tmp_path, expected_frameworks={"A"})

        assert "0.5" in str(excinfo.value)

    def test_the_zero_shot_baseline_is_held_to_the_same_domain(
        self, tmp_path: Path
    ) -> None:
        """The delta is trained minus baseline, so a poisoned baseline is a
        poisoned gate."""
        _write_fold(tmp_path, "A", [1, 0, 1], zero_shot=[1, 0, 7])

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(tmp_path, expected_frameworks={"A"})

        message = str(excinfo.value)
        assert "zero_shot" in message
        assert "7" in message

    def test_the_error_quotes_the_bad_values_without_dumping_the_array(
        self, tmp_path: Path
    ) -> None:
        """A 1,400-item fold must not print 1,400 numbers into a CI log."""
        _write_fold(tmp_path, "A", list(range(2, 40)))

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(tmp_path, expected_frameworks={"A"})

        message = str(excinfo.value)
        assert "2, 3, 4, 5, 6, 7, 8, 9" in message
        assert "and 30 more distinct values" in message

    def test_a_non_numeric_entry_is_named_rather_than_crashing(
        self, tmp_path: Path
    ) -> None:
        """A truncated or hand-edited record can hold a string or a null.

        The check must report those as offending values, not raise TypeError
        from inside a comparison and leave the reader with a stack trace.
        """
        _write_fold(tmp_path, "A", [1, 0], metrics={"hit_at_1": 0.5})
        path = tmp_path / "fold_A" / FOLD_RESULT_FILENAME
        record = json.loads(path.read_text(encoding="utf-8"))
        record["hit1_indicators"] = [1, None, "1"]
        record["n_eval_items"] = 3
        path.write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(tmp_path, expected_frameworks={"A"})

        message = str(excinfo.value)
        assert "None" in message
        assert "'1'" in message

    def test_legitimate_indicators_still_load(self, tmp_path: Path) -> None:
        """Ints and floats both, because two producers write each.

        run_single_fold casts to int; rescore_predictions.py writes
        hit1.tolist() off a float array. Both are legitimate and both must pass.
        """
        _write_fold(tmp_path, "A", [1, 0, 1])
        _write_fold(tmp_path, "B", [1.0, 0.0], zero_shot=[0.0, 0.0])

        records = load_fold_results(tmp_path, expected_frameworks={"A", "B"})

        assert [r["held_out_framework"] for r in records] == ["A", "B"]
        assert records[1]["hit1_indicators"] == [1.0, 0.0]


class TestSummaryAgreesWithIndicators:
    """A fold record states its hit@1 twice and nothing made them agree.

    metrics.hit_at_1 is what a human reads; hit1_indicators is what the micro
    average, the bootstrap CI and the gate are computed from. The domain check
    above catches an array that was never indicators. This catches the subtler
    producer-side case: a well-formed array that no longer describes the
    summary sitting beside it in the same file.
    """

    def test_a_summary_that_disagrees_with_its_own_indicators_is_refused(
        self, tmp_path: Path
    ) -> None:
        _write_fold(tmp_path, "A", [1, 0, 1, 0], metrics={"hit_at_1": 0.9})

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(tmp_path, expected_frameworks={"A"})

        message = str(excinfo.value)
        assert "fold_A" in message
        # Both numbers, because the reader cannot tell which one moved.
        assert "0.9" in message
        assert "0.5" in message

    def test_one_flipped_indicator_in_a_large_fold_is_caught(
        self, tmp_path: Path
    ) -> None:
        """The tolerance must not be wide enough to hide a real edit.

        1,000 items, one indicator changed after the summary was written: the
        two accounts differ by 0.001, six orders of magnitude above the
        float-noise tolerance.
        """
        indicators = [1] * 500 + [0] * 500
        _write_fold(tmp_path, "A", indicators, metrics={"hit_at_1": 0.501})

        with pytest.raises(ValueError, match="disagree"):
            load_fold_results(tmp_path, expected_frameworks={"A"})

    def test_float_noise_does_not_trip_it(self, tmp_path: Path) -> None:
        """The two numbers are computed in two modules, so allow the last bit.

        A fold that is genuinely one-third correct must load whether its
        summary was written by sum/n or by np.mean.
        """
        _write_fold(tmp_path, "A", [1, 0, 0], metrics={"hit_at_1": 1 / 3 + 1e-12})

        records = load_fold_results(tmp_path, expected_frameworks={"A"})

        assert len(records) == 1
        assert records[0]["metrics"]["hit_at_1"] == pytest.approx(1 / 3, abs=1e-9)

    def test_a_record_with_no_hit_at_1_is_refused(self, tmp_path: Path) -> None:
        """Skipping the check when the field is absent is how guards die.

        Every producer of this file writes metrics.hit_at_1; all 32 committed
        records carry it. A record without it was assembled somewhere else,
        which is exactly the case this check exists for.
        """
        _write_fold(tmp_path, "A", [1, 0], metrics={})

        with pytest.raises(ValueError, match="hit_at_1"):
            load_fold_results(tmp_path, expected_frameworks={"A"})

    def test_a_non_numeric_summary_is_refused(self, tmp_path: Path) -> None:
        _write_fold(tmp_path, "A", [1, 0])
        path = tmp_path / "fold_A" / FOLD_RESULT_FILENAME
        record = json.loads(path.read_text(encoding="utf-8"))
        record["metrics"]["hit_at_1"] = "0.5"
        path.write_text(json.dumps(record), encoding="utf-8")

        with pytest.raises(ValueError, match="hit_at_1"):
            load_fold_results(tmp_path, expected_frameworks={"A"})

    def test_the_zero_shot_summary_is_cross_checked_when_present(
        self, tmp_path: Path
    ) -> None:
        _write_fold(
            tmp_path, "A", [1, 0], zero_shot=[0, 0],
            zero_shot_metrics={"hit_at_1": 0.25},
        )

        with pytest.raises(ValueError) as excinfo:
            load_fold_results(tmp_path, expected_frameworks={"A"})

        message = str(excinfo.value)
        assert "zero_shot" in message
        assert "0.25" in message

    def test_a_zero_shot_block_of_indicators_alone_still_loads(
        self, tmp_path: Path
    ) -> None:
        """The block is legitimately allowed to carry indicators and nothing
        else; the pairing check is what makes it useful, not the summary."""
        _write_fold(tmp_path, "A", [1, 0], zero_shot=[0, 0])

        assert len(load_fold_results(tmp_path, expected_frameworks={"A"})) == 1

    def test_a_fold_with_no_eval_items_is_refused(self, tmp_path: Path) -> None:
        """Zero items passes the length check (0 == 0) and divides by zero here."""
        fold_dir = tmp_path / "fold_A"
        fold_dir.mkdir(parents=True)
        (fold_dir / FOLD_RESULT_FILENAME).write_text(
            json.dumps({
                "held_out_framework": "A",
                "metrics": {"hit_at_1": 0.0},
                "hit1_indicators": [],
                "n_eval_items": 0,
            }),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="empty"):
            load_fold_results(tmp_path, expected_frameworks={"A"})


class TestAggregateRangeGuard:
    """The loader is not on every path into the aggregate.

    run_experiment hands aggregate_fold_results the in-memory dicts
    run_single_fold returned; they never touch a file and never meet the
    domain check. One guard is not enough, which is the entire lesson of the
    7.0 replay above.
    """

    def test_a_micro_mean_outside_zero_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="outside"):
            aggregate_fold_results([
                {"held_out_framework": "A", "hit1_indicators": [7, 7],
                 "n_eval_items": 2},
            ])

    def test_the_macro_figure_is_checked_independently_of_the_micro(self) -> None:
        """A large clean fold can drag the micro back into range on its own.

        100 zeros pooled with a single 3.0 gives a micro of 0.0297, which no
        range check on the micro alone would ever flag. The macro average of
        the same two folds is 1.5.
        """
        folds = [
            {"held_out_framework": "Big", "hit1_indicators": [0] * 100,
             "n_eval_items": 100},
            {"held_out_framework": "Tiny", "hit1_indicators": [3],
             "n_eval_items": 1},
        ]

        with pytest.raises(ValueError, match="macro"):
            aggregate_fold_results(folds)

    def test_a_perfect_fold_is_not_a_false_positive(self) -> None:
        """1.0 is a legitimate hit rate and sits exactly on the bound."""
        result = aggregate_fold_results([
            {"held_out_framework": "A", "hit1_indicators": [1, 1, 1],
             "n_eval_items": 3},
            {"held_out_framework": "B", "hit1_indicators": [1, 1],
             "n_eval_items": 2},
        ])

        assert result["mean"] == pytest.approx(1.0)
        assert result["macro_mean"] == pytest.approx(1.0)

    def test_a_shut_out_fold_is_not_a_false_positive(self) -> None:
        """And so is 0.0, on the other bound."""
        result = aggregate_fold_results([
            {"held_out_framework": "A", "hit1_indicators": [0, 0],
             "n_eval_items": 2},
        ])

        assert result["mean"] == pytest.approx(0.0)
        assert result["macro_mean"] == pytest.approx(0.0)


class TestTheGuardsAcceptRealResults:
    """A false positive here destroys a paid run's output.

    These assertions sit on the path that reads the campaign's real results,
    so they are held to the committed evidence: every fold_result.json git
    knows about under results/phase1b must still load, and must still load
    for the reasons the guards state rather than by accident. If a real
    record ever fails one of them, the record is the finding -- the assertion
    does not get loosened to make the suite green.
    """

    def _committed_records(self) -> dict[Path, list[Path]]:
        """Committed fold records, grouped by experiment directory.

        git rather than a glob: a developer's uncommitted scratch run under
        results/phase1b is not evidence anyone published, and sweeping it in
        would fail this test for a reason that has nothing to do with the
        guards.
        """
        proc = subprocess.run(
            ["git", "ls-files", "results/phase1b/*/fold_*/" + FOLD_RESULT_FILENAME],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        by_experiment: dict[Path, list[Path]] = {}
        for line in proc.stdout.splitlines():
            path = PROJECT_ROOT / line.strip()
            by_experiment.setdefault(path.parent.parent, []).append(path)
        return by_experiment

    def test_every_committed_fold_record_still_loads(self) -> None:
        by_experiment = self._committed_records()

        # The published LOFO arms and the Campaign 2 arm the remaining
        # configurations are built from. Naming them means a checkout that
        # lost the results fails here instead of passing over an empty set.
        names = {d.name for d in by_experiment}
        assert {"c2_A1_prose_sw_bge", "lofo_prose", "lofo_title_only"} <= names

        checked = 0
        for experiment, paths in sorted(by_experiment.items()):
            expected = {
                json.loads(p.read_text(encoding="utf-8"))["held_out_framework"]
                for p in paths
            }
            records = load_fold_results(experiment, expected_frameworks=expected)
            assert len(records) == len(paths), experiment
            checked += len(records)

        assert checked >= 30, f"only {checked} committed fold records were checked"

    def test_every_committed_record_is_genuinely_in_domain(self) -> None:
        """Stated here too, so the evidence survives a change to the loader."""
        for paths in self._committed_records().values():
            for path in paths:
                record = json.loads(path.read_text(encoding="utf-8"))
                indicators = record["hit1_indicators"]
                assert set(indicators) <= {0, 1}, path
                assert record["metrics"]["hit_at_1"] == pytest.approx(
                    sum(indicators) / len(indicators), abs=1e-9,
                ), path
                zero_shot = record.get("zero_shot") or {}
                if "hit1_indicators" in zero_shot:
                    assert set(zero_shot["hit1_indicators"]) <= {0, 1}, path


class TestTheStalenessCheckReadsWhatTheFoldWrote:
    """A fold hashes the corpus it TRAINED ON. The check hashed a different file.

    fold_input_digests writes all_controls_sha256 = merged_corpus_sha256(),
    which resolves through merged_corpus_path() to the licensed overlay at
    data/processed/licensed/all_controls.json whenever that overlay is staged
    -- and it must be staged, because a run without it silently trains on a
    corpus missing every restricted framework. tract.staleness compared that
    digest against a hardcoded data/processed/all_controls.json. The overlay is
    the tracked corpus plus the restricted frameworks, so the two files can
    never hold the same bytes, and during a real campaign EVERY fold was
    therefore reported stale. Replayed against the arms as configured, with the
    overlay on disk: A1 prose+sw refused on all_controls_sha256, A3
    prose+sw+qwen refused on all_controls_sha256, and A5 --no-prose proceeded
    only because it records no corpus digest at all. A flawless five-fold run
    produced no number anyone was allowed to quote.

    Both directions are asserted on purpose. Widening the check until A1 passes
    was the cheap fix and the wrong one: what makes the aggregate worth quoting
    is that a corpus which really did move is still caught, and the arms above
    would have been just as green under a check that had simply stopped
    looking.
    """

    # The corpus digest every committed Campaign 2 fold record carries, read
    # from results/phase1b/c2_A1_prose_sw_bge/fold_ASVS/fold_result.json. It
    # names a real corpus that is genuinely no longer on disk, which makes it a
    # far better inverse case than 64 zeros: a check can reject an obvious
    # sentinel and still bless a plausible one.
    _PRE_REBUILD_CORPUS_SHA256 = (
        "776be12eb54289e9b94c7eecc061922fbf28ff39dcf4d61108e65d8d144aeb41"
    )

    def _campaign(
        self, root: Path, inputs: dict[str, str | None],
    ) -> Path:
        """Two folds of one arm, recording the digests a real producer wrote."""
        for framework in ("ASVS", "CAPEC"):
            _write_fold(root, framework, [1, 0], inputs=inputs)
        return root / "fold_ASVS" / FOLD_RESULT_FILENAME

    def test_the_corpus_a_fold_trained_on_reads_back_as_current(
        self, tmp_path: Path
    ) -> None:
        """The campaign-fatal case, stated against the real producer.

        Not a hand-written digest: fold_input_digests is the function
        run_single_fold calls, so whatever it writes here is byte-for-byte what
        a paid fold would have written.
        """
        digests = fold_input_digests(
            with_prose=True, with_stopwords=True, with_framework_identity=False,
        )
        assert digests["all_controls_sha256"] == merged_corpus_sha256()

        status = check_result(self._campaign(tmp_path, digests))

        assert not status.is_stale, (
            f"a fold that recorded the corpus it read was called stale: "
            f"{[(s.field, s.path) for s in status.stale]}"
        )
        assert status.is_checkable

    def test_a_corpus_that_really_moved_is_still_reported_stale(
        self, tmp_path: Path
    ) -> None:
        """The inverse, so the fix cannot be a blanket weakening."""
        digests = dict(fold_input_digests(
            with_prose=True, with_stopwords=True, with_framework_identity=False,
        ))
        digests["all_controls_sha256"] = self._PRE_REBUILD_CORPUS_SHA256

        status = check_result(self._campaign(tmp_path, digests))

        assert status.is_stale
        moved = {item.field: item for item in status.stale}
        assert "all_controls_sha256" in moved
        assert moved["all_controls_sha256"].recorded == self._PRE_REBUILD_CORPUS_SHA256
        # The digest the reader offers as "current" has to be the digest the
        # writer would record today, or the report names a file nobody read.
        assert moved["all_controls_sha256"].current == merged_corpus_sha256()
        # Only the corpus moved; blaming the curated links as well would send a
        # reader to re-run a parser that is fine.
        assert "curated_links_sha256" not in moved

    def test_the_check_resolves_the_path_the_producer_hashed(self) -> None:
        """One source of truth, asserted rather than assumed.

        The two sides used to name the file independently, in two modules, and
        nothing said they had to agree.
        """
        assert TRACKED_INPUTS["all_controls_sha256"] == merged_corpus_path()

    def test_the_check_follows_the_corpus_rather_than_a_literal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The regression has to bite in a checkout with no overlay staged.

        Without the licensed source on disk merged_corpus_path() falls back to
        the tracked corpus -- which is exactly the literal the check used to
        hold -- so every assertion above would pass straight over the bug on a
        CI runner. Pointing the producer at a corpus neither path names is the
        only way to state "the reader resolves whatever the writer resolved"
        independently of which corpus this particular checkout happens to hold.
        """
        corpus = tmp_path / "corpus" / "all_controls.json"
        corpus.parent.mkdir(parents=True)
        corpus.write_text('{"frameworks": []}', encoding="utf-8")
        monkeypatch.setattr(staleness, "merged_corpus_path", lambda: corpus)

        digests: dict[str, str | None] = {
            "all_controls_sha256": hashlib.sha256(
                corpus.read_bytes()
            ).hexdigest(),
        }

        assert not check_result(self._campaign(tmp_path, digests)).is_stale

    def test_every_digest_a_fold_records_is_a_digest_the_check_reads(
        self,
    ) -> None:
        """A field the writer writes and the reader ignores cannot go stale.

        fold_input_digests writes four; TRACKED_INPUTS listed three, so a
        framework-identity token set that changed between folds was invisible
        to the only instrument that looks. Asserted structurally, because the
        symptom of the omission is silence.
        """
        written = set(fold_input_digests(
            with_prose=True, with_stopwords=True, with_framework_identity=True,
        ))
        assert written == set(TRACKED_INPUTS)

    def test_a_clean_campaign_reaches_an_aggregate(self, tmp_path: Path) -> None:
        """End to end through the gate that actually refused.

        check_result reports; _assert_results_are_current is what turns the
        report into a RuntimeError and costs a campaign its result.
        """
        from scripts.phase1b import runpod_parallel as rpp

        self._campaign(tmp_path, fold_input_digests(
            with_prose=True, with_stopwords=True, with_framework_identity=False,
        ))

        rpp._assert_results_are_current(tmp_path, allow_stale=False)

    def test_a_campaign_measured_on_a_corpus_that_moved_is_refused(
        self, tmp_path: Path
    ) -> None:
        digests = dict(fold_input_digests(
            with_prose=True, with_stopwords=True, with_framework_identity=False,
        ))
        digests["all_controls_sha256"] = self._PRE_REBUILD_CORPUS_SHA256
        self._campaign(tmp_path, digests)

        from scripts.phase1b import runpod_parallel as rpp

        with pytest.raises(RuntimeError, match="all_controls_sha256"):
            rpp._assert_results_are_current(tmp_path, allow_stale=False)

    def test_every_committed_record_can_still_be_spoken_about(self) -> None:
        """Widening TRACKED_INPUTS must not silence the 32 published records.

        is_checkable is a ratio against the number of tracked inputs, so adding
        a fourth field moves the bar under every record already on disk. The
        five title-only folds record the fewest digests of any of them and are
        the ones this would fall over first.
        """
        proc = subprocess.run(
            ["git", "ls-files", "results/phase1b/*/fold_*/" + FOLD_RESULT_FILENAME],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        paths = [PROJECT_ROOT / line.strip() for line in proc.stdout.splitlines()]
        assert len(paths) >= 30, f"only {len(paths)} committed records found"

        for path in paths:
            assert check_result(path).is_checkable, path
