"""Tests for tract/training/tracking.py.

The point of this module is that it fails closed. Phase 0's init_wandb returns
None and warns when the key is missing, which is right for a local probe and
wrong for a fleet: pods that each degrade to untracked spend the whole budget
before anyone notices the runs are not appearing.
"""
from __future__ import annotations

from typing import Any

import pytest

from tract.training.tracking import (
    WANDB_KEY_LENGTH,
    finish_run,
    log_fold,
    resolve_api_key,
)

VALID_KEY = "a" * WANDB_KEY_LENGTH


class TestResolveApiKey:

    def test_a_well_formed_key_from_the_environment_is_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", VALID_KEY)
        assert resolve_api_key() == VALID_KEY

    def test_surrounding_whitespace_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key pasted from a terminal usually arrives with a newline."""
        monkeypatch.setenv("WANDB_API_KEY", f"  {VALID_KEY}\n")
        assert resolve_api_key() == VALID_KEY

    def test_a_username_where_a_key_belongs_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real failure this was written for.

        `pass wandb/api-key` held a 5-character username. Without a shape
        check that value reaches wandb.init, which falls back to an
        interactive login prompt and hangs a headless pod until its deadline.
        """
        monkeypatch.setenv("WANDB_API_KEY", "rockc")
        with pytest.raises(RuntimeError, match="not a WandB API key"):
            resolve_api_key()

    def test_a_right_length_non_hex_value_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "z" * WANDB_KEY_LENGTH)
        with pytest.raises(RuntimeError, match="not a WandB API key"):
            resolve_api_key()

    def test_the_error_says_where_the_bad_value_came_from(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "short")
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            resolve_api_key()

    def test_uppercase_hex_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Built rather than written literally: a 40-char hex string in a source
        # file is what secret scanners are for, and an allowlist entry to
        # silence a test fixture would weaken the scanner for real code.
        key = (VALID_KEY[:-6] + "abcdef").upper()
        monkeypatch.setenv("WANDB_API_KEY", key)
        assert resolve_api_key() == key


class _FakeRun:
    """Captures what would have been sent to WandB."""

    url = "https://wandb.test/run"

    def __init__(self) -> None:
        self.logged: list[dict[str, Any]] = []
        self.exit_code: int | None = None

    def log(self, data: dict[str, Any]) -> None:
        self.logged.append(data)

    def finish(self, exit_code: int = 0) -> None:
        self.exit_code = exit_code


class TestLogFold:

    def _record(self) -> dict[str, Any]:
        return {
            "held_out_framework": "MITRE ATLAS",
            "n_eval_items": 40,
            "n_training_pairs": 3000,
            "elapsed_s": 900.0,
            "metrics": {"hit_at_1": 0.6, "hit_at_5": 0.9, "mrr": 0.7},
            "zero_shot": {"metrics": {"hit_at_1": 0.25}},
            "text_selection": {
                "prose_fraction": 0.85,
                "by_source": {"description": 30, "full_text": 4},
                "n_truncated_at_encoder_budget": 2,
            },
        }

    def test_metrics_are_flattened_to_scalars(self) -> None:
        """WandB charts a scalar, not a nested object."""
        run = _FakeRun()
        log_fold(run, self._record())

        payload = run.logged[0]
        assert payload["metrics/hit_at_1"] == 0.6
        assert payload["fold/held_out_framework"] == "MITRE ATLAS"
        assert payload["text_selection/by_source/description"] == 30

    def test_the_paired_delta_is_logged(self) -> None:
        """The trained-versus-baseline gap is the result, not either side."""
        run = _FakeRun()
        log_fold(run, self._record())

        payload = run.logged[0]
        assert payload["zero_shot/hit_at_1"] == 0.25
        assert payload["delta/hit_at_1"] == pytest.approx(0.35)

    def test_no_delta_is_invented_without_a_baseline(self) -> None:
        record = self._record()
        del record["zero_shot"]
        run = _FakeRun()
        log_fold(run, record)

        assert "delta/hit_at_1" not in run.logged[0]

    def test_an_untracked_run_is_a_no_op(self) -> None:
        """Callers should not need a branch around every log site."""
        log_fold(None, self._record())
        finish_run(None)

    def test_finish_propagates_the_exit_code(self) -> None:
        run = _FakeRun()
        finish_run(run, exit_code=1)
        assert run.exit_code == 1


class TestStableRunId:
    """WandB names are not unique keys; ids are.

    Without a stable id, re-running the logging step after a partial collect
    creates a second copy of every fold already present, and the project shows
    two populations of the same experiment.
    """

    def test_the_same_fold_yields_the_same_id(self) -> None:
        from tract.training.tracking import stable_run_id

        first = stable_run_id("campaign", "prose", "MITRE ATLAS")
        second = stable_run_id("campaign", "prose", "MITRE ATLAS")
        assert first == second

    def test_different_arms_do_not_collide(self) -> None:
        from tract.training.tracking import stable_run_id

        assert stable_run_id("c", "prose", "ATLAS") != stable_run_id(
            "c", "prose-stopwords", "ATLAS"
        )

    def test_different_folds_do_not_collide(self) -> None:
        from tract.training.tracking import stable_run_id

        assert stable_run_id("c", "prose", "ATLAS") != stable_run_id(
            "c", "prose", "NIST AI 100-2"
        )

    def test_different_campaigns_do_not_collide(self) -> None:
        from tract.training.tracking import stable_run_id

        assert stable_run_id("a", "prose", "ATLAS") != stable_run_id(
            "b", "prose", "ATLAS"
        )

    def test_parts_cannot_be_confused_by_concatenation(self) -> None:
        """("ab","c") and ("a","bc") must not hash alike."""
        from tract.training.tracking import stable_run_id

        assert stable_run_id("ab", "c") != stable_run_id("a", "bc")

    def test_the_id_is_wandb_safe(self) -> None:
        from tract.training.tracking import stable_run_id

        run_id = stable_run_id("c", "prose", "OWASP Top10 for LLM")
        assert run_id.isalnum()
        assert len(run_id) == 16

    def test_a_self_hosted_key_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Self-hosted WandB issues "local-" prefixed keys.

        Requiring bare hex rejected a legitimate credential on any non-cloud
        server, which is a preflight that blocks the run for no reason.
        """
        from tract.training.tracking import SELF_HOSTED_PREFIX

        key = f"{SELF_HOSTED_PREFIX}{VALID_KEY}"
        monkeypatch.setenv("WANDB_API_KEY", key)
        assert resolve_api_key() == key

    def test_a_date_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """What actually landed in `pass` on the second attempt."""
        monkeypatch.setenv("WANDB_API_KEY", "09/12/2026")
        with pytest.raises(RuntimeError, match="not a WandB API key"):
            resolve_api_key()

    def test_a_prefixed_key_of_the_wrong_length_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tract.training.tracking import SELF_HOSTED_PREFIX

        monkeypatch.setenv("WANDB_API_KEY", f"{SELF_HOSTED_PREFIX}abc123")
        with pytest.raises(RuntimeError, match="not a WandB API key"):
            resolve_api_key()
