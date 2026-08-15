"""Tests for tract/training/tracking.py.

The module fails closed on credentials and open on the network. Phase 0's
init_wandb returns None and warns when the key is missing, which is right for
a local probe and wrong for a fleet: pods that each degrade to untracked spend
the whole budget before anyone notices the runs are not appearing.
"""
from __future__ import annotations

from typing import Any

import pytest

from tract.training.tracking import (
    finish_run,
    log_fold,
    resolve_api_key,
    stable_run_id,
)

LEGACY_KEY = "a" * 40
# Shaped like the key actually in use: long, mixed-case, prefixed.
MODERN_KEY = "wandb" + "Ab3" * 27


class TestResolveApiKey:
    """Shape checking here is deliberately weak.

    It used to require 40 hex digits, which was the legacy WandB format. The
    current key is longer, mixed-case and prefixed, so that check rejected a
    working credential three times running and taught nothing except to
    distrust the preflight. Guessing a vendor's credential layout is a losing
    game. The offline check catches only the cheap, common mistake -- wrong
    text pasted at a hidden prompt -- and verify_credential settles the rest
    by asking WandB.
    """

    def test_a_legacy_hex_key_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", LEGACY_KEY)
        assert resolve_api_key() == LEGACY_KEY

    def test_a_modern_prefixed_key_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """86 mixed-case characters -- the shape that was wrongly rejected."""
        monkeypatch.setenv("WANDB_API_KEY", MODERN_KEY)
        assert resolve_api_key() == MODERN_KEY

    def test_a_self_hosted_key_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", f"local-{LEGACY_KEY}")
        assert resolve_api_key() == f"local-{LEGACY_KEY}"

    def test_surrounding_whitespace_is_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key pasted from a terminal usually arrives with a newline."""
        monkeypatch.setenv("WANDB_API_KEY", f"  {LEGACY_KEY}\n")
        assert resolve_api_key() == LEGACY_KEY

    def test_a_username_where_a_key_belongs_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """What `pass wandb/api-key` actually held on the first attempt."""
        monkeypatch.setenv("WANDB_API_KEY", "rockc")
        with pytest.raises(RuntimeError, match="not a WandB API key"):
            resolve_api_key()

    def test_a_date_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """And on the second attempt."""
        monkeypatch.setenv("WANDB_API_KEY", "09/12/2026")
        with pytest.raises(RuntimeError, match="not a WandB API key"):
            resolve_api_key()

    def test_pasted_prose_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Long enough to pass a length check, but plainly not a token."""
        monkeypatch.setenv("WANDB_API_KEY", "the quick brown fox jumped over it")
        with pytest.raises(RuntimeError, match="whitespace"):
            resolve_api_key()

    def test_the_error_says_where_the_bad_value_came_from(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("WANDB_API_KEY", "short")
        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            resolve_api_key()

    def test_the_error_suggests_echo_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hidden prompt is how the wrong text got pasted three times."""
        monkeypatch.setenv("WANDB_API_KEY", "short")
        with pytest.raises(RuntimeError, match=r"-e"):
            resolve_api_key()


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
        assert stable_run_id("campaign", "prose", "MITRE ATLAS") == stable_run_id(
            "campaign", "prose", "MITRE ATLAS"
        )

    def test_different_arms_do_not_collide(self) -> None:
        assert stable_run_id("c", "prose", "ATLAS") != stable_run_id(
            "c", "prose-stopwords", "ATLAS"
        )

    def test_different_folds_do_not_collide(self) -> None:
        assert stable_run_id("c", "prose", "ATLAS") != stable_run_id(
            "c", "prose", "NIST AI 100-2"
        )

    def test_different_campaigns_do_not_collide(self) -> None:
        assert stable_run_id("a", "prose", "ATLAS") != stable_run_id(
            "b", "prose", "ATLAS"
        )

    def test_parts_cannot_be_confused_by_concatenation(self) -> None:
        """("ab","c") and ("a","bc") must not hash alike."""
        assert stable_run_id("ab", "c") != stable_run_id("a", "bc")

    def test_the_id_is_wandb_safe(self) -> None:
        run_id = stable_run_id("c", "prose", "OWASP Top10 for LLM")
        assert run_id.isalnum()
        assert len(run_id) == 16


class TestVerifyCredential:
    """The check that actually settles whether tracking will work."""

    def _response(self, status: int, payload: dict[str, Any]) -> Any:
        class _Resp:
            status_code = status
            text = str(payload)

            def json(self) -> dict[str, Any]:
                return payload

        return _Resp()

    def test_a_good_key_returns_the_viewer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests

        from tract.training import tracking

        monkeypatch.setenv("WANDB_API_KEY", LEGACY_KEY)
        monkeypatch.setattr(requests, "post", lambda *a, **k: self._response(
            200, {"data": {"viewer": {"username": "rockl", "entity": "rockcyber"}}},
        ))
        assert tracking.verify_credential() == {
            "username": "rockl", "entity": "rockcyber",
        }

    def test_a_rejected_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import requests

        from tract.training import tracking

        monkeypatch.setenv("WANDB_API_KEY", LEGACY_KEY)
        monkeypatch.setattr(requests, "post",
                            lambda *a, **k: self._response(401, {}))
        with pytest.raises(RuntimeError, match="rejected"):
            tracking.verify_credential()

    def test_an_authenticated_request_with_no_viewer_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A key that identifies no account cannot own a run."""
        import requests

        from tract.training import tracking

        monkeypatch.setenv("WANDB_API_KEY", LEGACY_KEY)
        monkeypatch.setattr(requests, "post", lambda *a, **k: self._response(
            200, {"data": {"viewer": None}},
        ))
        with pytest.raises(RuntimeError, match="no viewer"):
            tracking.verify_credential()

    def test_an_unreachable_server_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import requests

        from tract.training import tracking

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise OSError("network down")

        monkeypatch.setenv("WANDB_API_KEY", LEGACY_KEY)
        monkeypatch.setattr(requests, "post", _boom)
        with pytest.raises(RuntimeError, match="Could not reach WandB"):
            tracking.verify_credential()
