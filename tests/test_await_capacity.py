"""Tests for the unattended capacity watcher -- the thing that spends money.

This module shipped with no tests at all, and on the night of 2026-08-27 it
provisioned four fleets, trained zero folds and cost about $40. The final
failure was not a training bug: `run` aborted in the bootstrap barrier, nothing
was ever computed, and the watcher then called `collect` anyway. Collect rsyncs
a remote results directory that a bootstrap abort never created, rsync exits 23,
and a non-zero collect is what selects the "NOT tearing down. Pods hold
uncollected results." branch. Five idle pods were left billing on a fleet
holding nothing, and a human had to terminate them by hand.

Every assertion below is about that class of decision: what does the watcher do
to a BILLING fleet when a stage fails. Nothing here provisions anything,
contacts RunPod, runs systemd-run, or loads a model -- the orchestrator, the two
gates, the account sweep and subprocess itself are all replaced with fakes,
because the failure worth testing is the branch taken, not the subprocess that
reveals it.

The one thing that is real is the module import, which opens a log file at
import time. It is redirected below so a `pytest` run during a live campaign
does not interleave test output into the incident log the operator reads.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

# Set BEFORE the import: await_capacity resolves LOG_DIR and attaches its
# FileHandler at module scope, so a fixture would be far too late.
os.environ.setdefault(
    "TRACT_CAMPAIGN2_LOG_DIR",
    tempfile.mkdtemp(prefix="tract-await-capacity-tests-"),
)

from scripts.phase0.runpod_provision import (  # noqa: E402 - same
    SSH_POLL_TIMEOUT_S as provision_ssh_poll_timeout,
)
from scripts.phase1b import await_capacity as ac  # noqa: E402 - see above
from scripts.phase1b import reaper_guard as rg  # noqa: E402 - same
from scripts.phase1b import runpod_parallel as rp  # noqa: E402 - same

ARM = ac.ARMS["A1"]


class FakeOrchestrator:
    """Stands in for the runpod_parallel subprocess. Records the stages run."""

    def __init__(self, **rcs: int) -> None:
        self.rcs = rcs
        self.calls: list[str] = []

    def __call__(self, *args: str) -> int:
        self.calls.append(args[0])
        return self.rcs.get(args[0], 0)


@pytest.fixture
def fleet(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """A fleet that provisions cleanly and passes both gates.

    The sweep is counted rather than performed: it is the only thing in
    attempt_arm that would otherwise reach the RunPod API.
    """
    swept: list[int] = []

    def sweep() -> int:
        swept.append(1)
        return 0

    monkeypatch.setattr(ac, "arm_the_reaper", lambda: None)
    monkeypatch.setattr(ac, "gate_a_gpu_is_fast_enough", lambda: True)
    monkeypatch.setattr(ac, "gate_b_ssh_actually_authenticates", lambda: True)
    monkeypatch.setattr(ac, "sweep_account", sweep)
    return {"swept": swept}


class TestFleetIsNeverStrandedHoldingNothing:
    """The 2026-08-27 incident, and the case that must keep working."""

    def test_bootstrap_failure_tears_the_fleet_down(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """`run` failed with nothing computed: the fleet must not survive.

        This is the incident. A bootstrap abort and a partially failed run are
        the same exit code from `run`, so the watcher cannot tell them apart --
        which is why the answer for both is teardown. A partial fold set cannot
        produce a LOFO number either way.
        """
        # The incident's exact shape: `run` aborts in the bootstrap barrier and
        # a collect against a directory that was never created exits non-zero.
        orch = FakeOrchestrator(run=1, collect=1)
        monkeypatch.setattr(ac, "orchestrator", orch)

        outcome = ac.attempt_arm(ARM)

        assert "teardown" in orch.calls, (
            f"the fleet was left up after a failed run; stages were {orch.calls}"
        )
        assert fleet["swept"], "teardown was not followed by an account sweep"
        assert outcome is ac.ArmOutcome.FLEET_DOWN

    def test_bootstrap_failure_does_not_call_collect(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Collect on a fleet that computed nothing is what strands it.

        rsync exits 23 on a remote directory that was never created, and a
        non-zero collect is the signal the finally block reads as "the pods
        hold paid-for work".
        """
        orch = FakeOrchestrator(run=1, collect=1)
        monkeypatch.setattr(ac, "orchestrator", orch)

        ac.attempt_arm(ARM)

        assert "collect" not in orch.calls, (
            f"collect ran on a fleet that never trained: {orch.calls}"
        )

    def test_collect_failure_after_training_holds_the_fleet(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The genuine case: folds ran, so a pod may hold the only copy."""
        orch = FakeOrchestrator(collect=1)
        monkeypatch.setattr(ac, "orchestrator", orch)

        outcome = ac.attempt_arm(ARM)

        assert outcome is ac.ArmOutcome.FLEET_HELD
        assert orch.calls == ["provision", "run", "collect"], (
            f"a fleet holding uncollected folds was torn down: {orch.calls}"
        )
        assert not fleet["swept"], "the sweep terminated a held fleet"

    def test_a_complete_arm_collects_then_tears_down(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        orch = FakeOrchestrator()
        monkeypatch.setattr(ac, "orchestrator", orch)

        outcome = ac.attempt_arm(ARM)

        assert outcome is ac.ArmOutcome.COMPLETE
        assert orch.calls == ["provision", "run", "collect", "teardown"]

    def test_a_run_that_timed_out_tears_down(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A wedged `run` is a failed run: the watcher stopped watching it."""
        orch = FakeOrchestrator(run=ac.STAGE_TIMEOUT_RC)
        monkeypatch.setattr(ac, "orchestrator", orch)

        assert ac.attempt_arm(ARM) is ac.ArmOutcome.FLEET_DOWN
        assert "teardown" in orch.calls

    @pytest.mark.parametrize("gate", [
        "gate_a_gpu_is_fast_enough", "gate_b_ssh_actually_authenticates",
    ])
    def test_a_failed_gate_tears_the_fleet_down(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch, gate: str,
    ) -> None:
        orch = FakeOrchestrator()
        monkeypatch.setattr(ac, "orchestrator", orch)
        monkeypatch.setattr(ac, gate, lambda: False)

        assert ac.attempt_arm(ARM) is ac.ArmOutcome.FLEET_DOWN
        assert orch.calls == ["provision", "teardown"]

    def test_a_failed_provision_still_sweeps_the_account(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """provision half-creates fleets; the state file is not the record."""
        orch = FakeOrchestrator(provision=1)
        monkeypatch.setattr(ac, "orchestrator", orch)

        assert ac.attempt_arm(ARM) is ac.ArmOutcome.FLEET_DOWN
        assert orch.calls == ["provision", "teardown"]
        assert fleet["swept"]


class TestTheReaperIsAPrecondition:
    """No independent backstop, no fleet."""

    def test_arm_the_reaper_raises_when_systemd_run_fails(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*args: object, **kwargs: object) -> None:
            raise subprocess.CalledProcessError(1, "systemd-run", stderr=b"nope")

        monkeypatch.setattr(subprocess, "run", boom)

        with pytest.raises(RuntimeError, match="reaper"):
            ac.arm_the_reaper()

    def test_a_missing_systemd_run_also_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def missing(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("systemd-run")

        monkeypatch.setattr(subprocess, "run", missing)

        with pytest.raises(RuntimeError):
            ac.arm_the_reaper()

    def test_nothing_is_provisioned_when_the_reaper_will_not_arm(
        self, fleet: dict[str, object], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        orch = FakeOrchestrator()
        monkeypatch.setattr(ac, "orchestrator", orch)

        def refuse() -> None:
            raise RuntimeError("could not arm the reaper")

        monkeypatch.setattr(ac, "arm_the_reaper", refuse)

        outcome = ac.attempt_arm(ARM)

        assert outcome is ac.ArmOutcome.REFUSED
        assert orch.calls == [], (
            f"a fleet was created without a backstop: {orch.calls}"
        )

    def test_the_first_check_is_short_and_tied_to_the_guard(self) -> None:
        """One number, imported, and a first look inside the hour."""
        assert ac.FIRST_REAPER_CHECK_SECONDS <= rg.REARM_SECONDS
        assert 30 * 60 <= ac.FIRST_REAPER_CHECK_SECONDS <= 45 * 60

    def test_the_armed_unit_asks_for_that_window(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The old command said 8h while the guard's own cadence said 2h."""
        seen: dict[str, list[str]] = {}

        def capture(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0, b"", b"")

        monkeypatch.setattr(subprocess, "run", capture)

        ac.arm_the_reaper()

        on_active = [a for a in seen["cmd"] if a.startswith("--on-active=")]
        assert on_active == [f"--on-active={ac.FIRST_REAPER_CHECK}"]
        assert "--on-active=8h" not in seen["cmd"]


class TestNoStageCanWedgeTheWatcher:
    """subprocess.run without a timeout waits forever, and MAX_WALL_SECONDS is
    only re-read at the top of the polling loop."""

    def _fake_run(self, seen: dict[str, object]) -> object:
        def run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            seen["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(cmd, 0)
        return run

    @pytest.mark.parametrize("stage", ["provision", "run", "collect", "teardown"])
    def test_every_stage_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch, stage: str,
    ) -> None:
        seen: dict[str, object] = {}
        monkeypatch.setattr(subprocess, "run", self._fake_run(seen))

        assert ac.orchestrator(stage) == 0
        assert seen["timeout"] == ac.STAGE_TIMEOUT_S[stage], (
            f"{stage} ran with timeout={seen['timeout']!r}"
        )

    def test_the_bounds_exceed_what_the_orchestrator_may_legitimately_take(
        self,
    ) -> None:
        """A timeout that fires on healthy work destroys paid-for folds."""
        assert ac.STAGE_TIMEOUT_S["run"] > rp.FOLD_TIMEOUT_S + rp.BOOTSTRAP_DEADLINE_S
        # Collect walks the roster serially and every pod can pay the full
        # pull ladder, so the wall has to cover all five of them.
        assert ac.STAGE_TIMEOUT_S["collect"] > (
            ac.FLEET_SIZE * rp.RSYNC_PULL_ATTEMPTS * rp.RSYNC_PULL_TIMEOUT_S
        )
        assert ac.STAGE_TIMEOUT_S["provision"] > provision_ssh_poll_timeout

    def test_a_wedged_stage_is_reported_as_failed(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    ) -> None:
        def wedge(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            raise subprocess.TimeoutExpired(cmd, float(kwargs["timeout"]))  # type: ignore[arg-type]

        monkeypatch.setattr(subprocess, "run", wedge)

        with caplog.at_level(logging.ERROR):
            rc = ac.orchestrator("run", "--config-name", "c2r_A1_prose_sw_bge")

        assert rc == ac.STAGE_TIMEOUT_RC
        assert rc != 0, "a wedged stage must not read as success"
        assert "run" in caplog.text

    def test_an_unknown_stage_is_a_programming_error(self) -> None:
        with pytest.raises(ValueError, match="aggregate"):
            ac.orchestrator("aggregate")


class TestTheWatcherStopsWhenItShould:
    """main() decides whether to create ANOTHER fleet."""

    @pytest.fixture(autouse=True)
    def _no_waiting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ac, "survey", lambda: [("NVIDIA H100 80GB HBM3", "High")])

        def never_sleep(seconds: float) -> None:
            raise AssertionError(
                f"the watcher slept {seconds}s instead of deciding"
            )

        monkeypatch.setattr(time, "sleep", never_sleep)

    def test_a_held_fleet_stops_the_watcher(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Provisioning over a held fleet overwrites the only record of it.

        provision() rewrites .pod_state.json, and both fleets carry the same
        pod names, so the held fleet becomes invisible to teardown the moment
        the next attempt starts.
        """
        attempts: list[str] = []

        def held(arm: ac.Arm) -> ac.ArmOutcome:
            attempts.append(arm.key)
            return ac.ArmOutcome.FLEET_HELD

        monkeypatch.setattr(ac, "attempt_arm", held)

        assert ac.main(["--arm", "A1", "--confirm"]) == ac.EXIT_FLEET_HELD
        assert attempts == ["A1"], "a second fleet was provisioned over a held one"

    def test_a_refused_arm_stops_the_watcher(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(ac, "attempt_arm", lambda arm: ac.ArmOutcome.REFUSED)

        assert ac.main(["--arm", "A1", "--confirm"]) == ac.EXIT_REFUSED

    def test_a_complete_arm_exits_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(ac, "attempt_arm", lambda arm: ac.ArmOutcome.COMPLETE)

        assert ac.main(["--arm", "A1", "--confirm"]) == ac.EXIT_OK

    def test_a_dry_run_provisions_nothing(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def forbidden(arm: ac.Arm) -> ac.ArmOutcome:
            raise AssertionError("a dry run provisioned a fleet")

        monkeypatch.setattr(ac, "attempt_arm", forbidden)

        assert ac.main(["--arm", "A1"]) == ac.EXIT_OK

    def test_retries_stay_at_three(self) -> None:
        """Pinned deliberately.

        Cheap, fast failure multiplied by a retry budget is how one capacity
        window becomes several fleets. The premortem's finding is explicit that
        this number must not rise until the stranding fix above has landed and
        been exercised.
        """
        assert ac.MAX_PROVISION_ATTEMPTS == 3

    def test_a_retryable_failure_is_retried_to_the_cap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        attempts: list[str] = []

        def down(arm: ac.Arm) -> ac.ArmOutcome:
            attempts.append(arm.key)
            return ac.ArmOutcome.FLEET_DOWN

        monkeypatch.setattr(ac, "attempt_arm", down)
        # Overrides the no-sleep guard above: this is the one test where the
        # watcher is SUPPOSED to go back to polling between attempts.
        monkeypatch.setattr(time, "sleep", lambda seconds: None)

        assert ac.main(["--arm", "A1", "--confirm"]) == ac.EXIT_GAVE_UP
        assert len(attempts) == ac.MAX_PROVISION_ATTEMPTS


class TestLoggingStaysOutOfTheIncidentLog:
    def test_the_log_directory_is_redirectable(self) -> None:
        """Asserted so the redirect at the top of this file cannot rot away."""
        assert ac.LOG_DIR == Path(os.environ["TRACT_CAMPAIGN2_LOG_DIR"])
