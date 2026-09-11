"""The single-pod trainer must not be able to strand a billing GPU.

`runpod_retrain` is the only single-pod training path in the repository, which
is the shape Gate 2 needs -- four arms, one model each, run in sequence. It had
none of the controls `runpod_parallel` and even the small `smoke_on_pod` probe
carry:

  * `full_pipeline` was a bare sequence, `provision(); run(); collect();
    teardown()`. Any raise between the first and last orphans the pod, and
    `_ssh` here has no retry ladder, so a transient SSH error is enough.
  * `find_fastest_available` was called with no `max_usd_per_hour`, whose own
    docstring warns the fallback is "largest VRAM wins, which can select a part
    several times the rate of an H100".
  * No wall clock.

And the failure was worse than unguarded. The reaper's name sweep covers
`tract-p1b-fold*` and `tract-p1b-val-fold*` only, so `tract-p1c-retrain` was
invisible to it -- while `_is_orchestrator_argv` matched only `runpod_parallel`,
so a live retrain read as NO ORCHESTRATOR, `running_pod_count()` returned 0, and
the guard took the quiet branch and DISARMED after three checks. It concluded the
campaign was over while the pod trained.

Three arms is three independent draws on that.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from scripts.phase1c import runpod_retrain as rt


class TestTeardownIsGuaranteed:
    def test_full_pipeline_has_a_finally(self) -> None:
        source = inspect.getsource(rt.full_pipeline)
        assert "finally:" in source, (
            "full_pipeline tears down only on the happy path; any raise "
            "between provision and teardown leaves a billing pod behind"
        )

    def test_teardown_is_inside_the_finally(self) -> None:
        source = inspect.getsource(rt.full_pipeline)
        after = source.split("finally:", 1)[1]
        assert "teardown" in after

    def test_keyboard_interrupt_is_covered(self) -> None:
        """Ctrl-C at 2am is the single most likely way this is interrupted.

        `except Exception` does not catch it -- KeyboardInterrupt derives from
        BaseException -- so a bare except-Exception teardown leaves the pod up
        in exactly the case an operator is watching.
        """
        source = inspect.getsource(rt.full_pipeline)
        assert "finally:" in source, "a finally covers BaseException; an except clause may not"


class TestSpendIsBounded:
    def test_a_price_ceiling_is_passed_to_gpu_selection(self) -> None:
        source = inspect.getsource(rt.provision)
        assert "max_usd_per_hour" in source, (
            "find_fastest_available falls back to 'largest VRAM wins', which "
            "its own docstring warns can select a part several times the rate "
            "of an H100"
        )

    def test_the_ceiling_is_a_named_constant(self) -> None:
        assert isinstance(rt.MAX_USD_PER_HOUR, float)
        assert 0 < rt.MAX_USD_PER_HOUR < 20

    def test_a_wall_clock_deadline_exists(self) -> None:
        assert isinstance(rt.MAX_RUN_HOURS, (int, float))
        assert 0 < rt.MAX_RUN_HOURS <= 48


class TestTheReaperCanSeeThisPod:
    def test_the_pod_name_is_in_the_swept_family(self) -> None:
        """The sweep is the recovery path when the state file is gone.

        It covers tract-p1b-fold* and tract-p1b-val-fold*. A pod named outside
        that family is invisible to the one command that cleans up after a
        dead orchestrator.
        """
        from scripts.phase1b.reaper_guard import expected_pod_names

        assert rt.POD_NAME in expected_pod_names(), (
            f"{rt.POD_NAME} is not swept by the reaper; a dead orchestrator "
            "leaves it billing with no recovery path"
        )

    def test_a_live_retrain_counts_as_an_orchestrator(self) -> None:
        """Otherwise the guard disarms after three quiet checks, ~6 hours.

        running_pod_count() returns 0 for a pod it cannot name, the guard reads
        that as 'campaign over', and it stands down while the pod trains.
        """
        from scripts.phase1b.reaper_guard import _is_orchestrator_argv

        assert _is_orchestrator_argv(
            ["python3", "-m", "scripts.phase1c.runpod_retrain", "full"]
        ), "a running retrain does not register as an orchestrator"

    def test_the_parallel_orchestrator_still_registers(self) -> None:
        from scripts.phase1b.reaper_guard import _is_orchestrator_argv

        assert _is_orchestrator_argv(
            ["python3", "-m", "scripts.phase1b.runpod_parallel", "run"]
        )

    def test_an_unrelated_process_does_not_register(self) -> None:
        """The guard must not be disarmed by anything that merely mentions it."""
        from scripts.phase1b.reaper_guard import _is_orchestrator_argv

        assert not _is_orchestrator_argv(
            ["vim", "scripts/phase1c/runpod_retrain.py"]
        )
        assert not _is_orchestrator_argv(["python3", "-m", "pytest"])


class TestCollectFailureIsNotSilent:
    def test_a_failed_collect_does_not_reach_teardown_quietly(self) -> None:
        """Arm 2 overwriting arm 1's un-collected output destroys paid work.

        collect() used to warn and continue, and full_pipeline then tore the pod
        down regardless -- so the result that cost money was gone and the log
        said "Failed to collect ... continuing".
        """
        source = inspect.getsource(rt.full_pipeline)
        assert "results_are_safe" in source or "raise" in source, (
            "a collect failure must be visible before the pod is destroyed"
        )


class TestEveryDriverIsRecognised:
    """A driver the guard cannot name gets its own pod reaped mid-run.

    The decision branch reaps when NO orchestrator is alive and pods ARE
    running. `tract-p2c-gate2` is in expected_pod_names(), so
    running_pod_count() sees it -- which means a runner missing from the
    orchestrator set produces exactly the reap condition, and the guard
    destroys the work it exists to bound. The failure is silent and it is
    destructive in both directions: omit an entry and a live run dies, make it
    over-broad and the guard stands down forever.
    """

    @pytest.mark.parametrize("module", [
        "scripts.phase1b.runpod_parallel",
        "scripts.phase1c.runpod_retrain",
        "scripts.phase2c.run_gate2",
    ])
    def test_each_driver_registers(self, module: str) -> None:
        from scripts.phase1b.reaper_guard import _is_orchestrator_argv

        assert _is_orchestrator_argv(["python3", "-m", module, "full"])

    @pytest.mark.parametrize("argv", [
        ["vim", "scripts/phase2c/run_gate2.py"],
        ["python3", "-m", "pytest"],
        ["python3", "-c", "import scripts.phase2c.run_gate2"],
        ["grep", "-r", "runpod_parallel", "."],
    ])
    def test_merely_mentioning_a_driver_does_not_register(
        self, argv: list[str]
    ) -> None:
        from scripts.phase1b.reaper_guard import _is_orchestrator_argv

        assert not _is_orchestrator_argv(argv)

    def test_every_driver_module_has_a_swept_pod_name(self) -> None:
        """A driver whose pod is unswept has no recovery path when it dies."""
        from scripts.phase1b.reaper_guard import expected_pod_names
        from scripts.phase1c.runpod_retrain import POD_NAME

        assert POD_NAME in expected_pod_names()
