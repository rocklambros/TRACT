"""Tests for the reaper guard -- the only bound on an orphaned GPU fleet.

The guard shipped with zero tests and four defects were found in it in a single
evening, which is the argument for this file. Two of those defects were failures
to RE-ARM, and one was a false positive that made the guard stand down forever;
both classes are invisible to a human reading the code, because the guard's whole
job is what it does two hours from now.

Nothing here provisions anything, contacts RunPod, or runs systemd-run. The
processes spawned below are sleepers wearing the orchestrator's name, because
argv classification is the layer that actually failed -- a fake argv list would
have agreed with the broken code that `tail -f runpod_parallel.log` is a live
orchestrator only if the fake had been written to include a log filename, which
is exactly the case nobody thinks to invent. So that one runs for real, through
/proc, the way the guard reads it in production.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from scripts.phase1b import reaper_guard as rg

# A spawned process is visible in /proc the instant fork returns, but its
# cmdline is still the PARENT's argv until exec completes. Polling for the
# expected argv is what makes these tests deterministic rather than a race that
# passes on an idle laptop and flakes on a loaded Jetson.
SPAWN_TIMEOUT_S: float = 10.0
POLL_INTERVAL_S: float = 0.02

# A sleeper long enough to outlive the assertions, short enough that a test
# killed with SIGKILL mid-run cannot leave a process on the box for an hour.
SLEEPER_SOURCE: str = "import time\ntime.sleep(60)\n"


def _read_cmdline(pid: int) -> list[str]:
    """The exec argv of *pid*, exactly as the guard reads it."""
    with open(f"/proc/{pid}/cmdline", "rb") as handle:
        raw = handle.read().decode("utf-8", "replace")
    return [arg for arg in raw.split("\x00") if arg]


def _await_cmdline(pid: int, expected: list[str]) -> None:
    """Block until /proc shows *expected*, or fail loudly with what it shows."""
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    seen: list[str] = []
    while time.monotonic() < deadline:
        try:
            seen = _read_cmdline(pid)
        except FileNotFoundError:  # pragma: no cover - the process died early
            raise AssertionError(
                f"pid {pid} exited before it could be classified; it was "
                f"expected to run {expected}"
            )
        if seen == expected:
            return
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(
        f"pid {pid} never showed the expected argv. Wanted {expected}, "
        f"/proc shows {seen}."
    )


@pytest.fixture
def spawn() -> Iterator[Callable[..., subprocess.Popen[bytes]]]:
    """Start a real process, wait until /proc agrees, and always reap it."""
    started: list[subprocess.Popen[bytes]] = []

    def _spawn(
        args: list[str], cwd: Path | None = None,
    ) -> subprocess.Popen[bytes]:
        env = dict(os.environ)
        env["USE_TF"] = "0"
        # A stray PYTHONPATH could let `-m scripts.phase1b.runpod_parallel`
        # resolve to the REAL orchestrator, whose bare invocation provisions a
        # fleet. Dropping it means the worst case is ModuleNotFoundError.
        env.pop("PYTHONPATH", None)
        proc = subprocess.Popen(
            args,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        started.append(proc)
        _await_cmdline(proc.pid, args)
        return proc

    yield _spawn

    for proc in started:
        proc.kill()
        proc.wait(timeout=SPAWN_TIMEOUT_S)


@pytest.fixture
def fake_orchestrator_script(tmp_path: Path) -> Path:
    """A harmless sleeper whose basename is the orchestrator's, exactly."""
    script = tmp_path / "runpod_parallel.py"
    script.write_text(SLEEPER_SOURCE, encoding="utf-8")
    return script


@pytest.fixture
def fake_orchestrator_package(tmp_path: Path) -> Path:
    """A sleeper importable as `scripts.phase1b.runpod_parallel` from tmp_path.

    The real module cannot be spawned to test the module form: `-m` runs it as
    __main__, argparse defaults the action to "full", and the test suite would
    provision five H100s. This shadow package has the identical dotted name and
    sleeps instead.
    """
    package = tmp_path / "scripts" / "phase1b"
    package.mkdir(parents=True)
    (tmp_path / "scripts" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "runpod_parallel.py").write_text(SLEEPER_SOURCE, encoding="utf-8")
    return tmp_path


@pytest.fixture
def guard_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the quiet-streak state at tmp_path instead of the live runtime dir."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    return runtime / rg.STATE_DIRNAME


class TestOrchestratorPids:
    """Classification through /proc, against processes that really exist."""

    pytestmark = pytest.mark.skipif(
        sys.platform != "linux", reason="the guard reads /proc"
    )

    def test_tailing_the_orchestrator_log_is_not_an_orchestrator(
        self, tmp_path: Path, spawn: Callable[..., subprocess.Popen[bytes]],
    ) -> None:
        """The incident: `tail -f results/phase1b/runpod_parallel.log`.

        That is the log an operator watches during a campaign, and the first
        version matched "runpod_parallel" anywhere in argv -- so watching the
        run convinced the guard the run was alive, permanently, and the fleet
        had no bound at all.
        """
        log = tmp_path / "results" / "phase1b" / "runpod_parallel.log"
        log.parent.mkdir(parents=True)
        log.write_text("INFO provisioning\n", encoding="utf-8")

        proc = spawn(["tail", "-f", str(log)])

        assert proc.poll() is None, "the tail exited before it was classified"
        assert proc.pid not in rg.orchestrator_pids()

    def test_a_reader_open_on_the_orchestrator_source_is_not_an_orchestrator(
        self,
        fake_orchestrator_script: Path,
        spawn: Callable[..., subprocess.Popen[bytes]],
    ) -> None:
        """Reading runpod_parallel.py is not running it, whatever the reader."""
        proc = spawn(["tail", "-f", str(fake_orchestrator_script)])

        assert proc.poll() is None
        assert proc.pid not in rg.orchestrator_pids()

    def test_a_module_invocation_is_alive(
        self,
        fake_orchestrator_package: Path,
        spawn: Callable[..., subprocess.Popen[bytes]],
    ) -> None:
        """`python -m scripts.phase1b.runpod_parallel` is the runbook's form."""
        proc = spawn(
            [sys.executable, "-m", rg.ORCHESTRATOR_MODULE],
            cwd=fake_orchestrator_package,
        )

        assert proc.poll() is None, "the shadow package failed to import"
        assert proc.pid in rg.orchestrator_pids()

    def test_a_bare_script_invocation_is_alive(
        self,
        fake_orchestrator_script: Path,
        spawn: Callable[..., subprocess.Popen[bytes]],
    ) -> None:
        """No action word means argparse's default, which is the whole pipeline."""
        proc = spawn([sys.executable, str(fake_orchestrator_script)])

        assert proc.poll() is None
        assert proc.pid in rg.orchestrator_pids()

    @pytest.mark.parametrize("action", ["price", "aggregate", "track"])
    def test_local_and_read_only_actions_are_not_a_live_fleet(
        self,
        action: str,
        fake_orchestrator_script: Path,
        spawn: Callable[..., subprocess.Popen[bytes]],
    ) -> None:
        proc = spawn([sys.executable, str(fake_orchestrator_script), action])

        assert proc.poll() is None
        assert proc.pid not in rg.orchestrator_pids()

    def test_another_generation_of_the_guard_is_not_an_orchestrator(
        self, tmp_path: Path, spawn: Callable[..., subprocess.Popen[bytes]],
    ) -> None:
        """Re-arming overlaps generations; a guard must never see its sibling."""
        sibling = tmp_path / "reaper_guard.py"
        sibling.write_text(SLEEPER_SOURCE, encoding="utf-8")

        proc = spawn([sys.executable, str(sibling), "--confirm"])

        assert proc.poll() is None
        assert proc.pid not in rg.orchestrator_pids()

    def test_the_guard_never_reports_itself(self) -> None:
        assert os.getpid() not in rg.orchestrator_pids()


class TestReadArgv:
    """/proc is a live filesystem: entries vanish mid-scan and reads can fail."""

    def test_a_pid_that_does_not_exist_reads_as_nothing(self) -> None:
        # Above /proc/sys/kernel/pid_max on any sane box, so nothing owns it.
        assert rg._read_argv(2**30) == []

    def test_any_os_error_reads_as_nothing_rather_than_killing_the_guard(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A read of /proc can fail with EIO, which is not one of the three
        named exceptions the first version caught -- and an uncaught one there
        takes down the whole check, re-arm included."""
        def unreadable(*args: object, **kwargs: object) -> object:
            raise OSError(5, "Input/output error")

        monkeypatch.setattr(rg, "open", unreadable, raising=False)

        assert rg._read_argv(os.getpid()) == []

    def test_a_real_cmdline_survives_the_nul_terminator(self) -> None:
        argv = rg._read_argv(os.getpid())

        assert argv, "the test runner has a command line"
        assert "" not in argv


class TestIsOrchestratorArgv:
    """The classifier itself, on argv shapes too awkward to spawn."""

    @pytest.mark.parametrize("argv", [
        ["tail", "-f", "/home/rock/x/results/phase1b/runpod_parallel.log"],
        ["less", "/home/rock/x/scripts/phase1b/runpod_parallel.py"],
        ["grep", "-n", "reap", "scripts/phase1b/runpod_parallel.py"],
        ["vim", "scripts/phase1b/runpod_parallel.py"],
        ["rsync", "-a", "pod:/w/results/runpod_parallel.log", "."],
        ["/usr/bin/python3", "-m", "scripts.phase1b.reaper_guard", "--confirm"],
        ["/usr/bin/python3", "-m", "scripts.phase1b.runpod_parallel_notes"],
        ["/usr/bin/python3", "scripts/phase1b/runpod_parallel_helper.py"],
        ["/usr/bin/python3", "-m", "pytest", "tests/test_runpod_safety.py"],
        [],
    ])
    def test_not_the_orchestrator(self, argv: list[str]) -> None:
        assert rg._is_orchestrator_argv(argv) is False

    @pytest.mark.parametrize("argv", [
        ["/usr/bin/python3", "-m", "scripts.phase1b.runpod_parallel"],
        ["python3", "-m", "scripts.phase1b.runpod_parallel", "full"],
        ["python", "-u", "-m", "scripts.phase1b.runpod_parallel", "run"],
        ["python3.12", "-m", "scripts.phase1b.runpod_parallel", "provision"],
        ["/usr/bin/python3", "scripts/phase1b/runpod_parallel.py"],
        ["/usr/bin/python3", "scripts/phase1b/runpod_parallel.py", "collect"],
        ["python3", "-um", "scripts.phase1b.runpod_parallel"],
    ])
    def test_is_the_orchestrator(self, argv: list[str]) -> None:
        assert rg._is_orchestrator_argv(argv) is True

    def test_a_python_process_pointed_at_the_log_is_not_the_orchestrator(
        self,
    ) -> None:
        """The basename test is exact. A .log is not the .py, whoever opened it."""
        assert rg._is_orchestrator_argv(
            ["/usr/bin/python3", "/home/rock/x/results/phase1b/runpod_parallel.log"]
        ) is False

    def test_a_repl_and_a_one_liner_are_not_the_orchestrator(self) -> None:
        assert rg._is_orchestrator_argv(["/usr/bin/python3"]) is False
        assert rg._is_orchestrator_argv(
            ["/usr/bin/python3", "-c", "print('runpod_parallel')"]
        ) is False

    def test_an_unrecognised_action_counts_as_alive(self) -> None:
        """Fail-safe: a delayed reap costs hours, a wrong reap costs the run."""
        assert rg._is_orchestrator_argv(
            ["python3", "-m", rg.ORCHESTRATOR_MODULE, "resurrect"]
        ) is True

    def test_an_action_word_before_the_target_is_not_the_action(self) -> None:
        """The action is what follows the target, not any word anywhere in argv.

        `python3 -X track -m ...` is a real command line -- CPython accepts
        arbitrary -X keys and runs it -- and reading the action from the whole
        of argv finds "track" there, calls the busiest possible process a
        read-only query, and reaps a fleet in mid-training.
        """
        assert rg._is_orchestrator_argv(
            ["python3", "-X", "track", "-m", rg.ORCHESTRATOR_MODULE]
        ) is True
        assert rg._is_orchestrator_argv(
            ["python3", "-X", "importtime", "-m", rg.ORCHESTRATOR_MODULE]
        ) is True


class TestIsPythonInterpreter:
    """argv[0] is the half of the strict test that rules out every reader."""

    @pytest.mark.parametrize("argv0", [
        "python",
        "python3",
        "python3.12",
        "python3.13t",
        "/usr/bin/python3",
        "/home/rock/anaconda3/bin/python3",
    ])
    def test_interpreters(self, argv0: str) -> None:
        assert rg._is_python_interpreter(argv0) is True

    @pytest.mark.parametrize("argv0", [
        "tail", "less", "grep", "vim", "rsync", "/usr/bin/vim",
        "pythonista", "python-config", "bash", "",
    ])
    def test_not_interpreters(self, argv0: str) -> None:
        assert rg._is_python_interpreter(argv0) is False


class TestPythonTarget:
    """What the interpreter was asked to run, and where argv said it."""

    def test_a_module_after_interpreter_flags(self) -> None:
        target = rg._python_target(
            ["python3", "-u", "-X", "importtime", "-m", "pkg.mod", "run"]
        )

        assert (target.kind, target.name) == ("module", "pkg.mod")
        # The action word has to be read from AFTER the module, index and all.
        assert target.index == 5

    def test_an_attached_module_name(self) -> None:
        target = rg._python_target(["python3", "-mpkg.mod"])

        assert (target.kind, target.name, target.index) == ("module", "pkg.mod", 1)

    def test_a_script_after_a_double_dash(self) -> None:
        target = rg._python_target(["python3", "--", "-weird-name.py"])

        assert (target.kind, target.name) == ("script", "-weird-name.py")

    def test_a_bare_interpreter_has_no_target(self) -> None:
        assert rg._python_target(["python3", "-u"]).kind == "none"


class TestExpectedPodNames:
    """Five test pods and five validation pods, under two different prefixes."""

    def test_covers_both_splits(self) -> None:
        names = rg.expected_pod_names()

        assert names == {
            "tract-p1b-fold0", "tract-p1b-fold1", "tract-p1b-fold2",
            "tract-p1b-fold3", "tract-p1b-fold4",
            "tract-p1b-val-fold0", "tract-p1b-val-fold1", "tract-p1b-val-fold2",
            "tract-p1b-val-fold3", "tract-p1b-val-fold4",
        }

    def test_validation_pods_are_not_missing(self) -> None:
        """Campaign 2 runs five validation rounds; missing them disarmed the guard."""
        names = rg.expected_pod_names()

        assert len([n for n in names if n.startswith("tract-p1b-val-")]) == 5
        assert len([n for n in names if not n.startswith("tract-p1b-val-")]) == 5


@pytest.fixture
def quiet_fleet(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """No orchestrator, no pods, and a rearm() that records instead of arming."""
    calls: list[str] = []
    monkeypatch.setattr(rg, "orchestrator_pids", lambda: [])
    monkeypatch.setattr(rg, "running_pod_count", lambda: 0)
    monkeypatch.setattr(rg, "rearm", lambda: calls.append("rearm"))
    return calls


class TestQuietCheckRearms:
    """The gap between arms looks exactly like a finished campaign."""

    def test_zero_pods_still_rearms(
        self, quiet_fleet: list[str], guard_state: Path,
    ) -> None:
        """Campaign 2 has five inter-arm gaps. The guard used to disarm in the first."""
        assert rg.main(["--confirm"]) == rg.EXIT_OK

        assert quiet_fleet == ["rearm"]

    def test_the_streak_is_persisted_outside_the_repo(
        self, quiet_fleet: list[str], guard_state: Path,
    ) -> None:
        rg.main(["--confirm"])

        state = json.loads(
            (guard_state / rg.QUIET_STREAK_FILENAME).read_text(encoding="utf-8")
        )
        assert state[rg.QUIET_STREAK_KEY] == 1
        assert state[rg.QUIET_UPDATED_KEY] > 0

    def test_it_disarms_once_the_quiet_streak_is_long_enough(
        self, quiet_fleet: list[str], guard_state: Path,
    ) -> None:
        """Re-arm through the gaps, but do not re-arm forever after the campaign."""
        for _ in range(rg.QUIET_CHECKS_BEFORE_DISARM + 3):
            assert rg.main(["--confirm"]) == rg.EXIT_OK

        assert len(quiet_fleet) == rg.QUIET_CHECKS_BEFORE_DISARM - 1

    def test_the_sentinel_disarms_immediately(
        self, quiet_fleet: list[str], guard_state: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        guard_state.mkdir(parents=True)
        (guard_state / rg.CAMPAIGN_COMPLETE_FILENAME).write_text(
            "campaign 2 done\n", encoding="utf-8"
        )

        with caplog.at_level(logging.INFO):
            assert rg.main(["--confirm"]) == rg.EXIT_OK

        assert quiet_fleet == []
        assert "campaign-complete" in caplog.text

    def test_a_live_orchestrator_resets_the_streak(
        self, quiet_fleet: list[str], guard_state: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for _ in range(rg.QUIET_CHECKS_BEFORE_DISARM - 1):
            rg.main(["--confirm"])
        monkeypatch.setattr(rg, "orchestrator_pids", lambda: [4242])
        rg.main(["--confirm"])
        monkeypatch.setattr(rg, "orchestrator_pids", lambda: [])
        quiet_fleet.clear()

        # The streak restarted, so the next quiet check re-arms rather than
        # disarming on a count inherited from the previous arm.
        assert rg.main(["--confirm"]) == rg.EXIT_OK
        assert quiet_fleet == ["rearm"]

    def test_running_pods_reset_the_streak(
        self, quiet_fleet: list[str], guard_state: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        for _ in range(rg.QUIET_CHECKS_BEFORE_DISARM - 1):
            rg.main(["--confirm"])
        monkeypatch.setattr(rg, "running_pod_count", lambda: 3)
        monkeypatch.setattr(
            "scripts.phase1b.runpod_parallel.reap", lambda confirm: None
        )
        assert rg.main(["--confirm"]) == rg.EXIT_REAPED

        assert rg._read_quiet_streak() == 0

    def test_a_stale_streak_does_not_carry_into_a_new_campaign(
        self, quiet_fleet: list[str], guard_state: Path,
    ) -> None:
        """A streak is consecutive in TIME. An entry from last week is not part of one."""
        guard_state.mkdir(parents=True)
        (guard_state / rg.QUIET_STREAK_FILENAME).write_text(
            json.dumps({
                rg.QUIET_STREAK_KEY: rg.QUIET_CHECKS_BEFORE_DISARM,
                rg.QUIET_UPDATED_KEY: time.time() - rg.QUIET_STREAK_TTL_S - 1,
            }, sort_keys=True),
            encoding="utf-8",
        )

        assert rg.main(["--confirm"]) == rg.EXIT_OK

        assert quiet_fleet == ["rearm"]
        assert rg._read_quiet_streak() == 1

    def test_a_truncated_streak_file_does_not_disarm_the_guard(
        self, quiet_fleet: list[str], guard_state: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Corrupt state must fail toward re-arming, never toward silence."""
        guard_state.mkdir(parents=True)
        (guard_state / rg.QUIET_STREAK_FILENAME).write_text(
            '{"quiet_checks": 2, "upda', encoding="utf-8"
        )

        with caplog.at_level(logging.ERROR):
            assert rg.main(["--confirm"]) == rg.EXIT_OK

        assert quiet_fleet == ["rearm"]
        assert "quiet-streak" in caplog.text

    def test_a_dry_run_neither_arms_nor_consumes_the_streak(
        self, quiet_fleet: list[str], guard_state: Path,
    ) -> None:
        """--confirm's contract is `report and change nothing`."""
        assert rg.main([]) == rg.EXIT_OK

        assert quiet_fleet == []
        assert not (guard_state / rg.QUIET_STREAK_FILENAME).exists()

    def test_a_missing_runtime_dir_is_not_fatal(
        self, quiet_fleet: list[str], monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """systemd --user always sets XDG_RUNTIME_DIR; a hand-run shell may not."""
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        monkeypatch.setattr(rg.tempfile, "gettempdir", lambda: str(tmp_path))

        assert rg.main(["--confirm"]) == rg.EXIT_OK

        assert quiet_fleet == ["rearm"]
        assert (tmp_path / rg.STATE_DIRNAME / rg.QUIET_STREAK_FILENAME).exists()


class TestReapFailureStillRearms:
    """A guard that dies is not a bound."""

    def test_an_exception_from_reap_still_rearms(
        self, monkeypatch: pytest.MonkeyPatch, guard_state: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A truncated .pod_state.json raises JSONDecodeError inside reap."""
        calls: list[str] = []

        def boom(confirm: bool) -> None:
            raise json.JSONDecodeError("Expecting value", "{trunc", 0)

        monkeypatch.setattr(rg, "orchestrator_pids", lambda: [])
        monkeypatch.setattr(rg, "running_pod_count", lambda: 4)
        monkeypatch.setattr(rg, "rearm", lambda: calls.append("rearm"))
        monkeypatch.setattr("scripts.phase1b.runpod_parallel.reap", boom)

        with caplog.at_level(logging.ERROR):
            assert rg.main(["--confirm"]) == rg.EXIT_ERROR

        assert calls == ["rearm"]
        assert "REAP FAILED" in caplog.text
        assert "4 running pod(s)" in caplog.text
        assert "STILL BILLING" in caplog.text
        # The traceback has to survive into the log: "reap failed" without the
        # exception is not something an operator can act on at 3am.
        assert "JSONDecodeError" in caplog.text

    def test_a_successful_reap_rearms_against_a_partial_termination(
        self, monkeypatch: pytest.MonkeyPatch, guard_state: Path,
    ) -> None:
        calls: list[str] = []
        reaped: list[bool] = []

        monkeypatch.setattr(rg, "orchestrator_pids", lambda: [])
        monkeypatch.setattr(rg, "running_pod_count", lambda: 2)
        monkeypatch.setattr(rg, "rearm", lambda: calls.append("rearm"))
        monkeypatch.setattr(
            "scripts.phase1b.runpod_parallel.reap",
            lambda confirm: reaped.append(confirm),
        )

        assert rg.main(["--confirm"]) == rg.EXIT_REAPED

        assert reaped == [True]
        assert calls == ["rearm"]

    def test_a_dry_run_never_reaps(
        self, monkeypatch: pytest.MonkeyPatch, guard_state: Path,
    ) -> None:
        def never(confirm: bool) -> None:
            raise AssertionError("reap ran without --confirm")

        monkeypatch.setattr(rg, "orchestrator_pids", lambda: [])
        monkeypatch.setattr(rg, "running_pod_count", lambda: 2)
        monkeypatch.setattr(rg, "rearm", lambda: None)
        monkeypatch.setattr("scripts.phase1b.runpod_parallel.reap", never)

        assert rg.main([]) == rg.EXIT_OK


class TestApiFailureRearms:
    """An unreadable API is not evidence that the fleet is down."""

    def test_an_unscannable_proc_stands_down_and_rearms(
        self, monkeypatch: pytest.MonkeyPatch, guard_state: Path,
    ) -> None:
        """Failing to ANSWER 'is the orchestrator alive?' is not a No."""
        calls: list[str] = []

        def unscannable() -> list[int]:
            raise OSError(5, "Input/output error")

        def never() -> int:
            raise AssertionError("pods were queried without a liveness answer")

        monkeypatch.setattr(rg, "orchestrator_pids", unscannable)
        monkeypatch.setattr(rg, "running_pod_count", never)
        monkeypatch.setattr(rg, "rearm", lambda: calls.append("rearm"))

        assert rg.main(["--confirm"]) == rg.EXIT_ERROR

        assert calls == ["rearm"]

    def test_a_broken_pod_query_rearms_and_does_not_count_as_quiet(
        self, monkeypatch: pytest.MonkeyPatch, guard_state: Path,
    ) -> None:
        calls: list[str] = []

        def unreachable() -> int:
            raise RuntimeError("RunPod API returned HTTP 502")

        monkeypatch.setattr(rg, "orchestrator_pids", lambda: [])
        monkeypatch.setattr(rg, "running_pod_count", unreachable)
        monkeypatch.setattr(rg, "rearm", lambda: calls.append("rearm"))

        assert rg.main(["--confirm"]) == rg.EXIT_ERROR

        assert calls == ["rearm"]
        assert rg._read_quiet_streak() == 0


class TestRearmInterval:
    """The systemd string and the staleness window must agree."""

    def test_the_rearm_string_is_derived_from_the_seconds(self) -> None:
        assert rg.REARM == "2h"
        assert rg.REARM_SECONDS == 7200
        assert rg.QUIET_STREAK_TTL_S > rg.REARM_SECONDS


class TestTheGuardWillNotKillLiveTraining:
    """A dead orchestrator does not mean idle pods.

    Folds run under `setsid nohup` so training outlives the SSH session that
    started it. "No orchestrator, pods up" is therefore the exact shape of a
    fleet mid-training with its driver crashed, and reaping on those two facts
    alone destroys the paid-for work the detachment exists to protect.
    """

    @staticmethod
    def _pod(name: str) -> dict[str, object]:
        return {
            "id": f"id-{name}",
            "name": name,
            "runtime": {"ports": [{"ip": "1.2.3.4", "publicPort": 2222, "privatePort": 22}]},
        }

    def test_a_busy_pod_stops_the_reap(self, monkeypatch) -> None:
        from scripts.phase1b import reaper_guard as g

        monkeypatch.setattr(g, "expected_pod_names", lambda: {"tract-p1b-val-fold0"})
        monkeypatch.setattr(
            "scripts.phase0.runpod_provision.get_running_pods",
            lambda: [self._pod("tract-p1b-val-fold0")],
        )
        monkeypatch.setattr(g, "pod_training_state", lambda pod: "BUSY")
        idle, reason = g.fleet_is_idle()
        assert idle is False
        assert "still training" in reason

    def test_an_unreachable_pod_stops_the_reap(self, monkeypatch) -> None:
        """Unreachable counts as busy: killing training cannot be undone."""
        from scripts.phase1b import reaper_guard as g

        monkeypatch.setattr(g, "expected_pod_names", lambda: {"tract-p1b-val-fold0"})
        monkeypatch.setattr(
            "scripts.phase0.runpod_provision.get_running_pods",
            lambda: [self._pod("tract-p1b-val-fold0")],
        )
        monkeypatch.setattr(g, "pod_training_state", lambda pod: "UNREACHABLE")
        idle, reason = g.fleet_is_idle()
        assert idle is False
        assert "cannot tell" in reason

    def test_one_busy_pod_protects_the_whole_fleet(self, monkeypatch) -> None:
        from scripts.phase1b import reaper_guard as g

        names = {f"tract-p1b-val-fold{i}" for i in range(5)}
        monkeypatch.setattr(g, "expected_pod_names", lambda: names)
        monkeypatch.setattr(
            "scripts.phase0.runpod_provision.get_running_pods",
            lambda: [self._pod(n) for n in sorted(names)],
        )
        monkeypatch.setattr(
            g, "pod_training_state",
            lambda pod: "BUSY" if pod["name"].endswith("3") else "IDLE",
        )
        idle, reason = g.fleet_is_idle()
        assert idle is False, "one training pod must protect the whole fleet"
        assert "fold3" in reason

    def test_an_all_idle_fleet_may_be_reaped(self, monkeypatch) -> None:
        from scripts.phase1b import reaper_guard as g

        names = {f"tract-p1b-val-fold{i}" for i in range(3)}
        monkeypatch.setattr(g, "expected_pod_names", lambda: names)
        monkeypatch.setattr(
            "scripts.phase0.runpod_provision.get_running_pods",
            lambda: [self._pod(n) for n in sorted(names)],
        )
        monkeypatch.setattr(g, "pod_training_state", lambda pod: "IDLE")
        idle, reason = g.fleet_is_idle()
        assert idle is True
        assert "idle" in reason

    def test_an_unreadable_api_is_not_evidence_of_idleness(self, monkeypatch) -> None:
        from scripts.phase1b import reaper_guard as g

        def boom() -> list[dict[str, object]]:
            raise RuntimeError("HTTP 502")

        monkeypatch.setattr(g, "expected_pod_names", lambda: {"tract-p1b-val-fold0"})
        monkeypatch.setattr("scripts.phase0.runpod_provision.get_running_pods", boom)
        idle, reason = g.fleet_is_idle()
        assert idle is False
        assert "could not list pods" in reason

    def test_a_pod_with_no_ssh_endpoint_is_unreachable_not_idle(self) -> None:
        """A pod in its first minute has no published port. Not finished."""
        from scripts.phase1b import reaper_guard as g

        assert g.pod_training_state({"name": "x", "runtime": {"ports": []}}) == "UNREACHABLE"
        assert g.pod_training_state({"name": "x"}) == "UNREACHABLE"
