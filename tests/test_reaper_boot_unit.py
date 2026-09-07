"""The reaper's spend bound must survive a reboot.

`scripts/phase1b/reaper_guard.py` is armed with `systemd-run --user`, which
creates a TRANSIENT unit. Transient units do not survive a reboot. Its
docstring reassures that the units run with `Linger=yes` "so it survives
logout" -- true, and irrelevant to power loss.

On 2026-08-30 this machine lost power mid-session. Verified afterwards:
`systemctl --user list-timers 'tract-reaper*'` returned 0 timers and
`$XDG_RUNTIME_DIR/tract-reaper/` was gone. The outage happened to land 72
minutes after teardown so nothing was billing, but folds run under `setsid
nohup` and `create_pod` sends no TTL, no auto-stop and no idle timeout -- so a
reboot during a live fleet leaves it running with no bound at all.

This pins a persistent boot-time unit that re-runs the guard after every boot.
The guard is already safe to run unconditionally: it stands down when an
orchestrator is alive and when no pods of this run exist, and only reaps when
the orchestrator is gone AND pods are still billing.
"""
from __future__ import annotations

import configparser
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIT_PATH = PROJECT_ROOT / "deploy" / "systemd" / "tract-reaper-boot.service"


class TestBootUnitExists:

    def test_unit_file_is_committed(self) -> None:
        assert UNIT_PATH.is_file(), (
            "The only spend bound that survives an orchestrator death is the "
            "reaper, and the only reaper that survives a reboot is a "
            "persistent unit. It must live in the repo, not in one operator's "
            "~/.config."
        )


class TestBootUnitContract:

    @pytest.fixture
    def unit(self) -> configparser.ConfigParser:
        parser = configparser.ConfigParser()
        parser.optionxform = str  # systemd keys are case-sensitive
        parser.read_string(UNIT_PATH.read_text(encoding="utf-8"))
        return parser

    def test_is_installed_for_the_user_session_at_boot(self, unit) -> None:
        # WantedBy=default.target is what makes `systemctl --user enable` run
        # it on every boot. Without it the file is inert.
        assert unit["Install"]["WantedBy"] == "default.target"

    def test_runs_the_guard_not_the_bare_reaper(self, unit) -> None:
        # `reap` has no liveness check and sweeps by name when the state file
        # is missing -- which is exactly the post-reboot situation. The guard
        # asks whether an orchestrator is alive and whether pods exist first.
        exec_start = unit["Service"]["ExecStart"]
        assert "scripts.phase1b.reaper_guard" in exec_start
        assert "--confirm" in exec_start

    def test_is_oneshot(self, unit) -> None:
        assert unit["Service"]["Type"] == "oneshot"

    def test_forces_the_pytorch_backend(self, unit) -> None:
        # USE_TF=0 everywhere a module under tract/ is imported: a TensorFlow
        # import deadlock in sentence-transformers would hang the guard.
        assert "USE_TF=0" in unit["Service"]["Environment"]

    def test_waits_for_the_network(self, unit) -> None:
        # The guard's first action is a RunPod API call. Firing before the
        # network is up turns the bound into a no-op that logs a failure.
        assert "network-online.target" in unit["Unit"]["After"]
