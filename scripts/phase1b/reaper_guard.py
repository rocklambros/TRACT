"""A reaper that will not kill a campaign that is still running.

    # arm it when you launch the campaign, not before:
    systemd-run --user --on-active=8h --unit=tract-reaper \
        /home/rock/anaconda3/bin/python3 -m scripts.phase1b.reaper_guard --confirm

    # see it, or cancel it:
    systemctl --user list-timers tract-reaper
    systemctl --user stop tract-reaper.timer

WHY THIS EXISTS. The runbook says to schedule `reap --confirm` at T+8h, and the
reasoning behind it is sound: `create_pod` sends no TTL, no auto-stop and no idle
timeout, so a dead orchestrator bills until a person acts. A scheduled reaper is
the only bound that survives the orchestrator dying.

But `reap` has no liveness guard. It terminates every pod matching this run's
names -- and when the state file is missing or stale it sweeps the account by
name instead -- without ever asking whether folds are still training. Fired at
T+8h against a campaign that legitimately ran long, it destroys paid-for GPU
hours whose per-item indicators exist only on those pods.

It is worse than that. `full_pipeline` leaves the fleet up DELIBERATELY on every
failure path, because "leaving pods up costs money; terminating them costs the
run", so that a failed fold can be retried on a pod that is still warm and still
has the model cached. An unguarded reaper is aimed squarely at that recovery
window.

So this wrapper asks two questions before it fires:

  1. Is an orchestrator process alive? If yes, something is driving the fleet.
     Re-arm and leave it alone.
  2. Are any of this run's pods actually running? If no, there is nothing to
     reap and no reason to touch anything.

Only when the orchestrator is gone AND pods are still billing does it reap. That
is exactly the P1 scenario the schedule exists for, and nothing else.

Re-arming rather than giving up matters: a campaign that outlives the first check
must still be bounded once it finally dies, or the guard has simply moved the
unbounded window later.

Owner: TRACT.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from typing import Final

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXIT_OK: Final[int] = 0
EXIT_REAPED: Final[int] = 10
EXIT_ERROR: Final[int] = 1

# Long enough that a re-arm is not a busy-wait, short enough that an orphaned
# fleet at $13.45/hr is not billing all night before the next look.
REARM: Final[str] = "2h"
UNIT: Final[str] = "tract-reaper"

# runpod_parallel's argparse `action` positional, and its default.
ALL_ACTIONS: Final[frozenset[str]] = frozenset(
    {"full", "provision", "run", "collect", "aggregate",
     "track", "teardown", "reap", "price"}
)
DEFAULT_ACTION: Final[str] = "full"

# Actions that do NOT drive a fleet: local computation or short read-only
# queries. Everything else -- including anything unrecognised -- counts as an
# orchestrator, which is the fail-SAFE direction: mistaking a dead process for
# a live one costs a delayed reap, while mistaking a live one for dead
# terminates a training fleet.
#
# This list is an allowlist for exactly that reason. The first version of this
# file used the opposite shape, an allowlist of orchestrator actions
# {run, provision, full, collect}, and it had a hole big enough to lose a
# campaign: `action` is `nargs="?"` with `default="full"`, so a bare
# `python -m scripts.phase1b.runpod_parallel` runs the WHOLE pipeline while the
# word "full" never appears in /proc/<pid>/cmdline. The guard saw no action
# word, matched nothing, and classified the busiest possible process as dead.
NON_FLEET_ACTIONS: Final[frozenset[str]] = frozenset(
    {"aggregate", "track", "price", "reap"}
)


def orchestrator_pids() -> list[int]:
    """PIDs of live runpod_parallel processes doing fleet work, excluding self.

    Reads /proc rather than shelling to pgrep, because pgrep's own -f match
    would find this process's command line and the exclusion would then depend
    on argument ordering.
    """
    me = os.getpid()
    found: list[int] = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as handle:
                argv = handle.read().decode("utf-8", "replace").split("\x00")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            # The process exited between listdir and open, or belongs to
            # another user. Neither is ours to reap against.
            continue
        if not any("runpod_parallel" in arg for arg in argv):
            continue
        if "reaper_guard" in " ".join(argv):
            continue
        # Absent an explicit action word, argparse supplies "full" -- so the
        # absence of a word means the whole pipeline, not nothing.
        action = next((arg for arg in argv if arg in ALL_ACTIONS), DEFAULT_ACTION)
        if action not in NON_FLEET_ACTIONS:
            found.append(pid)
    return found


def expected_pod_names() -> set[str]:
    """Every pod name this campaign can create, across BOTH splits.

    Derived from select_pod_configs rather than from POD_CONFIGS, and that
    distinction is the whole point. POD_CONFIGS holds the TEST fleet only --
    tract-p1b-fold0..4 -- while the validation split provisions
    tract-p1b-val-fold0..4 under a different prefix (select_pod_configs sets
    `prefix = "tract-p1b" if split == "test" else "tract-p1b-val"`).

    Campaign 2 runs five validation rounds and one test round. A guard matching
    POD_CONFIGS would see zero pods for five of the six, conclude there was
    nothing to reap, and DISARM ITSELF while five GPUs billed. The first version
    of this file did exactly that.

    `reap`'s own orphan sweep still matches POD_CONFIGS, so its name-based
    fallback is blind to validation pods too. That only bites when the state
    file is missing or stale -- which is the situation the fallback exists for.
    Fixing it belongs in runpod_parallel.py and is not this file's call to make;
    calling reap() here is still correct because reap reads the state file
    first, and this guard only calls it once pods are confirmed running.
    """
    from scripts.phase1b.runpod_parallel import select_pod_configs

    return {
        config["name"]
        for split in ("test", "validation")
        for config in select_pod_configs(None, split)
    }


def running_pod_count() -> int:
    """How many of this run's pods are up. Raises on an unusable API."""
    from scripts.phase0.runpod_provision import get_running_pods

    expected = expected_pod_names()
    return sum(1 for pod in get_running_pods() if pod.get("name") in expected)


def rearm() -> None:
    """Schedule the next look. A guard that gives up is not a bound.

    The unit name carries a per-generation suffix. A fixed name cannot work:
    the re-arm runs from INSIDE the current tract-reaper unit, which is still
    active, and systemd refuses with "Unit tract-reaper.timer already exists."
    Verified 2026-08-26 -- systemd-run exits non-zero and check=True lands in
    the error branch below, so the first version of this file announced the
    fleet was unbounded every single time it tried to re-arm.

    The shared `tract-reaper` prefix is what keeps them discoverable:
        systemctl --user list-timers 'tract-reaper*'
    """
    unit = f"{UNIT}-{int(time.time())}"
    try:
        subprocess.run(
            ["systemd-run", "--user", "--collect", f"--on-active={REARM}",
             f"--unit={unit}",
             sys.executable, "-m", "scripts.phase1b.reaper_guard", "--confirm"],
            check=True, capture_output=True, timeout=30,
        )
        logger.info("Re-armed for %s as %s.", REARM, unit)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        # Loud, because a failed re-arm silently removes the only bound on spend.
        detail = getattr(exc, "stderr", b"") or b""
        logger.error(
            "COULD NOT RE-ARM (%s: %s). The fleet is now unbounded if the "
            "orchestrator dies. Re-arm by hand or watch it yourself.",
            exc, detail.decode("utf-8", "replace").strip(),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--confirm", action="store_true",
                        help="actually reap; without it, report and change nothing")
    args = parser.parse_args(argv)

    pids = orchestrator_pids()
    if pids:
        logger.info(
            "Orchestrator alive (pid %s). Something is driving the fleet, so "
            "reaping now would destroy paid-for work. Standing down.",
            ", ".join(str(p) for p in pids),
        )
        if args.confirm:
            rearm()
        return EXIT_OK

    try:
        count = running_pod_count()
    except Exception as exc:  # noqa: BLE001 - any API failure must re-arm, not exit quietly
        logger.error("Could not read pod state (%s). Re-arming rather than "
                     "assuming the fleet is down.", exc)
        if args.confirm:
            rearm()
        return EXIT_ERROR

    if count == 0:
        logger.info("No orchestrator and no pods from this run are running. "
                    "Nothing to reap; not re-arming.")
        return EXIT_OK

    logger.warning(
        "No orchestrator process, but %d pod(s) from this run are still "
        "RUNNING and billing. This is the orphaned-fleet case the schedule "
        "exists for.", count,
    )
    if not args.confirm:
        logger.warning("Dry run. Re-run with --confirm to reap.")
        return EXIT_OK

    from scripts.phase1b.runpod_parallel import reap

    reap(confirm=True)
    logger.warning("Reaped an orphaned fleet of %d pod(s).", count)
    return EXIT_REAPED


if __name__ == "__main__":
    sys.exit(main())
