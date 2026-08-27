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

# Matched against the full command line. `price` and `reap` are excluded on
# purpose: both are short-lived read-only commands, and treating one as a live
# orchestrator would make the guard refuse to fire while a fleet burned.
ORCHESTRATOR_ACTIONS: Final[frozenset[str]] = frozenset(
    {"run", "provision", "full", "collect"}
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
        if ORCHESTRATOR_ACTIONS.intersection(argv):
            found.append(pid)
    return found


def running_pod_count() -> int:
    """How many of this run's pods are up. Raises on an unusable API."""
    from scripts.phase0.runpod_provision import get_running_pods
    from scripts.phase1b.runpod_parallel import POD_CONFIGS

    expected = {config["name"] for config in POD_CONFIGS}
    return sum(1 for pod in get_running_pods() if pod.get("name") in expected)


def rearm() -> None:
    """Schedule the next look. A guard that gives up is not a bound."""
    try:
        subprocess.run(
            ["systemd-run", "--user", f"--on-active={REARM}", f"--unit={UNIT}",
             sys.executable, "-m", "scripts.phase1b.reaper_guard", "--confirm"],
            check=True, capture_output=True, timeout=30,
        )
        logger.info("Re-armed for %s.", REARM)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        # Loud, because a failed re-arm silently removes the only bound on spend.
        logger.error(
            "COULD NOT RE-ARM (%s). The fleet is now unbounded if the "
            "orchestrator dies. Re-arm by hand or watch it yourself.", exc,
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
