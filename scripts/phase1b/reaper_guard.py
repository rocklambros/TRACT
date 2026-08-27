"""A reaper that will not kill a campaign that is still running.

    # arm it when you launch the campaign, not before.
    # --working-directory is NOT optional: systemd-run starts the unit in the
    # user's home, where `-m scripts.phase1b.reaper_guard` raises
    # ModuleNotFoundError while list-timers still reports it ARMED.
    systemd-run --user --on-active=8h --unit=tract-reaper \
        --working-directory="$PWD" --setenv=USE_TF=0 \
        "$(command -v python3)" -m scripts.phase1b.reaper_guard --confirm

    # see them (re-arms carry a generation suffix), or cancel:
    systemctl --user list-timers 'tract-reaper*'
    systemctl --user stop 'tract-reaper*.timer'

    # tell the guard the campaign is over, so the next quiet check disarms:
    mkdir -p "${XDG_RUNTIME_DIR:-/tmp}/tract-reaper"
    touch "${XDG_RUNTIME_DIR:-/tmp}/tract-reaper/campaign-complete"

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

That applies to the QUIET case too, and it is where the second version of this
file failed. Between two arms of a campaign the orchestrator is dead and the pods
are down -- which is byte-for-byte what a finished campaign looks like from here.
Returning EXIT_OK there without re-arming disarmed the guard in the first of
Campaign 2's five inter-arm gaps, and every arm after it would have run with no
bound at all. So a quiet check re-arms as well, and a streak counter decides when
quiet has gone on long enough to mean "finished" rather than "between arms". An
operator who knows the campaign is over says so with the sentinel above instead
of waiting the streak out.

Owner: TRACT.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

# Same derivation runpod_parallel.py uses. Needed because systemd-run starts a
# unit in the user's home unless told otherwise, and `-m scripts.phase1b...`
# only resolves from the repository root.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXIT_OK: Final[int] = 0
EXIT_REAPED: Final[int] = 10
EXIT_ERROR: Final[int] = 1

# Long enough that a re-arm is not a busy-wait, short enough that an orphaned
# fleet at $13.45/hr is not billing all night before the next look.
#
# Held in seconds and rendered for systemd, not written twice: the quiet-streak
# staleness window below is expressed in multiples of this interval, and two
# hand-maintained copies of "2h" would drift the moment someone tuned one.
REARM_SECONDS: Final[int] = 2 * 60 * 60
REARM: Final[str] = f"{REARM_SECONDS // 3600}h"
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

# The two ways the runbook, the docs and every script in this repo start the
# orchestrator. A process must BE one of these to count as alive; mentioning
# the name is not enough. See _is_orchestrator_argv for the incident.
ORCHESTRATOR_MODULE: Final[str] = "scripts.phase1b.runpod_parallel"
ORCHESTRATOR_SCRIPT: Final[str] = "runpod_parallel.py"

# python3, python3.12, python3.13t (free-threaded). Matched on the BASENAME of
# argv[0], so an absolute interpreter path from `command -v python3` or
# sys.executable resolves the same as a bare word.
PYTHON_BASENAME_RE: Final[re.Pattern[str]] = re.compile(r"^python[0-9.]*t?$")

# Interpreter options that swallow the NEXT argv element as their value, which
# would otherwise be misread as the script python was asked to run.
FLAGS_WITH_SEPARATE_VALUE: Final[frozenset[str]] = frozenset(
    {"-W", "-X", "--check-hash-based-pycs"}
)

# Where the quiet-streak counter lives. NOT in the repo: the systemd unit runs
# from the repository root, so a dotfile here would be untracked clutter inside
# a working tree that campaign artefacts are committed from. The user runtime
# dir is also wiped at logout, which resets the streak toward re-arming -- the
# safe direction -- rather than toward a stale disarm.
STATE_DIRNAME: Final[str] = "tract-reaper"
QUIET_STREAK_FILENAME: Final[str] = "quiet-streak.json"
CAMPAIGN_COMPLETE_FILENAME: Final[str] = "campaign-complete"
QUIET_STREAK_KEY: Final[str] = "quiet_checks"
QUIET_UPDATED_KEY: Final[str] = "updated_at"

# How many consecutive quiet checks (no orchestrator, no pods) mean the
# campaign is over rather than between arms. The whole trade sits in this
# number. Too few and the guard disarms inside an inter-arm gap -- Campaign 2
# has five, and a gap can run long when an arm is re-provisioned after a
# capacity error. Too many and the guard re-arms forever after the campaign
# ends, leaving a timer that fires weeks later against whatever pods happen to
# carry these names then, which is its own way to destroy paid-for work.
# Three checks at REARM apart is roughly four hours of silence before it lets
# go, and an operator who knows better touches CAMPAIGN_COMPLETE_FILENAME.
QUIET_CHECKS_BEFORE_DISARM: Final[int] = 3

# A streak is consecutive in TIME, not just in count. A live streak writes an
# entry every REARM; anything older than twice that belongs to a previous
# campaign and must not be inherited by this one, or the first quiet check of a
# new campaign would disarm on a count it never earned.
QUIET_STREAK_TTL_S: Final[int] = 2 * REARM_SECONDS


@dataclass(frozen=True)
class PythonTarget:
    """What an interpreter was actually asked to run, and where argv says so.

    `index` matters as much as `name`: the action word has to be read from what
    FOLLOWS the target, because that is where the program's own arguments
    begin. Scanning all of argv instead reads the interpreter's options as if
    they were the pipeline's -- and `python3 -X track -m ...` is a command line
    CPython accepts and runs, so the guard would call the busiest possible
    process a read-only query and reap a fleet in mid-training.
    """

    kind: Literal["module", "script", "inline", "none"]
    name: str
    index: int


def _is_python_interpreter(argv0: str) -> bool:
    """Is argv[0] a python interpreter, by basename?"""
    return bool(PYTHON_BASENAME_RE.match(Path(argv0).name))


def _python_target(argv: list[str]) -> PythonTarget:
    """Resolve what python would run from *argv*, skipping interpreter options.

    Deliberately understands the awkward forms as well as the tidy ones --
    `-um module`, `-mmodule`, `-X importtime -m module` -- because the guard
    that reads this cannot ask the process what it meant.
    """
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            if i + 1 < len(argv):
                return PythonTarget("script", argv[i + 1], i + 1)
            return PythonTarget("none", "", -1)
        if arg in FLAGS_WITH_SEPARATE_VALUE:
            i += 2
            continue
        if arg.startswith(("-W", "-X")):
            # The attached form, `-Wignore`; the separated form was consumed
            # above.
            i += 1
            continue
        if arg.startswith("--"):
            i += 1
            continue
        if arg.startswith("-") and len(arg) > 1:
            # A short-option cluster. `-m` and `-c` end it and take the rest of
            # the cluster, or the next element, as their value.
            for offset, letter in enumerate(arg[1:]):
                if letter in ("m", "c"):
                    kind: Literal["module", "inline"] = (
                        "module" if letter == "m" else "inline"
                    )
                    attached = arg[offset + 2:]
                    if attached:
                        return PythonTarget(kind, attached, i)
                    if i + 1 < len(argv):
                        return PythonTarget(kind, argv[i + 1], i + 1)
                    return PythonTarget("none", "", -1)
            i += 1
            continue
        return PythonTarget("script", arg, i)
    return PythonTarget("none", "", -1)


def _is_orchestrator_argv(argv: list[str]) -> bool:
    """Is this command line a live runpod_parallel doing fleet work?

    The test is a strict positive one: the process must BE the orchestrator.
    The first version asked only whether "runpod_parallel" appeared ANYWHERE in
    argv, and the string appears in far more than the orchestrator. Verified
    2026-08-27: `tail -f results/phase1b/runpod_parallel.log` -- the log an
    operator watches for the entire campaign -- was classified as a live
    orchestrator, so the guard stood down on every single check and the fleet
    had no bound at all for as long as anyone was paying attention to it.

    Blocklisting tail, less, grep and vim would have fixed that one command and
    nothing else; the next tool an operator reaches for would walk straight back
    into it. So the question asked here is what python was asked to RUN --
    `-m scripts.phase1b.runpod_parallel` or a script whose basename is
    runpod_parallel.py -- with argv[0] confirmed to be an interpreter. A reader
    fails both halves independently: `tail` is not python, and a .log is not the
    script.

    The strictness cuts the other way for a launch form this does not know
    about, and that direction is the dangerous one -- an unrecognised live
    orchestrator reads as dead. Wrappers are safe (nohup, timeout, setsid,
    systemd-run and `bash -c` all leave a real python child with a real argv,
    and that child is what matches), so what is left is an interpreter that
    does not answer to "python". If the campaign ever launches through one, it
    must be taught here, not discovered at 3am.

    Another generation of this guard needs no special case any more: its target
    module is scripts.phase1b.reaper_guard, which is not the orchestrator.
    """
    if not argv or not _is_python_interpreter(argv[0]):
        return False
    target = _python_target(argv)
    if target.kind == "module":
        if target.name != ORCHESTRATOR_MODULE:
            return False
    elif target.kind == "script":
        if Path(target.name).name != ORCHESTRATOR_SCRIPT:
            return False
    else:
        # A REPL or a `-c` one-liner is not the pipeline.
        return False
    # Absent an explicit action word, argparse supplies "full" -- so the
    # absence of a word means the whole pipeline, not nothing.
    action = next(
        (arg for arg in argv[target.index + 1:] if arg in ALL_ACTIONS),
        DEFAULT_ACTION,
    )
    return action not in NON_FLEET_ACTIONS


def _read_argv(pid: int) -> list[str]:
    """The exec argv of *pid*, or an empty list if it cannot be read.

    Every OSError answers "nothing", not just the three obvious ones. /proc is
    a live filesystem being enumerated while processes come and go: the entry
    disappears between listdir and open (FileNotFoundError), it belongs to
    another user (PermissionError), it exits mid-read (ProcessLookupError) --
    and rarer kernels return EIO, which is a bare OSError. Any of those
    escaping this loop would take down the whole check, re-arm included, over a
    process that was never ours to reap against.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read().decode("utf-8", "replace")
    except OSError:
        return []
    # cmdline is NUL-separated and NUL-terminated, so a naive split leaves a
    # trailing empty element on every process on the box.
    return [arg for arg in raw.split("\x00") if arg]


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
        if _is_orchestrator_argv(_read_argv(pid)):
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


# What `pgrep -f` looks for on the pod. The fold is launched as a detached
# `python -m scripts.phase1b.run_fold ...` under setsid, so its own name is what
# survives in the pod's process table.
FOLD_PROCESS_PATTERN: Final[str] = "run_fold"

# How long to wait for a pod to answer. Generous: a box saturated by training is
# slow to accept an SSH connection, and reading slowness as death is the exact
# mistake this function exists to avoid.
POD_PROBE_TIMEOUT_S: Final[int] = 45


def pod_training_state(pod: dict[str, object]) -> Literal["BUSY", "IDLE", "UNREACHABLE"]:
    """Whether this pod is actually running a fold right now.

    THE REASON THIS EXISTS. A fold is launched with `setsid nohup`, deliberately,
    so that training outlives the SSH session that started it -- a laptop
    sleeping or a wifi change used to kill an hour of paid GPU time. The
    consequence is that a pod can be training happily while the orchestrator on
    the Jetson is dead. Until 2026-08-27 this guard asked only two questions --
    is the orchestrator alive, are pods up -- and would therefore have answered
    "no orchestrator, pods running, reap" about a fleet in the middle of
    training, destroying exactly the paid-for work the detachment was added to
    protect.

    UNREACHABLE IS TREATED AS BUSY BY THE CALLER, and that is a deliberate
    trade. A pod that will not answer might be wedged and billing for nothing,
    which argues for reaping; or it might be a network blip in front of a
    healthy trainer, which argues for leaving it. Killing live training is
    unrecoverable and a wasted hour of billing is not, so the tie goes to not
    reaping -- loudly, so a human can settle it.
    """
    from scripts.phase1b.runpod_parallel import SSH_KEY

    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or [] if isinstance(runtime, dict) else []
    endpoint = next(
        (p for p in ports if isinstance(p, dict) and p.get("privatePort") == 22), None
    )
    if not endpoint or not endpoint.get("ip") or not endpoint.get("publicPort"):
        # No SSH endpoint published yet. That is the state a pod sits in for the
        # first minute or two of its life, when it is certainly not finished.
        return "UNREACHABLE"

    try:
        result = subprocess.run(
            ["ssh", "-o", "StrictHostKeyChecking=accept-new",
             "-o", f"UserKnownHostsFile={PROJECT_ROOT}/scripts/phase1b/.runpod_known_hosts",
             "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
             "-o", f"ConnectTimeout={POD_PROBE_TIMEOUT_S}",
             "-i", str(SSH_KEY),
             "-p", str(endpoint["publicPort"]), f"root@{endpoint['ip']}",
             f"pgrep -f {FOLD_PROCESS_PATTERN} >/dev/null && echo BUSY || echo IDLE"],
            capture_output=True, text=True, timeout=POD_PROBE_TIMEOUT_S + 15, check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("  probe %s: unreachable (%s)", pod.get("name"), exc)
        return "UNREACHABLE"

    answer = (result.stdout or "").strip().splitlines()
    if result.returncode != 0 or not answer:
        logger.warning("  probe %s: unreachable (rc=%d %s)", pod.get("name"),
                       result.returncode, (result.stderr or "").strip()[:80])
        return "UNREACHABLE"
    return "BUSY" if answer[-1] == "BUSY" else "IDLE"


def fleet_is_idle() -> tuple[bool, str]:
    """True only when every one of this run's pods is provably not training.

    Anything else -- one busy pod, one that will not answer, or an API that will
    not respond -- means do not reap. Returns the reason so the caller can say
    it out loud rather than standing down silently.
    """
    from scripts.phase0.runpod_provision import get_running_pods

    expected = expected_pod_names()
    try:
        mine = [p for p in get_running_pods() if p.get("name") in expected]
    except Exception as exc:  # noqa: BLE001 - an unreadable API is not evidence of idleness
        return False, f"could not list pods ({exc})"

    if not mine:
        return True, "no pods running"

    states = {str(p.get("name")): pod_training_state(p) for p in mine}
    for name, state in sorted(states.items()):
        logger.info("  %s: %s", name, state)

    busy = [n for n, s in states.items() if s == "BUSY"]
    unknown = [n for n, s in states.items() if s == "UNREACHABLE"]
    if busy:
        return False, f"still training: {', '.join(sorted(busy))}"
    if unknown:
        return False, (f"cannot tell (no answer from {', '.join(sorted(unknown))}). "
                       f"Refusing to reap on a guess -- killing live training is "
                       f"unrecoverable, an idle pod billing is not. Check by hand.")
    return True, f"all {len(states)} pod(s) idle"


def _state_dir() -> Path:
    """Where the quiet-streak counter and the sentinel live.

    Resolved per call rather than at import, so the answer follows the
    environment the unit is actually started in. systemd --user always exports
    XDG_RUNTIME_DIR; a hand-run shell on a headless box may not, and falling
    back to the temp dir keeps the streak working there instead of raising in
    the one code path whose job is to not die.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR", "")
    base = Path(runtime) if runtime else Path(tempfile.gettempdir())
    return base / STATE_DIRNAME


def _campaign_is_complete() -> bool:
    """Has an operator declared the campaign finished?

    An empty file, because the statement is its existence. The touch command is
    in the usage block at the top of this file, next to the systemd-run line
    that arms the guard in the first place.
    """
    return (_state_dir() / CAMPAIGN_COMPLETE_FILENAME).exists()


def _read_quiet_streak() -> int:
    """Consecutive quiet checks so far, or 0 if that cannot be established.

    Every failure here answers 0, which means "keep re-arming". That is the
    deliberate opposite of this project's fail-loud default, and the reason is
    the same one that put a try/except around reap() below: an exception raised
    anywhere in this check kills the process before it re-arms, and the guard is
    the only bound on an orphaned fleet. A counter this cannot parse must cost
    an extra re-arm, never the campaign. The failure is still logged at ERROR --
    loud, just not fatal.
    """
    path = _state_dir() / QUIET_STREAK_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        # First check of a campaign, or a runtime dir wiped at logout.
        return 0
    except OSError as exc:
        logger.error("Could not read %s (%s). Treating the streak as zero, "
                     "which keeps the guard armed.", path, exc)
        return 0

    try:
        state = json.loads(raw)
        count = state[QUIET_STREAK_KEY]
        updated_at = state[QUIET_UPDATED_KEY]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.error("Unusable quiet-streak state in %s (%s). Treating the "
                     "streak as zero, which keeps the guard armed.", path, exc)
        return 0

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        logger.error("Nonsensical quiet-streak count %r in %s. Treating the "
                     "streak as zero.", count, path)
        return 0
    if not isinstance(updated_at, (int, float)) or isinstance(updated_at, bool):
        logger.error("Nonsensical quiet-streak timestamp %r in %s. Treating "
                     "the streak as zero.", updated_at, path)
        return 0
    age = time.time() - float(updated_at)
    if age > QUIET_STREAK_TTL_S:
        logger.info("Quiet-streak state in %s is %.0fs old, older than the "
                    "%ds window. It belongs to an earlier campaign; starting "
                    "over.", path, age, QUIET_STREAK_TTL_S)
        return 0
    # int() rather than a bare return: `count` came out of json.loads as Any,
    # and --strict refuses to launder that through a declared -> int.
    return int(count)


def _write_quiet_streak(count: int) -> None:
    """Persist the streak. A failure to write costs a re-arm, not the guard."""
    # Imported here, not at module scope, for the same reason every other heavy
    # import in this file is: `-m scripts.phase1b.reaper_guard` has to start
    # even when the training stack underneath it is broken, because that is
    # exactly when an orphaned fleet is billing.
    from tract.io import atomic_write_json

    path = _state_dir() / QUIET_STREAK_FILENAME
    try:
        atomic_write_json(
            {QUIET_STREAK_KEY: count, QUIET_UPDATED_KEY: time.time()}, path
        )
    except OSError as exc:
        logger.error("Could not record the quiet streak in %s (%s). The guard "
                     "will keep re-arming, which is the safe direction.",
                     path, exc)


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
             # systemd-run does NOT inherit the caller's cwd. Without this the
             # re-armed unit starts in the user's home and dies in ~59ms with
             # "No module named 'scripts'" -- while `systemctl list-timers`
             # still shows it ARMED, so the checklist row that verifies the
             # bound passes on a unit that reaps nothing. Verified 2026-08-26.
             f"--working-directory={PROJECT_ROOT}",
             "--setenv=USE_TF=0",
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

    try:
        pids = orchestrator_pids()
    except Exception as exc:  # noqa: BLE001 - see the reap handler below
        # Failing to ANSWER question 1 is not a No. Reaping on an unscannable
        # /proc would terminate a fleet on the strength of a filesystem error.
        logger.error("Could not scan /proc for an orchestrator (%s). Standing "
                     "down and re-arming rather than guessing.", exc)
        if args.confirm:
            rearm()
        return EXIT_ERROR

    if pids:
        logger.info(
            "Orchestrator alive (pid %s). Something is driving the fleet, so "
            "reaping now would destroy paid-for work. Standing down.",
            ", ".join(str(p) for p in pids),
        )
        if args.confirm:
            # Seeing the orchestrator ends any streak: the campaign is running.
            _write_quiet_streak(0)
            rearm()
        return EXIT_OK

    try:
        count = running_pod_count()
    except Exception as exc:  # noqa: BLE001 - any API failure must re-arm, not exit quietly
        logger.error("Could not read pod state (%s). Re-arming rather than "
                     "assuming the fleet is down.", exc)
        if args.confirm:
            # An unreadable API is not evidence of quiet, so the streak does
            # not advance -- otherwise an API outage could time the guard out.
            rearm()
        return EXIT_ERROR

    if count == 0:
        if _campaign_is_complete():
            logger.info(
                "No orchestrator and no pods, and %s says the campaign is "
                "over. Nothing to reap and nothing left to bound; disarming.",
                _state_dir() / CAMPAIGN_COMPLETE_FILENAME,
            )
            return EXIT_OK

        streak = _read_quiet_streak() + 1
        if args.confirm:
            _write_quiet_streak(streak)
        if streak >= QUIET_CHECKS_BEFORE_DISARM:
            logger.info(
                "No orchestrator and no pods for %d consecutive checks about "
                "%s apart. That is a finished campaign, not a gap between "
                "arms; disarming.", streak, REARM,
            )
            return EXIT_OK

        logger.info(
            "No orchestrator and no pods from this run are running (quiet "
            "check %d of %d). The gap between two arms looks exactly like "
            "this, so re-arming rather than assuming the campaign is over.",
            streak, QUIET_CHECKS_BEFORE_DISARM,
        )
        if args.confirm:
            rearm()
        return EXIT_OK

    logger.warning(
        "No orchestrator process, but %d pod(s) from this run are still "
        "RUNNING and billing. Checking whether they are actually idle before "
        "touching anything.", count,
    )

    # THE THIRD QUESTION, and the one that protects paid-for work.
    #
    # A dead orchestrator does NOT mean idle pods. Folds are launched under
    # `setsid nohup` precisely so training survives the session that started it,
    # so "no orchestrator, pods up" is the exact shape of a fleet in the middle
    # of training with its driver crashed. Reaping on those two facts alone
    # would destroy hours of GPU time whose per-item indicators exist only on
    # those pods -- the work the detachment was added to protect.
    #
    # So each pod is asked directly whether a fold process is alive. Reap only
    # when every one of them says no. A pod that will not answer counts as busy:
    # killing live training cannot be undone, an idle pod billing for another
    # two hours can.
    idle, reason = fleet_is_idle()
    if not idle:
        logger.warning(
            "NOT REAPING -- %s. Re-arming; the fleet keeps running.", reason,
        )
        _write_quiet_streak(0)
        if args.confirm:
            rearm()
        return EXIT_OK

    logger.warning("Every pod is idle (%s) and no orchestrator is driving them. "
                   "This is the orphaned-fleet case the schedule exists for.",
                   reason)
    if not args.confirm:
        logger.warning("Dry run. Re-run with --confirm to reap.")
        return EXIT_OK

    from scripts.phase1b.runpod_parallel import reap

    # Pods are up, so whatever the counter said, this is not a quiet campaign.
    _write_quiet_streak(0)
    try:
        reap(confirm=True)
    except Exception as exc:  # noqa: BLE001 - a guard that dies is not a bound
        # reap() reads .pod_state.json and talks to the RunPod API, so a
        # truncated state file (JSONDecodeError) or an HTTP 502 lands here.
        # Unwrapped, either one killed the guard mid-reap and took the re-arm
        # with it -- leaving a fleet that is both orphaned and unwatched, which
        # is strictly worse than the situation this file exists to prevent.
        logger.exception(
            "REAP FAILED against %d running pod(s) (%s). The fleet is STILL "
            "BILLING. Re-arming so the next check tries again; terminate by "
            "hand if this repeats.", count, exc,
        )
        rearm()
        return EXIT_ERROR

    logger.warning("Reaped an orphaned fleet of %d pod(s).", count)
    # Re-arm even on success: terminate_pods can fail per pod, so "reap
    # returned" is not "every pod is gone". The next check confirms it, and the
    # quiet streak above guarantees these follow-ups end rather than run on
    # forever.
    rearm()
    return EXIT_REAPED


if __name__ == "__main__":
    sys.exit(main())
