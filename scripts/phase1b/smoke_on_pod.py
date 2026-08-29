"""Run the agentic smoke test on one rented pod, then destroy it.

The smoke test loads a model, and the standing rule is that anything allocating
a model runs on a pod. It is inference over six items and 522 hubs, so this
needs one pod for minutes, not a five-pod fleet for an hour -- `runpod_parallel`
is built around fold-per-pod fleets and has no single-pod path.

Two differences from the fleet bootstrap matter here:

`_rsync_to` excludes `results`, so the fold checkpoints this test exists to
score are NOT carried by the bootstrap. They are pushed separately, from the
config directory as the transfer root, which also keeps that exclude from
matching a path component the sender never sees.

The pod is named outside the `tract-p1b-*` family the reaper sweeps. That keeps
the reaper from destroying this pod mid-run, and it means the reaper will not
clean up after a crash either -- so teardown is in a `finally` that also runs on
KeyboardInterrupt, and a wall-clock cap terminates the pod even if the work
hangs.
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
import time
from typing import Any, Final

from scripts.phase0.runpod_provision import (
    create_pod,
    is_capacity_error,
    rank_available_gpus,
    terminate_pod,
)
from scripts.phase1b.runpod_parallel import (
    DOCKER_IMAGE,
    _bootstrap_pod,
    _rsync_from,
    _rsync_to,
    _ssh,
)
from tract.config import PHASE1B_RESULTS_DIR

logger = logging.getLogger(__name__)

POD_NAME: Final[str] = "tract-smoke-agentic"
# Inference on a 0.6B encoder over 528 short texts. The fleet's 48GB floor is a
# training constraint and would exclude cheap parts for no reason here.
MIN_VRAM_GB: Final[int] = 24
MAX_USD_PER_HOUR: Final[float] = 3.0
# Bootstrap installs torch and downloads the base model, which dominates. The
# scoring itself is minutes. Past this, the pod dies whatever it is doing.
WALL_CAP_S: Final[int] = 3600
SSH_RUN_TIMEOUT_S: Final[int] = 1800


def _provision() -> dict[str, Any]:
    """First candidate that accepts a pod. Capacity errors fall through."""
    candidates = rank_available_gpus(
        min_vram_gb=MIN_VRAM_GB, max_usd_per_hour=MAX_USD_PER_HOUR,
    )
    if not candidates:
        raise RuntimeError(
            f"No GPU at >= {MIN_VRAM_GB}GB under ${MAX_USD_PER_HOUR}/hr. "
            "Nothing was provisioned and nothing is billing."
        )
    logger.info("Candidates: %s", [(c[0], f"${c[1]:.2f}") for c in candidates[:5]])
    last: BaseException | None = None
    for gpu_id, price in candidates:
        try:
            logger.info("Trying %s at $%.2f/hr", gpu_id, price)
            # DOCKER_IMAGE, never create_pod's default. The default is
            # runpod/pytorch:2.4.0-py3.11, and requirements-train.txt is pinned
            # against the Python 3.12 inside the digest-pinned image. Taking
            # the default installed a transformers/sentence-transformers pair
            # that cannot import PreTrainedModel, and the bootstrap died on its
            # own verification step. A fleet and a one-off pod that disagree
            # about the image are not running the same stack.
            pod = create_pod(
                gpu_type_id=gpu_id, name=POD_NAME, image=DOCKER_IMAGE,
            )
            pod["role"] = "agentic-smoke"
            pod["usd_per_hour"] = price
            logger.info("Pod %s up: %s:%s on %s (%s)", pod["pod_id"],
                        pod["ip"], pod["port"], pod.get("gpu_type"),
                        pod.get("cloud_type"))
            return pod
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            if not is_capacity_error(exc):
                raise
            logger.warning("  %s has no capacity, falling through", gpu_id)
            last = exc
    raise RuntimeError(
        "Every candidate GPU returned a capacity error; no pod was created."
    ) from last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--stopwords", action="store_true")
    parser.add_argument("--no-prose", action="store_true")
    parser.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--keep-pod", action="store_true",
                        help="Leave the pod up after the run. It bills until "
                             "terminated by hand.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    local_config = PHASE1B_RESULTS_DIR / args.config_name
    checkpoints = sorted(
        d for d in local_config.glob("fold_*") if (d / "model" / "model").is_dir()
    )
    if not checkpoints:
        raise ValueError(
            f"No fold checkpoints under {local_config}; there is nothing to "
            "score and no reason to rent a pod."
        )
    logger.info("Will ship %d checkpoints from %s", len(checkpoints), local_config)

    flags = ""
    if args.stopwords:
        flags += " --stopwords"
    if args.no_prose:
        flags += " --no-prose"

    started = time.monotonic()
    pod = _provision()
    ip, port = pod["ip"], pod["port"]
    try:
        deadline = started + WALL_CAP_S
        _bootstrap_pod(pod, base_model=args.base_model, deadline=deadline)

        # Transfer root is the config directory, so relative paths inside it
        # never contain "results" and the bootstrap exclude cannot match them.
        # shlex.quote on every crossing into a remote shell. _run_fold_on_pod
        # quotes the same value at runpod_parallel.py:1262 and this path did
        # not; a config name carrying a space silently split the mkdir into two
        # arguments and the checkpoint probe then counted zero, aborting a run
        # that had already paid for bootstrap.
        quoted_name = shlex.quote(args.config_name)
        remote_config = f"/workspace/tract/results/phase1b/{args.config_name}/"
        quoted_config = shlex.quote(remote_config)
        _ssh(ip, port, f"mkdir -p {quoted_config}", deadline=deadline)
        _rsync_to(ip, port, f"{local_config}/", remote_config, deadline=deadline)

        # The quoted prefix abuts an UNQUOTED glob: shlex.quote wraps the path
        # in single quotes, and 'dir/'fold_* concatenates in the shell while
        # leaving the wildcard for the shell to expand. Quoting the whole word
        # would make it a literal filename and the probe would always count 0.
        probe = _ssh(
            ip, port,
            f"ls -d {quoted_config}fold_*/model/model 2>/dev/null | wc -l",
            deadline=deadline,
        )
        n_remote = (probe.stdout or "").strip().splitlines()[-1].strip()
        if n_remote != str(len(checkpoints)):
            raise RuntimeError(
                f"Shipped {len(checkpoints)} checkpoints but the pod reports "
                f"{n_remote!r}. Scoring a partial set would report a range over "
                "models that did not all arrive."
            )
        logger.info("All %d checkpoints present on the pod", len(checkpoints))

        # `set -o pipefail` is load-bearing, not decoration. _ssh runs this
        # through `bash -s` and checks the exit status; without pipefail the
        # status of `python ... | tail -40` is tail's, which is 0 whatever the
        # python did. A run that raised on the pod -- missing overlay, fixture
        # and corpus disagreeing, CUDA OOM -- returned success, and the failure
        # only surfaced later as a confusing "did not come back".
        _ssh(ip, port, (
            "set -o pipefail && cd /workspace/tract && USE_TF=0 PYTHONPATH=. "
            "python -m scripts.phase1b.run_agentic_smoke "
            f"--config-name {quoted_name}{flags} 2>&1 | tail -40"
        ), timeout=SSH_RUN_TIMEOUT_S, deadline=deadline)

        _rsync_from(
            ip, port,
            f"{remote_config}agentic_smoke_test.json",
            str(local_config / "agentic_smoke_test.json"),
        )
        out = local_config / "agentic_smoke_test.json"
        if not out.is_file():
            raise RuntimeError(
                f"Run reported success but {out} did not come back. Treat the "
                "run as not having happened."
            )
        logger.info("Collected %s", out)
        return 0
    finally:
        # Runs on success, on exception, and on KeyboardInterrupt. Nothing
        # else will clean up this pod: it is named outside the family the
        # reaper sweeps, precisely so the reaper cannot kill it mid-run.
        if args.keep_pod:
            logger.warning("--keep-pod: pod %s LEFT RUNNING and billing at "
                           "$%.2f/hr", pod["pod_id"], pod.get("usd_per_hour", 0))
        else:
            logger.info("Terminating pod %s", pod["pod_id"])
            terminate_pod(pod["pod_id"])
            logger.info("Pod terminated after %.1f min",
                        (time.monotonic() - started) / 60)


if __name__ == "__main__":
    sys.exit(main())
