"""Run the domain-shortcut probe on one rented pod, then destroy it.

The probe loads an encoder, and the standing rule is that anything allocating a
model runs on a pod. It is encode-only -- 147 anchors and 522 hub texts, no
gradients -- so this wants one pod for minutes, not a fold-per-pod fleet.

Simpler than `smoke_on_pod` in the one way that matters: the probe measures the
ZERO-SHOT encoder, so no fold checkpoints are shipped. That removes the
checkpoint-transfer step and the remote count probe guarding it, which is where
most of the smoke runner's complexity lives.

The pod is named outside the `tract-p1b-*` family the reaper sweeps, so the
reaper can neither kill it mid-run nor clean up after a crash. Teardown is in a
`finally` that also runs on KeyboardInterrupt, and a wall-clock cap terminates
the pod even if the work hangs.
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
    _ssh,
)
from tract.config import PHASE1B_RESULTS_DIR

logger = logging.getLogger(__name__)

POD_NAME: Final[str] = "tract-probe-domain-shortcut"
# Encode-only on a 0.6B model over ~670 short texts. The fleet's 48GB floor is a
# training constraint and would exclude cheap parts for no reason here.
MIN_VRAM_GB: Final[int] = 24
MAX_USD_PER_HOUR: Final[float] = 3.0
# Bootstrap installs torch and pulls the base model, which dominates; the encode
# itself is a couple of minutes. Past this the pod dies whatever it is doing.
WALL_CAP_S: Final[int] = 3600
SSH_RUN_TIMEOUT_S: Final[int] = 1800
BASE_MODEL: Final[str] = "Qwen/Qwen3-Embedding-0.6B"


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
            # runpod/pytorch:2.4.0-py3.11 and requirements-train.txt is pinned
            # against the Python 3.12 inside the digest-pinned image; taking the
            # default installs a transformers/sentence-transformers pair that
            # cannot import PreTrainedModel.
            pod = create_pod(gpu_type_id=gpu_id, name=POD_NAME, image=DOCKER_IMAGE)
            pod["role"] = "domain-shortcut-probe"
            pod["usd_per_hour"] = price
            logger.info("Pod %s up: %s:%s on %s (%s)", pod["pod_id"], pod["ip"],
                        pod["port"], pod.get("gpu_type"), pod.get("cloud_type"))
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
    parser.add_argument("--output-name", default="domain_shortcut_probe")
    parser.add_argument("--keep-pod", action="store_true",
                        help="Leave the pod up afterwards. It bills until "
                             "terminated by hand.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )

    started = time.monotonic()
    pod = _provision()
    ip, port = pod["ip"], pod["port"]
    try:
        deadline = started + WALL_CAP_S
        _bootstrap_pod(pod, base_model=BASE_MODEL, deadline=deadline)

        quoted_name = shlex.quote(args.output_name)
        # `set -o pipefail` is load-bearing. _ssh runs this through `bash -s`
        # and checks the exit status; without pipefail the status of
        # `python ... | tail -40` is tail's, which is 0 whatever the python did,
        # and a run that raised on the pod returns success.
        _ssh(ip, port, (
            "set -o pipefail && cd /workspace/tract && USE_TF=0 PYTHONPATH=. "
            "python -m scripts.phase1b.domain_shortcut_probe "
            f"--output-name {quoted_name} 2>&1 | tail -40"
        ), timeout=SSH_RUN_TIMEOUT_S, deadline=deadline)

        local_dir = PHASE1B_RESULTS_DIR / args.output_name
        local_dir.mkdir(parents=True, exist_ok=True)
        _rsync_from(
            ip, port,
            f"/workspace/tract/results/phase1b/{args.output_name}/probe_result.json",
            str(local_dir / "probe_result.json"),
        )
        out = local_dir / "probe_result.json"
        if not out.is_file():
            raise RuntimeError(
                f"Run reported success but {out} did not come back. Treat the "
                "run as not having happened."
            )
        logger.info("Collected %s", out)
        return 0
    finally:
        # Runs on success, on exception, and on KeyboardInterrupt. Nothing else
        # cleans up this pod: it is named outside the family the reaper sweeps,
        # precisely so the reaper cannot kill it mid-run.
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
