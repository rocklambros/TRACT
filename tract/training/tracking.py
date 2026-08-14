"""WandB tracking for the LOFO re-derivation.

Phase 0 logged every experiment to WandB through scripts/phase0/common.py.
Phase 1B did not: PHASE1B_WANDB_PROJECT was defined in tract/config.py and
referenced nowhere, so the LOFO path -- the one that spends the GPU budget and
produces the published number -- ran with no experiment tracking at all. The
only record of a fold was the JSON it left on the pod, which is exactly the
artifact that goes missing when collection fails.

This module fails closed. Phase 0's init_wandb returns None and logs a warning
when the key is absent or the import fails, which is right for a local probe
and wrong here: a fleet of pods that each degrade to untracked would burn the
whole budget before anyone noticed the runs were not appearing. When tracking
is requested, an initialization failure raises.

Owner: TRACT. See PRD Section 6.4.
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class WandbRun(Protocol):
    """The subset of the wandb run object this module uses."""

    url: str

    def log(self, data: dict[str, Any]) -> None: ...
    def finish(self, exit_code: int = 0) -> None: ...


def resolve_api_key() -> str:
    """Return the WandB API key from the environment or `pass`.

    Raises rather than returning a sentinel. Every caller here needs a usable
    key; handing back an empty string would push the failure into wandb.init(),
    which prompts for interactive login and hangs a headless pod until its
    deadline.
    """
    key = os.environ.get("WANDB_API_KEY", "").strip()
    source = "WANDB_API_KEY"
    if not key:
        source = "pass wandb/api-key"
        try:
            result = subprocess.run(
                ["pass", "wandb/api-key"],
                capture_output=True, text=True, timeout=10, check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "WANDB_API_KEY is unset and `pass` is not installed, so the "
                "WandB key cannot be resolved."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "WANDB_API_KEY is unset and `pass wandb/api-key` failed. "
                "Store the key with `pass insert wandb/api-key`."
            ) from exc
        key = result.stdout.strip()

    # A WandB key is 40 hex characters. Checking the shape here turns a
    # credential mistake into an error before a pod exists, rather than an
    # authentication failure on every pod after the fleet is already billing.
    # The entry in `pass` held a 5-character username when this was written,
    # which would have authenticated nowhere.
    if len(key) != WANDB_KEY_LENGTH or not all(
        c in "0123456789abcdefABCDEF" for c in key
    ):
        raise RuntimeError(
            f"The value from {source} is {len(key)} characters and is not "
            f"{WANDB_KEY_LENGTH} hex digits, so it is not a WandB API key. "
            "Get one from https://wandb.ai/authorize and store it with "
            "`pass insert -f wandb/api-key`."
        )
    return key


WANDB_KEY_LENGTH = 40


def stable_run_id(*parts: str) -> str:
    """Deterministic WandB run id for a (campaign, arm, fold) triple.

    WandB names are not unique keys: wandb.init creates a fresh run every call
    whatever the name, so re-running the logging step after a partial collect
    would have produced a second copy of every fold already present, and the
    project would show two populations of the same experiment. A run id IS a
    key, so deriving one from the identity of the fold makes re-logging an
    update rather than a duplicate.
    """
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    # WandB ids must be short and filesystem-safe.
    return digest[:16]


def init_run(
    project: str,
    name: str,
    config: dict[str, Any],
    tags: list[str] | None = None,
    entity: str | None = None,
    run_id: str | None = None,
) -> WandbRun:
    """Start a tracked run, or raise.

    Args:
        project: WandB project. One project per experiment campaign, so a
            re-derivation does not land beside the runs it supersedes.
        name: Run name, shown in the UI. Fold and arm belong here.
        config: Hyperparameters and provenance. Logged verbatim.
        tags: Filter handles, typically the arm and the held-out framework.
        entity: WandB team. None uses the key's default entity.
        run_id: Stable id. Supply one from stable_run_id() to make re-logging
            the same fold overwrite rather than duplicate.
    """
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed but tracking was requested. Install it, "
            "or drop --wandb to run untracked."
        ) from exc

    os.environ["WANDB_API_KEY"] = resolve_api_key()
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        config=config,
        tags=tags or [],
        reinit=True,
        id=run_id,
        # "allow" attaches to the existing run when the id is already present
        # and creates it otherwise, which is what makes re-logging idempotent.
        resume="allow" if run_id else None,
    )
    if run is None:
        raise RuntimeError(
            f"wandb.init returned None for project={project!r} name={name!r}. "
            "The run is not being tracked; refusing to continue."
        )
    logger.info("WandB run: %s", run.url)
    tracked: WandbRun = run
    return tracked


def log_fold(run: WandbRun | None, record: dict[str, Any]) -> None:
    """Log one fold's completed record.

    Accepts None so an untracked run needs no branch at the call site. The
    metrics are flattened because WandB charts a scalar, not a nested object;
    the nested blocks are logged alongside so nothing is lost.
    """
    if run is None:
        return

    payload: dict[str, Any] = {
        "fold/held_out_framework": record.get("held_out_framework"),
        "fold/n_eval_items": record.get("n_eval_items"),
        "fold/n_training_pairs": record.get("n_training_pairs"),
        "fold/elapsed_s": record.get("elapsed_s"),
    }
    for metric, value in (record.get("metrics") or {}).items():
        payload[f"metrics/{metric}"] = value

    # The zero-shot arm is the paired baseline. Logging the delta beside both
    # sides is what makes a regression visible in the UI without arithmetic.
    zero_shot_metrics = (record.get("zero_shot") or {}).get("metrics") or {}
    for metric, value in zero_shot_metrics.items():
        payload[f"zero_shot/{metric}"] = value
    trained_hit1 = (record.get("metrics") or {}).get("hit_at_1")
    baseline_hit1 = zero_shot_metrics.get("hit_at_1")
    if trained_hit1 is not None and baseline_hit1 is not None:
        payload["delta/hit_at_1"] = trained_hit1 - baseline_hit1

    for key, value in (record.get("text_selection") or {}).items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                payload[f"text_selection/{key}/{sub_key}"] = sub_value
        else:
            payload[f"text_selection/{key}"] = value

    run.log(payload)


def finish_run(run: WandbRun | None, exit_code: int = 0) -> None:
    """Close the run. A pod that dies without this leaves it marked running."""
    if run is None:
        return
    run.finish(exit_code=exit_code)
