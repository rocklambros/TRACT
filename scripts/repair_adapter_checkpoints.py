"""Make the adapter-only checkpoints under results/ loadable again.

    python -m scripts.repair_adapter_checkpoints --check
    python -m scripts.repair_adapter_checkpoints --dry-run
    python -m scripts.repair_adapter_checkpoints

Owner decision D2, answered (b) on 2026-08-26. Every checkpoint written before
`tract.training.checkpoint.save_sentence_transformer` existed is adapter-only
with no `config.json`, so `SentenceTransformer(dir)` raises "Unrecognized
model" on all of them. The weights are correct; the directory is not
self-describing. See tract/training/checkpoint.py for why transformers skips
the config write when a PEFT adapter is loaded.

WHAT THIS DOES NOT DO. It does not load a model, and it does not re-serialise
weights. It copies the backbone's own `config.json` in beside the adapter,
which is the repair named in `assert_loadable_checkpoint`'s error message. That
keeps it runnable on a machine that never allocates a model, and it leaves the
98 artifacts' weights byte-identical -- which matters, because nobody has
audited their provenance.

The backbone config comes from the local HuggingFace cache when it is there
and from the hub when it is not. Only `config.json` is fetched, never weights.

These checkpoints are ALSO stale against the rebuilt corpus, so a repaired
checkpoint is loadable and still not a thing to draw a conclusion from. The
decision record in claudedocs/jetson-runpod-start.md says so too.

Exit codes: 0 nothing left to repair, 1 repairs are outstanding under --check,
2 a checkpoint could not be repaired.

Owner: TRACT.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

from tract.config import PROJECT_ROOT
from tract.training.checkpoint import (
    ADAPTER_CONFIG_NAME,
    HF_CONFIG_NAME,
    repair_adapter_only_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ROOT: Final[Path] = PROJECT_ROOT / "results"

# Exit codes, so a caller can tell "clean" from "work outstanding" from "broke".
EXIT_CLEAN: Final[int] = 0
EXIT_OUTSTANDING: Final[int] = 1
EXIT_UNREPAIRABLE: Final[int] = 2


def find_adapter_only_checkpoints(root: Path) -> list[Path]:
    """Every adapter checkpoint under `root` that has no base config."""
    return sorted(
        p.parent
        for p in root.rglob(ADAPTER_CONFIG_NAME)
        if not (p.parent / HF_CONFIG_NAME).is_file()
    )


def declared_backbone(checkpoint_dir: Path) -> str:
    """The base model this adapter names.

    Raises:
        ValueError: The adapter config names no backbone, so there is nothing
            to resolve a config against and guessing one would be the exact
            wrong-backbone failure repair_adapter_only_checkpoint refuses.
    """
    adapter = json.loads(
        (checkpoint_dir / ADAPTER_CONFIG_NAME).read_text(encoding="utf-8")
    )
    base = adapter.get("base_model_name_or_path")
    if not base:
        raise ValueError(
            f"{checkpoint_dir}/{ADAPTER_CONFIG_NAME} names no "
            "base_model_name_or_path, so its backbone cannot be identified."
        )
    return str(base)


def resolve_base_config(repo_id: str) -> Path:
    """Local path to `repo_id`'s config.json, from the cache or the hub.

    Fetches one small JSON file. No weights, and no model is instantiated.
    """
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=HF_CONFIG_NAME))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"tree to sweep (default: {DEFAULT_ROOT})")
    parser.add_argument("--check", action="store_true",
                        help="report what is unrepaired and write nothing")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve every backbone config but write nothing")
    args = parser.parse_args(argv)

    if not args.root.is_dir():
        logger.error("%s is not a directory", args.root)
        return EXIT_UNREPAIRABLE

    pending = find_adapter_only_checkpoints(args.root)
    if not pending:
        logger.info("No adapter-only checkpoints under %s. Nothing to repair.",
                    args.root)
        return EXIT_CLEAN

    by_backbone: dict[str, list[Path]] = defaultdict(list)
    unidentified: list[tuple[Path, str]] = []
    for ckpt in pending:
        try:
            by_backbone[declared_backbone(ckpt)].append(ckpt)
        except (ValueError, json.JSONDecodeError) as exc:
            unidentified.append((ckpt, str(exc)))

    logger.info("%d adapter-only checkpoint(s) under %s across %d backbone(s):",
                len(pending), args.root, len(by_backbone))
    for repo_id, ckpts in sorted(by_backbone.items()):
        logger.info("  %4d  %s", len(ckpts), repo_id)
    for ckpt, reason in unidentified:
        logger.error("  UNIDENTIFIED %s: %s", ckpt, reason)

    if args.check:
        # Unidentified checkpoints are a harder failure than merely unrepaired
        # ones, so they outrank the outstanding-work code.
        if unidentified:
            return EXIT_UNREPAIRABLE
        return EXIT_OUTSTANDING

    repaired = 0
    failed: list[tuple[Path, str]] = []
    for repo_id, ckpts in sorted(by_backbone.items()):
        try:
            # Resolved once per backbone, not once per checkpoint: 95 of the 98
            # share one, and a fetch per checkpoint would be 98 round trips.
            config_path = resolve_base_config(repo_id)
        except Exception as exc:  # noqa: BLE001 - network, cache, auth, all fatal here
            logger.error("Could not resolve %s for %s: %s",
                         HF_CONFIG_NAME, repo_id, exc)
            failed.extend((c, f"unresolved backbone {repo_id}") for c in ckpts)
            continue

        logger.info("Backbone %s -> %s", repo_id, config_path)
        for ckpt in ckpts:
            if args.dry_run:
                logger.info("  DRY RUN would repair %s", ckpt)
                continue
            try:
                if repair_adapter_only_checkpoint(ckpt, config_path, repo_id):
                    repaired += 1
            except (ValueError, OSError) as exc:
                logger.error("  FAILED %s: %s", ckpt, exc)
                failed.append((ckpt, str(exc)))

    if args.dry_run:
        logger.info("DRY RUN: %d checkpoint(s) would be repaired, nothing written.",
                    len(pending) - len(unidentified))
        return EXIT_OUTSTANDING

    logger.info("Repaired %d of %d checkpoint(s).", repaired, len(pending))
    if failed or unidentified:
        logger.error("%d checkpoint(s) could not be repaired.",
                     len(failed) + len(unidentified))
        return EXIT_UNREPAIRABLE
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
