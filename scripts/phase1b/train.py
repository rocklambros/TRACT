"""CLI entrypoint for Phase 1B training.

Usage:
    python -m scripts.phase1b.train
    python -m scripts.phase1b.train --name my_experiment --lora-rank 32
    python -m scripts.phase1b.train --training-data ai-only --epochs 30
"""
from __future__ import annotations

import argparse
import logging

from tract.training.config import TrainingConfig
from tract.training.orchestrate import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1B model training")
    parser.add_argument("--name", type=str, default="phase1b_primary",
                        help="Experiment name (used for output dir and WandB)")
    parser.add_argument("--training-data", type=str, default="joint-tempscaled",
                        choices=["joint-tempscaled", "ai-only", "joint-flat", "two-stage-transfer"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hard-negatives", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hub-rep", type=str, default="path+name",
                        choices=["path+name", "path+name+desc", "path+name+standards"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        name=args.name,
        training_data=args.training_data,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        hard_negatives=args.hard_negatives,
        sampling_temperature=args.temperature,
        hub_rep_format=args.hub_rep,
        seed=args.seed,
    )

    logger.info("Config: %s", config.to_dict())
    result = run_experiment(config)

    agg = result["aggregate_hit1"]
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE: %s", config.name)
    logger.info("  Aggregate hit@1 (micro): %.4f [%.4f, %.4f] over n=%d",
                agg["mean"], agg["ci_low"], agg["ci_high"], agg["n_total"])
    logger.info("  Aggregate hit@1 (macro): %.4f", agg["macro_mean"])

    # gate_decision logs the full arithmetic, including both verdicts and any
    # disagreement between them. The baseline is the paired one measured on the
    # same items in the same run, not a constant carried over from a prior one:
    # the delta used to be computed against a hardcoded 0.399, which cannot
    # track a change in the eval set, the stack or the seed.
    if "gate" not in result:
        raise RuntimeError(
            "No gate decision in the experiment result. The run was made "
            "without the paired zero-shot baseline, so Gate 1 cannot be "
            "evaluated from it."
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
