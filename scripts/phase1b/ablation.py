"""Ablation sweep runner for Phase 1B.

Usage:
    python -m scripts.phase1b.ablation --ablation A1
    python -m scripts.phase1b.ablation --all
    python -m scripts.phase1b.ablation --list
"""
from __future__ import annotations

import argparse
import logging

from tract.training.config import TrainingConfig
from tract.training.orchestrate import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_ablation_configs(ablation_id: str) -> list[TrainingConfig]:
    """Generate configs for a specific ablation."""
    configs: dict[str, list[TrainingConfig]] = {
        "A1": [
            TrainingConfig(name="ablation_A1_ai_only", training_data="ai-only"),
            TrainingConfig(name="ablation_A1_joint_flat", training_data="joint-flat", sampling_temperature=float("inf")),
            TrainingConfig(name="ablation_A1_two_stage", training_data="two-stage-transfer"),
        ],
        "A2": [
            TrainingConfig(name="ablation_A2_neg0", hard_negatives=0),
            TrainingConfig(name="ablation_A2_neg1", hard_negatives=1),
            TrainingConfig(name="ablation_A2_neg5", hard_negatives=5),
        ],
        "A3": [
            TrainingConfig(name="ablation_A3_standards", hub_rep_format="path+name+standards"),
        ],
        "A4": [
            TrainingConfig(name="ablation_A4_ep5", max_epochs=5),
            TrainingConfig(name="ablation_A4_ep10", max_epochs=10),
            TrainingConfig(name="ablation_A4_ep30", max_epochs=30),
        ],
        "A5": [
            TrainingConfig(name="ablation_A5_lr1e4", learning_rate=1e-4),
            TrainingConfig(name="ablation_A5_lr1e3", learning_rate=1e-3),
        ],
        "A6": [
            TrainingConfig(name="ablation_A6_descriptions", hub_rep_format="path+name+desc"),
        ],
        "A7": [
            TrainingConfig(name="ablation_A7_rank4", lora_rank=4, lora_alpha=8),
            TrainingConfig(name="ablation_A7_rank8", lora_rank=8, lora_alpha=16),
            TrainingConfig(name="ablation_A7_rank32", lora_rank=32, lora_alpha=64),
        ],
        "A8": [
            TrainingConfig(name="ablation_A8_supcon"),
        ],
        "A9": [
            TrainingConfig(name="ablation_A9_full_ft", lora_rank=0),
        ],
        "A10": [
            TrainingConfig(name="ablation_A10_full_text", control_text_source="full_parsed"),
        ],
    }

    if ablation_id not in configs:
        raise ValueError(f"Unknown ablation: {ablation_id}. Valid: {sorted(configs.keys())}")

    return configs[ablation_id]


ABLATION_ORDER: list[str] = ["A1", "A6", "A10", "A2", "A5", "A7", "A4", "A8", "A9", "A3"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1B ablation sweep")
    parser.add_argument("--ablation", type=str, help="Run a specific ablation (A1-A10)")
    parser.add_argument("--all", action="store_true", help="Run all ablations in order")
    parser.add_argument("--list", action="store_true", help="List available ablations")
    args = parser.parse_args()

    if args.list:
        for aid in ABLATION_ORDER:
            cfgs = get_ablation_configs(aid)
            names = [c.name for c in cfgs]
            print(f"  {aid}: {', '.join(names)}")
        return

    ablations_to_run = ABLATION_ORDER if args.all else [args.ablation]

    for aid in ablations_to_run:
        cfgs = get_ablation_configs(aid)
        for config in cfgs:
            logger.info("Running ablation %s: %s", aid, config.name)
            run_experiment(config)


if __name__ == "__main__":
    main()
