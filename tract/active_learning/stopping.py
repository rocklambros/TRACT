"""Active learning stopping criteria evaluation."""
from __future__ import annotations

import logging

from tract.config import (
    PHASE1C_AL_ACCEPTANCE_GATE,
    PHASE1C_AL_CANARY_ACCURACY_GATE,
    PHASE1C_AL_HUB_DIVERSITY_GATE,
)

logger = logging.getLogger(__name__)


def evaluate_stopping_criteria(
    acceptance_rate: float,
    canary_accuracy: float,
    unique_hubs_accepted: int,
    acceptance_gate: float = PHASE1C_AL_ACCEPTANCE_GATE,
    canary_gate: float = PHASE1C_AL_CANARY_ACCURACY_GATE,
    diversity_gate: int = PHASE1C_AL_HUB_DIVERSITY_GATE,
) -> dict:
    """Evaluate whether AL loop should stop.

    All three criteria must be met:
    - Acceptance rate > gate (expert trusts predictions)
    - AI canary accuracy >= gate (expert applies judgment)
    - Hub diversity >= gate (model not concentrating on easy hubs)
    """
    criteria_met = {
        "acceptance_rate": acceptance_rate > acceptance_gate,
        "canary_accuracy": canary_accuracy >= canary_gate,
        "hub_diversity": unique_hubs_accepted >= diversity_gate,
    }
    should_stop = all(criteria_met.values())

    logger.info(
        "Stopping criteria: acceptance=%.1f%% (%s), canary=%.1f%% (%s), "
        "diversity=%d (%s) → %s",
        acceptance_rate * 100, "PASS" if criteria_met["acceptance_rate"] else "FAIL",
        canary_accuracy * 100, "PASS" if criteria_met["canary_accuracy"] else "FAIL",
        unique_hubs_accepted, "PASS" if criteria_met["hub_diversity"] else "FAIL",
        "STOP" if should_stop else "CONTINUE",
    )

    return {
        "should_stop": should_stop,
        "criteria_met": criteria_met,
        "values": {
            "acceptance_rate": acceptance_rate,
            "canary_accuracy": canary_accuracy,
            "unique_hubs_accepted": unique_hubs_accepted,
        },
        "gates": {
            "acceptance_rate": acceptance_gate,
            "canary_accuracy": canary_gate,
            "hub_diversity": diversity_gate,
        },
    }
