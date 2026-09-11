"""Gate 2's constants must resolve against the real corpus and the document.

Gate 1 has had its thresholds in `tract/config.py` with a drift test since
checkpoint 2. Gate 2 had none of them anywhere: the eval set, the threshold and
the arm roster existed only as prose in a file that was untracked when the
premortem opened, so editing the markdown after results existed would have left
no diff and failed no test.

Two distinct hazards are covered here.

The first is drift between the constants and the pre-registration -- the Gate 1
pattern, applied to Gate 2.

The second is specific to the firewall and is worse, because it fails silently.
The exclusion matches on `standard_name` as it appears in
`hub_links_curated.jsonl`, while `BRIDGE_AI_FRAMEWORK_IDS` holds framework *ids*.
The two vocabularies differ ("owasp_llm_top10" vs "OWASP Top10 for LLM"), and a
plausible-looking wrong name does not raise -- it matches nothing, holds out
nothing, trains on the framework it claims to exclude, and scores ~0.9. The
first draft of PHASE2C_GATE2_HELD_OUT contained exactly that mistake.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.phase0.common import load_curated_links
from tract.config import (
    BRIDGE_AI_FRAMEWORK_IDS,
    PHASE2C_GATE2_EVAL_FRAMEWORKS,
    PHASE2C_GATE2_HELD_OUT,
    PHASE2C_GATE2_N_CONFIGURATIONS,
    PHASE2C_GATE2_THRESHOLD,
    PROJECT_ROOT,
)

PREREG = PROJECT_ROOT / "docs" / "phase2c-preregistration.md"
PLAN = PROJECT_ROOT / "docs" / "phase2c-gate2-plan.md"


@pytest.fixture(scope="module")
def standard_names() -> frozenset[str]:
    """Every standard_name present in the curated gold links."""
    return frozenset(link.standard_name for link in load_curated_links())


class TestTheNamesResolve:
    """The silent-failure class. A name that matches nothing excludes nothing."""

    def test_every_held_out_name_exists_in_the_corpus(
        self, standard_names: frozenset[str]
    ) -> None:
        unresolved = PHASE2C_GATE2_HELD_OUT - standard_names
        assert not unresolved, (
            f"{sorted(unresolved)} appear in PHASE2C_GATE2_HELD_OUT but in no "
            "curated link. The exclusion matches on standard_name, so these "
            "hold out nothing and the arm trains on a framework it claims to "
            "firewall."
        )

    def test_every_eval_name_exists_in_the_corpus(
        self, standard_names: frozenset[str]
    ) -> None:
        unresolved = PHASE2C_GATE2_EVAL_FRAMEWORKS - standard_names
        assert not unresolved, f"{sorted(unresolved)} match no curated link"

    def test_the_held_out_set_is_the_whole_ai_region(self) -> None:
        """Eight frameworks, not the five-name LOFO roster.

        The five-name AI_FRAMEWORK_NAMES roster is the eval roster. Holding out
        only those leaves ENISA, ETSI and BIML in training, which is 52 of 56
        scored hubs still supervised.
        """
        assert len(PHASE2C_GATE2_HELD_OUT) == len(BRIDGE_AI_FRAMEWORK_IDS) == 8

    def test_the_eval_set_is_held_out(self) -> None:
        """Scoring a framework that trained is the firewall breach itself."""
        assert PHASE2C_GATE2_EVAL_FRAMEWORKS <= PHASE2C_GATE2_HELD_OUT


class TestTheThresholdIsNotGate1s:
    def test_threshold_is_zero_not_the_phase1b_delta(self) -> None:
        """gate_decision defaults to 0.10 against zero-shot. This is not that."""
        from tract.config import PHASE1B_GATE_HIT1_DELTA

        assert PHASE2C_GATE2_THRESHOLD == 0.0
        assert PHASE2C_GATE2_THRESHOLD != PHASE1B_GATE_HIT1_DELTA

    def test_two_configurations_are_corrected_for(self) -> None:
        """A1-vs-A0 and A1-vs-A0R. A0' is a noise floor, not a hypothesis."""
        assert PHASE2C_GATE2_N_CONFIGURATIONS == 2


class TestNoDriftFromTheDocuments:
    """Each constant's value must be findable in the pre-registration."""

    @pytest.mark.parametrize(
        "needle",
        [
            "ENISA + BIML + ETSI",
            "74 items",
            "62 scored hubs",
            "negative control",
            "NO VERDICT",
        ],
    )
    def test_plan_states_the_design(self, needle: str) -> None:
        assert needle in PLAN.read_text(encoding="utf-8"), (
            f"{needle!r} is not in {PLAN.name}; the plan and the constants have "
            "drifted"
        )

    @pytest.mark.parametrize(
        "needle",
        [
            "Amendment 1",
            "ENISA + BIML + ETSI, 74 items over 62 scored",
            "A0R",
            "noise-floor arm A0",
            "NO VERDICT",
            "withdrawal window",
        ],
    )
    def test_preregistration_records_the_amendment(self, needle: str) -> None:
        assert needle in PREREG.read_text(encoding="utf-8"), (
            f"{needle!r} is not in {PREREG.name}; Amendment 1 does not record "
            "what the constants encode"
        )

    def test_amendment_states_what_was_known_when_written(self) -> None:
        """Campaign 2's authorising clause was written after its results existed.

        An amendment without this section is the same shape.
        """
        text = PREREG.read_text(encoding="utf-8")
        assert "## What was known when this was written" in text
        assert "No Gate 2 arm has been trained" in text
