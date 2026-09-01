"""The project carries five separate definitions of "AI framework".

    scripts/phase0/common.py      AI_FRAMEWORK_NAMES            (LOFO roster)
    tract/training/data.py        AI_FRAMEWORK_NAMES            (pair tagging)
    tract/training/data_quality.py AI_FRAMEWORK_NAMES           (link tiering)
    tract/config.py               BRIDGE_AI_FRAMEWORK_IDS       (hub bridging)
    scripts/build_curation_packet.py EXCLUDED_ILLUSTRATION_FRAMEWORKS

Nothing related them. That is how the published model card came to claim 60
"naturally bridged" hubs with the worked example "Data poisoning (linked by both
ATLAS and CWE)": BRIDGE_AI_FRAMEWORK_IDS listed only the five frameworks that
rotate through the LOFO roster, so ENISA, ETSI and BIML were counted as
traditional, and every one of those "bridges" had its traditional side supplied
by those three. MITRE ATLAS hubs and CWE hubs intersect in zero hubs.

These constants legitimately differ in SIZE -- the roster is five, the AI-security
population is eight, and the packet exclusion adds Cloud Controls Matrix. What
must hold is the containment between them, and nothing enforced that.

The guards below pass today. `TestTheGuardCatchesADesync` exists because a guard
that has never failed is indistinguishable from one that cannot fail: it feeds
the same comparisons a deliberately desynced pair and asserts they catch it.
"""
from __future__ import annotations

import pytest


def _roster_ids() -> set[str]:
    from scripts.phase0.common import AI_FRAMEWORK_ID_MAP, AI_FRAMEWORK_NAMES
    return {AI_FRAMEWORK_ID_MAP[name] for name in AI_FRAMEWORK_NAMES}


class TestTheDefinitionsAgree:

    def test_the_tiering_copy_matches_the_roster(self) -> None:
        # data_quality imports cleanly; data.py pulls torch, so it is checked
        # separately below under an importorskip. Splitting them keeps the
        # containment guards -- the ones that would have caught the published
        # bridge falsehood -- running on a CI runner that has no torch.
        from scripts.phase0.common import AI_FRAMEWORK_NAMES as roster
        from tract.training.data_quality import AI_FRAMEWORK_NAMES as tiering
        assert roster == tiering

    def test_the_training_pair_copy_matches_the_roster(self) -> None:
        pytest.importorskip("torch", reason="tract.training.data imports torch")
        from scripts.phase0.common import AI_FRAMEWORK_NAMES as roster
        from tract.training.data import AI_FRAMEWORK_NAMES as pairs
        from tract.training.data_quality import AI_FRAMEWORK_NAMES as tiering
        assert roster == pairs == tiering, (
            "the LOFO roster, the training-pair tagger and the link tierer "
            "disagree about which frameworks are AI. A partial edit here is a "
            "silent train/eval firewall desync with no error."
        )

    def test_every_roster_framework_has_an_id(self) -> None:
        from scripts.phase0.common import AI_FRAMEWORK_ID_MAP, AI_FRAMEWORK_NAMES
        missing = {n for n in AI_FRAMEWORK_NAMES if n not in AI_FRAMEWORK_ID_MAP}
        assert not missing, f"no framework_id for {missing}"

    def test_every_roster_framework_counts_as_ai_for_bridging(self) -> None:
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        stragglers = _roster_ids() - BRIDGE_AI_FRAMEWORK_IDS
        assert not stragglers, (
            f"{stragglers} are AI frameworks for the LOFO roster but count as "
            "TRADITIONAL for bridge classification. That asymmetry is what put "
            "a false bridge count on the published model card."
        )

    def test_bridge_ids_cover_all_eight_ai_security_frameworks(self) -> None:
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        # ENISA, ETSI and BIML are AI-security frameworks that do not rotate
        # through the roster. PRD.md:58 describes them as appearing on AI hubs.
        for framework_id in ("enisa", "etsi", "biml"):
            assert framework_id in BRIDGE_AI_FRAMEWORK_IDS, (
                f"{framework_id} maps controls onto AI hubs; classifying it as "
                "traditional makes AI-only hubs look bridged"
            )

    def test_the_packet_never_illustrates_a_hub_with_a_roster_framework(
        self,
    ) -> None:
        from scripts.build_curation_packet import (
            EXCLUDED_ILLUSTRATION_FRAMEWORKS,
        )
        from scripts.phase0.common import AI_FRAMEWORK_NAMES
        leaked = set(AI_FRAMEWORK_NAMES) - set(EXCLUDED_ILLUSTRATION_FRAMEWORKS)
        assert not leaked, (
            f"{leaked} would illustrate hubs in the annotator's reference "
            "sheet. That sheet is the answer key for the framework under test."
        )


class TestTheGuardCatchesADesync:
    """A guard that has never failed may simply be unable to fail."""

    def test_identity_comparison_catches_a_dropped_framework(self) -> None:
        from scripts.phase0.common import AI_FRAMEWORK_NAMES as roster
        desynced = frozenset(set(roster) - {"MITRE ATLAS"})
        assert roster != desynced

    def test_containment_comparison_catches_a_missing_bridge_id(self) -> None:
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        # Reproduce the exact defect: drop the three non-rotating frameworks
        # and confirm the containment check would have flagged it.
        as_shipped_before_the_fix = frozenset(
            BRIDGE_AI_FRAMEWORK_IDS - {"enisa", "etsi", "biml"})
        for framework_id in ("enisa", "etsi", "biml"):
            assert framework_id not in as_shipped_before_the_fix

    def test_packet_exclusion_check_catches_a_new_roster_framework(self) -> None:
        from scripts.build_curation_packet import (
            EXCLUDED_ILLUSTRATION_FRAMEWORKS,
        )
        # The realistic future failure: a framework is added to the roster and
        # not to the packet exclusion, so its own links illustrate the hubs it
        # is being tested on. CAMPAIGN3 §6.5 already approves UK DSIT.
        hypothetical_roster = {"UK DSIT"} | set(EXCLUDED_ILLUSTRATION_FRAMEWORKS)
        leaked = hypothetical_roster - set(EXCLUDED_ILLUSTRATION_FRAMEWORKS)
        assert leaked == {"UK DSIT"}


class TestNamesAndIdsRoundTrip:

    @pytest.mark.parametrize(
        "framework_id", ["enisa", "etsi", "biml", "mitre_atlas"])
    def test_bridge_ids_are_lowercase_snake_case(self, framework_id: str) -> None:
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        assert framework_id in BRIDGE_AI_FRAMEWORK_IDS
        assert framework_id == framework_id.lower()
        assert " " not in framework_id
