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


# The guard bodies, as functions, so the "would this catch a desync" tests can
# run the SAME code against perturbed constants instead of asserting set
# algebra about them. The previous version of that class asserted
# `x not in (S - {x})`, which is true for every S including the empty set.
def _ai_names() -> frozenset[str]:
    from scripts.phase0.common import AI_FRAMEWORK_NAMES
    return frozenset(AI_FRAMEWORK_NAMES)


def containment_violations(roster: set[str], bridge: frozenset[str]) -> set[str]:
    """Roster frameworks that bridge classification would call traditional."""
    return roster - set(bridge)


def missing_ai_security_ids(bridge: frozenset[str]) -> set[str]:
    """The three non-rotating AI frameworks, if absent from bridge ids."""
    return {f for f in ("enisa", "etsi", "biml") if f not in bridge}


def packet_leaks(roster: set[str], excluded: frozenset[str]) -> set[str]:
    """Roster frameworks that would illustrate hubs in the annotator sheet."""
    return roster - set(excluded)


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
        stragglers = containment_violations(
            _roster_ids(), BRIDGE_AI_FRAMEWORK_IDS)
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
        leaked = packet_leaks(
            set(AI_FRAMEWORK_NAMES), EXCLUDED_ILLUSTRATION_FRAMEWORKS)
        assert not leaked, (
            f"{leaked} would illustrate hubs in the annotator's reference "
            "sheet. That sheet is the answer key for the framework under test."
        )


class TestTheGuardCatchesADesync:
    """A guard that has never failed may simply be unable to fail.

    The first version of this class could not fail either. Two of its three
    tests asserted set algebra -- `x not in (S - {x})` holds for every S,
    including the empty set and total garbage -- and the third was an INVERTED
    TRIPWIRE: it asserted `({"UK DSIT"} | E) - E == {"UK DSIT"}`, which breaks
    the moment somebody correctly adds UK DSIT to the exclusion set. An
    engineer performing the remediation the test's own comment prescribes would
    have seen a red test named "catches a new roster framework" and reverted
    the fix, leaking the answer key for the framework under test.

    These run the actual guard bodies against deliberately desynced constants.
    """

    def test_containment_guard_fires_on_the_real_2026_08_defect(self) -> None:
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        as_shipped = frozenset(BRIDGE_AI_FRAMEWORK_IDS - {"enisa", "etsi", "biml"})
        # The containment check alone did NOT catch it -- the roster's five were
        # a subset of the old five. This records which guard actually fires.
        assert containment_violations(_roster_ids(), as_shipped) == set()
        assert missing_ai_security_ids(as_shipped) == {"enisa", "etsi", "biml"}

    def test_containment_guard_fires_when_a_roster_id_is_dropped(self) -> None:
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        crippled = frozenset(BRIDGE_AI_FRAMEWORK_IDS - {"mitre_atlas"})
        assert containment_violations(_roster_ids(), crippled) == {"mitre_atlas"}

    def test_containment_guard_fires_on_an_empty_bridge_set(self) -> None:
        assert containment_violations(_roster_ids(), frozenset()) == _roster_ids()

    def test_packet_guard_fires_on_an_unexcluded_roster_framework(self) -> None:
        from scripts.build_curation_packet import (
            EXCLUDED_ILLUSTRATION_FRAMEWORKS,
        )
        # A framework joins the LOFO roster and nobody adds it to the packet
        # exclusion. CAMPAIGN3 §6.5 already approves UK DSIT.
        roster = set(_ai_names()) | {"UK DSIT"}
        assert packet_leaks(roster, EXCLUDED_ILLUSTRATION_FRAMEWORKS) == {
            "UK DSIT"}

    def test_packet_guard_goes_quiet_once_the_fix_is_applied(self) -> None:
        # The property the previous version got backwards: applying the
        # remediation must SILENCE the guard, not trip it.
        from scripts.build_curation_packet import (
            EXCLUDED_ILLUSTRATION_FRAMEWORKS,
        )
        roster = set(_ai_names()) | {"UK DSIT"}
        fixed = frozenset(set(EXCLUDED_ILLUSTRATION_FRAMEWORKS) | {"UK DSIT"})
        assert packet_leaks(roster, fixed) == set()


class TestNamesAndIdsRoundTrip:

    @pytest.mark.parametrize(
        "framework_id", ["enisa", "etsi", "biml", "mitre_atlas"])
    def test_bridge_ids_are_lowercase_snake_case(self, framework_id: str) -> None:
        from tract.config import BRIDGE_AI_FRAMEWORK_IDS
        assert framework_id in BRIDGE_AI_FRAMEWORK_IDS
        assert framework_id == framework_id.lower()
        assert " " not in framework_id
