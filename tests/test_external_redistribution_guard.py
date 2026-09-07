"""Both annotator packets leave the building. Only one had a licence check, and
it consulted the wrong tier.

An annotator packet is redistribution: control prose goes to a person outside
the project. This repository already encodes which frameworks may not be
redistributed -- `REDISTRIBUTION_RESERVED_FRAMEWORK_IDS` -- and neither packet
builder read it.

    REDISTRIBUTION_RESERVED : csa_aicm, csa_ccm, etsi, iso_27001
    OVERLAY                 : dsomm, etsi, iso_27001

`scripts/build_curation_packet.py` had no licence refusal at all, and its
comment reasons that "none of the four curation targets is a RESTRICTED
framework". RESTRICTED/OVERLAY is the **git-tracking** tier: it answers "may
this text exist in the repository", not "may this text be sent to a third
party". `csa_aicm` is a default target of that script, its recorded licence is
"Proprietary ... no redistribution", and `claudedocs/curation-package.md`
instructs the owner to run the command before mailing sheets out.

`scripts/build_bridge_packet.py` did refuse, on OVERLAY, reasoning explicitly
that external redistribution needs a broader set than RESTRICTED. Right
instinct, wrong constant: OVERLAY is broader in one direction (it adds the
GPL'd dsomm) and narrower in the one that matters (it drops both CSA
frameworks).

UNDETERMINED is treated as refusing, not permitting. `tract/config.py` says so
itself: "Guessing a permissive licence for a source that never granted one is
the mistake this table exists to make visible."
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tract.config import (
    OVERLAY_FRAMEWORK_IDS,
    REDISTRIBUTION_RESERVED_FRAMEWORK_IDS,
    UNDETERMINED_LICENSE,
)
from tract.licensing import (
    externally_redistributable,
    refuse_external_redistribution,
)


class TestTheTierGapIsReal:
    """Pin the gap, so the fix is not defended by an assumption."""

    def test_the_csa_frameworks_are_reserved_but_not_overlay(self) -> None:
        gap = REDISTRIBUTION_RESERVED_FRAMEWORK_IDS - OVERLAY_FRAMEWORK_IDS
        assert gap == {"csa_aicm", "csa_ccm"}, (
            f"The reserved-but-not-overlay gap is now {sorted(gap)}. An "
            "OVERLAY-based guard permits exactly this set to be redistributed."
        )


class TestTheGuardRefusesWhatItMust:
    @pytest.mark.parametrize(
        "framework_id", sorted(REDISTRIBUTION_RESERVED_FRAMEWORK_IDS)
    )
    def test_every_reserved_framework_is_refused(self, framework_id: str) -> None:
        assert not externally_redistributable(framework_id)
        with pytest.raises(ValueError, match="redistribut"):
            refuse_external_redistribution(framework_id)

    @pytest.mark.parametrize("framework_id", sorted(OVERLAY_FRAMEWORK_IDS))
    def test_every_overlay_framework_is_refused(self, framework_id: str) -> None:
        """The old guard's set stays refused; this widens, never narrows."""
        assert not externally_redistributable(framework_id)

    def test_an_undetermined_licence_is_refused(self) -> None:
        """A source that never granted terms has not granted these terms."""
        from tract.config import FRAMEWORK_LICENSES

        undetermined = sorted(
            fid
            for fid, lic in FRAMEWORK_LICENSES.items()
            if lic == UNDETERMINED_LICENSE
        )
        assert undetermined, "no UNDETERMINED entries; this test is now vacuous"
        for framework_id in undetermined:
            assert not externally_redistributable(framework_id), framework_id

    def test_an_unknown_framework_is_refused(self) -> None:
        """Unknown is not permission."""
        assert not externally_redistributable("not_a_framework")


class TestTheGuardPermitsSomething:
    """Guards the guard: a function that refuses everything blocks the round."""

    def test_at_least_one_framework_is_redistributable(self) -> None:
        from tract.config import FRAMEWORK_LICENSES

        permitted = [
            fid for fid in FRAMEWORK_LICENSES if externally_redistributable(fid)
        ]
        assert permitted, (
            "No framework is externally redistributable, so no annotation "
            "round of any kind can be run. Either the licence table or this "
            "guard is wrong."
        )


class TestBothPacketBuildersUseIt:
    def test_the_bridge_packet_refuses_a_reserved_framework(
        self, tmp_path: Path
    ) -> None:
        from scripts.build_bridge_packet import build_bridge_packet

        with pytest.raises(ValueError, match="redistribut"):
            build_bridge_packet(tmp_path, framework_id="csa_aicm")
        assert not list(tmp_path.glob("*.csv")), (
            "A refused framework still wrote sheets; the guard runs too late."
        )

    def test_the_curation_packet_refuses_a_reserved_framework(
        self, tmp_path: Path
    ) -> None:
        """csa_aicm is a DEFAULT target of this script.

        Its recorded licence is "Proprietary ... no redistribution", and the
        curation handbook tells the owner to run this before mailing sheets.
        """
        from scripts.build_curation_packet import build_curation_packet

        with pytest.raises(ValueError, match="redistribut"):
            build_curation_packet(tmp_path, frameworks=["csa_aicm"])
        assert not list(tmp_path.rglob("*.csv"))

    @pytest.mark.parametrize("framework_id", ["etsi", "iso_27001", "dsomm", "csa_ccm"])
    def test_the_curation_packet_refuses_each_licensed_framework(
        self, tmp_path: Path, framework_id: str
    ) -> None:
        from scripts.build_curation_packet import build_curation_packet

        with pytest.raises(ValueError, match="redistribut"):
            build_curation_packet(tmp_path, frameworks=[framework_id])

    def test_the_curation_packet_default_targets_are_all_permitted_or_refused_loudly(
        self,
    ) -> None:
        """The defaults must not be a mix that half-succeeds.

        A partial packet -- permitted subset written, refused one skipped -- is
        the shape that gets mailed by mistake, so one refusal fails the call.
        """
        from scripts.build_curation_packet import CURATION_TARGETS

        refused = sorted(
            f for f in CURATION_TARGETS if not externally_redistributable(f)
        )
        assert refused == ["aiuc_1", "csa_aicm"], (
            f"Default curation targets that cannot be redistributed: {refused}. "
            "This pins the measured state, not an aspiration. csa_aicm is "
            "refused on a recorded proprietary no-redistribution licence and "
            "has no override; aiuc_1 is a commercial standard nobody has "
            "adjudicated. cosai (CC-BY-4.0) and nist_ai_rmf (adjudicated "
            "2026-09-06) are clear. Adjudicating an UNDETERMINED entry is an "
            "OWNER decision recorded in tract/config.py, not a test edit."
        )


class TestThePhase2CDefaultIsAdjudicated:
    """Owner adjudication, 2026-09-06, replacing a block this suite once pinned.

    `build_bridge_packet`'s default framework is nist_800_53. Before the
    adjudication its recorded licence was UNDETERMINED, so the documented
    Phase 2C command refused and this class asserted that it did.

    The table had been internally inconsistent for one publisher: nist_800_63
    and nist_ssdf were recorded as US Government works while nist_800_53,
    nist_ai_100_2, nist_ai_600_1 and nist_ai_rmf were not. All six are NIST
    publications authored by US federal employees, and the owner recorded the
    same terms for the four.
    """

    def test_the_bridge_packet_default_runs_without_an_override(self) -> None:
        refuse_external_redistribution("nist_800_53")
        assert externally_redistributable("nist_800_53")

    def test_the_nist_licence_table_is_now_internally_consistent(self) -> None:
        """The inconsistency this suite previously pinned is closed."""
        from tract.config import FRAMEWORK_LICENSES

        nist = [
            "nist_800_53", "nist_800_63", "nist_ssdf",
            "nist_ai_100_2", "nist_ai_600_1", "nist_ai_rmf",
        ]
        undetermined = [
            f for f in nist if FRAMEWORK_LICENSES.get(f) == UNDETERMINED_LICENSE
        ]
        assert not undetermined, (
            f"NIST entries left unadjudicated: {undetermined}. Six NIST "
            "publications by the same publisher should not carry different "
            "licence determinations."
        )


class TestTheGuardCannotBeBypassedByFormatting:
    """A recorded prohibition must not be defeated by whitespace or a capital.

    'CSA_AICM', 'csa_aicm ' and ' csa_aicm' all missed the reserved set and
    classified as "undetermined" -- which IS overridable with
    allow_undetermined, so the docstring's "no override" was false for any
    caller who did not type the id exactly.
    """

    @pytest.mark.parametrize(
        "variant", ["CSA_AICM", "csa_aicm ", " csa_aicm", "Csa_Aicm", "CSA_aicm"]
    )
    def test_a_reserved_id_stays_reserved_under_formatting(
        self, variant: str
    ) -> None:
        from tract.licensing import redistribution_status

        assert redistribution_status(variant) == "reserved"
        with pytest.raises(ValueError, match="no override"):
            refuse_external_redistribution(variant, allow_undetermined=True)

    def test_an_unknown_id_raises_rather_than_becoming_overridable(self) -> None:
        """Unknown is not "nobody checked" -- it is a typo or an absent entry."""
        from tract.licensing import redistribution_status

        with pytest.raises(KeyError, match="not_a_framework"):
            redistribution_status("not_a_framework")

    def test_the_filter_helper_still_returns_false_for_unknown(self) -> None:
        """Callers filtering a list must not have to catch KeyError."""
        assert externally_redistributable("not_a_framework") is False


class TestTheAdjudicatedNistEntries:
    """Owner adjudication, 2026-09-06. All six NIST entries now agree."""

    @pytest.mark.parametrize(
        "framework_id",
        ["nist_800_53", "nist_800_63", "nist_ssdf", "nist_ai_100_2",
         "nist_ai_600_1", "nist_ai_rmf"],
    )
    def test_every_nist_entry_is_recorded_as_a_government_work(
        self, framework_id: str
    ) -> None:
        from tract.config import FRAMEWORK_LICENSES

        assert "US Government work" in FRAMEWORK_LICENSES[framework_id]
        assert externally_redistributable(framework_id)

    def test_the_phase_2c_default_no_longer_needs_an_override(self) -> None:
        """build_bridge_packet's default framework runs without a flag."""
        refuse_external_redistribution("nist_800_53")


class TestTheCurationHubSheetIsAlsoGuarded:
    """The hub sheet ships to the same annotator and had no licence filter.

    `build_curation_packet` guarded each CONTROL sheet and then called
    `build_hub_sheet`, which was unguarded. That sheet illustrates each hub with
    example control titles, filtered by a DISPLAY-NAME blocklist that named the
    AI frameworks and CCM -- and never named ISO 27001 or DSOMM. Measured before
    the fix: 45 ISO 27001 rows and 11 DSOMM rows. ISO's recorded licence is "no
    reproduction without prior written permission".

    Two independent concerns were conflated. The blocklist is about
    CONTAMINATION (an AI framework's section name hands over part of the
    answer); redistribution is a different question and now has its own filter,
    keyed on framework id so an alias cannot slip past it.
    """

    @pytest.fixture(scope="class")
    def hub_sheet_examples(self, tmp_path_factory: pytest.TempPathFactory) -> str:
        import csv
        import logging

        from scripts.build_curation_packet import build_hub_sheet

        logging.disable(logging.INFO)
        try:
            path = build_hub_sheet(tmp_path_factory.mktemp("hubsheet"))
        finally:
            logging.disable(logging.NOTSET)
        with path.open(encoding="utf-8") as handle:
            return " ".join(
                row["example_controls_already_mapped_here"]
                for row in csv.DictReader(handle)
            )

    @pytest.mark.parametrize(
        "name", ["ISO 27001", "DSOMM", "DevSecOps Maturity", "ETSI"]
    )
    def test_no_licensed_framework_illustrates_a_hub(
        self, hub_sheet_examples: str, name: str
    ) -> None:
        assert name not in hub_sheet_examples

    def test_every_illustrating_framework_is_redistributable(
        self, hub_sheet_examples: str
    ) -> None:
        """Positive form: whatever remains must be permitted, not merely
        absent from a blocklist."""
        from tract.config import OPENCRE_FRAMEWORK_ID_MAP

        named = {
            name
            for name in OPENCRE_FRAMEWORK_ID_MAP
            if f"{name}:" in hub_sheet_examples
        }
        assert named, "no framework illustrates any hub; the sheet is empty"
        for name in named:
            framework_id = OPENCRE_FRAMEWORK_ID_MAP[name]
            assert externally_redistributable(framework_id), (
                f"{name} ({framework_id}) illustrates a hub but may not be "
                "redistributed."
            )

    def test_the_sheet_is_still_useful(self, hub_sheet_examples: str) -> None:
        """Guards the guard: filtering everything out would also pass above."""
        assert len(hub_sheet_examples) > 10_000
