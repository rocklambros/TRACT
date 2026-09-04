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
        assert refused == ["aiuc_1", "csa_aicm", "nist_ai_rmf"], (
            f"Default curation targets that cannot be redistributed: {refused}. "
            "This pins the measured state, not an aspiration. Three of four "
            "defaults are refused: csa_aicm on a recorded proprietary "
            "no-redistribution licence, aiuc_1 and nist_ai_rmf on UNDETERMINED. "
            "Only cosai (CC-BY-4.0) is clear. Adjudicating an UNDETERMINED "
            "entry is an OWNER decision recorded in tract/config.py, not a "
            "test edit."
        )


class TestTheRoundCannotRunUntilLicencesAreAdjudicated:
    """The guard blocks Phase 2C's own default. That is the finding, not a bug.

    `build_bridge_packet`'s default framework is nist_800_53, whose recorded
    licence is UNDETERMINED, so the documented Phase 2C command now refuses.

    That is the conservative-correct behaviour and it is deliberately loud:
    NIST SP 800-53 is very likely a US Government work not subject to
    copyright -- the licence table already records `nist_800_63` and
    `nist_ssdf` exactly that way -- but the table leaves `nist_800_53`,
    `nist_ai_100_2`, `nist_ai_600_1` and `nist_ai_rmf` UNDETERMINED. The table
    is internally inconsistent for one publisher, and resolving that is an
    owner decision about four licences, not a guess this guard should make.
    """

    def test_the_bridge_packet_default_is_currently_refused(self) -> None:
        from scripts.build_bridge_packet import build_bridge_packet
        import inspect

        default = inspect.signature(build_bridge_packet).parameters
        assert "framework_id" in default
        assert not externally_redistributable("nist_800_53"), (
            "nist_800_53 is now redistributable. If an owner adjudicated it, "
            "good -- update this test's docstring. If the guard weakened, that "
            "is a regression."
        )

    def test_the_nist_licence_table_is_still_internally_inconsistent(self) -> None:
        """Pin the inconsistency so adjudication closes it deliberately."""
        from tract.config import FRAMEWORK_LICENSES

        adjudicated = {
            f
            for f in ("nist_800_63", "nist_ssdf")
            if FRAMEWORK_LICENSES.get(f, "") != UNDETERMINED_LICENSE
        }
        undetermined = {
            f
            for f in ("nist_800_53", "nist_ai_100_2", "nist_ai_600_1", "nist_ai_rmf")
            if FRAMEWORK_LICENSES.get(f) == UNDETERMINED_LICENSE
        }
        assert adjudicated and undetermined, (
            "The NIST licence entries are now consistent. If all were "
            "adjudicated, delete this test; it exists to keep a known "
            "inconsistency visible until someone decides."
        )
