"""A committed packet needs provenance, and a licence check that runs on commit.

Two gaps closed by one file.

**Lineage.** A returning annotator sheet could not be tied to the packet it came
from. If the hub roster shifts between builds -- which is the entire point of
this round -- nothing recorded which 78 hubs a given annotator actually saw, and
`created_at` was a free-text string an operator typed.

**A committed packet is redistribution that git makes permanent.** The handbook
tells the coordinator to build outside the working tree precisely so a packet
cannot be swept in by `git add -A`. This one is committed deliberately, because
the target machine needs it -- NIST 800-53 is a US Government work not subject to
copyright, verified before the copy. But nothing stopped the NEXT packet, of a
framework whose prose may not be redistributed, from following it in. The
fingerprint gate does not cover it either: it fingerprints dsomm, etsi and
iso_27001 only, so a CSA packet would sail through.

So every packet carries a manifest naming its framework, and a test refuses any
tracked packet whose framework is not redistributable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from tract.config import PROJECT_ROOT
from tract.licensing import externally_redistributable, redistribution_status

PACKETS_DIR: Final[Path] = PROJECT_ROOT / "packets"
MANIFEST_NAME: Final[str] = "manifest.json"


def _committed_packets() -> list[Path]:
    if not PACKETS_DIR.is_dir():
        return []
    return sorted(p for p in PACKETS_DIR.iterdir() if p.is_dir())


PACKETS: Final[list[Path]] = _committed_packets()


class TestTheBuilderEmitsAManifest:
    def test_a_freshly_built_packet_has_one(self, tmp_path: Path) -> None:
        from scripts.build_bridge_packet import build_bridge_packet

        build_bridge_packet(tmp_path, framework_id="nist_800_53")
        assert (tmp_path / MANIFEST_NAME).is_file()

    def test_it_records_what_a_returning_sheet_must_be_tied_to(
        self, tmp_path: Path
    ) -> None:
        from scripts.build_bridge_packet import build_bridge_packet

        build_bridge_packet(tmp_path, framework_id="nist_800_53")
        manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        for key in ("framework_id", "built_at", "git_sha", "n_hubs",
                    "n_controls", "files"):
            assert key in manifest, f"the manifest omits {key}"
        assert manifest["framework_id"] == "nist_800_53"
        assert manifest["n_hubs"] == 78

    def test_it_pins_every_file_by_digest(self, tmp_path: Path) -> None:
        """So a sheet can be tied to the exact bytes an annotator was sent."""
        import hashlib

        from scripts.build_bridge_packet import build_bridge_packet

        build_bridge_packet(tmp_path, framework_id="nist_800_53")
        manifest = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert set(manifest["files"]) == {
            "ai_hubs.csv", "controls.csv", "annotate.csv"
        }
        for name, digest in manifest["files"].items():
            actual = hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            assert actual == digest, f"{name} does not match its recorded digest"

    def test_the_manifest_itself_carries_no_prose(self, tmp_path: Path) -> None:
        """It is metadata; control text belongs in the sheets."""
        from scripts.build_bridge_packet import build_bridge_packet

        build_bridge_packet(tmp_path, framework_id="nist_800_53")
        raw = (tmp_path / MANIFEST_NAME).read_text(encoding="utf-8")
        assert len(raw) < 2_000


class TestEveryCommittedPacketIsRedistributable:
    """The guard that makes committing a packet safe to repeat."""

    def test_at_least_one_packet_is_committed(self) -> None:
        """Guards the guard: an empty directory passes everything below."""
        assert PACKETS, (
            "no packet is committed under packets/. If that is deliberate, "
            "delete this file rather than letting it pass vacuously."
        )

    @pytest.mark.parametrize("packet", PACKETS, ids=lambda p: p.name)
    def test_it_has_a_manifest(self, packet: Path) -> None:
        assert (packet / MANIFEST_NAME).is_file(), (
            f"{packet.name} is committed with no manifest, so nothing records "
            "which framework's prose it contains or whether that may be "
            "redistributed."
        )

    @pytest.mark.parametrize("packet", PACKETS, ids=lambda p: p.name)
    def test_its_framework_may_be_redistributed(self, packet: Path) -> None:
        """The load-bearing assertion.

        A packet in git is published to everyone who clones the repository.
        The fingerprint gate does not cover this: it fingerprints dsomm, etsi
        and iso_27001 only, so a CSA packet would pass it.
        """
        manifest = json.loads((packet / MANIFEST_NAME).read_text(encoding="utf-8"))
        framework_id = manifest["framework_id"]
        assert externally_redistributable(framework_id), (
            f"packets/{packet.name} contains {framework_id} prose, whose "
            f"redistribution status is {redistribution_status(framework_id)!r}. "
            "Committing it publishes that text to everyone who clones this "
            "repository. Remove it, or record a licence that permits it."
        )

    @pytest.mark.parametrize("packet", PACKETS, ids=lambda p: p.name)
    def test_its_files_match_their_recorded_digests(self, packet: Path) -> None:
        """A packet edited after the fact is no longer the packet it claims."""
        import hashlib

        manifest = json.loads((packet / MANIFEST_NAME).read_text(encoding="utf-8"))
        for name, digest in manifest["files"].items():
            path = packet / name
            assert path.is_file(), f"{packet.name}/{name} is missing"
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            assert actual == digest, (
                f"{packet.name}/{name} does not match the digest in its "
                "manifest, so it was changed after it was built."
            )

    @pytest.mark.parametrize("packet", PACKETS, ids=lambda p: p.name)
    def test_it_carries_no_answers(self, packet: Path) -> None:
        """A committed packet with answers in it would be a published answer key."""
        import csv

        sheet = packet / "annotate.csv"
        if not sheet.is_file():
            pytest.skip(f"{packet.name} has no annotate.csv")
        with sheet.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                assert not row["cre_id"], (
                    f"{packet.name}/annotate.csv carries a filled answer "
                    f"({row['control_id']} -> {row['cre_id']}). A committed "
                    "packet must be blank."
                )
