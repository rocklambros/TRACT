"""The tracked merged corpus must not carry licensed framework prose.

Gitignoring data/processed/frameworks/iso_27001.json guards the narrow
channel. merge_all_controls globs that same directory, so the merged
all_controls.json is the wide one: it is git-tracked, and inlining a
restricted framework there commits normative ISO control statements into a
CC0 repository.

The merge therefore writes two artifacts. The tracked one excludes every
framework in RESTRICTED_FRAMEWORK_IDS. The gitignored overlay under
data/processed/licensed/ carries the full corpus for local training.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from parsers.merge_all_controls import main as merge_main
from tract.config import RESTRICTED_FRAMEWORK_IDS


def _framework(framework_id: str, fetched_date: str, description: str) -> dict[str, object]:
    return {
        "framework_id": framework_id,
        "framework_name": framework_id.upper(),
        "version": "1.0",
        "source_url": "https://example.com",
        "fetched_date": fetched_date,
        "mapping_unit_level": "control",
        "controls": [
            {
                "control_id": "1.1",
                "title": "Some title",
                "description": description,
            }
        ],
    }


# Stands in for a restricted framework's control statement. Invented prose,
# not a quotation of any licensed source, since this test only needs
# something long enough to be treated as a statement rather than a title.
RESTRICTED_PROSE = (
    "Access credentials for regional facilities shall be issued, reviewed, "
    "and revoked under a documented process approved by the security lead."
)


@pytest.fixture
def corpus(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A frameworks dir with one public and one restricted framework."""
    frameworks_dir = tmp_path / "frameworks"
    frameworks_dir.mkdir()
    restricted_id = sorted(RESTRICTED_FRAMEWORK_IDS)[0]
    (frameworks_dir / "public_fw.json").write_text(
        json.dumps(_framework("public_fw", "2026-01-01", "A public control statement.")),
        encoding="utf-8",
    )
    (frameworks_dir / f"{restricted_id}.json").write_text(
        json.dumps(_framework(restricted_id, "2026-08-15", RESTRICTED_PROSE)),
        encoding="utf-8",
    )
    return frameworks_dir, tmp_path / "processed", tmp_path / "licensed"


class TestRestrictedFrameworksStayOutOfTheTrackedMerge:
    def test_tracked_artifact_excludes_restricted_frameworks(
        self, corpus: tuple[Path, Path, Path]
    ) -> None:
        frameworks_dir, processed_dir, licensed_dir = corpus
        merge_main(
            frameworks_dir=frameworks_dir,
            output_dir=processed_dir,
            licensed_dir=licensed_dir,
        )

        tracked = json.loads(
            (processed_dir / "all_controls.json").read_text(encoding="utf-8")
        )
        ids = {f["framework_id"] for f in tracked["frameworks"]}
        assert ids == {"public_fw"}
        assert RESTRICTED_PROSE not in json.dumps(tracked)
        assert tracked["framework_count"] == 1
        assert tracked["total_controls"] == 1

    def test_overlay_carries_the_full_corpus(
        self, corpus: tuple[Path, Path, Path]
    ) -> None:
        frameworks_dir, processed_dir, licensed_dir = corpus
        merge_main(
            frameworks_dir=frameworks_dir,
            output_dir=processed_dir,
            licensed_dir=licensed_dir,
        )

        overlay = json.loads(
            (licensed_dir / "all_controls.json").read_text(encoding="utf-8")
        )
        ids = {f["framework_id"] for f in overlay["frameworks"]}
        assert ids == {"public_fw", sorted(RESTRICTED_FRAMEWORK_IDS)[0]}
        assert RESTRICTED_PROSE in json.dumps(overlay)

    def test_generated_date_comes_only_from_included_frameworks(
        self, corpus: tuple[Path, Path, Path]
    ) -> None:
        """The tracked date must not flip when an untracked file appears.

        The restricted framework carries the later fetched_date. Taking the
        max over every file on disk would make the tracked artifact's bytes
        depend on whether a gitignored file happens to exist locally.
        """
        frameworks_dir, processed_dir, licensed_dir = corpus
        merge_main(
            frameworks_dir=frameworks_dir,
            output_dir=processed_dir,
            licensed_dir=licensed_dir,
        )

        tracked = json.loads(
            (processed_dir / "all_controls.json").read_text(encoding="utf-8")
        )
        overlay = json.loads(
            (licensed_dir / "all_controls.json").read_text(encoding="utf-8")
        )
        assert tracked["generated_date"] == "2026-01-01"
        assert overlay["generated_date"] == "2026-08-15"

    def test_a_stale_overlay_is_removed_when_no_restricted_source_remains(
        self, corpus: tuple[Path, Path, Path]
    ) -> None:
        """A left-behind overlay would shadow the tracked corpus forever.

        Readers prefer the overlay when it exists. If the restricted source is
        deleted and the overlay survives, every reader silently keeps using a
        corpus nothing can regenerate.
        """
        frameworks_dir, processed_dir, licensed_dir = corpus
        merge_main(
            frameworks_dir=frameworks_dir,
            output_dir=processed_dir,
            licensed_dir=licensed_dir,
        )
        assert (licensed_dir / "all_controls.json").exists()

        (frameworks_dir / f"{sorted(RESTRICTED_FRAMEWORK_IDS)[0]}.json").unlink()
        merge_main(
            frameworks_dir=frameworks_dir,
            output_dir=processed_dir,
            licensed_dir=licensed_dir,
        )
        assert not (licensed_dir / "all_controls.json").exists()


def test_every_restricted_framework_is_excluded_not_just_the_first(
    tmp_path: Path,
) -> None:
    """The split must be driven by the set, not by one hardcoded framework.

    RESTRICTED_FRAMEWORK_IDS grew from one member to two when ETSI's notice
    was read. The fixtures above stage a single restricted file, so they would
    keep passing against a merge that only ever excluded ISO. This one stages
    every member and checks each of them individually.
    """
    frameworks_dir = tmp_path / "frameworks"
    frameworks_dir.mkdir()
    (frameworks_dir / "public_fw.json").write_text(
        json.dumps(_framework("public_fw", "2026-01-01", "A public control statement.")),
        encoding="utf-8",
    )
    for framework_id in sorted(RESTRICTED_FRAMEWORK_IDS):
        (frameworks_dir / f"{framework_id}.json").write_text(
            json.dumps(_framework(framework_id, "2026-08-15", RESTRICTED_PROSE)),
            encoding="utf-8",
        )

    processed_dir, licensed_dir = tmp_path / "processed", tmp_path / "licensed"
    merge_main(
        frameworks_dir=frameworks_dir,
        output_dir=processed_dir,
        licensed_dir=licensed_dir,
    )

    tracked = json.loads(
        (processed_dir / "all_controls.json").read_text(encoding="utf-8")
    )
    overlay = json.loads(
        (licensed_dir / "all_controls.json").read_text(encoding="utf-8")
    )
    tracked_ids = {f["framework_id"] for f in tracked["frameworks"]}
    overlay_ids = {f["framework_id"] for f in overlay["frameworks"]}

    assert tracked_ids == {"public_fw"}
    assert overlay_ids == {"public_fw"} | set(RESTRICTED_FRAMEWORK_IDS)
    assert tracked["framework_count"] == 1
    assert overlay["framework_count"] == 1 + len(RESTRICTED_FRAMEWORK_IDS)
    assert RESTRICTED_PROSE not in json.dumps(tracked)


def test_restricted_ids_have_one_source_of_truth() -> None:
    """The test suite must not carry its own copy of the restricted list.

    A second copy drifts. The merge step and the tracking test have to agree
    about which frameworks are licensed or the gate has a hole in it.
    """
    import tests.test_licensed_text_not_tracked as tracking_test

    assert tracking_test.RESTRICTED_FRAMEWORK_IDS is RESTRICTED_FRAMEWORK_IDS
