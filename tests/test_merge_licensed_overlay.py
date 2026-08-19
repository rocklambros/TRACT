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
from typing import Any

import pytest

from parsers.merge_all_controls import main as merge_main
from tract.config import CONDITIONAL_FRAMEWORK_IDS, RESTRICTED_FRAMEWORK_IDS


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

# Stands in for a conditional framework's control statement, and distinct from
# RESTRICTED_PROSE so a test cannot pass by finding the wrong string. Invented,
# for the same reason: this file must not quote any licensed source.
CONDITIONAL_PROSE = (
    "Pipeline build agents shall run with the narrowest privilege set the "
    "stage requires, and every widening shall be recorded with its owner."
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


class TestConditionalFrameworkProseStaysOutOfTheTrackedMerge:
    """The same defect one tier down, and the proof that closing it is real.

    The merge filtered on RESTRICTED_FRAMEWORK_IDS alone. The seven conditional
    frameworks carry GPL-3.0 and CC BY-SA text, their per-framework files are
    gitignored, and the merge inlined them into the tracked corpus anyway. No
    test fired, because they carry no prose today. That is not a control, it is
    an accident that has been holding.

    These tests stage the accident's end: a conditional framework that DOES
    carry prose. Without the widened filter the tracked artifact carries it
    verbatim, so the assertions below are reachable in both directions rather
    than restating what the current data already happens to satisfy.

    Identifiers and titles survive in the tracked artifact, and that is the
    tier's design rather than an oversight: a mapping is a fact about two
    documents, and OpenCRE already publishes these titles.
    """

    @staticmethod
    def _conditional_id() -> str:
        conditional = sorted(CONDITIONAL_FRAMEWORK_IDS)
        assert conditional, "no conditional framework to exercise the filter"
        return conditional[0]

    def _run(
        self, tmp_path: Path, description: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        frameworks_dir = tmp_path / "frameworks"
        frameworks_dir.mkdir()
        (frameworks_dir / "public_fw.json").write_text(
            json.dumps(
                _framework("public_fw", "2026-01-01", "A public control statement.")
            ),
            encoding="utf-8",
        )
        conditional_id = self._conditional_id()
        (frameworks_dir / f"{conditional_id}.json").write_text(
            json.dumps(_framework(conditional_id, "2026-02-02", description)),
            encoding="utf-8",
        )
        processed_dir, licensed_dir = tmp_path / "processed", tmp_path / "licensed"
        merge_main(
            frameworks_dir=frameworks_dir,
            output_dir=processed_dir,
            licensed_dir=licensed_dir,
        )
        return (
            json.loads(
                (processed_dir / "all_controls.json").read_text(encoding="utf-8")
            ),
            json.loads(
                (licensed_dir / "all_controls.json").read_text(encoding="utf-8")
            ),
        )

    def test_the_tracked_artifact_excludes_conditional_prose(
        self, tmp_path: Path,
    ) -> None:
        tracked, _ = self._run(tmp_path, CONDITIONAL_PROSE)
        assert CONDITIONAL_PROSE not in json.dumps(tracked), (
            "a conditional framework's control statement reached the tracked "
            "corpus. This repository is CC0 and the source is GPL-3.0 or CC "
            "BY-SA, so committing it asserts rights the project does not hold."
        )

    def test_the_overlay_keeps_conditional_prose(self, tmp_path: Path) -> None:
        """Training must not lose the text. Only publication does."""
        _, overlay = self._run(tmp_path, CONDITIONAL_PROSE)
        assert CONDITIONAL_PROSE in json.dumps(overlay)

    def test_identifiers_and_titles_survive_in_the_tracked_artifact(
        self, tmp_path: Path,
    ) -> None:
        """Withheld prose, not a withheld framework.

        Dropping the framework would remove 341 controls from the real corpus
        and shift every downstream count. The tier's own rule is that the
        mapping stays tracked, so the control keeps its id and its title.
        """
        tracked, _ = self._run(tmp_path, CONDITIONAL_PROSE)
        by_id = {f["framework_id"]: f for f in tracked["frameworks"]}
        conditional_id = self._conditional_id()
        assert conditional_id in by_id, (
            "the conditional framework vanished from the tracked corpus. Its "
            "assignments are published and need its identifiers."
        )
        control = by_id[conditional_id]["controls"][0]
        assert control["control_id"] == "1.1"
        assert control["title"] == "Some title"
        assert control["description"] == "Some title"
        assert tracked["framework_count"] == 2
        assert tracked["total_controls"] == 2

    def test_a_conditional_framework_without_prose_is_untouched(
        self, tmp_path: Path,
    ) -> None:
        """The other direction: no prose, no redaction, same object.

        Every one of the seven carries title-only stubs today, so this is the
        path the real corpus takes, and it is what makes the widened filter
        byte-identical on current data instead of a silent rewrite.
        """
        tracked, _ = self._run(tmp_path, "Some title")
        by_id = {f["framework_id"]: f for f in tracked["frameworks"]}
        control = by_id[self._conditional_id()]["controls"][0]
        assert control["description"] == "Some title"
        assert tracked["total_controls"] == 2

    def test_full_text_is_withheld_even_when_the_description_is_a_title(
        self, tmp_path: Path,
    ) -> None:
        """The second channel. A stub description with prose beside it.

        sanitize_control moves anything over DESCRIPTION_MAX_LENGTH into
        full_text, so a filter that looked only at description would publish
        the longest statements in the corpus and nothing else.
        """
        frameworks_dir = tmp_path / "frameworks"
        frameworks_dir.mkdir()
        conditional_id = self._conditional_id()
        framework = _framework(conditional_id, "2026-02-02", "Some title")
        controls = framework["controls"]
        assert isinstance(controls, list)
        controls[0]["full_text"] = CONDITIONAL_PROSE
        (frameworks_dir / f"{conditional_id}.json").write_text(
            json.dumps(framework), encoding="utf-8",
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
        assert CONDITIONAL_PROSE not in json.dumps(tracked)
        assert CONDITIONAL_PROSE in json.dumps(overlay)

    def test_every_conditional_framework_is_filtered_not_just_the_first(
        self, tmp_path: Path,
    ) -> None:
        """Driven by the set. A per-framework hole would survive the tests above."""
        frameworks_dir = tmp_path / "frameworks"
        frameworks_dir.mkdir()
        for framework_id in sorted(CONDITIONAL_FRAMEWORK_IDS):
            (frameworks_dir / f"{framework_id}.json").write_text(
                json.dumps(_framework(framework_id, "2026-02-02", CONDITIONAL_PROSE)),
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
        offenders = [
            f["framework_id"]
            for f in tracked["frameworks"]
            if any(
                c.get("description") != c.get("title")
                for c in f["controls"]
            )
        ]
        assert not offenders, f"{offenders} kept their prose in the tracked corpus"
        assert tracked["framework_count"] == len(CONDITIONAL_FRAMEWORK_IDS)


def test_restricted_ids_have_one_source_of_truth() -> None:
    """The test suite must not carry its own copy of the restricted list.

    A second copy drifts. The merge step and the tracking test have to agree
    about which frameworks are licensed or the gate has a hole in it.
    """
    import tests.test_licensed_text_not_tracked as tracking_test

    assert tracking_test.RESTRICTED_FRAMEWORK_IDS is RESTRICTED_FRAMEWORK_IDS
