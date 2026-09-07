"""The packet an annotator receives must be fillable into what the importer reads.

Measured before this landed:

    packet emits  : control_id, control_title, control_text
    importer wants: control_id, cre_id, confidence, rationale
    overlap       : control_id

So a volunteer received a sheet with no answer columns and had to invent the
response format, and the import then rejected whatever they invented. The two
halves of the round were built against different schemas and nothing connected
them, because no test ever carried a file from one to the other.

That is the same shape as the packet-versus-handbook mismatch and as
`publish-hf`, which was documented and could not run: each side was tested in
isolation and the join was tested by nobody.

These tests carry a real packet through a real fill to a real import. The
load-bearing one is `test_a_filled_packet_imports_cleanly` -- it is the only
test in the suite that touches both sides.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.build_bridge_packet import (
    ANNOTATION_SHEET_NAME,
    ANSWER_FIELDS,
    build_bridge_packet,
)
from scripts.import_bridge_links import REQUIRED_COLUMNS, import_bridge_links

FRAMEWORK = "nist_800_53"


@pytest.fixture(scope="module")
def packet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("packet")
    build_bridge_packet(out, framework_id=FRAMEWORK)
    return out


class TestThePacketIsFillable:
    def test_it_emits_an_annotation_sheet(self, packet: Path) -> None:
        assert (packet / ANNOTATION_SHEET_NAME).is_file()

    def test_the_annotation_sheet_carries_every_column_the_importer_requires(
        self, packet: Path
    ) -> None:
        """The join, asserted directly rather than inferred."""
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            header = set(csv.DictReader(handle).fieldnames or [])
        missing = set(REQUIRED_COLUMNS) - header
        assert not missing, (
            f"The annotator cannot supply {sorted(missing)} in the sheet they "
            "were given, so their filled file cannot be imported."
        )

    def test_it_carries_the_control_text_beside_the_answer(
        self, packet: Path
    ) -> None:
        """One file to work in. A separate reference sheet means transcription."""
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert rows[0]["control_text"].strip()
        assert rows[0]["control_title"].strip()

    def test_every_answer_column_starts_empty(self, packet: Path) -> None:
        """A pre-filled answer is a suggestion, and a suggestion is Tier 3."""
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                for field in ANSWER_FIELDS:
                    assert row[field] == "", (
                        f"{field} arrives pre-filled with {row[field]!r}. "
                        "Anything in an answer column is a suggestion."
                    )

    def test_one_row_per_control(self, packet: Path) -> None:
        with (packet / "controls.csv").open(encoding="utf-8") as handle:
            reference = sum(1 for _ in csv.DictReader(handle))
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            annotation = sum(1 for _ in csv.DictReader(handle))
        assert reference == annotation


class TestTheRoundTrip:
    """Build a packet, fill it the way a volunteer would, import it."""

    def _fill(self, packet: Path, out: Path, answers: list[tuple[str, str, str]]) -> Path:
        """Write a filled sheet: (cre_id, confidence, rationale) per leading row."""
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        for row, (hub, confidence, rationale) in zip(rows, answers):
            row["cre_id"] = hub
            row["confidence"] = confidence
            row["rationale"] = rationale
        filled = out / "filled.csv"
        with filled.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows[: len(answers)])
        return filled

    @staticmethod
    def _hub() -> str:
        from scripts.build_bridge_packet import ai_hub_ids

        return ai_hub_ids()[0]

    def test_a_filled_packet_imports_cleanly(
        self, packet: Path, tmp_path: Path
    ) -> None:
        """The load-bearing test: the only one that touches both sides."""
        hub = self._hub()
        filled = self._fill(
            packet, tmp_path,
            [(hub, "3", "boundary protection maps to this hub")],
        )
        links = import_bridge_links(
            filled, tmp_path / "out.jsonl",
            framework_id=FRAMEWORK, annotator_id="vol-01",
            created_at="2026-09-07T12:00:00Z",
        )
        assert len(links) == 1
        assert links[0].cre_id == hub
        assert links[0].tier == 2

    def test_a_none_answer_round_trips(self, packet: Path, tmp_path: Path) -> None:
        """The handbook calls NONE expected. It must survive the whole path."""
        from scripts.import_bridge_links import NO_HUB_SENTINEL

        hub = self._hub()
        filled = self._fill(
            packet, tmp_path,
            [
                (NO_HUB_SENTINEL, "3", "nothing in the AI region fits this"),
                (hub, "2", "partial fit, recorded with low confidence"),
            ],
        )
        out = tmp_path / "out.jsonl"
        links = import_bridge_links(
            filled, out, framework_id=FRAMEWORK, annotator_id="vol-01",
            created_at="2026-09-07T12:00:00Z",
        )
        assert len(links) == 1, "NONE is a judgement, not a link"
        reviewed = out.with_suffix(".reviewed.json")
        assert reviewed.is_file()

    def test_an_untouched_packet_is_refused_rather_than_imported_empty(
        self, packet: Path, tmp_path: Path
    ) -> None:
        """A volunteer who returns the sheet unfilled must not read as a round.

        Every answer column empty is indistinguishable, in the corpus, from a
        round that was never run.
        """
        import shutil

        untouched = tmp_path / "untouched.csv"
        shutil.copy(packet / ANNOTATION_SHEET_NAME, untouched)
        with pytest.raises(ValueError):
            import_bridge_links(
                untouched, tmp_path / "out.jsonl", framework_id=FRAMEWORK,
                annotator_id="vol-01", created_at="2026-09-07T12:00:00Z",
            )


class TestTheReferenceSheetStillCarriesNoAnswers:
    """Narrowed, not deleted.

    The original test forbade "cre", "hub", "gold", "answer" and "label" in the
    control sheet's header, which would now reject the annotation sheet's own
    empty `cre_id` column. The property that matters is that no sheet carries a
    ground-truth VALUE, and `test_every_answer_column_starts_empty` above holds
    that for the annotation sheet.
    """

    def test_the_reference_control_sheet_has_no_answer_column(
        self, packet: Path
    ) -> None:
        with (packet / "controls.csv").open(encoding="utf-8") as handle:
            header = set(csv.DictReader(handle).fieldnames or [])
        assert header == {"control_id", "control_title", "control_text"}


class TestThePacketCarriesTheFullestTextAvailable:
    """19% of the packet was truncated mid-word before this.

    `all_controls.json` caps `description` at 2,000 characters, so 58 of the 300
    NIST 800-53 controls ended mid-word -- one finishes "Procedures can be
    documente". An annotator reading that is judging a control on partial text,
    and the whole task is a judgement about what the control is FOR.

    Every one of those 58 carries a `full_text` field that is longer (median
    +535 characters, up to +3,508), and across the framework `full_text` is
    never shorter than `description`. So preferring it recovers the truncated
    ones with no case where it loses anything.

    CLAUDE.md already required this: "Consider all available prose, always.
    Prefer a control's full text over its title everywhere text is selected."
    """

    def test_no_control_text_is_cut_at_the_description_cap(
        self, packet: Path
    ) -> None:
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        at_cap = [r for r in rows if len(r["control_text"]) == 2000]
        assert not at_cap, (
            f"{len(at_cap)} controls sit at exactly the 2,000-character "
            "description cap, so they are truncated mid-sentence. Prefer "
            "full_text where it is longer."
        )

    def test_it_matches_the_longest_text_the_corpus_holds(
        self, packet: Path
    ) -> None:
        """Checked against the source, not against a copied expectation."""
        import json

        from tract.config import PROCESSED_DIR

        corpus = json.loads(
            (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
        )
        framework = next(
            f for f in corpus["frameworks"] if f["framework_id"] == FRAMEWORK
        )
        longest = {
            c["control_id"]: max(
                (c.get("description") or "").strip(),
                (c.get("full_text") or "").strip(),
                key=len,
            )
            for c in framework["controls"]
        }
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        short = [
            r["control_id"] for r in rows
            if len(r["control_text"]) < len(longest.get(r["control_id"], ""))
        ]
        assert not short, (
            f"{len(short)} controls carry less text than the corpus holds for "
            f"them, e.g. {short[:5]}"
        )

    def test_the_reference_sheet_agrees_with_the_annotation_sheet(
        self, packet: Path
    ) -> None:
        """Two sheets showing the same control differently is worse than one."""
        with (packet / "controls.csv").open(encoding="utf-8") as handle:
            reference = {r["control_id"]: r["control_text"] for r in csv.DictReader(handle)}
        with (packet / ANNOTATION_SHEET_NAME).open(encoding="utf-8") as handle:
            annotation = {r["control_id"]: r["control_text"] for r in csv.DictReader(handle)}
        assert reference == annotation
