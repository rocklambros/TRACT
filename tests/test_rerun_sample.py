"""The re-run sample must carry no signal, and the sheet must arrive blank.

A second round by the same annotators on the same controls cannot be an
independent measurement -- anchoring is unavoidable. What makes it usable is
that anchoring runs toward the NULL: it makes people repeat themselves, so
movement is a lower bound on the instruction effect and stillness is only weak
evidence against it.

That argument collapses if the SAMPLE itself carries the hypothesis. Sending
only the controls someone marked `NONE` says "you said NONE too often" as
plainly as a sentence would, and it would deliver the exact nudge the re-run
exists to remove. So the sample is balanced by construction and these tests
hold it there.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.build_rerun_sample import balanced_sample, build_rerun_packet

FRAMEWORK = "nist_800_53"


@pytest.fixture(scope="module")
def round1(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A filled round-1 sheet: 12 linked, 60 NONE."""
    from scripts.build_bridge_packet import ai_hub_ids, build_bridge_packet

    pk = tmp_path_factory.mktemp("r1_packet")
    build_bridge_packet(pk, framework_id=FRAMEWORK)
    hubs = ai_hub_ids()
    with (pk / "annotate.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0].keys())

    out = tmp_path_factory.mktemp("r1_filled") / "vol-01.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows[:72]):
            row = dict(row)
            row["cre_id"] = hubs[i % len(hubs)] if i < 12 else "NONE"
            row["confidence"] = "3"
            row["rationale"] = "round one"
            writer.writerow(row)
    return out


class TestTheSampleIsBalanced:
    def test_equal_numbers_from_each_side(self, round1: Path) -> None:
        _, counts = balanced_sample(round1, "vol-01")
        assert counts["n_from_linked"] == counts["n_from_none"] == 12
        assert counts["n_sampled"] == 24

    def test_it_refuses_a_sheet_it_cannot_balance(self, tmp_path: Path) -> None:
        """Better no re-run than a sample whose shape states the hypothesis."""
        path = tmp_path / "all_linked.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["control_id", "control_title", "control_text",
                            "cre_id", "confidence", "rationale"],
            )
            writer.writeheader()
            writer.writerow({"control_id": "AC-1", "control_title": "t",
                             "control_text": "x", "cre_id": "010-108",
                             "confidence": "3", "rationale": "r"})
        with pytest.raises(ValueError, match="cannot be balanced"):
            balanced_sample(path, "vol-01")

    def test_it_refuses_a_sheet_with_no_links(self, tmp_path: Path) -> None:
        path = tmp_path / "all_none.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["control_id", "control_title", "control_text",
                            "cre_id", "confidence", "rationale"],
            )
            writer.writeheader()
            for i in range(5):
                writer.writerow({"control_id": f"AC-{i}", "control_title": "t",
                                 "control_text": "x", "cre_id": "NONE",
                                 "confidence": "3", "rationale": "r"})
        with pytest.raises(ValueError, match="no linked controls"):
            balanced_sample(path, "vol-01")


class TestTheSheetArrivesBlank:
    def test_no_answer_column_carries_a_value(self, round1: Path) -> None:
        rows, _ = balanced_sample(round1, "vol-01")
        for row in rows:
            for field in ("cre_id", "confidence", "rationale"):
                assert row[field] == "", (
                    f"{field} arrives carrying {row[field]!r} -- the annotator "
                    "would see their previous answer."
                )

    def test_the_control_text_is_preserved(self, round1: Path) -> None:
        rows, _ = balanced_sample(round1, "vol-01")
        assert all(r["control_text"].strip() for r in rows)
        assert all(r["control_id"].strip() for r in rows)


class TestTheOrderingLeaksNothing:
    def test_linked_and_none_rows_are_interleaved(self, round1: Path) -> None:
        """Unshuffled, position alone would reveal the previous answer."""
        rows, _ = balanced_sample(round1, "vol-01")
        with round1.open(encoding="utf-8") as handle:
            previous = {
                r["control_id"]: r["cre_id"].strip().upper()
                for r in csv.DictReader(handle)
            }
        was_linked = [previous[r["control_id"]] != "NONE" for r in rows]
        # A clean split would put all True first or all False first.
        first_half = sum(was_linked[: len(was_linked) // 2])
        assert 1 <= first_half <= len(was_linked) // 2 - 1, (
            "previously-linked rows cluster in the ordering, so position "
            "reveals what the annotator said last time"
        )

    def test_it_is_deterministic_for_a_given_annotator(
        self, round1: Path
    ) -> None:
        a, _ = balanced_sample(round1, "vol-01")
        b, _ = balanced_sample(round1, "vol-01")
        assert [r["control_id"] for r in a] == [r["control_id"] for r in b]

    def test_two_annotators_get_different_orderings(self, round1: Path) -> None:
        a, _ = balanced_sample(round1, "vol-01")
        b, _ = balanced_sample(round1, "vol-02")
        assert [r["control_id"] for r in a] != [r["control_id"] for r in b]


class TestTheCompositionIsNotSentToTheAnnotator:
    def test_the_packet_records_composition_in_a_separate_file(
        self, round1: Path, tmp_path: Path
    ) -> None:
        """It is the one thing that would reveal the hypothesis."""
        build_rerun_packet(tmp_path, "vol-01", round1, FRAMEWORK)
        composition = tmp_path / "sample_composition.json"
        assert composition.is_file()
        payload = json.loads(composition.read_text(encoding="utf-8"))
        assert payload["n_from_linked"] == payload["n_from_none"]

    def test_no_csv_reveals_the_composition(
        self, round1: Path, tmp_path: Path
    ) -> None:
        build_rerun_packet(tmp_path, "vol-01", round1, FRAMEWORK)
        for csv_path in tmp_path.glob("*.csv"):
            text = csv_path.read_text(encoding="utf-8").lower()
            for term in ("round", "previous", "n_from", "sample", "linked"):
                assert term not in text.splitlines()[0], (
                    f"{csv_path.name} header mentions {term!r}"
                )

    def test_the_brief_tells_the_coordinator_not_to_send_it(self) -> None:
        from tract.config import PROJECT_ROOT

        brief = (PROJECT_ROOT / "docs" / "phase2c-rerun-brief.md").read_text(
            encoding="utf-8"
        )
        assert "sample_composition.json" in brief
        assert "Do NOT send" in brief or "do not send" in brief.lower()
