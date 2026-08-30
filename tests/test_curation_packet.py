"""The annotation packet must contain nothing a model produced or answered.

A curation round is Tier 2 only if the annotator worked blind. One model-derived
artifact in the packet makes the whole round Tier 3 and useless for the gate in
`results/phase1b/CAMPAIGN3.md`, and the cost is paid before anyone notices --
the labels look identical either way.

The specific trap: `results/review/hub_reference.md` covers all 522 hubs, reads
well, and is the obvious thing to send. 400 of its hub descriptions were written
by an LLM conditioned on the gold links.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from scripts.build_curation_packet import (
    EXCLUDED_ILLUSTRATION_FRAMEWORKS,
    build_control_sheet,
    build_hub_sheet,
)


@pytest.fixture(scope="module")
def packet(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("packet")
    build_hub_sheet(out)
    build_control_sheet(out, "csa_aicm")
    return out


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class TestHubSheetIsNotModelDerived:
    def test_every_hub_appears_exactly_once(self, packet: Path) -> None:
        rows = _rows(packet / "hub_reference_sheet.csv")
        ids = [r["hub_id"] for r in rows]
        assert len(ids) == len(set(ids))
        assert len(ids) == 522, f"expected the full hub tree, got {len(ids)}"

    def test_no_excluded_framework_illustrates_a_hub(self, packet: Path) -> None:
        """AI frameworks are the answer; CCM is csa_aicm's parent corpus.

        203 of csa_aicm's 243 control ids come from CCM and 91 carry
        byte-identical statements, so a CCM example beside a CCM-derived
        control is the answer wearing another framework's name.
        """
        rows = _rows(packet / "hub_reference_sheet.csv")
        blob = " ".join(r["example_controls_already_mapped_here"] for r in rows)
        for framework in EXCLUDED_ILLUSTRATION_FRAMEWORKS:
            assert framework not in blob, (
                f"{framework!r} illustrates a hub in the annotator's reference "
                "sheet; that is either the answer or its parent corpus"
            )

    def test_examples_are_titles_not_generated_prose(self, packet: Path) -> None:
        """Each example must be `Framework: section`, a value copied from a link.

        Generated descriptions are the contamination this packet exists to
        avoid, and they do not have this shape.
        """
        rows = _rows(packet / "hub_reference_sheet.csv")
        shape = re.compile(r"^[^:]+: .+$")
        checked = 0
        for row in rows:
            for example in filter(None, row["example_controls_already_mapped_here"].split(" | ")):
                assert shape.match(example), f"unexpected example form: {example!r}"
                checked += 1
        assert checked > 500, f"only {checked} examples checked; sheet looks empty"

    def test_some_hubs_have_no_examples_and_that_is_reported_honestly(
        self, packet: Path,
    ) -> None:
        """148 hubs have no safe illustration. They must be blank, not invented."""
        rows = _rows(packet / "hub_reference_sheet.csv")
        blank = [r for r in rows if not r["example_controls_already_mapped_here"]]
        assert blank, (
            "every hub has an example, which means the exclusion filter is not "
            "running -- 148 hubs have no non-AI non-CCM link"
        )


class TestControlSheetCarriesNoAnswer:
    def test_answer_columns_are_empty(self, packet: Path) -> None:
        rows = _rows(packet / "annotate_csa_aicm.csv")
        assert rows
        for row in rows:
            for column, value in row.items():
                if column.startswith("ANSWER"):
                    assert value == "", (
                        f"{column} is pre-filled with {value!r}; the annotator "
                        "is being shown an answer"
                    )

    def test_no_prediction_or_confidence_column_exists(self, packet: Path) -> None:
        """The shape of the leak, not just its current absence.

        A future edit adding `predicted_hub` or `model_confidence` would keep
        every ANSWER column empty and still make the round Tier 3.
        """
        with (packet / "annotate_csa_aicm.csv").open(encoding="utf-8") as handle:
            header = handle.readline().lower()
        for forbidden in (
            "predict", "confidence_score", "similarity", "rank", "top_k",
            "suggested", "candidate", "model",
        ):
            if forbidden == "confidence_score":
                assert forbidden not in header
                continue
            assert forbidden not in header, (
                f"column matching {forbidden!r} in the annotator's sheet: "
                f"{header.strip()}"
            )

    def test_every_control_is_present_and_numbered(self, packet: Path) -> None:
        rows = _rows(packet / "annotate_csa_aicm.csv")
        assert len(rows) == 243, f"csa_aicm has 243 controls, sheet has {len(rows)}"
        assert [int(r["row"]) for r in rows] == list(range(1, 244))

    def test_every_control_carries_readable_text(self, packet: Path) -> None:
        """An empty statement is an unanswerable row that still costs time."""
        rows = _rows(packet / "annotate_csa_aicm.csv")
        empty = [r["control_id"] for r in rows if len(r["control_statement"]) < 20]
        assert not empty, f"controls with no usable statement: {empty[:5]}"
