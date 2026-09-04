"""The importer is the tier boundary. Everything it accepts becomes Tier 2 gold.

A filled annotator sheet is untrusted input: it has been through a spreadsheet,
an email client and a human. This is the only place that can refuse a row, and
what it writes goes straight into training supervision and the Gate 1 count.

So it validates rather than coerces. An unknown hub id, an unknown control id, a
duplicate mapping, a missing annotator -- each raises, naming the row, rather
than being dropped. Silently skipping a malformed record would report a smaller
corpus as though it were the whole one, and the count is what Gate 1 measures.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.import_bridge_links import import_bridge_links

ANNOTATOR = "vol-01"
WHEN = "2026-09-04T12:00:00Z"


def _known() -> tuple[str, str]:
    """A real hub id and a real NIST 800-53 control id."""
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


def _csv(
    tmp_path: Path,
    *,
    cre_id: str | None = None,
    control_id: str | None = None,
    duplicate: bool = False,
    confidence: str = "3",
    rationale: str = "boundary protection maps to this hub",
    name: str = "filled.csv",
) -> Path:
    hub, control = _known()
    row = {
        "control_id": control_id if control_id is not None else control,
        "cre_id": cre_id if cre_id is not None else hub,
        "confidence": confidence,
        "rationale": rationale,
    }
    path = tmp_path / name
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
        if duplicate:
            writer.writerow(row)
    return path


def _import(src: Path, out: Path) -> None:
    import_bridge_links(
        src, out, framework_id="nist_800_53", annotator_id=ANNOTATOR, created_at=WHEN,
    )


class TestProvenanceIsStampedAtTheBoundary:
    def test_every_row_carries_tier_2_and_provenance(self, tmp_path: Path) -> None:
        out = tmp_path / "o.jsonl"
        _import(_csv(tmp_path), out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines
        for line in lines:
            row = json.loads(line)
            assert row["tier"] == 2
            assert row["annotator_id"] == ANNOTATOR
            assert row["created_at"] == WHEN

    def test_the_output_loads_back_through_the_bridge_loader(
        self, tmp_path: Path
    ) -> None:
        """Round-trip through the real reader, not a hand-parsed copy."""
        from tract.bridge.links import load_bridge_links

        out = tmp_path / "o.jsonl"
        _import(_csv(tmp_path), out)
        links = load_bridge_links(out)
        assert len(links) == 1
        assert links[0].tier == 2
        assert links[0].framework_id == "nist_800_53"

    def test_the_annotator_id_is_required(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="annotator_id"):
            import_bridge_links(
                _csv(tmp_path),
                tmp_path / "o.jsonl",
                framework_id="nist_800_53",
                annotator_id="",
                created_at=WHEN,
            )


class TestItRefusesBadRows:
    def test_refuses_an_unknown_hub_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown hub"):
            _import(_csv(tmp_path, cre_id="000-000"), tmp_path / "o.jsonl")

    def test_refuses_an_unknown_control_id(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="unknown control"):
            _import(_csv(tmp_path, control_id="ZZ-99"), tmp_path / "o.jsonl")

    def test_refuses_a_duplicate_row(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            _import(_csv(tmp_path, duplicate=True), tmp_path / "o.jsonl")

    def test_refuses_a_confidence_outside_the_scale(self, tmp_path: Path) -> None:
        """D4 sets a 1-3 scale. 4 and 5 are off it, not merely high."""
        for value in ("0", "4", "5", "7"):
            with pytest.raises(ValueError, match="confidence"):
                _import(_csv(tmp_path, confidence=value), tmp_path / "o.jsonl")

    def test_accepts_every_value_on_the_scale(self, tmp_path: Path) -> None:
        for value in ("1", "2", "3"):
            _import(_csv(tmp_path, confidence=value), tmp_path / "o.jsonl")

    def test_the_scale_matches_the_gate_one_floor(self) -> None:
        """The floor must sit inside the scale or it can never bind."""
        from scripts.import_bridge_links import (
            CONFIDENCE_MAX,
            CONFIDENCE_MIN,
            GATE1_CONFIDENCE_FLOOR,
        )

        assert CONFIDENCE_MIN < GATE1_CONFIDENCE_FLOOR <= CONFIDENCE_MAX
        assert (CONFIDENCE_MIN, CONFIDENCE_MAX) == (1, 3)

    def test_refuses_a_non_numeric_confidence(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="confidence"):
            _import(_csv(tmp_path, confidence="high"), tmp_path / "o.jsonl")

    def test_refuses_an_empty_rationale(self, tmp_path: Path) -> None:
        """A rationale is what makes a disputed link reviewable later."""
        with pytest.raises(ValueError, match="rationale"):
            _import(_csv(tmp_path, rationale="   "), tmp_path / "o.jsonl")

    def test_the_error_names_the_row(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="row 2"):
            _import(_csv(tmp_path, cre_id="000-000"), tmp_path / "o.jsonl")

    def test_refuses_an_empty_sheet(self, tmp_path: Path) -> None:
        """Zero links would import cleanly and de-orphan nothing, silently."""
        path = tmp_path / "empty.csv"
        path.write_text(
            "control_id,cre_id,confidence,rationale\n", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="no rows"):
            _import(path, tmp_path / "o.jsonl")


class TestNothingIsWrittenOnRejection:
    def test_a_rejected_sheet_leaves_no_output(self, tmp_path: Path) -> None:
        """Atomic in the sense that matters: no partial corpus on disk."""
        out = tmp_path / "o.jsonl"
        with pytest.raises(ValueError):
            _import(_csv(tmp_path, cre_id="000-000"), out)
        assert not out.exists()

    def test_a_rejected_sheet_does_not_clobber_an_existing_output(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "o.jsonl"
        _import(_csv(tmp_path), out)
        before = out.read_text(encoding="utf-8")
        with pytest.raises(ValueError):
            _import(_csv(tmp_path, cre_id="000-000", name="bad.csv"), out)
        assert out.read_text(encoding="utf-8") == before


class TestTheImportIsDeterministic:
    def test_two_imports_of_the_same_sheet_are_byte_identical(
        self, tmp_path: Path
    ) -> None:
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        src = _csv(tmp_path)
        _import(src, a)
        _import(src, b)
        assert a.read_bytes() == b.read_bytes()
