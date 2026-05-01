"""Tests for tract.export.opencre_csv — CSV generation (spec §3)."""
from __future__ import annotations

import csv
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from tract.export.opencre_csv import generate_opencre_csv, write_opencre_csv


def _make_row(
    hub_id: str = "607-671",
    hub_name: str = "Protect against injection",
    framework_id: str = "mitre_atlas",
    section_id: str = "AML.M0015",
    title: str = "Adversarial Input Detection",
    description: str = "Detect adversarial inputs",
) -> dict:
    return {
        "hub_id": hub_id,
        "hub_name": hub_name,
        "framework_id": framework_id,
        "section_id": section_id,
        "title": title,
        "description": description,
    }


class TestGenerateOpencreCsv:
    def test_header_uses_opencre_name(self) -> None:
        csv_text = generate_opencre_csv([_make_row()], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        header = next(reader)
        assert header[0] == "CRE 0"
        assert header[1] == "MITRE ATLAS|name"
        assert header[2] == "MITRE ATLAS|id"
        assert header[3] == "MITRE ATLAS|description"
        assert header[4] == "MITRE ATLAS|hyperlink"

    def test_cre0_pipe_delimited(self) -> None:
        csv_text = generate_opencre_csv([_make_row()], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        row = next(reader)
        assert row[0] == "607-671|Protect against injection"

    def test_standard_columns_populated(self) -> None:
        csv_text = generate_opencre_csv([_make_row()], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        row = next(reader)
        assert row[1] == "Adversarial Input Detection"
        assert row[2] == "AML.M0015"
        assert row[3] == "Detect adversarial inputs"
        assert "atlas.mitre.org" in row[4]

    def test_sorted_by_hub_framework_section(self) -> None:
        rows = [
            _make_row(hub_id="999-999", section_id="B"),
            _make_row(hub_id="111-111", section_id="A"),
            _make_row(hub_id="111-111", section_id="C"),
        ]
        csv_text = generate_opencre_csv(rows, "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        data_rows = list(reader)
        assert data_rows[0][0].startswith("111-111")
        assert data_rows[1][0].startswith("111-111")
        assert data_rows[2][0].startswith("999-999")
        assert data_rows[0][2] == "A"
        assert data_rows[1][2] == "C"

    def test_empty_rows(self) -> None:
        csv_text = generate_opencre_csv([], "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        header = next(reader)
        assert len(header) == 5
        remaining = list(reader)
        assert remaining == []

    def test_multiple_hubs_same_control(self) -> None:
        rows = [
            _make_row(hub_id="111-111", hub_name="Hub A"),
            _make_row(hub_id="222-222", hub_name="Hub B"),
        ]
        csv_text = generate_opencre_csv(rows, "mitre_atlas")
        reader = csv.reader(StringIO(csv_text))
        next(reader)
        data_rows = list(reader)
        assert len(data_rows) == 2

    def test_unknown_framework_raises(self) -> None:
        with pytest.raises(KeyError):
            generate_opencre_csv([_make_row()], "nonexistent_framework")

    def test_new_framework_csv(self) -> None:
        row = _make_row(framework_id="csa_aicm", section_id="AICM-01")
        csv_text = generate_opencre_csv([row], "csa_aicm")
        reader = csv.reader(StringIO(csv_text))
        header = next(reader)
        assert header[1] == "CSA AI Controls Matrix|name"


class TestWriteOpencreCsv:
    def test_creates_file(self, tmp_path: Path) -> None:
        rows = [_make_row()]
        result = write_opencre_csv(rows, "mitre_atlas", tmp_path)
        assert result.exists()
        assert result.suffix == ".csv"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "subdir" / "nested"
        result = write_opencre_csv([_make_row()], "mitre_atlas", out)
        assert result.exists()

    def test_file_content_matches_generate(self, tmp_path: Path) -> None:
        rows = [_make_row()]
        result = write_opencre_csv(rows, "mitre_atlas", tmp_path)
        expected = generate_opencre_csv(rows, "mitre_atlas")
        with open(result, encoding="utf-8", newline="") as f:
            actual = f.read()
        assert actual == expected
