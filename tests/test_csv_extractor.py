"""Tests for tract.prepare.csv_extractor — CSV format extraction."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

from tract.prepare.csv_extractor import CsvExtractor
from tract.schema import Control


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    path = tmp_path / "framework.csv"
    path.write_text(
        'control_id,title,description\n'
        'TC-01,Access Control,"Enforce access control policies for all system components and users"\n'
        'TC-02,Data Encryption,"Encrypt sensitive data at rest and in transit using standard algorithms"\n'
        'TC-03,Audit Logging,"Maintain comprehensive audit logs of security events for forensic analysis"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def alternate_columns_csv(tmp_path: Path) -> Path:
    path = tmp_path / "alt_cols.csv"
    path.write_text(
        'id,name,text\n'
        'A1,Access Control,"Enforce access control policies for all system components"\n'
        'A2,Encryption,"Encrypt data at rest and in transit using industry standards"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def csv_with_empty_rows(tmp_path: Path) -> Path:
    path = tmp_path / "blanks.csv"
    path.write_text(
        'control_id,title,description\n'
        'TC-01,Access Control,"Enforce access control policies for all system components and users"\n'
        ',,\n'
        '  , , \n'
        'TC-02,Data Encryption,"Encrypt sensitive data at rest and in transit using standard algorithms"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def csv_with_full_text(tmp_path: Path) -> Path:
    path = tmp_path / "full_text.csv"
    path.write_text(
        'control_id,title,description,full_text\n'
        'TC-01,Access Control,"Short desc for access","Full detailed text about access control policies"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def csv_with_bom(tmp_path: Path) -> Path:
    path = tmp_path / "bom.csv"
    path.write_bytes(
        b'\xef\xbb\xbf'
        b'control_id,title,description\n'
        b'TC-01,Access Control,Enforce access control policies\n'
    )
    return path


@pytest.fixture
def csv_missing_columns(tmp_path: Path) -> Path:
    path = tmp_path / "missing.csv"
    path.write_text(
        'foo,bar,baz\n'
        '1,2,3\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def tsv_file(tmp_path: Path) -> Path:
    path = tmp_path / "framework.tsv"
    path.write_text(
        'control_id\ttitle\tdescription\n'
        'TC-01\tAccess Control\tEnforce access control policies for all system components and users\n',
        encoding="utf-8",
    )
    return path


class TestCsvExtractor:
    def test_extract_standard_columns(self, sample_csv: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(sample_csv)
        assert len(controls) == 3
        assert all(isinstance(c, Control) for c in controls)
        assert controls[0].control_id == "TC-01"
        assert controls[0].title == "Access Control"
        assert "access control" in controls[0].description.lower()

    def test_extract_alternate_columns(self, alternate_columns_csv: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(alternate_columns_csv)
        assert len(controls) == 2
        assert controls[0].control_id == "A1"
        assert controls[0].title == "Access Control"

    def test_skips_empty_rows(self, csv_with_empty_rows: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(csv_with_empty_rows)
        assert len(controls) == 2

    def test_full_text_column(self, csv_with_full_text: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(csv_with_full_text)
        assert len(controls) == 1
        assert controls[0].full_text == "Full detailed text about access control policies"

    def test_missing_columns_raises(self, csv_missing_columns: Path) -> None:
        extractor = CsvExtractor()
        with pytest.raises(ValueError, match="(?i)column"):
            extractor.extract(csv_missing_columns)

    def test_tsv_file(self, tsv_file: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(tsv_file)
        assert len(controls) == 1
        assert controls[0].control_id == "TC-01"

    def test_bom_handling(self, csv_with_bom: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(csv_with_bom)
        assert len(controls) == 1
        assert controls[0].control_id == "TC-01"

    def test_returns_list_of_control_objects(self, sample_csv: Path) -> None:
        extractor = CsvExtractor()
        controls = extractor.extract(sample_csv)
        for ctrl in controls:
            assert isinstance(ctrl, Control)
            assert ctrl.control_id
            assert ctrl.description
