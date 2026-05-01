"""Tests for tract.prepare.json_extractor — JSON format extraction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.prepare.json_extractor import JsonExtractor
from tract.schema import Control


@pytest.fixture
def framework_output_json(tmp_path: Path) -> Path:
    data = {
        "framework_id": "test_fw",
        "framework_name": "Test Framework",
        "version": "1.0",
        "source_url": "https://example.com",
        "fetched_date": "2026-05-01",
        "mapping_unit_level": "control",
        "controls": [
            {"control_id": "TC-01", "title": "Access Control", "description": "Enforce access control policies for system components and users"},
            {"control_id": "TC-02", "title": "Encryption", "description": "Encrypt sensitive data at rest and in transit using standard algorithms"},
        ],
    }
    path = tmp_path / "framework_output.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def array_json(tmp_path: Path) -> Path:
    data = [
        {"id": "TC-01", "name": "Access Control", "description": "Enforce access control policies for components"},
        {"id": "TC-02", "name": "Encryption", "description": "Encrypt data at rest and in transit securely"},
    ]
    path = tmp_path / "array.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def nested_controls_json(tmp_path: Path) -> Path:
    data = {
        "name": "Some Framework",
        "controls": [
            {"control_id": "TC-01", "title": "Access", "body": "Enforce access control policies for system users"},
            {"control_id": "TC-02", "title": "Encrypt", "text": "Encrypt data at rest and in transit using standards"},
        ],
    }
    path = tmp_path / "nested.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def nested_items_json(tmp_path: Path) -> Path:
    data = {
        "items": [
            {"section_id": "S-01", "title": "First", "desc": "Description of first control for testing"},
        ],
    }
    path = tmp_path / "items.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def nested_data_json(tmp_path: Path) -> Path:
    data = {
        "data": [
            {"id": "D-01", "name": "Item", "text": "Description of the data item for extraction testing"},
        ],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def unrecognizable_json(tmp_path: Path) -> Path:
    data = {"config": {"setting": True}, "metadata": {"author": "test"}}
    path = tmp_path / "unrecognizable.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestJsonExtractor:
    def test_passthrough_framework_output(self, framework_output_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(framework_output_json)
        assert len(controls) == 2
        assert controls[0].control_id == "TC-01"
        assert controls[0].title == "Access Control"

    def test_array_of_objects(self, array_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(array_json)
        assert len(controls) == 2
        assert controls[0].control_id == "TC-01"
        assert controls[0].title == "Access Control"

    def test_nested_controls_key(self, nested_controls_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(nested_controls_json)
        assert len(controls) == 2
        assert controls[0].control_id == "TC-01"
        assert "access control" in controls[0].description.lower()

    def test_nested_items_key(self, nested_items_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(nested_items_json)
        assert len(controls) == 1
        assert controls[0].control_id == "S-01"

    def test_nested_data_key(self, nested_data_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(nested_data_json)
        assert len(controls) == 1
        assert controls[0].control_id == "D-01"

    def test_unrecognizable_raises(self, unrecognizable_json: Path) -> None:
        extractor = JsonExtractor()
        with pytest.raises(ValueError, match="(?i)no recognizable"):
            extractor.extract(unrecognizable_json)

    def test_returns_control_objects(self, array_json: Path) -> None:
        extractor = JsonExtractor()
        controls = extractor.extract(array_json)
        for ctrl in controls:
            assert isinstance(ctrl, Control)
            assert ctrl.control_id
            assert ctrl.description
