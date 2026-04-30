"""Tests for unmapped control loading."""
from __future__ import annotations

import json

import pytest
from pathlib import Path


class TestLoadUnmappedControls:
    def test_returns_list_of_dicts(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        fw_file = fw_dir / "csa_aicm.json"
        fw_file.write_text(json.dumps({
            "framework_name": "CSA AI Controls Matrix",
            "framework_id": "csa_aicm",
            "controls": [
                {"control_id": "A01", "title": "Audit", "description": "Audit desc", "full_text": "Full audit text"},
                {"control_id": "A02", "title": "Access", "description": "Access desc", "full_text": ""},
            ],
        }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm"],
            framework_display_names={"csa_aicm": "CSA AI Controls Matrix"},
        )
        assert len(controls) == 2
        assert controls[0]["control_id"] == "csa_aicm:A01"
        assert controls[0]["framework"] == "CSA AI Controls Matrix"
        assert "control_text" in controls[0]
        assert len(controls[0]["control_text"]) > 0

    def test_control_text_uses_full_text_when_available(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        fw_file = fw_dir / "csa_aicm.json"
        fw_file.write_text(json.dumps({
            "framework_name": "CSA AI Controls Matrix",
            "framework_id": "csa_aicm",
            "controls": [
                {"control_id": "A01", "title": "Audit", "description": "Short", "full_text": "Very long full text here"},
            ],
        }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm"],
            framework_display_names={"csa_aicm": "CSA AI Controls Matrix"},
        )
        assert "Very long full text here" in controls[0]["control_text"]

    def test_control_text_falls_back_to_description(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        fw_file = fw_dir / "csa_aicm.json"
        fw_file.write_text(json.dumps({
            "framework_name": "CSA AI Controls Matrix",
            "framework_id": "csa_aicm",
            "controls": [
                {"control_id": "A01", "title": "Audit", "description": "Description text here", "full_text": ""},
            ],
        }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm"],
            framework_display_names={"csa_aicm": "CSA AI Controls Matrix"},
        )
        assert "Description text here" in controls[0]["control_text"]

    def test_multiple_frameworks(self, tmp_path: Path) -> None:
        from tract.active_learning.unmapped import load_unmapped_controls

        fw_dir = tmp_path / "frameworks"
        fw_dir.mkdir()
        for fid, count in [("csa_aicm", 3), ("mitre_atlas", 2)]:
            fw_file = fw_dir / f"{fid}.json"
            fw_file.write_text(json.dumps({
                "framework_name": f"FW {fid}",
                "framework_id": fid,
                "controls": [
                    {"control_id": f"C{i}", "title": f"T{i}", "description": f"D{i}", "full_text": ""}
                    for i in range(count)
                ],
            }), encoding="utf-8")

        controls = load_unmapped_controls(
            frameworks_dir=fw_dir,
            framework_file_ids=["csa_aicm", "mitre_atlas"],
            framework_display_names={"csa_aicm": "FW csa_aicm", "mitre_atlas": "FW mitre_atlas"},
        )
        assert len(controls) == 5
