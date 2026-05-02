"""Tests for publish-hf CLI command and orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestPublishGate:

    def test_rejects_missing_bridge_report(self, tmp_path) -> None:
        from tract.publish import check_publication_gate
        with pytest.raises(ValueError, match="bridge_report.json"):
            check_publication_gate(tmp_path / "bridge_report.json")

    def test_rejects_pending_candidates(self, tmp_path) -> None:
        from tract.publish import check_publication_gate
        report = {
            "counts": {"total": 1, "accepted": 0, "rejected": 0},
            "candidates": [{"status": "pending"}],
        }
        report_path = tmp_path / "bridge_report.json"
        report_path.write_text(json.dumps(report))
        with pytest.raises(ValueError, match="pending"):
            check_publication_gate(report_path)

    def test_accepts_zero_bridges(self, tmp_path) -> None:
        from tract.publish import check_publication_gate
        report = {
            "counts": {"total": 1, "accepted": 0, "rejected": 1},
            "candidates": [{"status": "rejected"}],
        }
        report_path = tmp_path / "bridge_report.json"
        report_path.write_text(json.dumps(report))
        check_publication_gate(report_path)

    def test_accepts_all_reviewed_with_hierarchy(self, tmp_path) -> None:
        from tract.publish import check_publication_gate
        report = {
            "counts": {"total": 2, "accepted": 1, "rejected": 1},
            "candidates": [
                {"status": "accepted", "ai_hub_id": "AI-1", "trad_hub_id": "T-1"},
                {"status": "rejected", "ai_hub_id": "AI-2", "trad_hub_id": "T-2"},
            ],
        }
        report_path = tmp_path / "bridge_report.json"
        report_path.write_text(json.dumps(report))

        hier = {
            "version": "1.1",
            "hubs": {
                "AI-1": {"related_hub_ids": ["T-1"]},
                "T-1": {"related_hub_ids": ["AI-1"]},
            },
        }
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier))
        check_publication_gate(report_path, hierarchy_path=hier_path)

    def test_rejects_accepted_bridges_without_hierarchy_update(self, tmp_path) -> None:
        from tract.publish import check_publication_gate
        report = {
            "counts": {"total": 1, "accepted": 1, "rejected": 0},
            "candidates": [
                {"status": "accepted", "ai_hub_id": "AI-1", "trad_hub_id": "T-1"},
            ],
        }
        report_path = tmp_path / "bridge_report.json"
        report_path.write_text(json.dumps(report))

        hier = {"version": "1.0", "hubs": {"AI-1": {"related_hub_ids": []}}}
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier))
        with pytest.raises(ValueError, match="version"):
            check_publication_gate(report_path, hierarchy_path=hier_path)


class TestPublishHFCLIParsing:

    def test_subcommand_exists(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo"])
        assert args.command == "publish-hf"
        assert args.repo_id == "test/repo"

    def test_dry_run_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--dry-run"])
        assert args.dry_run is True

    def test_skip_upload_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--skip-upload"])
        assert args.skip_upload is True

    def test_gpu_hours_param(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--gpu-hours", "2.5"])
        assert args.gpu_hours == 2.5
