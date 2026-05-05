"""Tests for tract export-canonical CLI subcommand."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from tract.crosswalk.schema import create_database
from tract.crosswalk.store import (
    insert_assignments,
    insert_controls,
    insert_frameworks,
    insert_hubs,
)


@pytest.fixture
def cli_db(tmp_path):
    """DB with data for CLI integration tests."""
    db_path = tmp_path / "cli_test.db"
    create_database(db_path)
    insert_frameworks(db_path, [
        {"id": "fw1", "name": "FW1", "version": "1.0", "fetch_date": "2026-05-04", "control_count": 2},
    ])
    insert_hubs(db_path, [
        {"id": "h1", "name": "Hub 1", "path": "R > H1", "parent_id": None},
    ])
    insert_controls(db_path, [
        {"id": "fw1:c1", "framework_id": "fw1", "section_id": "c1",
         "title": "Control 1", "description": "Desc 1", "full_text": None},
        {"id": "fw1:c2", "framework_id": "fw1", "section_id": "c2",
         "title": "Control 2", "description": "Desc 2", "full_text": None},
    ])
    insert_assignments(db_path, [
        {"control_id": "fw1:c1", "hub_id": "h1", "confidence": 0.8,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
        {"control_id": "fw1:c2", "hub_id": "h1", "confidence": 0.7,
         "in_conformal_set": 1, "is_ood": 0, "provenance": "active_learning_round_2",
         "source_link_id": None, "model_version": None, "review_status": "accepted"},
    ])
    return db_path


class TestExportCanonicalCLI:
    def test_parser_registered(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export-canonical", "--dry-run"])
        assert args.command == "export-canonical"
        assert args.dry_run is True

    def test_parser_with_embeddings_flag(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export-canonical", "--with-embeddings"])
        assert args.with_embeddings is True

    def test_parser_framework_filter(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export-canonical", "--framework", "csa_aicm"])
        assert args.framework == "csa_aicm"

    def test_handler_dry_run(self, cli_db, tmp_path, capsys) -> None:
        from tract.cli import _cmd_export_canonical, build_parser

        output_dir = tmp_path / "output"
        parser = build_parser()
        args = parser.parse_args([
            "export-canonical",
            "--framework", "fw1",
            "--output-dir", str(output_dir),
            "--dry-run",
        ])

        with patch("tract.cli.PHASE1C_CROSSWALK_DB_PATH", cli_db), \
             patch("tract.cli.PHASE1D_ARTIFACTS_PATH", tmp_path / "nonexistent.npz"), \
             patch("tract.export.opencre_names.TRACT_TO_OPENCRE_NAME", {"fw1": "FW1"}), \
             patch("tract.export.opencre_names.HYPERLINK_TEMPLATES", {"fw1": "https://example.com/{section_id}"}):
            _cmd_export_canonical(args)

        assert not (output_dir / "fw1" / "snapshot.json").exists()
        captured = capsys.readouterr()
        assert "fw1" in captured.out
