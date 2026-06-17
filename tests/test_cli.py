"""Tests for tract CLI argument parsing and output."""
from __future__ import annotations

import json
import sys
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest


class TestArgParsing:
    def test_assign_text(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "test control text"])
        assert args.command == "assign"
        assert args.text == "test control text"

    def test_assign_file(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "--file", "controls.txt"])
        assert args.command == "assign"
        assert args.file == "controls.txt"

    def test_assign_top_k(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--top-k", "10"])
        assert args.top_k == 10

    def test_assign_raw_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--raw"])
        assert args.raw is True

    def test_assign_verbose_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--verbose"])
        assert args.verbose is True

    def test_assign_json_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["assign", "text", "--json"])
        assert args.json is True

    def test_compare_frameworks(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["compare", "--framework", "fw_a", "--framework", "fw_b"])
        assert args.command == "compare"
        assert args.framework == ["fw_a", "fw_b"]

    def test_ingest_file(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["ingest", "--file", "framework.json"])
        assert args.command == "ingest"
        assert args.file == "framework.json"

    def test_ingest_force(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["ingest", "--file", "f.json", "--force"])
        assert args.force is True

    def test_export_format(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["export", "--format", "csv"])
        assert args.command == "export"
        assert args.format == "csv"

    def test_hierarchy_hub(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["hierarchy", "--hub", "646-285"])
        assert args.command == "hierarchy"
        assert args.hub == "646-285"

    def test_propose_hubs(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["propose-hubs", "--budget", "20"])
        assert args.command == "propose-hubs"
        assert args.budget == 20

    def test_propose_hubs_name_with_llm(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["propose-hubs", "--name-with-llm"])
        assert args.name_with_llm is True

    def test_review_proposals(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["review-proposals", "--round", "1"])
        assert args.command == "review-proposals"
        assert args.round == 1

    def test_review_dry_run(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["review-proposals", "--round", "1", "--dry-run"])
        assert args.dry_run is True

    def test_tutorial(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["tutorial"])
        assert args.command == "tutorial"

    def test_accept_review(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["accept", "--review", "review.json"])
        assert args.command == "accept"
        assert args.review == "review.json"

    def test_accept_force(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["accept", "--review", "r.json", "--force"])
        assert args.force is True

    def test_all_commands_have_help(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        for cmd in ["assign", "compare", "ingest", "accept", "export", "hierarchy",
                     "propose-hubs", "review-proposals", "tutorial", "validate", "prepare"]:
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([cmd, "--help"])
            assert exc_info.value.code == 0


class TestOutputFormatting:
    def test_format_predictions_table(self) -> None:
        from tract.cli import format_predictions_table
        from tract.inference import HubPrediction

        preds = [
            HubPrediction("646-285", "AI compliance", "R > AI compliance", 0.523, 0.847, True, False),
            HubPrediction("220-442", "Access minimum", "R > Access", 0.412, 0.631, True, False),
        ]
        output = format_predictions_table(preds, raw=False, verbose=False)
        assert "646-285" in output
        assert "AI compliance" in output
        assert "0.847" in output

    def test_format_predictions_raw(self) -> None:
        from tract.cli import format_predictions_table
        from tract.inference import HubPrediction

        preds = [
            HubPrediction("646-285", "AI compliance", "R > AI compliance", 0.523, 0.847, True, False),
        ]
        output = format_predictions_table(preds, raw=True, verbose=False)
        assert "0.523" in output
        assert "Hub Similarity" in output

    def test_format_predictions_json(self) -> None:
        from tract.cli import format_predictions_json
        from tract.inference import HubPrediction

        preds = [
            HubPrediction("646-285", "AI compliance", "R > AI compliance", 0.523, 0.847, True, False),
        ]
        output = format_predictions_json(preds)
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert parsed[0]["hub_id"] == "646-285"


class TestExportOpenCRECLI:
    def test_opencre_flag_recognized(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre", "--dry-run"])
        assert args.opencre is True
        assert args.dry_run is True

    def test_opencre_with_framework_filter(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre", "--framework", "nist_ai_600_1"])
        assert args.opencre is True
        assert args.framework == "nist_ai_600_1"

    def test_opencre_with_output_dir(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre", "--output-dir", "/tmp/test"])
        assert args.output_dir == "/tmp/test"

    def test_opencre_proposals_flag_recognized(self) -> None:
        from tract.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["export", "--opencre-proposals", "--output-dir", "/tmp/test"])
        assert args.opencre_proposals is True


import importlib.util
import tract.cli as cli


def test_require_runtime_exits_before_download(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)  # torch "absent"
    called = {"download": False}
    monkeypatch.setattr(cli, "ensure_deployment_model",
                        lambda: called.__setitem__("download", True), raising=False)
    with pytest.raises(SystemExit) as e:
        cli._require_inference_runtime()
    assert e.value.code == cli.EXIT_MISSING_RUNTIME
    assert called["download"] is False  # guard fired before any resolve/download


def test_download_pins_revision(tmp_path, monkeypatch):
    import argparse
    from unittest.mock import patch
    from tract import config

    monkeypatch.setattr(config, "PHASE1D_DEPLOYMENT_MODEL_DIR", tmp_path / "dm")
    monkeypatch.setattr(config, "PHASE1C_CROSSWALK_DB_PATH", tmp_path / "x.db")
    with patch("huggingface_hub.hf_hub_download") as dl:   # _cmd_download imports it locally
        dl.return_value = str(tmp_path / "f")
        cli._cmd_download(argparse.Namespace(model_only=True, force=True))
        assert dl.call_count >= 1
        for _, kwargs in dl.call_args_list:
            assert kwargs.get("revision") == config.TRACT_MODEL_PINNED_REVISION


def test_version_is_a_nonempty_string():
    import tract
    assert isinstance(tract.__version__, str) and tract.__version__


def test_tract_version_flag(capsys):
    import tract.cli as cli
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("tract ")
    assert "model:" in out


# ── C1: _resolve_model_or_exit maps exceptions to exit codes ─────────────────


class TestResolveModelOrExit:
    """_resolve_model_or_exit translates OfflineModelError → EXIT_OFFLINE
    and ModelIntegrityError → EXIT_INTEGRITY instead of letting them escape
    as uncaught tracebacks.
    """

    def test_offline_error_exits_with_exit_offline(self, monkeypatch, capsys):
        from tract.model_resolver import OfflineModelError

        monkeypatch.setattr(
            "tract.model_resolver.ensure_deployment_model",
            lambda: (_ for _ in ()).throw(OfflineModelError("hub unreachable")),
        )
        with pytest.raises(SystemExit) as exc:
            cli._resolve_model_or_exit()
        assert exc.value.code == cli.EXIT_OFFLINE
        err = capsys.readouterr().err
        assert "hub unreachable" in err

    def test_integrity_error_exits_with_exit_integrity(self, monkeypatch, capsys):
        from tract.model_resolver import ModelIntegrityError

        monkeypatch.setattr(
            "tract.model_resolver.ensure_deployment_model",
            lambda: (_ for _ in ()).throw(ModelIntegrityError("sha mismatch")),
        )
        with pytest.raises(SystemExit) as exc:
            cli._resolve_model_or_exit()
        assert exc.value.code == cli.EXIT_INTEGRITY
        err = capsys.readouterr().err
        assert "sha mismatch" in err

    def test_success_returns_resolved_model(self, monkeypatch):
        from pathlib import Path
        from tract.model_resolver import ResolvedModel

        fake = ResolvedModel(path=Path("/fake"), revision="abc123", source="local")
        monkeypatch.setattr(
            "tract.model_resolver.ensure_deployment_model",
            lambda: fake,
        )
        result = cli._resolve_model_or_exit()
        assert result is fake

    def test_exit_offline_constant_is_imported(self):
        """EXIT_OFFLINE and EXIT_INTEGRITY must be importable from cli module."""
        assert hasattr(cli, "EXIT_OFFLINE"), "EXIT_OFFLINE not imported in cli.py"
        assert hasattr(cli, "EXIT_INTEGRITY"), "EXIT_INTEGRITY not imported in cli.py"
        from tract.config import EXIT_OFFLINE, EXIT_INTEGRITY
        assert cli.EXIT_OFFLINE == EXIT_OFFLINE
        assert cli.EXIT_INTEGRITY == EXIT_INTEGRITY
