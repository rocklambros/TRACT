"""Tests for Phase 3 CLI subcommands — argument parsing only."""
from __future__ import annotations

import pytest

from tract.cli import build_parser


class TestImportGroundTruth:
    def test_dry_run_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["import-ground-truth", "--dry-run"])
        assert args.command == "import-ground-truth"
        assert args.dry_run is True

    def test_default_no_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["import-ground-truth"])
        assert args.command == "import-ground-truth"
        assert args.dry_run is False


class TestReviewExport:
    def test_custom_model_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["review-export", "--model-dir", "/tmp/model"])
        assert args.command == "review-export"
        assert args.model_dir == "/tmp/model"

    def test_custom_output(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["review-export", "--output", "/tmp/out"])
        assert args.command == "review-export"
        assert args.output == "/tmp/out"

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["review-export"])
        assert args.command == "review-export"
        assert args.output is not None
        assert args.model_dir is not None


class TestReviewValidate:
    def test_requires_input(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["review-validate"])

    def test_accepts_input(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["review-validate", "--input", "review.json"])
        assert args.command == "review-validate"
        assert args.input == "review.json"


class TestReviewImport:
    def test_requires_reviewer(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["review-import", "--input", "f.json"])

    def test_requires_input(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["review-import", "--reviewer", "expert_1"])

    def test_accepts_both(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["review-import", "--input", "f.json", "--reviewer", "expert_1"]
        )
        assert args.command == "review-import"
        assert args.input == "f.json"
        assert args.reviewer == "expert_1"


class TestPublishDataset:
    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["publish-dataset"])
        assert args.command == "publish-dataset"
        assert args.dry_run is False
        assert args.skip_upload is False

    def test_dry_run(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["publish-dataset", "--dry-run"])
        assert args.dry_run is True

    def test_skip_upload(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["publish-dataset", "--skip-upload"])
        assert args.skip_upload is True

    def test_custom_repo_id(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["publish-dataset", "--repo-id", "user/repo"])
        assert args.repo_id == "user/repo"


@pytest.mark.parametrize("cmd", [
    "import-ground-truth",
    "review-export",
    "review-validate",
    "review-import",
    "publish-dataset",
])
class TestHelpExitsCleanly:
    def test_help(self, cmd: str) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args([cmd, "--help"])
        assert exc_info.value.code == 0
