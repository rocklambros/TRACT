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
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--zero-shot-results", "zs.json"])
        assert args.command == "publish-hf"
        assert args.repo_id == "test/repo"

    def test_dry_run_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--zero-shot-results", "zs.json", "--dry-run"])
        assert args.dry_run is True

    def test_skip_upload_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--zero-shot-results", "zs.json", "--skip-upload"])
        assert args.skip_upload is True

    def test_gpu_hours_param(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--zero-shot-results", "zs.json", "--gpu-hours", "2.5"])
        assert args.gpu_hours == 2.5


class TestAIBOMValidationPin:
    """The AIBOM step clones and executes third-party code as the publishing
    user, on the host that holds the pass store. These tests hold the line on
    what it is allowed to run and what it is allowed to hand that code.
    """

    @staticmethod
    def _fake_subprocess_run(calls, head_sha, run_result=None):
        """Stand in for subprocess.run: record argv, never touch the network."""
        import subprocess

        def _run(cmd, **kwargs):
            cwd = kwargs.get("cwd")
            if cwd is not None:
                # The temp tree is deleted before the test can look at it, so
                # the workdir listing is taken here, while it still exists.
                kwargs = {
                    **kwargs,
                    "cwd_entries": sorted(p.name for p in Path(cwd).iterdir()),
                }
            calls.append((list(cmd), kwargs))
            if cmd[0] == "git":
                stdout = f"{head_sha}\n" if "rev-parse" in cmd else ""
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
            if run_result is None:
                raise AssertionError(f"validator was executed unexpectedly: {cmd}")
            return run_result

        return _run

    def test_branch_name_pin_is_refused_before_anything_runs(
        self, tmp_path, monkeypatch,
    ) -> None:
        import subprocess

        from tract.publish import _validate_aibom

        (tmp_path / "README.md").write_text("# card", encoding="utf-8")
        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", self._fake_subprocess_run(calls, "a" * 40),
        )

        with pytest.raises(ValueError, match="40-character commit SHA"):
            _validate_aibom(tmp_path, "main")

        assert calls == [], "a branch pin must not reach git clone"

    def test_pin_is_checked_before_the_pipeline_does_any_work(self) -> None:
        """publish_to_huggingface calls the same check above step 1, so a bad
        pin costs nothing rather than costing an adapter merge.
        """
        from tract.publish import _require_aibom_pin

        with pytest.raises(ValueError, match="40-character commit SHA"):
            _require_aibom_pin("")
        with pytest.raises(ValueError, match="40-character commit SHA"):
            _require_aibom_pin("A" * 40)
        _require_aibom_pin("0123456789abcdef0123456789abcdef01234567")

    def test_checkout_off_the_pin_refuses_to_execute(
        self, tmp_path, monkeypatch,
    ) -> None:
        import subprocess

        from tract.publish import _validate_aibom

        (tmp_path / "README.md").write_text("# card", encoding="utf-8")
        calls: list = []
        monkeypatch.setattr(
            subprocess, "run", self._fake_subprocess_run(calls, "b" * 40),
        )

        with pytest.raises(ValueError, match="not the pinned"):
            _validate_aibom(tmp_path, "a" * 40)

        assert all(cmd[0] == "git" for cmd, _ in calls)

    def test_validator_gets_no_credentials_and_this_interpreter(
        self, tmp_path, monkeypatch,
    ) -> None:
        import subprocess
        import sys

        from tract.publish import _validate_aibom

        (tmp_path / "README.md").write_text("# card", encoding="utf-8")
        monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-value")
        calls: list = []
        ok = subprocess.CompletedProcess(["validator"], 0, stdout="score 100", stderr="")
        monkeypatch.setattr(
            subprocess, "run",
            self._fake_subprocess_run(calls, "a" * 40, run_result=ok),
        )

        _validate_aibom(tmp_path, "a" * 40)

        cmd, kwargs = calls[-1]
        assert cmd[0] == sys.executable, "must not resolve 'python' through PATH"
        assert cmd[1:3] == ["-m", "aibom_generator"]
        env = kwargs["env"]
        assert "HF_TOKEN" not in env
        assert "ANTHROPIC_API_KEY" not in env
        assert "hf_secret_value" not in "".join(env.values())
        # The workdir is housekeeping, not a boundary — PYTHONPATH puts the
        # clone on sys.path either way. It should still hold only the card.
        assert kwargs["cwd_entries"] == ["README_to_validate.md"]
        assert env["PYTHONPATH"].endswith("aibom-generator")

    def test_nonzero_validator_exit_blocks_the_publish(
        self, tmp_path, monkeypatch,
    ) -> None:
        import subprocess

        from tract.publish import _validate_aibom

        (tmp_path / "README.md").write_text("# card", encoding="utf-8")
        bad = subprocess.CompletedProcess(["validator"], 3, stdout="", stderr="missing fields")
        monkeypatch.setattr(
            subprocess, "run",
            self._fake_subprocess_run([], "a" * 40, run_result=bad),
        )

        with pytest.raises(ValueError, match="AIBOM validation failed"):
            _validate_aibom(tmp_path, "a" * 40)

    def test_flags_default_to_off_and_unpinned(self) -> None:
        from tract.cli import build_parser

        args = build_parser().parse_args(["publish-hf", "--repo-id", "test/repo", "--zero-shot-results", "zs.json"])
        assert args.validate_aibom is False
        assert args.aibom_commit == ""
