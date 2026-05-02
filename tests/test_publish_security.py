"""Tests for tract.publish.security — pre-upload secret detection."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


class TestScanForSecrets:

    def test_detects_api_key_in_py(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "script.py").write_text('API_KEY = "sk-abc123def456ghi789jkl012mno345"')
        findings = scan_for_secrets(tmp_path)
        assert len(findings) > 0
        assert any("sk-" in f.matched_text for f in findings)

    def test_detects_hf_token(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "config.yaml").write_text("token: hf_AbCdEfGhIjKlMnOpQrStUvWx")
        findings = scan_for_secrets(tmp_path)
        assert len(findings) > 0

    def test_detects_home_path(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "script.py").write_text('MODEL_PATH = "/home/rock/models/tract"')
        findings = scan_for_secrets(tmp_path)
        assert len(findings) > 0

    def test_detects_email(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "README.md").write_text("Contact: user@example.com")
        findings = scan_for_secrets(tmp_path)
        assert len(findings) > 0

    def test_detects_env_assignment(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "script.py").write_text('HF_TOKEN = "my-secret-token"')
        findings = scan_for_secrets(tmp_path)
        assert len(findings) > 0

    def test_clean_dir_passes(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "script.py").write_text("def predict(text): return text")
        (tmp_path / "README.md").write_text("# Model Card\nA fine model.")
        findings = scan_for_secrets(tmp_path)
        assert findings == []

    def test_ignores_json_data_files(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "cre_hierarchy.json").write_text(
            json.dumps({"hubs": {"h1": {"hierarchy_path": "Home > Security"}}})
        )
        findings = scan_for_secrets(tmp_path)
        assert findings == []

    def test_scans_bridge_report_reviewer_notes(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        report = {
            "candidates": [
                {"status": "accepted", "reviewer_notes": "Reviewed by user@company.com"}
            ]
        }
        (tmp_path / "bridge_report.json").write_text(json.dumps(report))
        findings = scan_for_secrets(tmp_path)
        assert len(findings) > 0

    def test_rejects_git_directory(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")
        findings = scan_for_secrets(tmp_path)
        assert any(".git" in f.matched_text for f in findings)

    def test_rejects_adapter_config(self, tmp_path) -> None:
        from tract.publish.security import scan_for_secrets
        (tmp_path / "adapter_config.json").write_text("{}")
        findings = scan_for_secrets(tmp_path)
        assert any("adapter_config" in f.matched_text for f in findings)
