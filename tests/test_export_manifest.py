"""Tests for export manifest generation."""
from __future__ import annotations

import json

import pytest


class TestExportManifest:
    def test_manifest_has_required_fields(self) -> None:
        from tract.export.manifest import build_manifest
        per_framework = {
            "csa_aicm": {"exported": 10, "excluded_confidence": 5, "excluded_ood": 0,
                         "excluded_ground_truth": 0, "excluded_null_confidence": 0, "excluded_not_accepted": 0},
        }
        staleness = {"status": "pass", "upstream_hub_count": 522}
        manifest = build_manifest(
            per_framework_stats=per_framework, confidence_floor=0.30,
            confidence_overrides={"mitre_atlas": 0.35}, staleness_result=staleness,
            model_adapter_hash="abc123",
        )
        assert manifest["confidence_floor"] == 0.30
        assert manifest["confidence_overrides"] == {"mitre_atlas": 0.35}
        assert manifest["staleness_check"]["status"] == "pass"
        assert manifest["total_exported"] == 10
        assert manifest["total_excluded"] == 5
        assert "export_date" in manifest
        assert "tract_git_sha" in manifest
        assert manifest["model_adapter_hash"] == "abc123"

    def test_manifest_sums_across_frameworks(self) -> None:
        from tract.export.manifest import build_manifest
        per_framework = {
            "fw1": {"exported": 10, "excluded_confidence": 5, "excluded_ood": 1,
                    "excluded_ground_truth": 2, "excluded_null_confidence": 0, "excluded_not_accepted": 0},
            "fw2": {"exported": 20, "excluded_confidence": 3, "excluded_ood": 0,
                    "excluded_ground_truth": 0, "excluded_null_confidence": 1, "excluded_not_accepted": 0},
        }
        manifest = build_manifest(
            per_framework_stats=per_framework, confidence_floor=0.30,
            confidence_overrides={}, staleness_result={"status": "pass", "upstream_hub_count": 100},
            model_adapter_hash="abc",
        )
        assert manifest["total_exported"] == 30
        assert manifest["total_excluded"] == 12

    def test_manifest_is_json_serializable(self) -> None:
        from tract.export.manifest import build_manifest
        manifest = build_manifest(
            per_framework_stats={}, confidence_floor=0.30,
            confidence_overrides={}, staleness_result={"status": "pass", "upstream_hub_count": 0},
            model_adapter_hash="abc",
        )
        serialized = json.dumps(manifest, sort_keys=True, indent=2)
        roundtripped = json.loads(serialized)
        assert roundtripped["confidence_floor"] == 0.30

    def test_manifest_git_sha_format(self) -> None:
        from tract.export.manifest import build_manifest
        manifest = build_manifest(
            per_framework_stats={}, confidence_floor=0.30,
            confidence_overrides={}, staleness_result={"status": "pass", "upstream_hub_count": 0},
            model_adapter_hash="abc",
        )
        sha = manifest["tract_git_sha"]
        assert isinstance(sha, str)
        assert len(sha) >= 7 or sha == "unknown"
