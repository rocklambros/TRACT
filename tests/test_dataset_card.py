"""Tests for HuggingFace dataset card generation."""
from __future__ import annotations

from pathlib import Path

import pytest

from tract.dataset.card import generate_dataset_card

SAMPLE_FRAMEWORK_METADATA: list[dict] = [
    {
        "framework_id": "fw_alpha",
        "framework_name": "Alpha Framework",
        "total_controls": 50,
        "assigned_controls": 45,
        "assignment_count": 45,
        "coverage_type": "ground_truth",
    },
    {
        "framework_id": "fw_beta",
        "framework_name": "Beta Framework",
        "total_controls": 30,
        "assigned_controls": 25,
        "assignment_count": 25,
        "coverage_type": "model_prediction",
    },
]

SAMPLE_REVIEW_METRICS: dict = {
    "coverage": {
        "total_predictions": 100,
        "reviewed": 90,
        "pending": 10,
        "completion_pct": 90.0,
    },
    "overall": {
        "accepted": 70,
        "rejected": 10,
        "reassigned": 10,
        "accepted_pct": 77.8,
        "rejected_pct": 11.1,
        "reassigned_pct": 11.1,
    },
    "calibration": {
        "quality_score": 0.85,
        "total_reviewed": 20,
        "agreed": 17,
        "disagreements": [],
    },
}

SAMPLE_BUNDLE_STATS: dict = {
    "total_rows": 5000,
    "frameworks": 31,
    "files": ["crosswalk_v1.0.jsonl", "framework_metadata.json"],
}


@pytest.fixture()
def card_path(tmp_path: Path) -> Path:
    """Generate a dataset card and return its path."""
    return generate_dataset_card(
        staging_dir=tmp_path,
        framework_metadata=SAMPLE_FRAMEWORK_METADATA,
        review_metrics=SAMPLE_REVIEW_METRICS,
        bundle_stats=SAMPLE_BUNDLE_STATS,
    )


class TestYAMLFrontmatter:
    def test_starts_with_yaml_delimiter(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert content.startswith("---\n")

    def test_has_language_field(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "language: en" in content

    def test_has_license_field(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "license: cc-by-sa-4.0" in content

    def test_has_task_categories(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "task_categories:" in content
        assert "text-classification" in content

    def test_has_tags(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        for tag in ["security", "crosswalk", "CRE", "AI-security", "framework-mapping"]:
            assert tag in content

    def test_frontmatter_closed(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        parts = content.split("---")
        assert len(parts) >= 3, "YAML frontmatter must be delimited by --- on both sides"


class TestSectionsPresent:
    @pytest.mark.parametrize(
        "section",
        [
            "What Is This",
            "Quick Start",
            "Dataset Structure",
            "Framework Coverage",
            "How It Was Made",
            "Review Methodology",
            "Limitations",
            "License",
            "Citation",
        ],
    )
    def test_section_present(self, card_path: Path, section: str) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert f"## {section}" in content


class TestFrameworkCoverageTable:
    def test_alpha_framework_in_table(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "Alpha Framework" in content
        assert "| Alpha Framework | 50 | 45 | ground_truth |" in content

    def test_beta_framework_in_table(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "Beta Framework" in content
        assert "| Beta Framework | 30 | 25 | model_prediction |" in content

    def test_table_has_header(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "| Framework |" in content


class TestQuickStart:
    def test_load_dataset_present(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "load_dataset" in content
        assert 'load_dataset("rockCO78/tract-crosswalk-dataset")' in content


class TestCitation:
    def test_bibtex_present(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "@dataset{" in content

    def test_bibtex_has_author(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "Lambros, Rock" in content

    def test_bibtex_has_year(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "2026" in content


class TestDynamicContent:
    def test_total_rows_appears(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "5,000" in content

    def test_framework_count_appears(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "31" in content

    def test_review_rates_appear(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "77.8%" in content
        assert "11.1%" in content

    def test_calibration_quality_appears(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "85%" in content
        assert "calibration" in content.lower()

    def test_reviewed_count_appears(self, card_path: Path) -> None:
        content = card_path.read_text(encoding="utf-8")
        assert "90" in content


class TestOutputFile:
    def test_written_to_readme(self, card_path: Path) -> None:
        assert card_path.name == "README.md"
        assert card_path.exists()

    def test_returns_path(self, tmp_path: Path) -> None:
        result = generate_dataset_card(
            staging_dir=tmp_path,
            framework_metadata=SAMPLE_FRAMEWORK_METADATA,
            review_metrics=SAMPLE_REVIEW_METRICS,
            bundle_stats=SAMPLE_BUNDLE_STATS,
        )
        assert isinstance(result, Path)
        assert result == tmp_path / "README.md"
