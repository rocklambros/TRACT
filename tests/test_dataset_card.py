"""Tests for HuggingFace dataset card generation."""
from __future__ import annotations

from pathlib import Path

import pytest

from tract.dataset.card import generate_dataset_card
from tract.licensing import NOTICE_FILENAME, published_license_frontmatter

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
        """The card declares `other` with a name and a link, not one identifier.

        It used to declare `cc-by-sa-4.0` over content drawn from 31
        publishers, one of them GPL-3.0 and two of whom reserve
        redistribution. Read from tract.licensing so this test and the model
        card's cannot pin two different answers.
        """
        content = card_path.read_text(encoding="utf-8")
        assert published_license_frontmatter() in content
        assert "license: cc-by-sa-4.0" not in content, (
            "the withdrawn CC BY-SA 4.0 grant is back in the dataset card"
        )

    def test_the_license_link_names_a_file_the_bundle_ships(self) -> None:
        """A link is only useful if it resolves inside the published artifact.

        `license_link: NOTICE` is a relative reference. The dataset bundle
        copies NOTICE into the staging directory, and
        tests/test_dataset_bundle.py asserts it lands there. This test holds
        the two ends of that agreement together: pointing the card at a
        filename the bundler does not write turns it red.
        """
        from tract.licensing import PUBLISHED_LICENSE_LINK

        assert PUBLISHED_LICENSE_LINK == NOTICE_FILENAME

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
            "Known Limitations",
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


class TestNoUnwarrantedReviewClaim:
    """The dataset is not human-reviewed as a whole, and the card must not say it is.

    An earlier card claimed every assignment was expert-reviewed. That was false:
    most rows are links imported from OpenCRE, and the reviewed subset is the
    model predictions only, assessed by one reviewer. The claim was corrected in
    the published artifact, but the generator kept producing the original wording,
    so the next publish would have silently restored it. These tests pin the
    correction to the generator rather than to the artifact it emits.
    """

    def test_headline_does_not_call_the_whole_dataset_human_reviewed(
        self, card_path: Path
    ) -> None:
        headline = card_path.read_text(encoding="utf-8").split("---")[2]
        lowered = headline.lower()
        assert "human-reviewed crosswalk" not in lowered
        assert "not human-reviewed" in lowered

    def test_review_stage_states_a_single_reviewer(self, card_path: Path) -> None:
        text = card_path.read_text(encoding="utf-8")
        assert "**single** cybersecurity domain expert" in text
        assert "inter-rater reliability is unmeasured" in text.lower()

    def test_review_stage_scopes_itself_to_model_predictions(
        self, card_path: Path
    ) -> None:
        text = card_path.read_text(encoding="utf-8")
        assert "not the imported OpenCRE links" in text

    def test_limitations_still_disclose_the_single_reviewer(
        self, card_path: Path
    ) -> None:
        # Defence in depth: the headline and the methodology section can both be
        # rewritten without touching Limitations, so assert it independently.
        text = card_path.read_text(encoding="utf-8").lower()
        assert "single reviewer" in text
