"""Tests for tract.publish.model_card — AIBOM-compliant model card generation."""
from __future__ import annotations

from pathlib import Path

import pytest

SAMPLE_FOLD_RESULTS = [
    {"fold": "MITRE ATLAS", "hit1": 0.279, "zs_hit1": 0.273, "n": 43, "hit_any": 0.35},
    {"fold": "NIST AI 100-2", "hit1": 0.429, "zs_hit1": 0.107, "n": 28, "hit_any": 0.50},
    {"fold": "OWASP AI Exchange", "hit1": 0.762, "zs_hit1": 0.619, "n": 63, "hit_any": 0.82},
    {"fold": "OWASP Top10 for LLM", "hit1": 0.333, "zs_hit1": 0.333, "n": 6, "hit_any": 0.50},
    {"fold": "OWASP Top10 for ML", "hit1": 0.714, "zs_hit1": 0.429, "n": 7, "hit_any": 0.86},
]

# The card computes the aggregate hit@1 interval by fold-stratified bootstrap,
# so each fold must carry its per-item indicators. Derived from the fold's own
# hit1 and n to keep the fixture self-consistent.
for _f in SAMPLE_FOLD_RESULTS:
    _hits = round(_f["hit1"] * _f["n"])
    _f["hit1_indicators"] = [1.0] * _hits + [0.0] * (_f["n"] - _hits)

SAMPLE_CALIBRATION = {
    "t_deploy": 0.074,
    "ood_threshold": 0.568,
    "conformal_quantile": 0.997,
}

SAMPLE_ECE = {"ece": 0.079, "ece_ci": {"ci_low": 0.049, "ci_high": 0.111}}

# hub_classification is required, not optional: the card refuses to publish a
# hub-classification table it did not measure. See
# TestBridgeSectionIsMeasuredNotFabricated for why.
SAMPLE_BRIDGE = {
    "counts": {"accepted": 5, "rejected": 58, "total": 63},
    "hub_classification": {
        "ai_only": 83, "trad_only": 380, "naturally_bridged": 0, "unlinked": 59,
    },
}


class TestGenerateModelCard:

    def test_creates_readme(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path,
            fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION,
            ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE,
            gpu_hours=2.5,
        )
        assert (tmp_path / "README.md").exists()

    def test_contains_model_description(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "TRACT" in content
        assert "CRE" in content
        assert "bi-encoder" in content.lower() or "bi_encoder" in content.lower()

    def test_contains_lofo_table(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "MITRE ATLAS" in content
        assert "0.279" in content
        assert "hit@any" in content.lower() or "hit_any" in content.lower()

    def test_contains_calibration(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "0.074" in content or "0.0738" in content
        assert "0.079" in content

    def test_contains_limitations(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "limitation" in content.lower()

    def test_contains_license(self, tmp_path) -> None:
        """The front matter declares `other`, and the body says why.

        This card declared `license: mit` while the dataset card built from the
        same 31 sources declared `cc-by-sa-4.0`. `mit` stated the base model's
        terms as though they covered the fine-tuned weights, the bundled
        hierarchy and the hub descriptions.

        The MIT reference survives in the body because the base model really is
        MIT and its terms travel with the weights. What must not survive is the
        front-matter declaration, so this asserts on both directions.
        """
        from tract.licensing import (
            NOTICE_FILENAME,
            published_license_frontmatter,
        )
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        front_matter = content.split("---", 2)[1]

        assert published_license_frontmatter() in front_matter
        assert "license: mit" not in front_matter, (
            "the withdrawn MIT declaration is back in the model card's YAML"
        )
        assert NOTICE_FILENAME in content, (
            "the card does not point a reader at the per-framework terms"
        )
        assert "MIT" in content

    def test_both_published_cards_declare_the_same_licence(
        self, tmp_path,
    ) -> None:
        """Four declarations across three artifacts stated three answers.

        Rendering both cards and comparing their licence blocks is what makes
        that class of drift impossible to reintroduce by editing one file.
        """
        from tract.dataset.card import generate_dataset_card
        from tract.licensing import published_license_frontmatter
        from tract.publish.model_card import generate_model_card

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        generate_model_card(
            model_dir, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        generate_dataset_card(
            dataset_dir,
            framework_metadata=[],
            review_metrics={},
            bundle_stats={"total_rows": 0, "frameworks": 0},
        )

        block = published_license_frontmatter()
        for card in (model_dir / "README.md", dataset_dir / "README.md"):
            assert block in card.read_text(encoding="utf-8"), (
                f"{card.parent.name} card does not carry the shared licence "
                f"block. The two published cards must not state different "
                f"terms for work drawn from the same sources."
            )

    def test_contains_bridge_summary(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "bridge" in content.lower()
        assert "5" in content  # accepted count

    def test_no_secrets_in_output(self, tmp_path) -> None:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "/home/rock" not in content
        assert "sk-" not in content
        assert "hf_" not in content


class TestErratumSurvivesRegeneration:
    """README.md links to #erratum-2026-08-15 on the published model card.

    The erratum lived only in the uploaded artifact, not in this generator, so
    the next publish would have dropped the section and left the repository
    pointing at an anchor that no longer resolved. These tests pin the erratum
    to the generator and pin the heading text that produces the anchor.
    """

    def _card(self, tmp_path) -> str:
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path,
            fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION,
            ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE,
            gpu_hours=2.5,
        )
        return (tmp_path / "README.md").read_text(encoding="utf-8")

    def test_heading_produces_the_anchor_the_readme_links_to(self, tmp_path) -> None:
        # GitHub and HuggingFace slugify "## Erratum 2026-08-15" to
        # "#erratum-2026-08-15". Changing this heading breaks README.md:48.
        assert "## Erratum 2026-08-15" in self._card(tmp_path)

    def test_states_the_figures_are_withdrawn(self, tmp_path) -> None:
        card = self._card(tmp_path).lower()
        assert "withdrawn" in card
        assert "pre-registered gate" in card

    def test_names_the_specific_audit_failures(self, tmp_path) -> None:
        card = self._card(tmp_path)
        assert "arithmetic on the point estimate" in card
        assert "-0.0004" in card
        assert "1,265" in card

    def test_scopes_the_review_claim(self, tmp_path) -> None:
        # The card is hard-wrapped, so claims span newlines. Collapse whitespace
        # before matching or the assertion depends on where the wrap happens to
        # fall, which is not what is being tested.
        card = " ".join(self._card(tmp_path).split())
        assert "single reviewer" in card
        assert "13 of 20" in card
        assert "Inter-rater reliability is not measured" in card
        assert "imported rather than reviewed here" in card


class TestBridgeSectionIsMeasuredNotFabricated:
    """The hub-classification table was hardcoded prose, and it was wrong.

    The published card carried `| Naturally bridged (both) | 60 | "Data
    poisoning" (linked by both ATLAS and CWE) |`. Measured against the curated
    links, MITRE ATLAS hubs and CWE hubs intersect in ZERO hubs, so the worked
    example named a hub that does not exist. The 60 came from
    BRIDGE_AI_FRAMEWORK_IDS listing only the five rotating frameworks, which
    put ENISA, ETSI and BIML on the traditional side; under the eight-framework
    definition the count is 0, as PRD.md:58 has always said.

    These pin the same contract the rest of this module already enforces via
    `_measured`: a published figure comes from a measurement, or the card
    refuses to build.
    """

    CLASSIFICATION = {
        "ai_only": 83, "trad_only": 380, "naturally_bridged": 0, "unlinked": 59,
    }

    def _card(self, tmp_path, classification=None):
        from tract.publish.model_card import generate_model_card
        summary = dict(SAMPLE_BRIDGE)
        if classification is None:
            summary.pop("hub_classification", None)
        else:
            summary["hub_classification"] = classification
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=summary, gpu_hours=2.5,
        )
        return (tmp_path / "README.md").read_text()

    def test_does_not_present_the_atlas_cwe_example_as_fact(
        self, tmp_path,
    ) -> None:
        # The retraction is allowed — required, even — to quote the claim it
        # withdraws. What must never reappear is the claim standing alone as a
        # table example, which is how it was published.
        card = " ".join(self._card(tmp_path, self.CLASSIFICATION).split())
        assert "| Naturally bridged (both) | 60 |" not in card
        if "ATLAS and CWE" in card:
            assert "superseded" in card.lower()
            assert "was wrong" in card.lower()

    def test_renders_the_measured_classification_counts(self, tmp_path) -> None:
        card = self._card(tmp_path, self.CLASSIFICATION)
        assert "| AI-only | 83 |" in card
        assert "| Traditional-only | 380 |" in card
        assert "| Naturally bridged (both) | 0 |" in card
        assert "| Unlinked (structural) | 59 |" in card

    def test_does_not_carry_the_superseded_literal_counts(self, tmp_path) -> None:
        card = self._card(tmp_path, self.CLASSIFICATION)
        assert "| AI-only | 21 |" not in card
        assert "| Naturally bridged (both) | 60 |" not in card

    def test_similarity_matrix_shape_follows_the_measured_counts(
        self, tmp_path,
    ) -> None:
        # The method section said "21 AI-only hubs x 382 traditional-only hubs
        # (8,022 pairs)" as literals, so it disagreed with the table above it
        # the moment either number moved.
        card = " ".join(self._card(tmp_path, self.CLASSIFICATION).split())
        assert "83 AI-only hubs x 380 traditional-only hubs" in card
        assert "31,540 pairs" in card

    def test_refuses_to_build_without_a_measured_classification(
        self, tmp_path,
    ) -> None:
        with pytest.raises(ValueError, match="hub_classification"):
            self._card(tmp_path, None)

    def test_discloses_that_phase2b_ran_under_the_old_definition(
        self, tmp_path,
    ) -> None:
        card = " ".join(self._card(tmp_path, self.CLASSIFICATION).split())
        assert "21 AI-only hubs" in card
        assert "superseded" in card.lower()
