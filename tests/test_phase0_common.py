"""Tests for scripts/phase0/common.py — data loading and hierarchy."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "phase0_mini_cres.json"


@pytest.fixture
def mini_cres() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_build_hierarchy(mini_cres: dict) -> None:
    from scripts.phase0.common import build_hierarchy

    tree = build_hierarchy(mini_cres["cres"])

    assert len(tree.hubs) == 9
    assert len(tree.roots) == 3
    assert set(tree.roots) == {"ROOT-A", "ROOT-B", "ROOT-C"}
    assert tree.depth["ROOT-A"] == 0
    assert tree.depth["HUB-A1"] == 1
    assert tree.parent["HUB-A1"] == "ROOT-A"
    assert "HUB-A1" in tree.children["ROOT-A"]

    path = tree.hierarchy_path("HUB-A1")
    assert path == "Root A > Hub A1"


def test_extract_hub_standard_links(mini_cres: dict) -> None:
    from scripts.phase0.common import build_hierarchy, extract_hub_standard_links

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])

    assert len(links) == 13
    alpha_links = [l for l in links if l.standard_name == "Framework Alpha"]
    assert len(alpha_links) == 3


def test_build_evaluation_corpus() -> None:
    from scripts.phase0.common import (
        HubStandardLink,
        build_evaluation_corpus,
    )

    links = [
        HubStandardLink(cre_id="HUB-A1", cre_name="Hub A1", standard_name="Framework Alpha", section_id="ALPHA-1", section_name="Alpha Section 1"),
        HubStandardLink(cre_id="HUB-A2", cre_name="Hub A2", standard_name="Framework Alpha", section_id="ALPHA-2", section_name="Alpha Section 2"),
        HubStandardLink(cre_id="HUB-B1", cre_name="Hub B1", standard_name="Framework Beta", section_id="BETA-1", section_name="Beta Section 1"),
    ]
    ai_frameworks = {"Framework Alpha", "Framework Beta"}

    corpus = build_evaluation_corpus(links, ai_frameworks, parsed_controls={})

    assert len(corpus) == 3
    assert corpus[0].control_text == "Alpha Section 1"
    assert corpus[0].ground_truth_hub_id == "HUB-A1"
    assert corpus[0].valid_hub_ids == frozenset({"HUB-A1"})
    assert corpus[0].framework_name == "Framework Alpha"
    assert corpus[0].track == "all"


def test_build_evaluation_corpus_with_parsed_controls() -> None:
    from scripts.phase0.common import (
        HubStandardLink,
        build_evaluation_corpus,
    )

    links = [
        HubStandardLink(cre_id="HUB-A1", cre_name="Hub A1", standard_name="Framework Alpha", section_id="ALPHA-1", section_name="Alpha Section 1"),
    ]
    parsed = {("Framework Alpha", "ALPHA-1"): "Full parsed description of Alpha control 1."}

    corpus = build_evaluation_corpus(links, {"Framework Alpha"}, parsed_controls=parsed)

    assert corpus[0].control_text == "Full parsed description of Alpha control 1."
    assert corpus[0].track == "full-text"


def test_build_hub_texts_with_firewall(mini_cres: dict) -> None:
    from scripts.phase0.common import build_hierarchy, extract_hub_standard_links, build_hub_texts

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])

    texts = build_hub_texts(tree, links, held_out_framework="Framework Alpha")

    assert "HUB-A1" in texts
    assert "Alpha Section 1" not in texts["HUB-A1"]
    assert "Beta Section 1" in texts["HUB-A1"]
    assert "Hub A1" in texts["HUB-A1"]


def test_build_hub_texts_without_firewall(mini_cres: dict) -> None:
    from scripts.phase0.common import build_hierarchy, extract_hub_standard_links, build_hub_texts

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])

    texts = build_hub_texts(tree, links, held_out_framework=None)

    assert "HUB-A1" in texts
    assert "Alpha Section 1" in texts["HUB-A1"]


def test_score_predictions() -> None:
    from scripts.phase0.common import score_predictions

    predictions = [
        ["HUB-A", "HUB-B", "HUB-C", "HUB-D", "HUB-E"],
        ["HUB-X", "HUB-A", "HUB-Y", "HUB-Z", "HUB-W"],
        ["HUB-P", "HUB-Q", "HUB-R", "HUB-S", "HUB-T"],
    ]
    ground_truth = ["HUB-A", "HUB-A", "HUB-U"]

    metrics = score_predictions(predictions, ground_truth)

    assert metrics["hit_at_1"] == pytest.approx(1 / 3)
    assert metrics["hit_at_5"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_bootstrap_ci() -> None:
    import numpy as np
    from scripts.phase0.common import bootstrap_ci

    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    result = bootstrap_ci(values, n_resamples=1000, seed=42)

    assert 0.3 < result["mean"] < 0.9
    assert result["ci_low"] < result["mean"]
    assert result["ci_high"] > result["mean"]
    assert result["ci_low"] >= 0.0
    assert result["ci_high"] <= 1.0


def test_lofo_integration(mini_cres: dict) -> None:
    """Full pipeline integration test on mini fixture."""
    from scripts.phase0.common import (
        build_hierarchy,
        extract_hub_standard_links,
        build_evaluation_corpus,
        build_lofo_folds,
        aggregate_lofo_metrics,
    )

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])
    ai_names = {"Framework Alpha", "Framework Beta", "Framework Gamma"}
    corpus = build_evaluation_corpus(links, ai_names, parsed_controls={})

    assert len(corpus) == 8

    folds = build_lofo_folds(tree, links, corpus, ai_names, template="default")
    assert len(folds) == 3

    for fold in folds:
        assert fold.held_out_framework in ai_names
        for item in fold.eval_items:
            assert item.framework_name == fold.held_out_framework

    fold_results: list[dict] = []
    for fold in folds:
        preds: dict[int, list[str]] = {}
        for i, item in enumerate(fold.eval_items):
            preds[i] = [item.ground_truth_hub_id, "WRONG-1", "WRONG-2"]
        fold_results.append(preds)

    metrics = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
    assert metrics["hit_at_1"]["mean"] == 1.0
    assert metrics["hit_at_5"]["mean"] == 1.0
    assert metrics["mrr"]["mean"] == 1.0


def test_lofo_hub_firewall(mini_cres: dict) -> None:
    """Verify hub firewall excludes held-out framework's linked standards."""
    from scripts.phase0.common import (
        build_hierarchy,
        extract_hub_standard_links,
        build_hub_texts,
    )

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])

    texts_no_firewall = build_hub_texts(tree, links, held_out_framework=None)
    texts_alpha_out = build_hub_texts(tree, links, held_out_framework="Framework Alpha")

    assert "Alpha Section 1" in texts_no_firewall.get("HUB-A1", "")
    assert "Alpha Section 1" not in texts_alpha_out.get("HUB-A1", "")
    assert "Beta Section 1" in texts_alpha_out.get("HUB-A1", "")


def test_lofo_wrong_predictions(mini_cres: dict) -> None:
    """Verify metrics are 0 when all predictions are wrong."""
    from scripts.phase0.common import (
        build_hierarchy,
        extract_hub_standard_links,
        build_evaluation_corpus,
        build_lofo_folds,
        aggregate_lofo_metrics,
    )

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])
    ai_names = {"Framework Alpha", "Framework Beta", "Framework Gamma"}
    corpus = build_evaluation_corpus(links, ai_names, parsed_controls={})
    folds = build_lofo_folds(tree, links, corpus, ai_names)

    fold_results: list[dict] = []
    for fold in folds:
        preds: dict[int, list[str]] = {}
        for i in range(len(fold.eval_items)):
            preds[i] = ["NONEXISTENT-1", "NONEXISTENT-2", "NONEXISTENT-3"]
        fold_results.append(preds)

    metrics = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
    assert metrics["hit_at_1"]["mean"] == 0.0
    assert metrics["hit_at_5"]["mean"] == 0.0
    assert metrics["mrr"]["mean"] == 0.0


def test_path_template(mini_cres: dict) -> None:
    """Verify path template includes hierarchy path."""
    from scripts.phase0.common import build_hierarchy, extract_hub_standard_links, build_hub_texts

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])

    texts = build_hub_texts(tree, links, template="path")

    assert "Root A > Hub A1" in texts["HUB-A1"]
    assert "|" in texts["HUB-A1"]


class TestBuildEvaluationCorpusDedup:
    """Test deduplicated evaluation corpus with multi-label ground truth."""

    def test_deduplicates_identical_text_same_hub(self) -> None:
        from scripts.phase0.common import build_evaluation_corpus, HubStandardLink

        links = [
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model",
            ),
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model",
            ),
        ]
        corpus = build_evaluation_corpus(links, {"MITRE ATLAS"}, {})
        assert len(corpus) == 1
        assert corpus[0].valid_hub_ids == frozenset({"hub-1"})

    def test_multi_label_collects_all_valid_hubs(self) -> None:
        from scripts.phase0.common import build_evaluation_corpus, HubStandardLink

        links = [
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model",
            ),
            HubStandardLink(
                cre_id="hub-2", cre_name="Hub 2",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model",
            ),
        ]
        corpus = build_evaluation_corpus(links, {"MITRE ATLAS"}, {})
        assert len(corpus) == 1
        assert corpus[0].valid_hub_ids == frozenset({"hub-1", "hub-2"})
        assert corpus[0].ground_truth_hub_id in {"hub-1", "hub-2"}

    def test_different_texts_not_deduplicated(self) -> None:
        from scripts.phase0.common import build_evaluation_corpus, HubStandardLink

        links = [
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model",
            ),
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="AML.M0009",
                section_name="Different Control",
            ),
        ]
        corpus = build_evaluation_corpus(links, {"MITRE ATLAS"}, {})
        assert len(corpus) == 2

    def test_cross_framework_not_deduplicated(self) -> None:
        from scripts.phase0.common import build_evaluation_corpus, HubStandardLink

        links = [
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="X",
                section_name="Same Name",
            ),
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="NIST AI 100-2", section_id="Y",
                section_name="Same Name",
            ),
        ]
        corpus = build_evaluation_corpus(links, {"MITRE ATLAS", "NIST AI 100-2"}, {})
        assert len(corpus) == 2


class TestMultiLabelScoring:
    """Test multi-label-aware scoring functions."""

    def test_hit1_accepts_any_valid_hub(self) -> None:
        from scripts.phase0.common import score_predictions

        predictions = [["hub-2", "hub-1", "hub-3"]]
        ground_truth = ["hub-1"]
        valid_hub_sets = [frozenset({"hub-1", "hub-2"})]
        result = score_predictions(predictions, ground_truth, valid_hub_sets)
        assert result["hit_at_1"] == 1.0

    def test_hit1_fails_when_no_valid_hub_at_top(self) -> None:
        from scripts.phase0.common import score_predictions

        predictions = [["hub-3", "hub-1"]]
        ground_truth = ["hub-1"]
        valid_hub_sets = [frozenset({"hub-1", "hub-2"})]
        result = score_predictions(predictions, ground_truth, valid_hub_sets)
        assert result["hit_at_1"] == 0.0

    def test_mrr_uses_first_valid_hub(self) -> None:
        from scripts.phase0.common import score_predictions

        predictions = [["hub-3", "hub-2", "hub-1"]]
        ground_truth = ["hub-1"]
        valid_hub_sets = [frozenset({"hub-1", "hub-2"})]
        result = score_predictions(predictions, ground_truth, valid_hub_sets)
        assert result["mrr"] == 0.5  # hub-2 at position 2

    def test_backward_compat_without_valid_sets(self) -> None:
        from scripts.phase0.common import score_predictions

        predictions = [["hub-1", "hub-2"]]
        ground_truth = ["hub-1"]
        result = score_predictions(predictions, ground_truth)
        assert result["hit_at_1"] == 1.0

    def test_ndcg_multi_label(self) -> None:
        from scripts.phase0.common import score_predictions

        predictions = [["hub-3", "hub-2", "hub-1"]]
        ground_truth = ["hub-1"]
        valid_hub_sets = [frozenset({"hub-1", "hub-2"})]
        result = score_predictions(predictions, ground_truth, valid_hub_sets)
        import math
        expected_ndcg = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert result["ndcg_at_10"] == pytest.approx(expected_ndcg)

    def test_hit5_multi_label(self) -> None:
        from scripts.phase0.common import score_predictions

        predictions = [["a", "b", "c", "hub-2", "e"]]
        ground_truth = ["hub-1"]
        valid_hub_sets = [frozenset({"hub-1", "hub-2"})]
        result = score_predictions(predictions, ground_truth, valid_hub_sets)
        assert result["hit_at_5"] == 1.0
