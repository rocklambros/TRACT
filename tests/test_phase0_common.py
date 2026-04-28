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
