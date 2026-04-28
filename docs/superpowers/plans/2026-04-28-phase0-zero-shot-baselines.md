# Phase 0: Zero-Shot Baseline Experiments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run four zero-shot baseline experiments (embedding similarity, LLM probe, hierarchy paths, hub descriptions) to determine whether CRE hub assignment is feasible and whether a trained model can improve over zero-shot approaches.

**Architecture:** Shared `scripts/phase0/common.py` provides data loading, CRE hierarchy traversal, LOFO evaluation, and bootstrap CI computation. Each experiment is a standalone script that imports from common.py, runs predictions, scores them, and writes results to `results/phase0/`. RunPod GPU pods run embedding experiments; LLM experiments run locally via Claude API.

**Tech Stack:** Python 3.11, torch, transformers, sentence-transformers, anthropic SDK, numpy, scipy, scikit-learn

---

## File Structure

```
scripts/phase0/
├── __init__.py                    # Makes scripts/phase0 importable
├── common.py                      # Shared: data loading, hierarchy, LOFO, metrics, bootstrap
├── exp1_embedding_baseline.py     # Experiment 1: multi-model embedding comparison
├── exp2_llm_probe.py              # Experiment 2: Opus feasibility ceiling
├── exp3_hierarchy_paths.py        # Experiment 3: path-enriched hub features
├── exp4_hub_descriptions.py       # Experiment 4: hub description pilot
├── run_summary.py                 # Aggregate results, print gate table
└── runpod_setup.sh                # RunPod provisioning + teardown

results/phase0/
├── exp1_embedding_baseline.json
├── exp2_llm_probe.json
├── exp3_hierarchy_paths.json
├── exp4_hub_descriptions.json
├── pilot_hub_descriptions.json
└── summary.json

tests/
├── test_phase0_common.py          # Tests for common.py
├── test_phase0_exp1.py            # Tests for experiment 1 scoring logic
└── fixtures/
    └── phase0_mini_cres.json      # Minimal CRE fixture for phase0 tests
```

---

### Task 1: Add Phase 0 dependencies to pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add phase0 optional dependency group**

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "mypy>=1.10",
    "types-PyYAML",
    "types-beautifulsoup4",
    "types-requests",
]
phase0 = [
    "torch>=2.2",
    "transformers>=4.40",
    "sentence-transformers>=3.0",
    "anthropic>=0.25",
    "numpy>=1.26",
    "scipy>=1.12",
    "scikit-learn>=1.4",
]
```

- [ ] **Step 2: Install the new dependencies**

Run: `pip install -e ".[phase0]"`
Expected: All packages install successfully.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add phase0 optional dependencies for zero-shot experiments"
```

---

### Task 2: Create Phase 0 test fixture

**Files:**
- Create: `tests/fixtures/phase0_mini_cres.json`

A minimal CRE hierarchy with 3 roots → 6 hubs, 3 "AI frameworks" with 2 links each = 6 evaluation links. Enough to test LOFO, hub text building, scoring, and bootstrap.

- [ ] **Step 1: Write the fixture**

```json
{
  "cres": [
    {
      "doctype": "CRE",
      "id": "ROOT-A",
      "name": "Root A",
      "tags": [],
      "links": [
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "HUB-A1", "name": "Hub A1", "tags": []}},
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "HUB-A2", "name": "Hub A2", "tags": []}}
      ]
    },
    {
      "doctype": "CRE",
      "id": "ROOT-B",
      "name": "Root B",
      "tags": [],
      "links": [
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "HUB-B1", "name": "Hub B1", "tags": []}},
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "HUB-B2", "name": "Hub B2", "tags": []}}
      ]
    },
    {
      "doctype": "CRE",
      "id": "ROOT-C",
      "name": "Root C",
      "tags": [],
      "links": [
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "HUB-C1", "name": "Hub C1", "tags": []}},
        {"ltype": "Contains", "document": {"doctype": "CRE", "id": "HUB-C2", "name": "Hub C2", "tags": []}}
      ]
    },
    {"doctype": "CRE", "id": "HUB-A1", "name": "Hub A1", "tags": [], "links": [
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Alpha", "sectionID": "ALPHA-1", "section": "Alpha Section 1"}},
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Beta", "sectionID": "BETA-1", "section": "Beta Section 1"}},
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Standard X", "sectionID": "X-1", "section": "X Section 1"}}
    ]},
    {"doctype": "CRE", "id": "HUB-A2", "name": "Hub A2", "tags": [], "links": [
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Alpha", "sectionID": "ALPHA-2", "section": "Alpha Section 2"}},
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Standard Y", "sectionID": "Y-1", "section": "Y Section 1"}}
    ]},
    {"doctype": "CRE", "id": "HUB-B1", "name": "Hub B1", "tags": [], "links": [
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Beta", "sectionID": "BETA-2", "section": "Beta Section 2"}},
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Standard X", "sectionID": "X-2", "section": "X Section 2"}}
    ]},
    {"doctype": "CRE", "id": "HUB-B2", "name": "Hub B2", "tags": [], "links": [
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Gamma", "sectionID": "GAMMA-1", "section": "Gamma Section 1"}},
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Standard Z", "sectionID": "Z-1", "section": "Z Section 1"}}
    ]},
    {"doctype": "CRE", "id": "HUB-C1", "name": "Hub C1", "tags": [], "links": [
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Alpha", "sectionID": "ALPHA-3", "section": "Alpha Section 3"}},
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Gamma", "sectionID": "GAMMA-2", "section": "Gamma Section 2"}}
    ]},
    {"doctype": "CRE", "id": "HUB-C2", "name": "Hub C2", "tags": [], "links": [
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Framework Beta", "sectionID": "BETA-3", "section": "Beta Section 3"}},
      {"ltype": "Linked To", "document": {"doctype": "Standard", "name": "Standard W", "sectionID": "W-1", "section": "W Section 1"}}
    ]}
  ],
  "fetch_timestamp": "2026-04-28T00:00:00Z",
  "total_cres": 9,
  "total_pages": 1
}
```

This gives us:
- 3 roots (ROOT-A, ROOT-B, ROOT-C) each with 2 leaf hubs = 6 hubs total
- 3 "AI frameworks" (Alpha: 3 links, Beta: 3 links, Gamma: 2 links) = 8 AI links
- 4 "non-AI standards" (X, Y, Z, W) = 4 background links
- LOFO holds out one of Alpha/Beta/Gamma at a time

- [ ] **Step 2: Commit**

```bash
git add tests/fixtures/phase0_mini_cres.json
git commit -m "test: add minimal CRE fixture for phase0 experiments"
```

---

### Task 3: Write common.py — Data Loading & CRE Hierarchy

**Files:**
- Create: `scripts/phase0/__init__.py`
- Create: `scripts/phase0/common.py`
- Test: `tests/test_phase0_common.py`

This task builds the first half of common.py: loading CRE data, building the hierarchy tree, extracting hub links, and building the evaluation corpus. The LOFO evaluator, metrics, and bootstrap come in Task 4.

- [ ] **Step 1: Write failing tests for data loading and hierarchy**

Create `tests/test_phase0_common.py`:

```python
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

    assert len(links) == 12
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_phase0_common.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.phase0'`

- [ ] **Step 3: Create `scripts/phase0/__init__.py`**

```python
```

(Empty file — just makes the directory a package.)

- [ ] **Step 4: Write data loading and hierarchy code in `common.py`**

```python
"""Phase 0 shared utilities — data loading, hierarchy, LOFO evaluation, metrics.

Provides the core infrastructure used by all four Phase 0 experiments:
- CRE hierarchy tree construction
- Hub link extraction and normalization
- Evaluation corpus builder with full-text / title-only tracks
- LOFO cross-validation harness
- Scoring metrics (hit@k, MRR, NDCG@10)
- Bootstrap confidence intervals
"""
from __future__ import annotations

import json
import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
OPENCRE_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "opencre" / "opencre_all_cres.json"
HUB_LINKS_PATH: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links.jsonl"
PROCESSED_FRAMEWORKS_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "frameworks"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase0"

# ── AI Framework Names (as they appear in OpenCRE) ─────────────────────────

AI_FRAMEWORK_NAMES: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS",
    "OWASP AI Exchange",
    "NIST AI 100-2",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
})

AI_FRAMEWORK_ID_MAP: Final[dict[str, str]] = {
    "MITRE ATLAS": "mitre_atlas",
    "OWASP AI Exchange": "owasp_ai_exchange",
    "NIST AI 100-2": "nist_ai_100_2",
    "OWASP Top10 for LLM": "owasp_llm_top10",
    "OWASP Top10 for ML": "owasp_ml_top10",
}

PARSED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "mitre_atlas",
    "owasp_ai_exchange",
    "owasp_llm_top10",
})

# ── Bootstrap Settings ──────────────────────────────────────────────────────

BOOTSTRAP_N_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_CI_LEVEL: Final[float] = 0.95
BOOTSTRAP_SEED: Final[int] = 42

# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class CREHierarchy:
    """In-memory CRE hub tree with parent/child/depth lookups."""

    hubs: dict[str, str] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    parent: dict[str, str] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    depth: dict[str, int] = field(default_factory=dict)
    branch: dict[str, str] = field(default_factory=dict)

    def hierarchy_path(self, hub_id: str) -> str:
        """Build 'Root > Parent > ... > Hub' path string."""
        parts: list[str] = []
        current = hub_id
        while current:
            parts.append(self.hubs[current])
            current = self.parent.get(current, "")
        return " > ".join(reversed(parts))

    def leaf_hub_ids(self) -> list[str]:
        """Return hub IDs that have no children."""
        return [hid for hid in self.hubs if hid not in self.children or not self.children[hid]]

    def branch_hub_ids(self, root_id: str) -> list[str]:
        """Return all hub IDs under a given root (including the root)."""
        result: list[str] = []
        queue = deque([root_id])
        while queue:
            current = queue.popleft()
            result.append(current)
            queue.extend(self.children.get(current, []))
        return result


@dataclass
class HubStandardLink:
    """One link between a CRE hub and a standard section."""

    cre_id: str
    cre_name: str
    standard_name: str
    section_id: str
    section_name: str


@dataclass
class EvalItem:
    """One evaluation data point: a control mapped to a ground-truth hub."""

    control_text: str
    ground_truth_hub_id: str
    ground_truth_hub_name: str
    framework_name: str
    section_id: str
    track: str  # "full-text" or "all"


# ── Hierarchy Builder ───────────────────────────────────────────────────────


def build_hierarchy(cres: list[dict]) -> CREHierarchy:
    """Build CRE hierarchy tree from raw OpenCRE CRE list."""
    tree = CREHierarchy()

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        cre_id = cre["id"]
        tree.hubs[cre_id] = cre["name"]

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        cre_id = cre["id"]
        for link in cre.get("links", []):
            if link.get("ltype") == "Contains" and link["document"].get("doctype") == "CRE":
                child_id = link["document"]["id"]
                if child_id in tree.hubs:
                    tree.children.setdefault(cre_id, []).append(child_id)
                    tree.parent[child_id] = cre_id

    contained_ids = set(tree.parent.keys())
    tree.roots = [hid for hid in tree.hubs if hid not in contained_ids]
    tree.roots.sort()

    queue: deque[str] = deque()
    for root_id in tree.roots:
        tree.depth[root_id] = 0
        tree.branch[root_id] = root_id
        queue.append(root_id)

    while queue:
        current = queue.popleft()
        for child_id in tree.children.get(current, []):
            if child_id not in tree.depth:
                tree.depth[child_id] = tree.depth[current] + 1
                tree.branch[child_id] = tree.branch[current]
                queue.append(child_id)

    logger.info(
        "Built hierarchy: %d hubs, %d roots, %d leaves",
        len(tree.hubs), len(tree.roots), len(tree.leaf_hub_ids()),
    )
    return tree


# ── Link Extraction ─────────────────────────────────────────────────────────

TRAINING_LINK_TYPES: Final[frozenset[str]] = frozenset({
    "linked to", "automatically linked to",
})


def extract_hub_standard_links(cres: list[dict]) -> list[HubStandardLink]:
    """Extract all standard-to-hub links from CRE data."""
    links: list[HubStandardLink] = []

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        cre_id = cre["id"]
        cre_name = cre["name"]

        for link in cre.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") != "Standard":
                continue

            ltype = link.get("ltype", "")
            if ltype.lower() not in TRAINING_LINK_TYPES:
                continue

            standard_name = doc.get("name", "")
            if not standard_name:
                continue

            section_id = doc.get("sectionID") or doc.get("section", "")
            section_name = doc.get("section") or doc.get("sectionID", "")

            links.append(HubStandardLink(
                cre_id=cre_id,
                cre_name=cre_name,
                standard_name=standard_name,
                section_id=str(section_id) if section_id else "",
                section_name=str(section_name) if section_name else "",
            ))

    logger.info("Extracted %d standard-to-hub links", len(links))
    return links


# ── Evaluation Corpus ───────────────────────────────────────────────────────


def normalize_owasp_aie_id(s: str) -> str:
    """Normalize OWASP AI Exchange IDs for matching."""
    return s.lower().replace("_", "").replace(" ", "").replace("-", "").replace("/", "")


def load_parsed_controls() -> dict[tuple[str, str], str]:
    """Load parsed control descriptions from processed frameworks.

    Returns dict mapping (standard_name, section_id) -> full description text.
    Uses normalized IDs for OWASP AI Exchange matching.
    """
    parsed: dict[tuple[str, str], str] = {}

    framework_to_standard: dict[str, str] = {
        "mitre_atlas": "MITRE ATLAS",
        "owasp_ai_exchange": "OWASP AI Exchange",
        "owasp_llm_top10": "OWASP Top10 for LLM",
    }

    for fw_id in PARSED_FRAMEWORK_IDS:
        path = PROCESSED_FRAMEWORKS_DIR / f"{fw_id}.json"
        if not path.exists():
            logger.warning("Parsed framework not found: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        standard_name = framework_to_standard[fw_id]
        for ctrl in data.get("controls", []):
            cid = ctrl["control_id"]
            desc = ctrl.get("full_text") or ctrl["description"]
            parsed[(standard_name, cid)] = desc

    logger.info("Loaded %d parsed control descriptions", len(parsed))
    return parsed


def build_evaluation_corpus(
    links: list[HubStandardLink],
    ai_framework_names: set[str] | frozenset[str],
    parsed_controls: dict[tuple[str, str], str],
) -> list[EvalItem]:
    """Build evaluation corpus from AI framework hub links.

    For each AI framework link, uses parsed full-text description if available,
    otherwise falls back to section_name (title-only).
    """
    corpus: list[EvalItem] = []

    for link in links:
        if link.standard_name not in ai_framework_names:
            continue

        control_text: str | None = None
        track = "all"

        key_direct = (link.standard_name, link.section_id)
        if key_direct in parsed_controls:
            control_text = parsed_controls[key_direct]
            track = "full-text"
        elif link.standard_name == "OWASP AI Exchange":
            norm_link = normalize_owasp_aie_id(link.section_id or link.section_name)
            for (std, cid), desc in parsed_controls.items():
                if std == "OWASP AI Exchange" and normalize_owasp_aie_id(cid) == norm_link:
                    control_text = desc
                    track = "full-text"
                    break

        if control_text is None:
            control_text = link.section_name or link.section_id
            track = "all"

        corpus.append(EvalItem(
            control_text=control_text,
            ground_truth_hub_id=link.cre_id,
            ground_truth_hub_name=link.cre_name,
            framework_name=link.standard_name,
            section_id=link.section_id,
            track=track,
        ))

    logger.info(
        "Built evaluation corpus: %d items (%d full-text, %d title-only)",
        len(corpus),
        sum(1 for e in corpus if e.track == "full-text"),
        sum(1 for e in corpus if e.track == "all"),
    )
    return corpus
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_phase0_common.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/phase0/__init__.py scripts/phase0/common.py tests/test_phase0_common.py
git commit -m "feat: add phase0 common.py with hierarchy builder, link extraction, eval corpus"
```

---

### Task 4: Write common.py — LOFO Evaluator, Metrics, and Bootstrap

**Files:**
- Modify: `scripts/phase0/common.py`
- Modify: `tests/test_phase0_common.py`

This task adds the LOFO cross-validation harness, scoring metrics, bootstrap CI computation, and hub text builder to common.py.

- [ ] **Step 1: Write failing tests for hub text builder and LOFO**

Append to `tests/test_phase0_common.py`:

```python
def test_build_hub_texts_with_firewall(mini_cres: dict) -> None:
    from scripts.phase0.common import (
        build_hierarchy,
        extract_hub_standard_links,
        build_hub_texts,
    )

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])

    texts = build_hub_texts(tree, links, held_out_framework="Framework Alpha")

    assert "HUB-A1" in texts
    assert "Framework Alpha" not in texts["HUB-A1"]
    assert "Framework Beta" in texts["HUB-A1"] or "Beta Section 1" in texts["HUB-A1"]
    assert "Hub A1" in texts["HUB-A1"]


def test_build_hub_texts_without_firewall(mini_cres: dict) -> None:
    from scripts.phase0.common import (
        build_hierarchy,
        extract_hub_standard_links,
        build_hub_texts,
    )

    tree = build_hierarchy(mini_cres["cres"])
    links = extract_hub_standard_links(mini_cres["cres"])

    texts = build_hub_texts(tree, links, held_out_framework=None)

    assert "HUB-A1" in texts
    assert "Alpha Section 1" in texts["HUB-A1"] or "Framework Alpha" in texts["HUB-A1"]


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
    from scripts.phase0.common import bootstrap_ci

    np.random.seed(42)
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])

    result = bootstrap_ci(values, n_resamples=1000, seed=42)

    assert 0.3 < result["mean"] < 0.9
    assert result["ci_low"] < result["mean"]
    assert result["ci_high"] > result["mean"]
    assert result["ci_low"] >= 0.0
    assert result["ci_high"] <= 1.0
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/test_phase0_common.py -v`
Expected: 4 old PASS, 4 new FAIL.

- [ ] **Step 3: Add hub text builder, metrics, and bootstrap to common.py**

Append to `scripts/phase0/common.py`:

```python
# ── Hub Text Builder ────────────────────────────────────────────────────────


def build_hub_texts(
    tree: CREHierarchy,
    links: list[HubStandardLink],
    held_out_framework: str | None = None,
    template: str = "default",
    descriptions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build text representation for each hub.

    Templates:
      - "default": "{hub_name}: {linked standard names}"
      - "path": "{hierarchy_path} | {hub_name}: {linked standard names}"
      - "description": "{hub_name}: {description}. Linked: {linked standard names}"

    The held_out_framework parameter enables LOFO firewall — links from that
    framework are excluded from hub text construction.
    """
    hub_standards: dict[str, list[str]] = defaultdict(list)
    for link in links:
        if held_out_framework and link.standard_name == held_out_framework:
            continue
        hub_standards[link.cre_id].append(link.section_name or link.section_id)

    texts: dict[str, str] = {}
    for hub_id, hub_name in tree.hubs.items():
        standard_names = hub_standards.get(hub_id, [])
        linked_text = ", ".join(sorted(set(standard_names))) if standard_names else ""

        if template == "path":
            path = tree.hierarchy_path(hub_id)
            texts[hub_id] = f"{path} | {hub_name}: {linked_text}" if linked_text else f"{path} | {hub_name}"
        elif template == "description" and descriptions and hub_id in descriptions:
            desc = descriptions[hub_id]
            texts[hub_id] = f"{hub_name}: {desc}. Linked: {linked_text}" if linked_text else f"{hub_name}: {desc}"
        else:
            texts[hub_id] = f"{hub_name}: {linked_text}" if linked_text else hub_name

    return texts


# ── Scoring Metrics ─────────────────────────────────────────────────────────


def _reciprocal_rank(predicted: list[str], truth: str) -> float:
    """Reciprocal rank of truth in predicted list. 0 if not found."""
    for i, p in enumerate(predicted):
        if p == truth:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(predicted: list[str], truth: str, k: int = 10) -> float:
    """NDCG@k for single-relevant-item retrieval."""
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p == truth:
            dcg = 1.0 / math.log2(i + 2)
            break
    idcg = 1.0 / math.log2(2)
    return dcg / idcg


def score_predictions(
    predictions: list[list[str]],
    ground_truth: list[str],
) -> dict[str, float]:
    """Score ranked predictions against ground truth hub IDs.

    Args:
        predictions: List of ranked hub ID lists (one per eval item).
        ground_truth: List of correct hub IDs (one per eval item).

    Returns:
        Dict with hit_at_1, hit_at_5, mrr, ndcg_at_10.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("Empty predictions list")
    if n != len(ground_truth):
        raise ValueError(f"Prediction count {n} != ground truth count {len(ground_truth)}")

    hit1 = sum(1 for pred, gt in zip(predictions, ground_truth) if pred and pred[0] == gt) / n
    hit5 = sum(1 for pred, gt in zip(predictions, ground_truth) if gt in pred[:5]) / n
    mrr = sum(_reciprocal_rank(pred, gt) for pred, gt in zip(predictions, ground_truth)) / n
    ndcg = sum(_ndcg_at_k(pred, gt) for pred, gt in zip(predictions, ground_truth)) / n

    return {
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "mrr": mrr,
        "ndcg_at_10": ndcg,
    }


# ── Bootstrap Confidence Intervals ──────────────────────────────────────────


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for the mean of values."""
    rng = np.random.default_rng(seed)
    n = len(values)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[indices].mean(axis=1)

    alpha = (1 - ci_level) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return {
        "mean": float(values.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def bootstrap_paired_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute bootstrap CI for the difference (B - A) on paired samples."""
    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length for paired bootstrap")

    rng = np.random.default_rng(seed)
    n = len(values_a)
    indices = rng.integers(0, n, size=(n_resamples, n))

    boot_a = values_a[indices].mean(axis=1)
    boot_b = values_b[indices].mean(axis=1)
    boot_deltas = boot_b - boot_a

    alpha = (1 - ci_level) / 2
    return {
        "delta_mean": float((values_b - values_a).mean()),
        "ci_low": float(np.percentile(boot_deltas, 100 * alpha)),
        "ci_high": float(np.percentile(boot_deltas, 100 * (1 - alpha))),
    }


# ── LOFO Cross-Validation ──────────────────────────────────────────────────

@dataclass
class LOFOFold:
    """One fold of LOFO cross-validation."""

    held_out_framework: str
    eval_items: list[EvalItem]
    hub_texts: dict[str, str]
    hub_ids: list[str]


def build_lofo_folds(
    tree: CREHierarchy,
    links: list[HubStandardLink],
    corpus: list[EvalItem],
    ai_framework_names: set[str] | frozenset[str],
    template: str = "default",
    descriptions: dict[str, str] | None = None,
) -> list[LOFOFold]:
    """Build LOFO folds — one per AI framework.

    For each fold:
    1. Hold out one AI framework
    2. Rebuild hub texts excluding that framework's links
    3. Collect eval items for the held-out framework
    """
    folds: list[LOFOFold] = []
    all_hub_ids = sorted(tree.hubs.keys())

    for framework_name in sorted(ai_framework_names):
        eval_items = [e for e in corpus if e.framework_name == framework_name]
        if not eval_items:
            logger.warning("No eval items for %s, skipping fold", framework_name)
            continue

        hub_texts = build_hub_texts(
            tree, links,
            held_out_framework=framework_name,
            template=template,
            descriptions=descriptions,
        )

        folds.append(LOFOFold(
            held_out_framework=framework_name,
            eval_items=eval_items,
            hub_texts=hub_texts,
            hub_ids=all_hub_ids,
        ))

        logger.info(
            "LOFO fold %s: %d eval items, %d hubs",
            framework_name, len(eval_items), len(hub_texts),
        )

    return folds


def aggregate_lofo_metrics(
    fold_results: list[dict[str, list[list[str]]]],
    folds: list[LOFOFold],
    track_filter: str | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate LOFO fold predictions into metrics with bootstrap CIs.

    Args:
        fold_results: List of dicts, one per fold, each mapping
            eval item index -> ranked hub ID predictions.
        folds: The LOFO folds (for ground truth and track info).
        track_filter: If set, only include items with this track ("full-text" or "all").
            None means include all items (the all-198 track).

    Returns:
        Dict with hit_at_1, hit_at_5, mrr, ndcg_at_10, each containing
        mean, ci_low, ci_high from bootstrap.
    """
    all_predictions: list[list[str]] = []
    all_ground_truth: list[str] = []

    for fold, results in zip(folds, fold_results):
        for i, item in enumerate(fold.eval_items):
            if track_filter == "full-text" and item.track != "full-text":
                continue
            all_predictions.append(results.get(str(i), results.get(i, [])))
            all_ground_truth.append(item.ground_truth_hub_id)

    if not all_predictions:
        return {m: {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0} for m in ["hit_at_1", "hit_at_5", "mrr", "ndcg_at_10"]}

    hit1_arr = np.array([
        1.0 if pred and pred[0] == gt else 0.0
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])
    hit5_arr = np.array([
        1.0 if gt in pred[:5] else 0.0
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])
    mrr_arr = np.array([
        _reciprocal_rank(pred, gt)
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])
    ndcg_arr = np.array([
        _ndcg_at_k(pred, gt)
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])

    return {
        "hit_at_1": bootstrap_ci(hit1_arr),
        "hit_at_5": bootstrap_ci(hit5_arr),
        "mrr": bootstrap_ci(mrr_arr),
        "ndcg_at_10": bootstrap_ci(ndcg_arr),
    }


# ── I/O Helpers ─────────────────────────────────────────────────────────────


def load_opencre_cres(path: Path | None = None) -> list[dict]:
    """Load CRE list from OpenCRE JSON dump."""
    p = path or OPENCRE_PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["cres"]


def save_results(results: dict, filename: str) -> Path:
    """Save results dict to results/phase0/ as formatted JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename

    import tempfile
    import os
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(results, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Saved results to %s", path)
    return path
```

- [ ] **Step 4: Run all tests**

Run: `python -m pytest tests/test_phase0_common.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase0/common.py tests/test_phase0_common.py
git commit -m "feat: add LOFO evaluator, scoring metrics, bootstrap CIs to common.py"
```

---

### Task 5: Experiment 1 — Multi-Model Embedding Baseline

**Files:**
- Create: `scripts/phase0/exp1_embedding_baseline.py`
- Test: `tests/test_phase0_exp1.py`

**Files referenced:**
- Read: `scripts/phase0/common.py` (imports all shared utilities)
- Read: `data/raw/opencre/opencre_all_cres.json` (CRE data)
- Read: `data/training/hub_links.jsonl` (hub links)
- Read: `data/processed/frameworks/*.json` (parsed controls)

- [ ] **Step 1: Write failing test for bi-encoder scoring logic**

Create `tests/test_phase0_exp1.py`:

```python
"""Tests for experiment 1 scoring logic (not model inference)."""
from __future__ import annotations

import numpy as np
import pytest


def test_rank_by_cosine_similarity() -> None:
    from scripts.phase0.exp1_embedding_baseline import rank_by_cosine_similarity

    hub_ids = ["HUB-A", "HUB-B", "HUB-C"]
    hub_embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    control_embedding = np.array([0.9, 0.1, 0.0])

    ranked = rank_by_cosine_similarity(control_embedding, hub_embeddings, hub_ids)

    assert ranked[0] == "HUB-A"
    assert len(ranked) == 3


def test_rank_by_nli_entailment() -> None:
    from scripts.phase0.exp1_embedding_baseline import rank_by_nli_scores

    hub_ids = ["HUB-A", "HUB-B", "HUB-C"]
    scores = np.array([0.2, 0.9, 0.5])

    ranked = rank_by_nli_scores(scores, hub_ids)

    assert ranked[0] == "HUB-B"
    assert ranked[1] == "HUB-C"
    assert ranked[2] == "HUB-A"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_phase0_exp1.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write experiment 1 script**

Create `scripts/phase0/exp1_embedding_baseline.py`:

```python
"""Experiment 1: Multi-model embedding baseline for CRE hub assignment.

Evaluates three off-the-shelf encoders on the hub assignment task:
- BAAI/bge-large-en-v1.5 (bi-encoder, 335M params, 1024 dim)
- Alibaba-NLP/gte-large-en-v1.5 (bi-encoder, 434M params, 1024 dim)
- cross-encoder/nli-deberta-v3-large (NLI cross-encoder, 304M params)

Bi-encoders rank hubs by cosine similarity.
Cross-encoder ranks by NLI entailment probability.
LOFO cross-validation with hub firewall on 5 AI frameworks.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Final

import numpy as np
import torch

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    EvalItem,
    LOFOFold,
    aggregate_lofo_metrics,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model Config ────────────────────────────────────────────────────────────

BIENCODER_MODELS: Final[list[dict[str, str]]] = [
    {"name": "bge-large-v1.5", "hf_id": "BAAI/bge-large-en-v1.5"},
    {"name": "gte-large-v1.5", "hf_id": "Alibaba-NLP/gte-large-en-v1.5"},
]

CROSSENCODER_MODEL: Final[dict[str, str]] = {
    "name": "deberta-v3-nli",
    "hf_id": "cross-encoder/nli-deberta-v3-large",
}

BATCH_SIZE: Final[int] = 32


# ── Ranking Functions ───────────────────────────────────────────────────────


def rank_by_cosine_similarity(
    control_embedding: np.ndarray,
    hub_embeddings: np.ndarray,
    hub_ids: list[str],
) -> list[str]:
    """Rank hub IDs by cosine similarity to a single control embedding."""
    control_norm = control_embedding / (np.linalg.norm(control_embedding) + 1e-10)
    hub_norms = hub_embeddings / (np.linalg.norm(hub_embeddings, axis=1, keepdims=True) + 1e-10)
    similarities = hub_norms @ control_norm
    ranked_indices = np.argsort(-similarities)
    return [hub_ids[i] for i in ranked_indices]


def rank_by_nli_scores(
    scores: np.ndarray,
    hub_ids: list[str],
) -> list[str]:
    """Rank hub IDs by NLI entailment scores (descending)."""
    ranked_indices = np.argsort(-scores)
    return [hub_ids[i] for i in ranked_indices]


# ── Bi-Encoder Pipeline ────────────────────────────────────────────────────


def run_biencoder(
    model_config: dict[str, str],
    folds: list[LOFOFold],
    corpus: list[EvalItem],
    device: str,
) -> list[dict]:
    """Run bi-encoder evaluation across all LOFO folds."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading bi-encoder: %s", model_config["hf_id"])
    model = SentenceTransformer(model_config["hf_id"], device=device)

    fold_results: list[dict] = []

    for fold in folds:
        logger.info("Fold: held out %s (%d items)", fold.held_out_framework, len(fold.eval_items))

        hub_texts_ordered = [fold.hub_texts[hid] for hid in fold.hub_ids]
        hub_embeddings = model.encode(
            hub_texts_ordered,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        hub_embeddings = np.array(hub_embeddings)

        control_texts = [item.control_text for item in fold.eval_items]
        control_embeddings = model.encode(
            control_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        control_embeddings = np.array(control_embeddings)

        predictions: dict[int, list[str]] = {}
        for i, ctrl_emb in enumerate(control_embeddings):
            ranked = rank_by_cosine_similarity(ctrl_emb, hub_embeddings, fold.hub_ids)
            predictions[i] = ranked[:20]

        fold_results.append(predictions)

    return fold_results


# ── Cross-Encoder Pipeline ──────────────────────────────────────────────────


def run_crossencoder(
    model_config: dict[str, str],
    folds: list[LOFOFold],
    corpus: list[EvalItem],
    device: str,
) -> list[dict]:
    """Run NLI cross-encoder evaluation across all LOFO folds."""
    from sentence_transformers import CrossEncoder

    logger.info("Loading cross-encoder: %s", model_config["hf_id"])
    model = CrossEncoder(model_config["hf_id"], device=device)

    fold_results: list[dict] = []

    for fold in folds:
        logger.info("Fold: held out %s (%d items)", fold.held_out_framework, len(fold.eval_items))
        n_hubs = len(fold.hub_ids)
        hub_texts_ordered = [fold.hub_texts[hid] for hid in fold.hub_ids]

        predictions: dict[int, list[str]] = {}

        for i, item in enumerate(fold.eval_items):
            pairs = [(item.control_text, ht) for ht in hub_texts_ordered]

            raw_scores = model.predict(pairs, batch_size=BATCH_SIZE, show_progress_bar=False)

            if raw_scores.ndim == 2:
                entailment_scores = raw_scores[:, -1]
            else:
                entailment_scores = raw_scores

            ranked = rank_by_nli_scores(np.array(entailment_scores), fold.hub_ids)
            predictions[i] = ranked[:20]

            if (i + 1) % 10 == 0:
                logger.info("  Cross-encoder progress: %d/%d", i + 1, len(fold.eval_items))

        fold_results.append(predictions)

    return fold_results


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Multi-model embedding baseline")
    parser.add_argument("--model", choices=["bge", "gte", "deberta", "all"], default="all",
                        help="Which model to run (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for model inference")
    args = parser.parse_args()

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    logger.info("Building LOFO folds (default template)...")
    folds = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")

    results: dict = {
        "experiment": "exp1_embedding_baseline",
        "models": {},
        "device": args.device,
    }

    models_to_run: list[tuple[str, dict[str, str], str]] = []
    if args.model in ("bge", "all"):
        models_to_run.append(("bge-large-v1.5", BIENCODER_MODELS[0], "biencoder"))
    if args.model in ("gte", "all"):
        models_to_run.append(("gte-large-v1.5", BIENCODER_MODELS[1], "biencoder"))
    if args.model in ("deberta", "all"):
        models_to_run.append(("deberta-v3-nli", CROSSENCODER_MODEL, "crossencoder"))

    for model_name, model_config, model_type in models_to_run:
        logger.info("=" * 60)
        logger.info("Running %s (%s)", model_name, model_type)
        start_time = time.time()

        if model_type == "biencoder":
            fold_results = run_biencoder(model_config, folds, corpus, args.device)
        else:
            fold_results = run_crossencoder(model_config, folds, corpus, args.device)

        elapsed = time.time() - start_time
        logger.info("%s completed in %.1f seconds", model_name, elapsed)

        metrics_all198 = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
        metrics_fulltext = aggregate_lofo_metrics(fold_results, folds, track_filter="full-text")

        per_fold: list[dict] = []
        for fold, preds in zip(folds, fold_results):
            fold_preds_list = [preds[i] for i in range(len(fold.eval_items))]
            fold_gt = [item.ground_truth_hub_id for item in fold.eval_items]
            from scripts.phase0.common import score_predictions
            fold_metrics = score_predictions(fold_preds_list, fold_gt)
            per_fold.append({
                "framework": fold.held_out_framework,
                "n_items": len(fold.eval_items),
                "metrics": fold_metrics,
            })

        results["models"][model_name] = {
            "hf_id": model_config["hf_id"],
            "model_type": model_type,
            "elapsed_seconds": round(elapsed, 1),
            "all_198": metrics_all198,
            "full_text": metrics_fulltext,
            "per_fold": per_fold,
        }

        logger.info(
            "%s all-198: hit@1=%.3f [%.3f, %.3f], hit@5=%.3f [%.3f, %.3f]",
            model_name,
            metrics_all198["hit_at_1"]["mean"],
            metrics_all198["hit_at_1"]["ci_low"],
            metrics_all198["hit_at_1"]["ci_high"],
            metrics_all198["hit_at_5"]["mean"],
            metrics_all198["hit_at_5"]["ci_low"],
            metrics_all198["hit_at_5"]["ci_high"],
        )

    save_results(results, "exp1_embedding_baseline.json")
    logger.info("Experiment 1 complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

Run: `python -m pytest tests/test_phase0_exp1.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase0/exp1_embedding_baseline.py tests/test_phase0_exp1.py
git commit -m "feat: add experiment 1 multi-model embedding baseline script"
```

---

### Task 6: Experiment 2 — LLM Probe (Opus)

**Files:**
- Create: `scripts/phase0/exp2_llm_probe.py`

This runs on the local machine using the Claude API. Two-stage approach per control: (1) branch shortlisting — one API call per root branch, asking for up to 20 candidates; (2) final ranking — one API call with all shortlisted candidates.

**Files referenced:**
- Read: `scripts/phase0/common.py` (all shared utilities)

- [ ] **Step 1: Write the experiment 2 script**

Create `scripts/phase0/exp2_llm_probe.py`:

```python
"""Experiment 2: LLM Probe — Opus feasibility ceiling for CRE hub assignment.

Two-stage approach per held-out control:
  Stage 1: Branch shortlisting — 5 API calls (one per CRE root branch),
           each asks Opus to select up to 20 candidate hubs.
  Stage 2: Final ranking — 1 API call with all shortlisted candidates,
           asks Opus to rank top 10.

Uses LOFO cross-validation with hub firewall.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import time
from typing import Final

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    CREHierarchy,
    EvalItem,
    LOFOFold,
    aggregate_lofo_metrics,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
    score_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL: Final[str] = "claude-opus-4-20250514"
MAX_CONCURRENT_REQUESTS: Final[int] = 5
SHORTLIST_PER_BRANCH: Final[int] = 20
FINAL_TOP_K: Final[int] = 10


def get_api_key() -> str:
    """Retrieve Anthropic API key from pass password manager."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    result = subprocess.run(
        ["pass", "anthropic/api_key"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def build_branch_prompt(
    control_text: str,
    branch_root_name: str,
    branch_hubs: list[dict[str, str]],
) -> str:
    """Build Stage 1 prompt for one branch."""
    hub_list = "\n".join(
        f"- [{h['id']}] {h['path']} | Linked: {h['linked']}"
        for h in branch_hubs
    )
    return f"""You are classifying a security control into the Common Requirements Enumeration (CRE) taxonomy.

CONTROL TEXT:
{control_text}

BRANCH: {branch_root_name}
The following CRE hubs belong to this branch. Each hub has an ID, hierarchy path, and linked standard names.

{hub_list}

Which of these hubs are relevant to this control? Return up to {SHORTLIST_PER_BRANCH} candidates ranked by relevance.

Respond with ONLY a JSON array of hub IDs, most relevant first. Example:
["123-456", "789-012", "345-678"]"""


def build_final_prompt(
    control_text: str,
    candidates: list[dict[str, str]],
) -> str:
    """Build Stage 2 prompt for final ranking."""
    candidate_list = "\n".join(
        f"- [{c['id']}] {c['path']} | Linked: {c['linked']}"
        for c in candidates
    )
    return f"""You are classifying a security control into the Common Requirements Enumeration (CRE) taxonomy.

CONTROL TEXT:
{control_text}

CANDIDATE HUBS (shortlisted from all branches):
{candidate_list}

Rank the top {FINAL_TOP_K} most relevant hubs for this control.

Respond with ONLY a JSON array of hub IDs, most relevant first. Example:
["123-456", "789-012", "345-678"]"""


def parse_hub_ids_from_response(text: str) -> list[str]:
    """Extract hub ID list from LLM response text."""
    json_match = re.search(r'\[.*?\]', text, re.DOTALL)
    if json_match:
        try:
            ids = json.loads(json_match.group())
            return [str(i) for i in ids if isinstance(i, (str, int))]
        except json.JSONDecodeError:
            pass
    return re.findall(r'[\d]+-[\d]+', text)


async def call_opus(
    client: object,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> str:
    """Make one Opus API call with rate limiting."""
    async with semaphore:
        response = await client.messages.create(
            model=MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


async def predict_single_control(
    client: object,
    control_text: str,
    tree: CREHierarchy,
    fold: LOFOFold,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Run two-stage prediction for one control."""
    all_candidates: list[dict[str, str]] = []

    branch_tasks = []
    for root_id in tree.roots:
        branch_hub_ids = tree.branch_hub_ids(root_id)
        branch_hubs = []
        for hid in branch_hub_ids:
            linked_names = []
            hub_text = fold.hub_texts.get(hid, "")
            if ":" in hub_text:
                linked_part = hub_text.split(":", 1)[1].strip()
                if linked_part:
                    linked_names = [n.strip() for n in linked_part.split(",")]
            branch_hubs.append({
                "id": hid,
                "path": tree.hierarchy_path(hid),
                "linked": ", ".join(linked_names) if linked_names else "(none)",
            })

        prompt = build_branch_prompt(
            control_text, tree.hubs[root_id], branch_hubs,
        )
        branch_tasks.append((root_id, prompt, branch_hubs))

    stage1_results: list[list[str]] = []
    for root_id, prompt, branch_hubs in branch_tasks:
        response_text = await call_opus(client, prompt, semaphore)
        hub_ids = parse_hub_ids_from_response(response_text)
        stage1_results.append(hub_ids[:SHORTLIST_PER_BRANCH])
        valid_ids = set(h["id"] for h in branch_hubs)
        for hid in hub_ids[:SHORTLIST_PER_BRANCH]:
            if hid in valid_ids:
                linked = next((h["linked"] for h in branch_hubs if h["id"] == hid), "(none)")
                all_candidates.append({
                    "id": hid,
                    "path": tree.hierarchy_path(hid),
                    "linked": linked,
                })

    if not all_candidates:
        return []

    final_prompt = build_final_prompt(control_text, all_candidates)
    final_response = await call_opus(client, final_prompt, semaphore)
    final_ranked = parse_hub_ids_from_response(final_response)

    valid_candidate_ids = set(c["id"] for c in all_candidates)
    return [hid for hid in final_ranked if hid in valid_candidate_ids][:FINAL_TOP_K]


async def run_fold_async(
    client: object,
    fold: LOFOFold,
    tree: CREHierarchy,
    max_concurrent: int,
) -> dict[int, list[str]]:
    """Run all controls in one LOFO fold."""
    semaphore = asyncio.Semaphore(max_concurrent)
    predictions: dict[int, list[str]] = {}

    for i, item in enumerate(fold.eval_items):
        ranked = await predict_single_control(
            client, item.control_text, tree, fold, semaphore,
        )
        predictions[i] = ranked
        logger.info(
            "  Control %d/%d: predicted %d hubs (truth: %s)",
            i + 1, len(fold.eval_items), len(ranked), item.ground_truth_hub_id,
        )

    return predictions


async def run_experiment(max_concurrent: int) -> dict:
    """Run the full LLM probe experiment."""
    import anthropic

    api_key = get_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    logger.info("Building LOFO folds...")
    folds = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")

    fold_results: list[dict[int, list[str]]] = []
    raw_predictions: list[dict] = []
    start_time = time.time()

    for fold in folds:
        logger.info("=" * 60)
        logger.info("Fold: held out %s (%d items)", fold.held_out_framework, len(fold.eval_items))
        fold_start = time.time()

        preds = await run_fold_async(client, fold, tree, max_concurrent)
        fold_results.append(preds)

        for i, item in enumerate(fold.eval_items):
            raw_predictions.append({
                "framework": fold.held_out_framework,
                "section_id": item.section_id,
                "ground_truth": item.ground_truth_hub_id,
                "predicted": preds.get(i, []),
                "hit_at_1": preds.get(i, [""])[0] == item.ground_truth_hub_id if preds.get(i) else False,
            })

        fold_elapsed = time.time() - fold_start
        logger.info("Fold completed in %.1f seconds", fold_elapsed)

    total_elapsed = time.time() - start_time
    logger.info("Total elapsed: %.1f seconds", total_elapsed)

    metrics_all198 = aggregate_lofo_metrics(fold_results, folds, track_filter=None)
    metrics_fulltext = aggregate_lofo_metrics(fold_results, folds, track_filter="full-text")

    per_fold: list[dict] = []
    for fold, preds in zip(folds, fold_results):
        fold_preds_list = [preds[i] for i in range(len(fold.eval_items))]
        fold_gt = [item.ground_truth_hub_id for item in fold.eval_items]
        fold_metrics = score_predictions(fold_preds_list, fold_gt)
        per_fold.append({
            "framework": fold.held_out_framework,
            "n_items": len(fold.eval_items),
            "metrics": fold_metrics,
        })

    results = {
        "experiment": "exp2_llm_probe",
        "model": MODEL,
        "elapsed_seconds": round(total_elapsed, 1),
        "all_198": metrics_all198,
        "full_text": metrics_fulltext,
        "per_fold": per_fold,
        "raw_predictions": raw_predictions,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2: Opus LLM Probe")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS,
                        help="Max concurrent API requests")
    args = parser.parse_args()

    results = asyncio.run(run_experiment(args.max_concurrent))
    save_results(results, "exp2_llm_probe.json")

    logger.info(
        "Opus all-198: hit@1=%.3f [%.3f, %.3f], hit@5=%.3f [%.3f, %.3f]",
        results["all_198"]["hit_at_1"]["mean"],
        results["all_198"]["hit_at_1"]["ci_low"],
        results["all_198"]["hit_at_1"]["ci_high"],
        results["all_198"]["hit_at_5"]["mean"],
        results["all_198"]["hit_at_5"]["ci_low"],
        results["all_198"]["hit_at_5"]["ci_high"],
    )
    logger.info("Experiment 2 complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/phase0/exp2_llm_probe.py
git commit -m "feat: add experiment 2 Opus LLM probe script"
```

---

### Task 7: Experiment 3 — Hierarchy Path Features

**Files:**
- Create: `scripts/phase0/exp3_hierarchy_paths.py`

Re-runs experiment 1's bi-encoder pipeline with path-enriched hub text template. Compares baseline vs path-enriched using paired bootstrap.

- [ ] **Step 1: Write experiment 3 script**

Create `scripts/phase0/exp3_hierarchy_paths.py`:

```python
"""Experiment 3: Hierarchy path features for CRE hub assignment.

Re-runs experiment 1's bi-encoder pipeline (BGE and GTE) with path-enriched
hub text: "{hierarchy_path} | {hub_name}: {linked standard names}".

Reports side-by-side metrics and paired bootstrap deltas vs baseline.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Final

import numpy as np
import torch

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    aggregate_lofo_metrics,
    bootstrap_paired_delta,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
    score_predictions,
    _reciprocal_rank,
    _ndcg_at_k,
)
from scripts.phase0.exp1_embedding_baseline import (
    BIENCODER_MODELS,
    run_biencoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_per_item_metrics(
    fold_results: list[dict],
    folds: list,
    track_filter: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-item metric arrays for paired bootstrap."""
    hit1_list: list[float] = []
    hit5_list: list[float] = []
    mrr_list: list[float] = []
    ndcg_list: list[float] = []

    for fold, results in zip(folds, fold_results):
        for i, item in enumerate(fold.eval_items):
            if track_filter == "full-text" and item.track != "full-text":
                continue
            pred = results.get(i, [])
            gt = item.ground_truth_hub_id
            hit1_list.append(1.0 if pred and pred[0] == gt else 0.0)
            hit5_list.append(1.0 if gt in pred[:5] else 0.0)
            mrr_list.append(_reciprocal_rank(pred, gt))
            ndcg_list.append(_ndcg_at_k(pred, gt))

    return {
        "hit_at_1": np.array(hit1_list),
        "hit_at_5": np.array(hit5_list),
        "mrr": np.array(mrr_list),
        "ndcg_at_10": np.array(ndcg_list),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3: Hierarchy path features")
    parser.add_argument("--model", choices=["bge", "gte", "all"], default="all",
                        help="Which model to run (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    folds_baseline = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")
    folds_path = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="path")

    results: dict = {
        "experiment": "exp3_hierarchy_paths",
        "models": {},
        "device": args.device,
    }

    models_to_run = []
    if args.model in ("bge", "all"):
        models_to_run.append(BIENCODER_MODELS[0])
    if args.model in ("gte", "all"):
        models_to_run.append(BIENCODER_MODELS[1])

    for model_config in models_to_run:
        model_name = model_config["name"]
        logger.info("=" * 60)

        logger.info("Running %s BASELINE...", model_name)
        start = time.time()
        baseline_fold_results = run_biencoder(model_config, folds_baseline, corpus, args.device)
        baseline_elapsed = time.time() - start

        logger.info("Running %s PATH-ENRICHED...", model_name)
        start = time.time()
        path_fold_results = run_biencoder(model_config, folds_path, corpus, args.device)
        path_elapsed = time.time() - start

        baseline_metrics_all = aggregate_lofo_metrics(baseline_fold_results, folds_baseline, track_filter=None)
        path_metrics_all = aggregate_lofo_metrics(path_fold_results, folds_path, track_filter=None)
        baseline_metrics_ft = aggregate_lofo_metrics(baseline_fold_results, folds_baseline, track_filter="full-text")
        path_metrics_ft = aggregate_lofo_metrics(path_fold_results, folds_path, track_filter="full-text")

        baseline_items_all = compute_per_item_metrics(baseline_fold_results, folds_baseline, track_filter=None)
        path_items_all = compute_per_item_metrics(path_fold_results, folds_path, track_filter=None)

        deltas_all: dict[str, dict[str, float]] = {}
        for metric_name in ["hit_at_1", "hit_at_5", "mrr", "ndcg_at_10"]:
            deltas_all[metric_name] = bootstrap_paired_delta(
                baseline_items_all[metric_name],
                path_items_all[metric_name],
            )

        results["models"][model_name] = {
            "hf_id": model_config["hf_id"],
            "baseline": {
                "all_198": baseline_metrics_all,
                "full_text": baseline_metrics_ft,
                "elapsed_seconds": round(baseline_elapsed, 1),
            },
            "path_enriched": {
                "all_198": path_metrics_all,
                "full_text": path_metrics_ft,
                "elapsed_seconds": round(path_elapsed, 1),
            },
            "deltas_all_198": deltas_all,
        }

        logger.info(
            "%s baseline all-198: hit@1=%.3f, hit@5=%.3f",
            model_name,
            baseline_metrics_all["hit_at_1"]["mean"],
            baseline_metrics_all["hit_at_5"]["mean"],
        )
        logger.info(
            "%s path all-198: hit@1=%.3f, hit@5=%.3f",
            model_name,
            path_metrics_all["hit_at_1"]["mean"],
            path_metrics_all["hit_at_5"]["mean"],
        )
        logger.info(
            "%s delta hit@1=%.3f [%.3f, %.3f]",
            model_name,
            deltas_all["hit_at_1"]["delta_mean"],
            deltas_all["hit_at_1"]["ci_low"],
            deltas_all["hit_at_1"]["ci_high"],
        )

    save_results(results, "exp3_hierarchy_paths.json")
    logger.info("Experiment 3 complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/phase0/exp3_hierarchy_paths.py
git commit -m "feat: add experiment 3 hierarchy path features script"
```

---

### Task 8: Experiment 4 — LLM Hub Description Pilot

**Files:**
- Create: `scripts/phase0/exp4_hub_descriptions.py`

Two stages: (1) Generate descriptions for the top 50 AI-linked leaf hubs using Opus. (2) Re-run best bi-encoder with description-enriched hub text on the subset of eval items mapping to described hubs.

- [ ] **Step 1: Write experiment 4 script**

Create `scripts/phase0/exp4_hub_descriptions.py`:

```python
"""Experiment 4: LLM hub description pilot.

Stage A: Generate 2-3 sentence descriptions for the top 50 AI-linked leaf hubs.
Stage B: Re-run best bi-encoder(s) with description-enriched hub text on the
         subset of evaluation items mapping to described hubs.

Hub descriptions are static features (no LOFO firewall on description generation).
Evaluation still uses LOFO for linked standard names.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
from collections import Counter
from typing import Final

import numpy as np
import torch

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    CREHierarchy,
    EvalItem,
    HubStandardLink,
    LOFOFold,
    aggregate_lofo_metrics,
    bootstrap_paired_delta,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
    save_results,
    score_predictions,
    _reciprocal_rank,
    _ndcg_at_k,
)
from scripts.phase0.exp1_embedding_baseline import (
    BIENCODER_MODELS,
    run_biencoder,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL: Final[str] = "claude-opus-4-20250514"
TOP_N_HUBS: Final[int] = 50
MAX_CONCURRENT_REQUESTS: Final[int] = 5


def get_api_key() -> str:
    """Retrieve Anthropic API key from pass password manager."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    result = subprocess.run(
        ["pass", "anthropic/api_key"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def select_top_hubs(
    links: list[HubStandardLink],
    tree: CREHierarchy,
    n: int = TOP_N_HUBS,
) -> list[str]:
    """Select top N leaf hubs by AI framework link count."""
    leaf_ids = set(tree.leaf_hub_ids())
    ai_link_counts: Counter[str] = Counter()
    for link in links:
        if link.standard_name in AI_FRAMEWORK_NAMES and link.cre_id in leaf_ids:
            ai_link_counts[link.cre_id] += 1

    return [hub_id for hub_id, _ in ai_link_counts.most_common(n)]


def build_description_prompt(
    hub_id: str,
    tree: CREHierarchy,
    links: list[HubStandardLink],
) -> str:
    """Build prompt for generating one hub description."""
    hub_name = tree.hubs[hub_id]
    path = tree.hierarchy_path(hub_id)

    linked_standards = sorted(set(
        link.section_name or link.section_id
        for link in links if link.cre_id == hub_id
    ))

    parent_id = tree.parent.get(hub_id)
    siblings: list[str] = []
    if parent_id:
        siblings = [
            tree.hubs[sid]
            for sid in tree.children.get(parent_id, [])
            if sid != hub_id
        ]

    return f"""You are writing concise descriptions for CRE (Common Requirements Enumeration) taxonomy hubs.

HUB: {hub_name}
HIERARCHY PATH: {path}
LINKED STANDARDS: {', '.join(linked_standards) if linked_standards else '(none)'}
SIBLING HUBS: {', '.join(siblings[:10]) if siblings else '(none)'}

Write a 2-3 sentence description covering:
(a) What this hub covers
(b) What distinguishes it from its siblings
(c) Its scope boundary (what it does NOT cover)

Be specific and technical. Use plain text, no markdown formatting."""


async def generate_descriptions(
    hub_ids: list[str],
    tree: CREHierarchy,
    links: list[HubStandardLink],
    max_concurrent: int,
) -> dict[str, str]:
    """Generate descriptions for selected hubs using Opus."""
    import anthropic

    api_key = get_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    descriptions: dict[str, str] = {}

    async def generate_one(hub_id: str) -> tuple[str, str]:
        prompt = build_description_prompt(hub_id, tree, links)
        async with semaphore:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return hub_id, response.content[0].text.strip()

    tasks = [generate_one(hid) for hid in hub_ids]
    results = await asyncio.gather(*tasks)

    for hub_id, description in results:
        descriptions[hub_id] = description
        logger.info("Generated description for %s (%s): %d chars",
                     hub_id, tree.hubs[hub_id], len(description))

    return descriptions


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 4: Hub description pilot")
    parser.add_argument("--model", choices=["bge", "gte", "all"], default="all",
                        help="Which bi-encoder model to evaluate (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT_REQUESTS)
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip description generation, load from existing file")
    args = parser.parse_args()

    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = extract_hub_standard_links(cres)
    parsed_controls = load_parsed_controls()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed_controls)

    top_hub_ids = select_top_hubs(links, tree, TOP_N_HUBS)
    logger.info("Selected %d hubs for description pilot", len(top_hub_ids))

    descriptions_path = save_results.__wrapped__ if hasattr(save_results, "__wrapped__") else None
    from scripts.phase0.common import RESULTS_DIR
    desc_file = RESULTS_DIR / "pilot_hub_descriptions.json"

    if args.skip_generation and desc_file.exists():
        logger.info("Loading existing descriptions from %s", desc_file)
        with open(desc_file, encoding="utf-8") as f:
            desc_data = json.load(f)
        descriptions = desc_data["descriptions"]
    else:
        logger.info("Generating descriptions for %d hubs...", len(top_hub_ids))
        descriptions = asyncio.run(
            generate_descriptions(top_hub_ids, tree, links, args.max_concurrent)
        )
        desc_data = {
            "model": MODEL,
            "n_hubs": len(descriptions),
            "descriptions": descriptions,
            "hub_details": {
                hid: {
                    "name": tree.hubs[hid],
                    "path": tree.hierarchy_path(hid),
                    "description": descriptions[hid],
                }
                for hid in descriptions
            },
        }
        save_results(desc_data, "pilot_hub_descriptions.json")

    described_hub_ids = set(descriptions.keys())
    eval_subset = [item for item in corpus if item.ground_truth_hub_id in described_hub_ids]
    logger.info(
        "Evaluation subset: %d items (of %d total) map to %d described hubs",
        len(eval_subset), len(corpus), len(described_hub_ids),
    )

    folds_baseline = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="default")
    folds_desc = build_lofo_folds(
        tree, links, corpus, AI_FRAMEWORK_NAMES,
        template="description", descriptions=descriptions,
    )

    results: dict = {
        "experiment": "exp4_hub_descriptions",
        "n_described_hubs": len(described_hub_ids),
        "n_eval_subset": len(eval_subset),
        "models": {},
        "device": args.device,
    }

    models_to_run = []
    if args.model in ("bge", "all"):
        models_to_run.append(BIENCODER_MODELS[0])
    if args.model in ("gte", "all"):
        models_to_run.append(BIENCODER_MODELS[1])

    for model_config in models_to_run:
        model_name = model_config["name"]
        logger.info("=" * 60)

        logger.info("Running %s BASELINE...", model_name)
        start = time.time()
        baseline_fold_results = run_biencoder(model_config, folds_baseline, corpus, args.device)
        baseline_elapsed = time.time() - start

        logger.info("Running %s DESCRIPTION-ENRICHED...", model_name)
        start = time.time()
        desc_fold_results = run_biencoder(model_config, folds_desc, corpus, args.device)
        desc_elapsed = time.time() - start

        def filter_to_described_hubs(fold_results, folds):
            """Extract per-item metrics only for items mapping to described hubs."""
            hit1, hit5, mrr_vals, ndcg_vals = [], [], [], []
            for fold, preds in zip(folds, fold_results):
                for i, item in enumerate(fold.eval_items):
                    if item.ground_truth_hub_id not in described_hub_ids:
                        continue
                    pred = preds.get(i, [])
                    gt = item.ground_truth_hub_id
                    hit1.append(1.0 if pred and pred[0] == gt else 0.0)
                    hit5.append(1.0 if gt in pred[:5] else 0.0)
                    mrr_vals.append(_reciprocal_rank(pred, gt))
                    ndcg_vals.append(_ndcg_at_k(pred, gt))
            return {
                "hit_at_1": np.array(hit1),
                "hit_at_5": np.array(hit5),
                "mrr": np.array(mrr_vals),
                "ndcg_at_10": np.array(ndcg_vals),
            }

        baseline_items = filter_to_described_hubs(baseline_fold_results, folds_baseline)
        desc_items = filter_to_described_hubs(desc_fold_results, folds_desc)

        from scripts.phase0.common import bootstrap_ci
        baseline_metrics_subset = {
            m: bootstrap_ci(baseline_items[m]) for m in baseline_items
        }
        desc_metrics_subset = {
            m: bootstrap_ci(desc_items[m]) for m in desc_items
        }

        deltas: dict[str, dict[str, float]] = {}
        for metric_name in baseline_items:
            deltas[metric_name] = bootstrap_paired_delta(
                baseline_items[metric_name],
                desc_items[metric_name],
            )

        results["models"][model_name] = {
            "hf_id": model_config["hf_id"],
            "baseline_subset": baseline_metrics_subset,
            "description_enriched_subset": desc_metrics_subset,
            "deltas_subset": deltas,
            "baseline_elapsed_seconds": round(baseline_elapsed, 1),
            "description_elapsed_seconds": round(desc_elapsed, 1),
        }

        logger.info(
            "%s described-hub subset: baseline hit@1=%.3f, desc hit@1=%.3f, delta=%.3f",
            model_name,
            baseline_metrics_subset["hit_at_1"]["mean"],
            desc_metrics_subset["hit_at_1"]["mean"],
            deltas["hit_at_1"]["delta_mean"],
        )

    save_results(results, "exp4_hub_descriptions.json")
    logger.info("Experiment 4 complete.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/phase0/exp4_hub_descriptions.py
git commit -m "feat: add experiment 4 hub description pilot script"
```

---

### Task 9: Summary Aggregation Script

**Files:**
- Create: `scripts/phase0/run_summary.py`

Reads all experiment result files, builds the summary table, evaluates gate criteria.

- [ ] **Step 1: Write run_summary.py**

Create `scripts/phase0/run_summary.py`:

```python
"""Aggregate Phase 0 experiment results and evaluate gate criteria.

Reads results from all four experiments and prints the summary table
from the design spec. Evaluates:
  (a) Opus hit@5 > 0.50 on all-198 → task is feasible
  (b) Best embedding hit@1 at least 0.10 below Opus hit@1 → room for trained model
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from scripts.phase0.common import RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

GATE_A_THRESHOLD: float = 0.50
GATE_B_THRESHOLD: float = 0.10


def load_result(filename: str) -> dict | None:
    path = RESULTS_DIR / filename
    if not path.exists():
        logger.warning("Result file not found: %s", path)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_metric(m: dict[str, float]) -> str:
    """Format metric as 'mean ± CI'."""
    return f"{m['mean']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}]"


def main() -> None:
    exp1 = load_result("exp1_embedding_baseline.json")
    exp2 = load_result("exp2_llm_probe.json")
    exp3 = load_result("exp3_hierarchy_paths.json")
    exp4 = load_result("exp4_hub_descriptions.json")

    rows: list[tuple[str, dict | None]] = []

    if exp1:
        for model_name, model_data in exp1["models"].items():
            rows.append((f"{model_name} (baseline)", model_data.get("all_198")))

    if exp3:
        for model_name, model_data in exp3["models"].items():
            rows.append((f"{model_name} + paths", model_data.get("path_enriched", {}).get("all_198")))

    if exp4:
        for model_name, model_data in exp4["models"].items():
            rows.append((f"{model_name} + descriptions", model_data.get("description_enriched_subset")))

    if exp2:
        rows.append(("Opus LLM probe", exp2.get("all_198")))

    print("\n" + "=" * 90)
    print("PHASE 0 SUMMARY — All-198 Track")
    print("=" * 90)
    print(f"{'Method':<30} {'hit@1':<22} {'hit@5':<22} {'MRR':<22} {'NDCG@10':<22}")
    print("-" * 90)

    for name, metrics in rows:
        if metrics is None:
            print(f"{name:<30} {'(missing)':<22} {'(missing)':<22} {'(missing)':<22} {'(missing)':<22}")
            continue
        h1 = fmt_metric(metrics.get("hit_at_1", {})) if "hit_at_1" in metrics else "(n/a)"
        h5 = fmt_metric(metrics.get("hit_at_5", {})) if "hit_at_5" in metrics else "(n/a)"
        mrr = fmt_metric(metrics.get("mrr", {})) if "mrr" in metrics else "(n/a)"
        ndcg = fmt_metric(metrics.get("ndcg_at_10", {})) if "ndcg_at_10" in metrics else "(n/a)"
        print(f"{name:<30} {h1:<22} {h5:<22} {mrr:<22} {ndcg:<22}")

    print("-" * 90)

    opus_hit5 = exp2["all_198"]["hit_at_5"]["mean"] if exp2 else None
    opus_hit1 = exp2["all_198"]["hit_at_1"]["mean"] if exp2 else None

    best_emb_hit1 = None
    best_emb_name = None
    if exp1:
        for model_name, model_data in exp1["models"].items():
            h1 = model_data["all_198"]["hit_at_1"]["mean"]
            if best_emb_hit1 is None or h1 > best_emb_hit1:
                best_emb_hit1 = h1
                best_emb_name = model_name

    print("\nGATE CRITERIA (all-198 track):")

    if opus_hit5 is not None:
        gate_a_pass = opus_hit5 > GATE_A_THRESHOLD
        print(f"  (a) Opus hit@5 = {opus_hit5:.3f} > {GATE_A_THRESHOLD:.2f}? {'PASS' if gate_a_pass else 'FAIL'}")
    else:
        print("  (a) Opus hit@5: MISSING (experiment 2 not run)")
        gate_a_pass = False

    if opus_hit1 is not None and best_emb_hit1 is not None:
        gap = opus_hit1 - best_emb_hit1
        gate_b_pass = gap > GATE_B_THRESHOLD
        print(f"  (b) Opus hit@1 ({opus_hit1:.3f}) - best embedding hit@1 ({best_emb_hit1:.3f}, {best_emb_name}) = {gap:.3f} > {GATE_B_THRESHOLD:.2f}? {'PASS' if gate_b_pass else 'FAIL'}")
    else:
        print("  (b) hit@1 gap: MISSING (need both experiments 1 and 2)")
        gate_b_pass = False

    print()
    if gate_a_pass and gate_b_pass:
        print(">>> BOTH GATES PASS — proceed to Phase 1.")
    elif not gate_a_pass and not gate_b_pass:
        print(">>> BOTH GATES FAIL — reassess architecture.")
    else:
        print(">>> PARTIAL PASS — review results before proceeding.")

    summary = {
        "gate_a": {"opus_hit5": opus_hit5, "threshold": GATE_A_THRESHOLD, "pass": gate_a_pass},
        "gate_b": {
            "opus_hit1": opus_hit1,
            "best_embedding_hit1": best_emb_hit1,
            "best_embedding_model": best_emb_name,
            "gap": (opus_hit1 - best_emb_hit1) if opus_hit1 and best_emb_hit1 else None,
            "threshold": GATE_B_THRESHOLD,
            "pass": gate_b_pass,
        },
        "proceed_to_phase1": gate_a_pass and gate_b_pass,
    }
    from scripts.phase0.common import save_results
    save_results(summary, "summary.json")

    print("\nSaved summary to results/phase0/summary.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/phase0/run_summary.py
git commit -m "feat: add phase0 summary aggregation with gate evaluation"
```

---

### Task 10: RunPod Setup Script

**Files:**
- Create: `scripts/phase0/runpod_setup.sh`

Provisions 3 RunPod GPU pods, installs dependencies, syncs code, runs experiments, pulls results, and tears down.

- [ ] **Step 1: Write runpod_setup.sh**

Create `scripts/phase0/runpod_setup.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

# Phase 0 RunPod GPU provisioning and experiment execution.
# Requires: runpodctl CLI, pass password manager, rsync
#
# Usage:
#   ./scripts/phase0/runpod_setup.sh [provision|run|collect|teardown|all]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/results/phase0"

RUNPOD_API_KEY="$(pass runpod/api_key)"
export RUNPOD_API_KEY

GPU_TYPE="NVIDIA A100 80GB PCIe"
DOCKER_IMAGE="runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04"
VOLUME_SIZE=50
DISK_SIZE=20

POD_NAMES=("tract-phase0-bge" "tract-phase0-gte" "tract-phase0-deberta")
POD_IDS_FILE="$SCRIPT_DIR/.pod_ids"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

provision() {
    log "Provisioning 3 GPU pods..."
    mkdir -p "$RESULTS_DIR"
    > "$POD_IDS_FILE"

    for name in "${POD_NAMES[@]}"; do
        log "Creating pod: $name"
        pod_id=$(runpodctl create pod \
            --name "$name" \
            --gpuType "$GPU_TYPE" \
            --gpuCount 1 \
            --imageName "$DOCKER_IMAGE" \
            --volumeSize "$VOLUME_SIZE" \
            --containerDiskSize "$DISK_SIZE" \
            --ports "22/tcp" \
            2>&1 | grep -oP 'pod "\K[^"]+')
        echo "$name=$pod_id" >> "$POD_IDS_FILE"
        log "Created $name: $pod_id"
    done

    log "Waiting for pods to be ready..."
    sleep 30

    for line in $(cat "$POD_IDS_FILE"); do
        name="${line%%=*}"
        pod_id="${line##*=}"
        log "Waiting for $name ($pod_id)..."
        for i in $(seq 1 30); do
            status=$(runpodctl get pod "$pod_id" 2>/dev/null | grep -oP 'status: \K\w+' || echo "unknown")
            if [ "$status" = "RUNNING" ]; then
                log "$name is RUNNING"
                break
            fi
            sleep 10
        done
    done
}

setup_pod() {
    local pod_id="$1"
    local name="$2"

    log "Setting up $name ($pod_id)..."

    local ssh_cmd="runpodctl ssh --podId $pod_id"

    $ssh_cmd "pip install -q sentence-transformers transformers torch numpy scipy scikit-learn"

    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/scripts/phase0/" /workspace/scripts/phase0/
    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/tract/" /workspace/tract/
    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/data/" /workspace/data/
    runpodctl send --podId "$pod_id" "$PROJECT_ROOT/pyproject.toml" /workspace/pyproject.toml

    $ssh_cmd "cd /workspace && pip install -e '.[phase0]'"
}

run_experiments() {
    log "Setting up pods and running experiments..."

    local bge_id gte_id deberta_id
    while IFS='=' read -r name pod_id; do
        case "$name" in
            tract-phase0-bge) bge_id="$pod_id" ;;
            tract-phase0-gte) gte_id="$pod_id" ;;
            tract-phase0-deberta) deberta_id="$pod_id" ;;
        esac
    done < "$POD_IDS_FILE"

    for line in $(cat "$POD_IDS_FILE"); do
        name="${line%%=*}"
        pod_id="${line##*=}"
        setup_pod "$pod_id" "$name" &
    done
    wait

    log "Phase A: Running baseline experiments in parallel..."
    runpodctl ssh --podId "$bge_id" "cd /workspace && python -m scripts.phase0.exp1_embedding_baseline --model bge" &
    runpodctl ssh --podId "$gte_id" "cd /workspace && python -m scripts.phase0.exp1_embedding_baseline --model gte" &
    runpodctl ssh --podId "$deberta_id" "cd /workspace && python -m scripts.phase0.exp1_embedding_baseline --model deberta" &
    wait
    log "Phase A complete."

    log "Phase B: Running path-enriched experiments..."
    runpodctl ssh --podId "$bge_id" "cd /workspace && python -m scripts.phase0.exp3_hierarchy_paths --model bge" &
    runpodctl ssh --podId "$gte_id" "cd /workspace && python -m scripts.phase0.exp3_hierarchy_paths --model gte" &
    wait
    log "Phase B complete."

    log "Phase C: Running description pilot on best model..."
    runpodctl ssh --podId "$bge_id" "cd /workspace && python -m scripts.phase0.exp4_hub_descriptions --model all"
    log "Phase C complete."
}

collect() {
    log "Collecting results from pods..."
    mkdir -p "$RESULTS_DIR"

    while IFS='=' read -r name pod_id; do
        log "Collecting from $name ($pod_id)..."
        runpodctl recv --podId "$pod_id" /workspace/results/phase0/ "$RESULTS_DIR/" || true
    done < "$POD_IDS_FILE"

    log "Results collected to $RESULTS_DIR"
    ls -la "$RESULTS_DIR"
}

teardown() {
    log "Tearing down pods..."

    while IFS='=' read -r name pod_id; do
        log "Removing $name ($pod_id)..."
        runpodctl remove pod "$pod_id" || true
    done < "$POD_IDS_FILE"

    rm -f "$POD_IDS_FILE"
    log "All pods removed."
}

case "${1:-all}" in
    provision) provision ;;
    run) run_experiments ;;
    collect) collect ;;
    teardown) teardown ;;
    all)
        provision
        run_experiments
        collect
        teardown
        ;;
    *) echo "Usage: $0 [provision|run|collect|teardown|all]"; exit 1 ;;
esac
```

- [ ] **Step 2: Make executable and commit**

```bash
chmod +x scripts/phase0/runpod_setup.sh
git add scripts/phase0/runpod_setup.sh
git commit -m "feat: add RunPod provisioning script for phase0 GPU experiments"
```

---

### Task 11: Add Phase 0 config constants

**Files:**
- Modify: `tract/config.py`

Add Phase 0-specific constants referenced by the experiment scripts.

- [ ] **Step 1: Add constants to config.py**

Append to `tract/config.py`:

```python
# ── Phase 0: Zero-Shot Baseline Settings ─────────────────────────────────

PHASE0_BOOTSTRAP_N_RESAMPLES: Final[int] = 10_000
PHASE0_BOOTSTRAP_CI_LEVEL: Final[float] = 0.95
PHASE0_BOOTSTRAP_SEED: Final[int] = 42

PHASE0_GATE_A_OPUS_HIT5_THRESHOLD: Final[float] = 0.50
PHASE0_GATE_B_HIT1_GAP_THRESHOLD: Final[float] = 0.10

PHASE0_LLM_PROBE_MODEL: Final[str] = "claude-opus-4-20250514"
PHASE0_LLM_PROBE_MAX_CONCURRENT: Final[int] = 5
PHASE0_LLM_SHORTLIST_PER_BRANCH: Final[int] = 20
PHASE0_LLM_FINAL_TOP_K: Final[int] = 10

PHASE0_DESCRIPTION_PILOT_N_HUBS: Final[int] = 50
```

- [ ] **Step 2: Commit**

```bash
git add tract/config.py
git commit -m "feat: add phase0 gate thresholds and experiment constants to config"
```

---

### Task 12: Integration test — Dry run with mini fixture

**Files:**
- Modify: `tests/test_phase0_common.py`

Add an integration test that exercises the full pipeline (data loading → hierarchy → LOFO folds → dummy predictions → scoring → bootstrap) on the mini fixture.

- [ ] **Step 1: Write integration test**

Append to `tests/test_phase0_common.py`:

```python
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
        for hub_id, hub_text in fold.hub_texts.items():
            assert fold.held_out_framework not in hub_text or fold.held_out_framework in tree.hubs.values()

    fold_results: list[dict] = []
    for fold in folds:
        preds: dict[int, list[str]] = {}
        for i, item in enumerate(fold.eval_items):
            preds[i] = [item.ground_truth_hub_id] + ["WRONG-1", "WRONG-2"]
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
```

- [ ] **Step 2: Run all phase0 tests**

Run: `python -m pytest tests/test_phase0_common.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_phase0_common.py
git commit -m "test: add phase0 LOFO integration and hub firewall tests"
```

---

### Task 13: End-to-end dry run on real data (no GPU, no API)

**Files:**
- No new files. Validates that all scripts load real data correctly.

- [ ] **Step 1: Test data loading and fold construction**

Run:
```bash
python -c "
from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    extract_hub_standard_links,
    load_opencre_cres,
    load_parsed_controls,
)

cres = load_opencre_cres()
tree = build_hierarchy(cres)
links = extract_hub_standard_links(cres)
parsed = load_parsed_controls()
corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, parsed)

print(f'CREs: {len(tree.hubs)}, roots: {len(tree.roots)}, leaves: {len(tree.leaf_hub_ids())}')
print(f'Links: {len(links)}, AI corpus: {len(corpus)}')
print(f'Full-text: {sum(1 for e in corpus if e.track == \"full-text\")}, title-only: {sum(1 for e in corpus if e.track == \"all\")}')

folds = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES)
for fold in folds:
    print(f'Fold {fold.held_out_framework}: {len(fold.eval_items)} items, {len(fold.hub_texts)} hubs')
    # Verify firewall
    for item in fold.eval_items:
        for hub_id, hub_text in fold.hub_texts.items():
            assert fold.held_out_framework not in hub_text.split(':')[1] if ':' in hub_text else True
print('All folds constructed and firewalled correctly.')
"
```

Expected output:
```
CREs: 522, roots: 5, leaves: 400
Links: ~4406, AI corpus: 198
Full-text: 125, title-only: 73
Fold MITRE ATLAS: 65 items, 522 hubs
Fold NIST AI 100-2: 45 items, 522 hubs
Fold OWASP AI Exchange: 65 items, 522 hubs
Fold OWASP Top10 for LLM: 13 items, 522 hubs
Fold OWASP Top10 for ML: 10 items, 522 hubs
All folds constructed and firewalled correctly.
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: All tests pass (existing + new phase0 tests).

---

## Execution Order

Experiments 1-3 (GPU) run on RunPod pods. Experiment 2 (LLM probe) runs locally in parallel.

| Phase | GPU Pod 1 | GPU Pod 2 | GPU Pod 3 | Local Machine |
|-------|-----------|-----------|-----------|---------------|
| A | BGE baseline (exp1 --model bge) | GTE baseline (exp1 --model gte) | DeBERTa baseline (exp1 --model deberta) | — |
| B | BGE + paths (exp3 --model bge) | GTE + paths (exp3 --model gte) | (idle/teardown) | LLM probe (exp2) |
| C | Best model + descriptions (exp4) | (teardown) | (teardown) | — |

After all experiments complete:
```bash
python -m scripts.phase0.run_summary
```

Estimated timeline: ~60-90 minutes wall-clock. Estimated cost: ~$70 ($10-12 RunPod + $55 Opus probe + $5 Opus descriptions).
