# Phase 2B: AI/Traditional Bridge + HuggingFace Publication — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify conceptual bridges between AI-specific and traditional CRE hubs, then publish the complete TRACT model to HuggingFace with a merged full model, bundled inference data, and an AIBOM-compliant model card.

**Architecture:** Two sequential workstreams. Bridge analysis: classify 522 hubs by framework type → compute 21×382 cosine matrix → extract top-3 per AI hub → generate LLM descriptions → expert review → commit to hierarchy. HuggingFace publication: merge LoRA adapters → bundle inference data → generate model card → write standalone scripts → security scan → upload. Publication gate requires completed bridge review.

**Tech Stack:** Python 3.12, sentence-transformers, peft, huggingface_hub, numpy, anthropic, pytest

**Spec:** `docs/superpowers/specs/2026-05-02-phase2b-hf-bridge-design.md`

---

## File Structure

### New Files

```
tract/bridge/__init__.py              — run_bridge_analysis() orchestrator
tract/bridge/classify.py              — classify_hubs() — AI/trad/both/unlinked split
tract/bridge/similarity.py            — compute_bridge_similarities(), extract_top_k()
tract/bridge/describe.py              — generate_bridge_descriptions(), generate_negative_descriptions()
tract/bridge/review.py                — load_candidates(), validate_candidates(), commit_bridges()

tract/publish/__init__.py             — publish_to_huggingface() orchestrator, bridge gate
tract/publish/merge.py                — merge_lora_adapters() — SentenceTransformer-aware PEFT merge
tract/publish/bundle.py               — bundle_inference_data() — copy/validate data files
tract/publish/model_card.py           — generate_model_card() — templated README.md
tract/publish/scripts.py              — write_predict_script(), write_train_script()
tract/publish/security.py             — scan_for_secrets() — context-aware regex scan

tests/test_bridge_classify.py         — Hub classification unit tests
tests/test_bridge_similarity.py       — Cosine matrix + top-K extraction tests
tests/test_bridge_describe.py         — LLM description generation tests (mocked API)
tests/test_bridge_review.py           — Candidate validation + commit tests
tests/test_bridge_integration.py      — End-to-end bridge pipeline test
tests/test_publish_merge.py           — LoRA merge validation tests
tests/test_publish_bundle.py          — Inference data bundle tests
tests/test_publish_model_card.py      — Model card content tests
tests/test_publish_scripts.py         — Standalone script content tests
tests/test_publish_security.py        — Secret detection tests
tests/test_publish_integration.py     — End-to-end publish pipeline test

tests/fixtures/bridge_mini_hub_links.json  — Synthetic hub links for bridge tests
```

### Modified Files

```
tract/config.py:277+                  — Add BRIDGE_* and HF_* constants
tract/hierarchy.py:20-34              — Add related_hub_ids to HubNode, bump version, update validate_integrity
tract/cli.py:27-32,1089-1101          — Add bridge and publish-hf subcommands
tests/test_hierarchy.py               — Test related_hub_ids backward compat + bidirectionality
```

### Existing Files Referenced (read-only)

```
tract/io.py                           — atomic_write_json(), load_json()
tract/sanitize.py                     — sanitize_text()
tract/inference.py                    — DeploymentArtifacts, load_deployment_artifacts()
tract/active_learning/model_io.py     — load_deployment_model(), SentenceTransformer load pattern
data/training/hub_links_by_framework.json    — dict[framework_id, list[link_dict]] with cre_id keys
data/processed/cre_hierarchy.json            — 522 hubs, version "1.0"
data/processed/hub_descriptions_reviewed.json — Hub semantic descriptions
results/phase1c/deployment_model/deployment_artifacts.npz  — (522,1024) unit-norm hub embeddings
results/phase1c/deployment_model/calibration.json          — t_deploy, ood_threshold, conformal_quantile
results/phase1c/deployment_model/model/model/              — SentenceTransformer + PEFT adapter
results/phase1c/calibration/ece_gate.json                  — ECE + bootstrap CI
results/phase1b/phase1b_textaware/fold_*_summary.json      — Per-fold LOFO metrics
results/phase1b/phase1b_textaware/corrected_metrics.json   — Multi-label-aware corrected metrics (generated in Task 0)
```

---

### Task 0: Generate Corrected Metrics for TEXTAWARE Experiment (Pre-requisite)

**Files:**
- Modify: `scripts/phase1b/rescore_predictions.py:92-98`
- Output: `results/phase1b/phase1b_textaware/corrected_metrics.json`

> **Why:** The existing `corrected_metrics.json` was generated for the PRIMARY experiment (n=197, different model). The deployment model comes from the TEXTAWARE experiment (n=147). Mixing metrics from different models in the model card is methodologically invalid. This task generates corrected metrics for the actual deployment model.

- [ ] **Step 1: Update rescore_predictions.py to include textaware**

In `scripts/phase1b/rescore_predictions.py`, update the `main()` function:

```python
def main() -> None:
    for experiment in ["phase1b_primary", "phase1b_textaware", "ablation_a6_descriptions"]:
        exp_dir = PHASE1B_RESULTS_DIR / experiment
        if exp_dir.exists():
            rescore_experiment(exp_dir)
        else:
            logger.warning("Experiment dir not found: %s", exp_dir)

    print("\n" + "=" * 60)
    print("RE-SCORING COMPLETE")
    for experiment in ["phase1b_primary", "phase1b_textaware", "ablation_a6_descriptions"]:
        corrected_path = PHASE1B_RESULTS_DIR / experiment / "corrected_metrics.json"
        if corrected_path.exists():
            data = load_json(corrected_path)
            agg = data["aggregate_hit1"]
            print(f"\n  {experiment}:")
            print(f"    Corrected hit@1: {agg['mean']:.3f} [{agg['ci_low']:.3f}, {agg['ci_high']:.3f}]")
            for fw, m in sorted(data["per_fold"].items()):
                print(f"    {fw}: hit@1={m['hit_at_1']:.3f} hit@5={m['hit_at_5']:.3f} MRR={m['mrr']:.3f}")
    print("=" * 60)
```

- [ ] **Step 2: Run rescore_predictions.py**

Run: `python scripts/phase1b/rescore_predictions.py`
Expected: Generates `results/phase1b/phase1b_textaware/corrected_metrics.json` with per_fold keys: `"MITRE ATLAS"`, `"NIST AI 100-2"`, `"OWASP AI Exchange"`, `"OWASP Top10 for LLM"`, `"OWASP Top10 for ML"`.

- [ ] **Step 3: Verify the output**

Run: `python3 -c "from tract.io import load_json; d = load_json('results/phase1b/phase1b_textaware/corrected_metrics.json'); print('Keys:', list(d['per_fold'].keys())); print('Aggregate:', d['aggregate_hit1'])"`
Expected: Keys match fold directory names. Aggregate should be close to 0.531 (the textaware micro-average).

- [ ] **Step 4: Commit**

```bash
git add scripts/phase1b/rescore_predictions.py results/phase1b/phase1b_textaware/corrected_metrics.json
git commit -m "data: generate multi-label corrected metrics for textaware experiment"
```

---

### Task 1: Config Constants + Test Fixture

**Files:**
- Modify: `tract/config.py:277+`
- Create: `tests/fixtures/bridge_mini_hub_links.json`

- [ ] **Step 1: Add BRIDGE_* and HF_* constants to config.py**

Add `import re` at the top of the file (after `from pathlib import Path`), then append after line 277 (after `PHASE5_GROUND_TRUTH_PROVENANCE`):

```python
# ── Phase 2B: Bridge Analysis ─────────────────────────────────────────

BRIDGE_AI_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "mitre_atlas", "owasp_ai_exchange", "nist_ai_100_2",
    "owasp_llm_top10", "owasp_ml_top10",
})
BRIDGE_TOP_K: Final[int] = 3
BRIDGE_LLM_MODEL: Final[str] = "claude-sonnet-4-20250514"
BRIDGE_LLM_TEMPERATURE: Final[float] = 0.0
BRIDGE_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "results" / "bridge"

# ── Phase 2B: HuggingFace Publication ─────────────────────────────────

HF_DEFAULT_REPO_ID: Final[str] = "rockCO78/tract-cre-assignment"
HF_STAGING_DIR: Final[Path] = PROJECT_ROOT / "build" / "hf_repo"
HF_BASE_MODEL: Final[str] = "BAAI/bge-large-en-v1.5"
HF_SCAN_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".py", ".md", ".txt", ".yaml", ".yml", ".json",
})
HF_SECRET_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"hf_[a-zA-Z0-9]{20,}"),
    re.compile(r"wandb_[a-zA-Z0-9]{10,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"/home/rock"),
    re.compile(r"/Users/rock"),
    re.compile(r"^pass\s+\w+/\w+", re.MULTILINE),
    re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    re.compile(r"(HF_TOKEN|WANDB_API_KEY|ANTHROPIC_API_KEY)\s*="),
]

PHASE1B_TEXTAWARE_RESULTS_DIR: Final[Path] = (
    PROJECT_ROOT / "results" / "phase1b" / "phase1b_textaware"
)
PHASE1B_CORRECTED_METRICS_PATH: Final[Path] = (
    PROJECT_ROOT / "results" / "phase1b" / "phase1b_textaware" / "corrected_metrics.json"
)
PHASE1C_ECE_GATE_PATH: Final[Path] = (
    PHASE1C_RESULTS_DIR / "calibration" / "ece_gate.json"
)
```

- [ ] **Step 2: Create test fixture file**

Create `tests/fixtures/bridge_mini_hub_links.json`:

```json
{
  "mitre_atlas": [
    {"cre_id": "AI-1", "cre_name": "AI Hub 1", "framework_id": "mitre_atlas", "link_type": "LinkedTo", "section_id": "T0001", "section_name": "Adversarial Example", "standard_name": "MITRE ATLAS"},
    {"cre_id": "AI-2", "cre_name": "AI Hub 2", "framework_id": "mitre_atlas", "link_type": "LinkedTo", "section_id": "T0002", "section_name": "Data Poisoning", "standard_name": "MITRE ATLAS"},
    {"cre_id": "BOTH-1", "cre_name": "Both Hub 1", "framework_id": "mitre_atlas", "link_type": "LinkedTo", "section_id": "T0003", "section_name": "Model Access", "standard_name": "MITRE ATLAS"}
  ],
  "owasp_ai_exchange": [
    {"cre_id": "AI-3", "cre_name": "AI Hub 3", "framework_id": "owasp_ai_exchange", "link_type": "LinkedTo", "section_id": "AIX-01", "section_name": "AI Exchange Control", "standard_name": "OWASP AI Exchange"}
  ],
  "asvs": [
    {"cre_id": "TRAD-1", "cre_name": "Trad Hub 1", "framework_id": "asvs", "link_type": "LinkedTo", "section_id": "V1.1", "section_name": "Secure SDLC", "standard_name": "ASVS"},
    {"cre_id": "TRAD-2", "cre_name": "Trad Hub 2", "framework_id": "asvs", "link_type": "LinkedTo", "section_id": "V1.2", "section_name": "Authentication", "standard_name": "ASVS"},
    {"cre_id": "TRAD-3", "cre_name": "Trad Hub 3", "framework_id": "asvs", "link_type": "LinkedTo", "section_id": "V1.3", "section_name": "Session Mgmt", "standard_name": "ASVS"},
    {"cre_id": "TRAD-4", "cre_name": "Trad Hub 4", "framework_id": "asvs", "link_type": "LinkedTo", "section_id": "V1.4", "section_name": "Access Control", "standard_name": "ASVS"},
    {"cre_id": "TRAD-5", "cre_name": "Trad Hub 5", "framework_id": "asvs", "link_type": "LinkedTo", "section_id": "V1.5", "section_name": "Input Validation", "standard_name": "ASVS"},
    {"cre_id": "BOTH-1", "cre_name": "Both Hub 1", "framework_id": "asvs", "link_type": "LinkedTo", "section_id": "V2.1", "section_name": "Model Access Ctrl", "standard_name": "ASVS"}
  ],
  "cwe": [
    {"cre_id": "TRAD-1", "cre_name": "Trad Hub 1", "framework_id": "cwe", "link_type": "LinkedTo", "section_id": "CWE-89", "section_name": "SQL Injection", "standard_name": "CWE"}
  ]
}
```

This gives: AI-only={AI-1, AI-2, AI-3}, trad-only={TRAD-1..5}, naturally-bridged={BOTH-1}. Tests add UNLINKED-1, UNLINKED-2 to all_hub_ids programmatically.

- [ ] **Step 3: Verify constants import**

Run: `python -c "from tract.config import BRIDGE_AI_FRAMEWORK_IDS, BRIDGE_TOP_K, HF_DEFAULT_REPO_ID, HF_SECRET_PATTERNS; print('OK:', len(HF_SECRET_PATTERNS), 'patterns')"`
Expected: `OK: 9 patterns`

- [ ] **Step 4: Commit**

```bash
git add tract/config.py tests/fixtures/bridge_mini_hub_links.json
git commit -m "feat(config): add Phase 2B bridge analysis and HF publication constants"
```

---

### Task 2: HubNode Schema Extension

**Files:**
- Modify: `tract/hierarchy.py:20-34,179-238`
- Modify: `tests/test_hierarchy.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_hierarchy.py`:

```python
class TestRelatedHubIds:

    def test_related_hub_ids_defaults_empty(self, hierarchy) -> None:
        for node in hierarchy.hubs.values():
            assert node.related_hub_ids == []

    def test_related_hub_ids_backward_compat(self) -> None:
        """Old JSON without related_hub_ids loads correctly."""
        from tract.hierarchy import HubNode
        raw = {
            "hub_id": "TEST-1",
            "name": "Test Hub",
            "parent_id": None,
            "children_ids": [],
            "depth": 0,
            "branch_root_id": "TEST-1",
            "hierarchy_path": "Test Hub",
            "is_leaf": True,
            "sibling_hub_ids": [],
        }
        node = HubNode.model_validate(raw)
        assert node.related_hub_ids == []

    def test_related_hub_ids_loads_when_present(self) -> None:
        from tract.hierarchy import HubNode
        raw = {
            "hub_id": "TEST-1",
            "name": "Test Hub",
            "parent_id": None,
            "children_ids": [],
            "depth": 0,
            "branch_root_id": "TEST-1",
            "hierarchy_path": "Test Hub",
            "is_leaf": True,
            "sibling_hub_ids": [],
            "related_hub_ids": ["TEST-2"],
        }
        node = HubNode.model_validate(raw)
        assert node.related_hub_ids == ["TEST-2"]

    def test_validate_integrity_rejects_dangling_related(self) -> None:
        from tract.hierarchy import CREHierarchy, HubNode
        hubs = {
            "A": HubNode(
                hub_id="A", name="Hub A", depth=0, branch_root_id="A",
                hierarchy_path="Hub A", is_leaf=True, related_hub_ids=["NONEXISTENT"],
            ),
        }
        hier = CREHierarchy(
            hubs=hubs, roots=["A"], label_space=["A"],
            fetch_timestamp="2026-01-01T00:00:00", data_hash="test",
            version="1.1",
        )
        with pytest.raises(ValueError, match="dangling related_hub_id"):
            hier.validate_integrity()

    def test_validate_integrity_rejects_asymmetric_related(self) -> None:
        from tract.hierarchy import CREHierarchy, HubNode
        hubs = {
            "A": HubNode(
                hub_id="A", name="Hub A", depth=0, branch_root_id="A",
                hierarchy_path="Hub A", is_leaf=True, related_hub_ids=["B"],
            ),
            "B": HubNode(
                hub_id="B", name="Hub B", depth=0, branch_root_id="B",
                hierarchy_path="Hub B", is_leaf=True, related_hub_ids=[],
            ),
        }
        hier = CREHierarchy(
            hubs=hubs, roots=["A", "B"], label_space=["A", "B"],
            fetch_timestamp="2026-01-01T00:00:00", data_hash="test",
            version="1.1",
        )
        with pytest.raises(ValueError, match="does not list"):
            hier.validate_integrity()

    def test_validate_integrity_accepts_symmetric_related(self) -> None:
        from tract.hierarchy import CREHierarchy, HubNode
        hubs = {
            "A": HubNode(
                hub_id="A", name="Hub A", depth=0, branch_root_id="A",
                hierarchy_path="Hub A", is_leaf=True, related_hub_ids=["B"],
            ),
            "B": HubNode(
                hub_id="B", name="Hub B", depth=0, branch_root_id="B",
                hierarchy_path="Hub B", is_leaf=True, related_hub_ids=["A"],
            ),
        }
        hier = CREHierarchy(
            hubs=hubs, roots=["A", "B"], label_space=["A", "B"],
            fetch_timestamp="2026-01-01T00:00:00", data_hash="test",
            version="1.1",
        )
        hier.validate_integrity()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_hierarchy.py::TestRelatedHubIds -v`
Expected: FAIL — `related_hub_ids` field does not exist on HubNode

- [ ] **Step 3: Add related_hub_ids field to HubNode**

In `tract/hierarchy.py`, add field to `HubNode` class (after `sibling_hub_ids` on line 33):

```python
    related_hub_ids: list[str] = Field(default_factory=list)
```

Change default version in `CREHierarchy` (line 46):

```python
    version: str = "1.1"
```

Add validation to `validate_integrity()` after the label_space sort check (after line 224):

```python
        # 5. Related hub IDs: exist and are bidirectional
        for hub_id, node in self.hubs.items():
            for related_id in node.related_hub_ids:
                if related_id not in self.hubs:
                    raise ValueError(
                        f"Hub {hub_id} has dangling related_hub_id: {related_id}"
                    )
                if hub_id not in self.hubs[related_id].related_hub_ids:
                    raise ValueError(
                        f"Hub {hub_id} has related_hub_id {related_id} but "
                        f"{related_id} does not list {hub_id}"
                    )
```

Renumber the existing "5. Expected counts" comment to "6. Expected counts".

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_hierarchy.py -v`
Expected: ALL PASS (including all existing tests — backward compat)

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `python -m pytest tests/ -q`
Expected: 553+ tests pass, 0 failures

- [ ] **Step 6: Commit**

```bash
git add tract/hierarchy.py tests/test_hierarchy.py
git commit -m "feat(hierarchy): add related_hub_ids field for bridge analysis"
```

---

### Task 3: Hub Classification

**Files:**
- Create: `tract/bridge/__init__.py`
- Create: `tract/bridge/classify.py`
- Create: `tests/test_bridge_classify.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bridge_classify.py`:

```python
"""Tests for tract.bridge.classify — hub classification by framework type."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"

ALL_HUB_IDS = [
    "AI-1", "AI-2", "AI-3",
    "BOTH-1",
    "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5",
    "UNLINKED-1", "UNLINKED-2",
]


@pytest.fixture
def classification():
    from tract.bridge.classify import classify_hubs
    return classify_hubs(FIXTURE_PATH, ALL_HUB_IDS)


class TestClassifyHubs:

    def test_ai_only_count(self, classification) -> None:
        assert len(classification.ai_only) == 3

    def test_ai_only_ids(self, classification) -> None:
        assert set(classification.ai_only) == {"AI-1", "AI-2", "AI-3"}

    def test_trad_only_count(self, classification) -> None:
        assert len(classification.trad_only) == 5

    def test_trad_only_ids(self, classification) -> None:
        assert set(classification.trad_only) == {"TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5"}

    def test_naturally_bridged(self, classification) -> None:
        assert classification.naturally_bridged == ["BOTH-1"]

    def test_unlinked(self, classification) -> None:
        assert set(classification.unlinked) == {"UNLINKED-1", "UNLINKED-2"}

    def test_all_lists_sorted(self, classification) -> None:
        assert classification.ai_only == sorted(classification.ai_only)
        assert classification.trad_only == sorted(classification.trad_only)
        assert classification.naturally_bridged == sorted(classification.naturally_bridged)
        assert classification.unlinked == sorted(classification.unlinked)

    def test_no_overlap(self, classification) -> None:
        sets = [
            set(classification.ai_only),
            set(classification.trad_only),
            set(classification.naturally_bridged),
            set(classification.unlinked),
        ]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert sets[i].isdisjoint(sets[j])

    def test_all_hubs_accounted_for(self, classification) -> None:
        total = (
            len(classification.ai_only)
            + len(classification.trad_only)
            + len(classification.naturally_bridged)
            + len(classification.unlinked)
        )
        assert total == len(ALL_HUB_IDS)

    def test_hub_not_in_links(self) -> None:
        """Hub with no links at all classified as unlinked."""
        from tract.bridge.classify import classify_hubs
        result = classify_hubs(FIXTURE_PATH, ["TOTALLY-NEW"])
        assert result.unlinked == ["TOTALLY-NEW"]
        assert result.ai_only == []
        assert result.trad_only == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bridge_classify.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tract.bridge'`

- [ ] **Step 3: Create package and implement classify module**

Create `tract/bridge/__init__.py`:

```python
"""TRACT bridge analysis — AI/traditional CRE hub bridge discovery."""
```

Create `tract/bridge/classify.py`:

```python
"""Classify CRE hubs as AI-only, traditional-only, naturally bridged, or unlinked."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tract.config import BRIDGE_AI_FRAMEWORK_IDS
from tract.io import load_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubClassification:
    ai_only: list[str]
    trad_only: list[str]
    naturally_bridged: list[str]
    unlinked: list[str]


def classify_hubs(
    hub_links_path: Path | str,
    all_hub_ids: list[str],
) -> HubClassification:
    """Classify hubs by which framework types link to them.

    Args:
        hub_links_path: Path to hub_links_by_framework.json
            (dict keyed by framework_id, values are lists of link dicts with 'cre_id')
        all_hub_ids: All hub IDs with embeddings (from deployment_artifacts.npz)

    Returns:
        HubClassification with sorted lists of hub IDs per category.
    """
    hub_links: dict[str, list[dict]] = load_json(hub_links_path)

    ai_hubs: set[str] = set()
    trad_hubs: set[str] = set()

    for framework_id, links in hub_links.items():
        for link in links:
            cre_id = link["cre_id"]
            if framework_id in BRIDGE_AI_FRAMEWORK_IDS:
                ai_hubs.add(cre_id)
            else:
                trad_hubs.add(cre_id)

    all_hub_set = set(all_hub_ids)
    linked_hubs = ai_hubs | trad_hubs

    result = HubClassification(
        ai_only=sorted(ai_hubs - trad_hubs),
        trad_only=sorted(trad_hubs - ai_hubs),
        naturally_bridged=sorted(ai_hubs & trad_hubs),
        unlinked=sorted(all_hub_set - linked_hubs),
    )

    logger.info(
        "Hub classification: %d AI-only, %d trad-only, %d bridged, %d unlinked",
        len(result.ai_only), len(result.trad_only),
        len(result.naturally_bridged), len(result.unlinked),
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bridge_classify.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/bridge/__init__.py tract/bridge/classify.py tests/test_bridge_classify.py
git commit -m "feat(bridge): add hub classification by framework type"
```

---

### Task 4: Similarity Computation

**Files:**
- Create: `tract/bridge/similarity.py`
- Create: `tests/test_bridge_similarity.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bridge_similarity.py`:

```python
"""Tests for tract.bridge.similarity — cosine matrix and top-K extraction."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def unit_embeddings():
    """11 unit-normalized 1024-dim vectors with fixed seed."""
    rng = np.random.default_rng(42)
    vecs = rng.standard_normal((11, 1024))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


@pytest.fixture
def hub_ids():
    return [
        "AI-1", "AI-2", "AI-3",
        "BOTH-1",
        "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5",
        "UNLINKED-1", "UNLINKED-2",
    ]


@pytest.fixture
def ai_only_ids():
    return ["AI-1", "AI-2", "AI-3"]


@pytest.fixture
def trad_only_ids():
    return ["TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5"]


class TestComputeBridgeSimilarities:

    def test_matrix_shape(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        matrix = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        assert matrix.shape == (3, 5)

    def test_values_bounded(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        matrix = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        assert matrix.min() >= -1.0 - 1e-6
        assert matrix.max() <= 1.0 + 1e-6

    def test_deterministic(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        m1 = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        m2 = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        np.testing.assert_array_equal(m1, m2)

    def test_dot_product_equals_cosine(self, unit_embeddings, hub_ids, ai_only_ids, trad_only_ids) -> None:
        from tract.bridge.similarity import compute_bridge_similarities
        matrix = compute_bridge_similarities(unit_embeddings, hub_ids, ai_only_ids, trad_only_ids)
        ai_idx = [hub_ids.index(h) for h in ai_only_ids]
        trad_idx = [hub_ids.index(h) for h in trad_only_ids]
        for i, ai_i in enumerate(ai_idx):
            for j, trad_j in enumerate(trad_idx):
                expected = float(np.dot(unit_embeddings[ai_i], unit_embeddings[trad_j]))
                np.testing.assert_almost_equal(matrix[i, j], expected, decimal=6)


class TestExtractTopK:

    def test_returns_k_per_hub(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.5, 0.3], [0.2, 0.8, 0.6]])
        ai_ids = ["AI-1", "AI-2"]
        trad_ids = ["TRAD-1", "TRAD-2", "TRAD-3"]
        candidates = extract_top_k(matrix, ai_ids, trad_ids, k=2)
        assert len(candidates) == 4
        ai1_cands = [c for c in candidates if c["ai_hub_id"] == "AI-1"]
        assert len(ai1_cands) == 2

    def test_sorted_descending_per_hub(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.5, 0.3, 0.7, 0.1]])
        candidates = extract_top_k(matrix, ["AI-1"], ["T1", "T2", "T3", "T4", "T5"], k=3)
        sims = [c["cosine_similarity"] for c in candidates]
        assert sims == sorted(sims, reverse=True)

    def test_rank_numbering(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.5, 0.3]])
        candidates = extract_top_k(matrix, ["AI-1"], ["T1", "T2", "T3"], k=3)
        ranks = [c["rank_for_ai_hub"] for c in candidates]
        assert ranks == [1, 2, 3]

    def test_top_k_identity(self) -> None:
        from tract.bridge.similarity import extract_top_k
        matrix = np.array([[0.9, 0.1, 0.5]])
        candidates = extract_top_k(matrix, ["AI-1"], ["T1", "T2", "T3"], k=2)
        trad_ids = [c["trad_hub_id"] for c in candidates]
        assert trad_ids == ["T1", "T3"]


class TestExtractNegatives:

    def test_one_per_ai_hub(self) -> None:
        from tract.bridge.similarity import extract_negatives
        matrix = np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.3]])
        negatives = extract_negatives(matrix, ["AI-1", "AI-2"], ["T1", "T2", "T3"])
        assert len(negatives) == 2

    def test_picks_lowest_similarity(self) -> None:
        from tract.bridge.similarity import extract_negatives
        matrix = np.array([[0.9, 0.1, 0.5]])
        negatives = extract_negatives(matrix, ["AI-1"], ["T1", "T2", "T3"])
        assert negatives[0]["trad_hub_id"] == "T2"
        assert abs(negatives[0]["cosine_similarity"] - 0.1) < 1e-6


class TestComputeSimilarityStats:

    def test_stats_keys(self) -> None:
        from tract.bridge.similarity import compute_similarity_stats
        matrix = np.array([[0.9, 0.5], [0.3, 0.7]])
        stats = compute_similarity_stats(matrix)
        assert "matrix_shape" in stats
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "percentiles" in stats

    def test_matrix_shape_value(self) -> None:
        from tract.bridge.similarity import compute_similarity_stats
        matrix = np.array([[0.9, 0.5], [0.3, 0.7]])
        stats = compute_similarity_stats(matrix)
        assert stats["matrix_shape"] == [2, 2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bridge_similarity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tract.bridge.similarity'`

- [ ] **Step 3: Implement similarity module**

Create `tract/bridge/similarity.py`:

```python
"""Cosine similarity computation and top-K extraction for bridge analysis."""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def compute_bridge_similarities(
    hub_embeddings: NDArray[np.floating],
    hub_ids: list[str],
    ai_only_ids: list[str],
    trad_only_ids: list[str],
) -> NDArray[np.floating]:
    """Compute cosine similarity matrix between AI-only and trad-only hubs.

    All hub embeddings are unit-normalized, so cosine = dot product.

    Returns:
        (n_ai, n_trad) float matrix of cosine similarities.
    """
    ai_indices = [hub_ids.index(h) for h in ai_only_ids]
    trad_indices = [hub_ids.index(h) for h in trad_only_ids]

    ai_emb = hub_embeddings[ai_indices]
    trad_emb = hub_embeddings[trad_indices]

    return ai_emb @ trad_emb.T


def extract_top_k(
    similarity_matrix: NDArray[np.floating],
    ai_hub_ids: list[str],
    trad_hub_ids: list[str],
    k: int = 3,
) -> list[dict]:
    """Extract top-K traditional matches per AI-only hub.

    Returns:
        List of candidate dicts sorted by (ai_hub_id, rank).
    """
    candidates: list[dict] = []
    for i, ai_id in enumerate(ai_hub_ids):
        row = similarity_matrix[i]
        top_k_indices = np.argsort(row)[-k:][::-1]
        for rank, j in enumerate(top_k_indices, 1):
            candidates.append({
                "ai_hub_id": ai_id,
                "trad_hub_id": trad_hub_ids[int(j)],
                "cosine_similarity": round(float(row[j]), 6),
                "rank_for_ai_hub": rank,
            })
    return candidates


def extract_negatives(
    similarity_matrix: NDArray[np.floating],
    ai_hub_ids: list[str],
    trad_hub_ids: list[str],
) -> list[dict]:
    """Extract bottom-1 traditional match per AI-only hub (negative controls).

    Returns:
        List of negative candidate dicts (one per AI hub).
    """
    negatives: list[dict] = []
    for i, ai_id in enumerate(ai_hub_ids):
        row = similarity_matrix[i]
        worst_idx = int(np.argmin(row))
        negatives.append({
            "ai_hub_id": ai_id,
            "trad_hub_id": trad_hub_ids[worst_idx],
            "cosine_similarity": round(float(row[worst_idx]), 6),
            "is_negative": True,
        })
    return negatives


def compute_similarity_stats(
    similarity_matrix: NDArray[np.floating],
) -> dict:
    """Compute summary statistics for the similarity matrix."""
    return {
        "matrix_shape": list(similarity_matrix.shape),
        "mean": round(float(np.mean(similarity_matrix)), 3),
        "std": round(float(np.std(similarity_matrix)), 3),
        "min": round(float(np.min(similarity_matrix)), 3),
        "max": round(float(np.max(similarity_matrix)), 3),
        "percentiles": {
            str(p): round(float(np.percentile(similarity_matrix, p)), 3)
            for p in [25, 50, 75, 90, 95, 99]
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bridge_similarity.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/bridge/similarity.py tests/test_bridge_similarity.py
git commit -m "feat(bridge): add cosine similarity computation and top-K extraction"
```

---

### Task 5: LLM Description Generation

**Files:**
- Create: `tract/bridge/describe.py`
- Create: `tests/test_bridge_describe.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bridge_describe.py`:

```python
"""Tests for tract.bridge.describe — LLM bridge description generation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"


@pytest.fixture
def hub_links():
    from tract.io import load_json
    return load_json(FIXTURE_PATH)


@pytest.fixture
def mini_hierarchy():
    from tract.hierarchy import HubNode, CREHierarchy
    hubs = {}
    for hid, name in [
        ("AI-1", "AI Hub 1"), ("AI-2", "AI Hub 2"), ("AI-3", "AI Hub 3"),
        ("TRAD-1", "Trad Hub 1"), ("TRAD-2", "Trad Hub 2"),
        ("BOTH-1", "Both Hub 1"),
    ]:
        hubs[hid] = HubNode(
            hub_id=hid, name=name, depth=0, branch_root_id=hid,
            hierarchy_path=name, is_leaf=True,
        )
    return CREHierarchy(
        hubs=hubs, roots=sorted(hubs), label_space=sorted(hubs),
        fetch_timestamp="2026-01-01T00:00:00", data_hash="test",
    )


class TestGenerateBridgeDescriptions:

    def test_adds_description_field(self, mocker, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="Both hubs address access control concerns.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mocker.patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client)

        candidates = [
            {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
             "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1},
        ]
        generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
        assert candidates[0]["description"] != ""

    def test_sanitizes_description(self, mocker, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="  Has  <b>tags</b>  and   spaces  ")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mocker.patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client)

        candidates = [
            {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
             "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1},
        ]
        generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
        assert "<b>" not in candidates[0]["description"]
        assert "  " not in candidates[0]["description"]

    def test_api_failure_sets_empty(self, mocker, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        mocker.patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client)

        candidates = [
            {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
             "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1},
        ]
        generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
        assert candidates[0]["description"] == ""

    def test_skips_existing_descriptions(self, mocker, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_bridge_descriptions
        mock_client = MagicMock()
        mocker.patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client)

        candidates = [
            {"ai_hub_id": "AI-1", "ai_hub_name": "AI Hub 1",
             "trad_hub_id": "TRAD-1", "trad_hub_name": "Trad Hub 1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "Already has one"},
        ]
        generate_bridge_descriptions(candidates, mini_hierarchy, hub_links)
        assert candidates[0]["description"] == "Already has one"
        mock_client.messages.create.assert_not_called()


class TestGenerateNegativeDescriptions:

    def test_returns_one_per_ai_hub(self, mocker, mini_hierarchy, hub_links) -> None:
        from tract.bridge.describe import generate_negative_descriptions
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text="These hubs are unrelated.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_resp
        mocker.patch("tract.bridge.describe.anthropic.Anthropic", return_value=mock_client)

        negatives_input = [
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-2", "cosine_similarity": 0.01, "is_negative": True},
        ]
        result = generate_negative_descriptions(negatives_input, mini_hierarchy, hub_links)
        assert len(result) == 1
        assert result[0]["is_negative"] is True
        assert result[0]["description"] != ""


class TestCountControlsForHub:

    def test_counts_across_frameworks(self, hub_links) -> None:
        from tract.bridge.describe import count_controls_for_hub
        count = count_controls_for_hub("TRAD-1", hub_links)
        assert count == 2  # asvs + cwe

    def test_zero_for_unknown_hub(self, hub_links) -> None:
        from tract.bridge.describe import count_controls_for_hub
        assert count_controls_for_hub("NONEXISTENT", hub_links) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bridge_describe.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement describe module**

Create `tract/bridge/describe.py`:

```python
"""LLM-generated bridge descriptions for candidate hub pairs."""
from __future__ import annotations

import logging

import anthropic

from tract.config import BRIDGE_LLM_MODEL, BRIDGE_LLM_TEMPERATURE
from tract.hierarchy import CREHierarchy
from tract.sanitize import sanitize_text

logger = logging.getLogger(__name__)


def count_controls_for_hub(
    cre_id: str,
    hub_links: dict[str, list[dict]],
) -> int:
    """Count total controls linked to a hub across all frameworks."""
    return sum(
        1 for links in hub_links.values()
        for link in links
        if link["cre_id"] == cre_id
    )


def _build_prompt(
    candidate: dict,
    hierarchy: CREHierarchy,
    hub_links: dict[str, list[dict]],
) -> str:
    ai_id = candidate["ai_hub_id"]
    trad_id = candidate["trad_hub_id"]
    ai_name = candidate.get("ai_hub_name", ai_id)
    trad_name = candidate.get("trad_hub_name", trad_id)

    ai_path = hierarchy.hubs[ai_id].hierarchy_path if ai_id in hierarchy.hubs else ai_name
    trad_path = hierarchy.hubs[trad_id].hierarchy_path if trad_id in hierarchy.hubs else trad_name

    n_ai = count_controls_for_hub(ai_id, hub_links)
    n_trad = count_controls_for_hub(trad_id, hub_links)

    return (
        "You are a security standards expert. Describe the conceptual bridge "
        "between these two CRE hubs in 2-3 sentences.\n\n"
        f'AI Security Hub: "{ai_name}" (path: {ai_path})\n'
        f"- Linked from {n_ai} controls in AI security frameworks\n\n"
        f'Traditional Security Hub: "{trad_name}" (path: {trad_path})\n'
        f"- Linked from {n_trad} controls in traditional security frameworks\n\n"
        "Explain why these hubs address related security concerns, "
        "despite originating from different domains. Be specific about "
        "the shared concepts."
    )


def generate_bridge_descriptions(
    candidates: list[dict],
    hierarchy: CREHierarchy,
    hub_links: dict[str, list[dict]],
) -> None:
    """Add 'description' field to each candidate. Modifies in-place.

    Skips candidates that already have a non-empty description (idempotent).
    On API failure, sets description to '' (does not crash).
    """
    client = anthropic.Anthropic()
    for candidate in candidates:
        if candidate.get("description"):
            continue
        prompt = _build_prompt(candidate, hierarchy, hub_links)
        try:
            response = client.messages.create(
                model=BRIDGE_LLM_MODEL,
                temperature=BRIDGE_LLM_TEMPERATURE,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            candidate["description"] = sanitize_text(raw_text, max_length=500)
        except Exception:
            logger.warning(
                "LLM description failed for %s -> %s",
                candidate["ai_hub_id"], candidate["trad_hub_id"],
            )
            candidate["description"] = ""


def generate_negative_descriptions(
    negatives: list[dict],
    hierarchy: CREHierarchy,
    hub_links: dict[str, list[dict]],
) -> list[dict]:
    """Add descriptions to negative control candidates.

    Args:
        negatives: List from extract_negatives() — each has ai_hub_id,
            trad_hub_id, cosine_similarity, is_negative.

    Returns:
        The same list with hub names and descriptions added.
    """
    for neg in negatives:
        ai_id = neg["ai_hub_id"]
        trad_id = neg["trad_hub_id"]
        neg["ai_hub_name"] = hierarchy.hubs[ai_id].name if ai_id in hierarchy.hubs else ai_id
        neg["trad_hub_name"] = hierarchy.hubs[trad_id].name if trad_id in hierarchy.hubs else trad_id

    generate_bridge_descriptions(negatives, hierarchy, hub_links)
    return negatives
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bridge_describe.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/bridge/describe.py tests/test_bridge_describe.py
git commit -m "feat(bridge): add LLM bridge description generation with negative controls"
```

---

### Task 6: Review & Commit

**Files:**
- Create: `tract/bridge/review.py`
- Create: `tests/test_bridge_review.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bridge_review.py`:

```python
"""Tests for tract.bridge.review — candidate validation and bridge commit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_hierarchy_data(hub_ids: list[str]) -> dict:
    """Build a minimal hierarchy dict for testing."""
    hubs = {}
    for hid in hub_ids:
        hubs[hid] = {
            "hub_id": hid, "name": f"Hub {hid}", "parent_id": None,
            "children_ids": [], "depth": 0, "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}", "is_leaf": True,
            "sibling_hub_ids": [], "related_hub_ids": [],
        }
    return {
        "hubs": hubs, "roots": sorted(hub_ids), "label_space": sorted(hub_ids),
        "fetch_timestamp": "2026-01-01T00:00:00", "data_hash": "test",
        "version": "1.1",
    }


def _make_candidates_data(candidates: list[dict]) -> dict:
    return {
        "generated_at": "2026-05-02T00:00:00",
        "method": "top_k_per_ai_hub",
        "top_k": 3,
        "similarity_stats": {"matrix_shape": [2, 3], "mean": 0.5},
        "candidates": candidates,
        "negative_controls": [],
        "unclassified_leaf_hubs": [],
    }


class TestValidateCandidates:

    def test_rejects_pending_status(self, tmp_path) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "pending", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert any("pending" in e.lower() for e in errors)

    def test_rejects_unknown_status(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "maybe", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert any("maybe" in e for e in errors)

    def test_rejects_nonexistent_hub_id(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "GHOST",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "accepted", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert any("GHOST" in e for e in errors)

    def test_accepts_all_reviewed(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "accepted", "reviewer_notes": ""},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert errors == []

    def test_accepts_rejected(self) -> None:
        from tract.bridge.review import validate_candidates
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.5, "rank_for_ai_hub": 1,
             "description": "test", "status": "rejected", "reviewer_notes": "weak"},
        ])
        errors = validate_candidates(candidates_data, hier_data)
        assert errors == []


class TestCommitBridges:

    def test_accepted_creates_bidirectional_links(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1", "TRAD-2"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.7, "rank_for_ai_hub": 1,
             "description": "bridge", "status": "accepted", "reviewer_notes": ""},
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-2",
             "cosine_similarity": 0.3, "rank_for_ai_hub": 2,
             "description": "weak", "status": "rejected", "reviewer_notes": "too weak"},
        ])
        report = commit_bridges(candidates_data, hier_path, tmp_path / "report.json")

        updated = json.loads(hier_path.read_text())
        assert "TRAD-1" in updated["hubs"]["AI-1"]["related_hub_ids"]
        assert "AI-1" in updated["hubs"]["TRAD-1"]["related_hub_ids"]
        assert "TRAD-2" not in updated["hubs"]["AI-1"]["related_hub_ids"]

    def test_zero_accepted_is_valid(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.3, "rank_for_ai_hub": 1,
             "description": "weak", "status": "rejected", "reviewer_notes": "no bridge"},
        ])
        report = commit_bridges(candidates_data, hier_path, tmp_path / "report.json")
        assert report["counts"]["accepted"] == 0

    def test_report_json_written(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))
        report_path = tmp_path / "bridge_report.json"

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.7, "rank_for_ai_hub": 1,
             "description": "yes", "status": "accepted", "reviewer_notes": ""},
        ])
        commit_bridges(candidates_data, hier_path, report_path)
        assert report_path.exists()
        report = json.loads(report_path.read_text())
        assert report["counts"]["accepted"] == 1
        assert report["counts"]["rejected"] == 0

    def test_hierarchy_validates_after_commit(self, tmp_path) -> None:
        from tract.bridge.review import commit_bridges
        from tract.hierarchy import CREHierarchy
        hier_data = _make_hierarchy_data(["AI-1", "TRAD-1"])
        hier_path = tmp_path / "cre_hierarchy.json"
        hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

        candidates_data = _make_candidates_data([
            {"ai_hub_id": "AI-1", "trad_hub_id": "TRAD-1",
             "cosine_similarity": 0.7, "rank_for_ai_hub": 1,
             "description": "yes", "status": "accepted", "reviewer_notes": ""},
        ])
        commit_bridges(candidates_data, hier_path, tmp_path / "report.json")
        hier = CREHierarchy.load(hier_path)
        assert "TRAD-1" in hier.hubs["AI-1"].related_hub_ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bridge_review.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement review module**

Create `tract/bridge/review.py`:

```python
"""Bridge candidate review validation and hierarchy commit."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json

logger = logging.getLogger(__name__)

VALID_STATUSES = {"accepted", "rejected"}


def validate_candidates(
    candidates_data: dict,
    hierarchy_data: dict,
) -> list[str]:
    """Validate reviewed candidates. Returns list of error messages (empty = valid).

    Checks:
    - All candidates have status 'accepted' or 'rejected' (no 'pending')
    - All hub IDs exist in the hierarchy
    - Required keys present in each candidate
    """
    errors: list[str] = []
    hub_ids = set(hierarchy_data.get("hubs", {}).keys())
    required_keys = {"ai_hub_id", "trad_hub_id", "status"}

    for i, candidate in enumerate(candidates_data.get("candidates", [])):
        missing = required_keys - set(candidate.keys())
        if missing:
            errors.append(f"Candidate {i}: missing keys {missing}")
            continue

        status = candidate["status"]
        if status not in VALID_STATUSES:
            errors.append(
                f"Candidate {i} ({candidate['ai_hub_id']} -> {candidate['trad_hub_id']}): "
                f"invalid status '{status}', must be one of {VALID_STATUSES}"
            )

        for key in ("ai_hub_id", "trad_hub_id"):
            if candidate[key] not in hub_ids:
                errors.append(
                    f"Candidate {i}: {key} '{candidate[key]}' not in hierarchy"
                )

    return errors


def commit_bridges(
    candidates_data: dict,
    hierarchy_path: Path,
    report_path: Path,
) -> dict:
    """Commit accepted bridges to hierarchy and write bridge_report.json.

    Args:
        candidates_data: Reviewed bridge_candidates.json content.
        hierarchy_path: Path to cre_hierarchy.json (modified in place).
        report_path: Path to write bridge_report.json.

    Returns:
        The bridge report dict.
    """
    hier_data = load_json(hierarchy_path)

    errors = validate_candidates(candidates_data, hier_data)
    if errors:
        raise ValueError(
            f"Candidate validation failed with {len(errors)} errors:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    accepted = [
        c for c in candidates_data["candidates"]
        if c["status"] == "accepted"
    ]
    rejected = [
        c for c in candidates_data["candidates"]
        if c["status"] == "rejected"
    ]

    if accepted:
        for bridge in accepted:
            ai_id = bridge["ai_hub_id"]
            trad_id = bridge["trad_hub_id"]

            ai_related = hier_data["hubs"][ai_id].get("related_hub_ids", [])
            if trad_id not in ai_related:
                ai_related.append(trad_id)
            hier_data["hubs"][ai_id]["related_hub_ids"] = sorted(ai_related)

            trad_related = hier_data["hubs"][trad_id].get("related_hub_ids", [])
            if ai_id not in trad_related:
                trad_related.append(ai_id)
            hier_data["hubs"][trad_id]["related_hub_ids"] = sorted(trad_related)

        hier_data["version"] = "1.1"

        updated = CREHierarchy.model_validate(hier_data)
        updated.validate_integrity()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "method": candidates_data.get("method", "top_k_per_ai_hub"),
            "top_k": candidates_data.get("top_k", 3),
        },
        "counts": {
            "total": len(candidates_data.get("candidates", [])),
            "accepted": len(accepted),
            "rejected": len(rejected),
        },
        "candidates": [
            {
                "ai_hub_id": c["ai_hub_id"],
                "trad_hub_id": c["trad_hub_id"],
                "cosine_similarity": c["cosine_similarity"],
                "status": c["status"],
            }
            for c in candidates_data["candidates"]
        ],
        "similarity_stats": candidates_data.get("similarity_stats", {}),
    }

    # Write report BEFORE hierarchy — a crash between leaves report
    # as ground truth; hierarchy can be re-derived from report.
    atomic_write_json(report, report_path)
    logger.info("Wrote bridge report to %s", report_path)

    if accepted:
        updated.save(hierarchy_path)
        logger.info("Updated hierarchy with %d bridges", len(accepted))

    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bridge_review.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/bridge/review.py tests/test_bridge_review.py
git commit -m "feat(bridge): add candidate validation and bridge commit to hierarchy"
```

---

### Task 7: Bridge Orchestrator + CLI Command

**Files:**
- Modify: `tract/bridge/__init__.py`
- Modify: `tract/cli.py:27-32,1089-1101`
- Create: `tests/test_bridge_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bridge_cli.py`:

```python
"""Tests for bridge CLI command and orchestrator."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _make_mini_artifacts(tmp_path: Path, hub_ids: list[str]) -> Path:
    """Create a minimal deployment_artifacts.npz."""
    n = len(hub_ids)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, 1024)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    path = tmp_path / "deployment_artifacts.npz"
    np.savez(
        str(path),
        hub_embeddings=emb,
        control_embeddings=np.zeros((10, 1024), dtype=np.float32),
        hub_ids=np.array(hub_ids),
        control_ids=np.array([f"ctrl-{i}" for i in range(10)]),
    )
    return path


def _make_mini_hierarchy(tmp_path: Path, hub_ids: list[str]) -> Path:
    """Create a minimal cre_hierarchy.json."""
    hubs = {}
    for hid in hub_ids:
        hubs[hid] = {
            "hub_id": hid, "name": f"Hub {hid}", "parent_id": None,
            "children_ids": [], "depth": 0, "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}", "is_leaf": True,
            "sibling_hub_ids": [], "related_hub_ids": [],
        }
    data = {
        "hubs": hubs, "roots": sorted(hub_ids), "label_space": sorted(hub_ids),
        "fetch_timestamp": "2026-01-01T00:00:00", "data_hash": "test",
        "version": "1.1",
    }
    path = tmp_path / "cre_hierarchy.json"
    path.write_text(json.dumps(data, sort_keys=True, indent=2))
    return path


FIXTURE_LINKS = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"
ALL_HUB_IDS = ["AI-1", "AI-2", "AI-3", "BOTH-1", "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5", "UNLINKED-1"]


class TestRunBridgeAnalysis:

    def test_generates_candidates_file(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        candidates_path = output_dir / "bridge_candidates.json"
        assert candidates_path.exists()
        data = json.loads(candidates_path.read_text())
        assert len(data["candidates"]) == 6  # 3 AI-only × top-2

    def test_candidates_all_pending(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        data = json.loads((output_dir / "bridge_candidates.json").read_text())
        for c in data["candidates"]:
            assert c["status"] == "pending"

    def test_includes_negative_controls(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        data = json.loads((output_dir / "bridge_candidates.json").read_text())
        assert len(data["negative_controls"]) == 3  # 1 per AI-only hub

    def test_includes_similarity_stats(self, tmp_path) -> None:
        from tract.bridge import run_bridge_analysis
        artifacts_path = _make_mini_artifacts(tmp_path, ALL_HUB_IDS)
        hier_path = _make_mini_hierarchy(tmp_path, ALL_HUB_IDS)
        output_dir = tmp_path / "output"

        run_bridge_analysis(
            artifacts_path=artifacts_path,
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=hier_path,
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )
        data = json.loads((output_dir / "bridge_candidates.json").read_text())
        assert "matrix_shape" in data["similarity_stats"]


class TestBridgeCLIParsing:

    def test_bridge_subcommand_exists(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["bridge", "--skip-descriptions"])
        assert args.command == "bridge"
        assert args.skip_descriptions is True

    def test_bridge_top_k_default(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["bridge"])
        assert args.top_k == 3

    def test_bridge_commit_mode(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["bridge", "--commit", "--candidates", "cands.json"])
        assert args.commit is True
        assert args.candidates == "cands.json"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bridge_cli.py -v`
Expected: FAIL — `run_bridge_analysis` not in `tract.bridge`, bridge subcommand not in CLI

- [ ] **Step 3: Implement bridge orchestrator**

Replace `tract/bridge/__init__.py` content:

```python
"""TRACT bridge analysis — AI/traditional CRE hub bridge discovery."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tract.bridge.classify import classify_hubs
from tract.bridge.describe import count_controls_for_hub
from tract.bridge.similarity import (
    compute_bridge_similarities,
    compute_similarity_stats,
    extract_negatives,
    extract_top_k,
)
from tract.io import atomic_write_json

logger = logging.getLogger(__name__)


def run_bridge_analysis(
    *,
    artifacts_path: Path,
    hub_links_path: Path,
    hierarchy_path: Path,
    output_dir: Path,
    top_k: int = 3,
    skip_descriptions: bool = False,
) -> Path:
    """Run the full bridge analysis pipeline.

    Returns:
        Path to bridge_candidates.json.
    """
    from tract.hierarchy import CREHierarchy
    from tract.io import load_json

    data = np.load(str(artifacts_path), allow_pickle=False)
    hub_embeddings = data["hub_embeddings"]
    hub_ids = list(data["hub_ids"])

    hub_links = load_json(hub_links_path)
    hierarchy = CREHierarchy.load(hierarchy_path)

    classification = classify_hubs(hub_links_path, hub_ids)
    logger.info(
        "Classified %d AI-only, %d trad-only, %d bridged, %d unlinked",
        len(classification.ai_only), len(classification.trad_only),
        len(classification.naturally_bridged), len(classification.unlinked),
    )

    sim_matrix = compute_bridge_similarities(
        hub_embeddings, hub_ids,
        classification.ai_only, classification.trad_only,
    )
    stats = compute_similarity_stats(sim_matrix)

    candidates = extract_top_k(
        sim_matrix, classification.ai_only, classification.trad_only, k=top_k,
    )

    for candidate in candidates:
        ai_id = candidate["ai_hub_id"]
        trad_id = candidate["trad_hub_id"]
        candidate["ai_hub_name"] = hierarchy.hubs[ai_id].name
        candidate["trad_hub_name"] = hierarchy.hubs[trad_id].name
        candidate["seed_evidence"] = {
            "ai_controls_linked": count_controls_for_hub(ai_id, hub_links),
            "trad_controls_linked": count_controls_for_hub(trad_id, hub_links),
        }
        candidate["status"] = "pending"
        candidate["reviewer_notes"] = ""

    negatives_raw = extract_negatives(
        sim_matrix, classification.ai_only, classification.trad_only,
    )
    for neg in negatives_raw:
        ai_id = neg["ai_hub_id"]
        trad_id = neg["trad_hub_id"]
        neg["ai_hub_name"] = hierarchy.hubs[ai_id].name
        neg["trad_hub_name"] = hierarchy.hubs[trad_id].name

    if not skip_descriptions:
        from tract.bridge.describe import (
            generate_bridge_descriptions,
            generate_negative_descriptions,
        )
        generate_bridge_descriptions(candidates, hierarchy, hub_links)
        generate_negative_descriptions(negatives_raw, hierarchy, hub_links)
    else:
        for c in candidates:
            c.setdefault("description", "")
        for n in negatives_raw:
            n.setdefault("description", "")

    unclassified_leaves = [
        h for h in classification.unlinked
        if h in hierarchy.hubs and hierarchy.hubs[h].is_leaf
    ]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "top_k_per_ai_hub",
        "top_k": top_k,
        "similarity_stats": stats,
        "candidates": candidates,
        "negative_controls": negatives_raw,
        "unclassified_leaf_hubs": unclassified_leaves,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "bridge_candidates.json"
    atomic_write_json(output, candidates_path)
    logger.info("Wrote %d candidates to %s", len(candidates), candidates_path)

    print(f"AI-only hubs: {len(classification.ai_only)}")
    print(f"Candidates generated: {len(candidates)}")
    print(f"Output: {candidates_path}")

    return candidates_path
```

- [ ] **Step 4: Add bridge subcommand to CLI**

In `tract/cli.py`, add after the `prepare` subparser definition (before `def _cmd_assign`):

Add the import at the top of the file (after existing config imports):

```python
from tract.config import (
    BRIDGE_OUTPUT_DIR,
    BRIDGE_TOP_K,
    # ... existing imports ...
)
```

Add the subparser (after the `prepare` subparser):

```python
    # ── bridge ──────────────────────────────────────────────────
    p_bridge = subparsers.add_parser(
        "bridge",
        help="Discover AI/traditional CRE hub bridges",
        epilog=(
            "Examples:\n"
            "  tract bridge --skip-descriptions\n"
            "  tract bridge --top-k 5\n"
            "  tract bridge --commit --candidates results/bridge/bridge_candidates.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_bridge.add_argument("--output-dir", default=str(BRIDGE_OUTPUT_DIR), help="Output directory")
    p_bridge.add_argument("--top-k", type=int, default=BRIDGE_TOP_K, help="Top-K matches per AI hub")
    p_bridge.add_argument("--skip-descriptions", action="store_true", help="Skip LLM descriptions")
    p_bridge.add_argument("--commit", action="store_true", help="Commit reviewed candidates")
    p_bridge.add_argument("--candidates", help="Path to reviewed bridge_candidates.json (for --commit)")
```

Add the handler function:

```python
def _cmd_bridge(args: argparse.Namespace) -> None:
    if args.commit:
        if not args.candidates:
            print("Error: --commit requires --candidates <path>", file=sys.stderr)
            sys.exit(1)
        from tract.bridge.review import commit_bridges
        from tract.io import load_json

        candidates_path = Path(args.candidates)
        candidates_data = load_json(candidates_path)
        hierarchy_path = PROCESSED_DIR / "cre_hierarchy.json"
        report_path = Path(args.output_dir) / "bridge_report.json"

        report = commit_bridges(candidates_data, hierarchy_path, report_path)
        print(f"Accepted: {report['counts']['accepted']}")
        print(f"Rejected: {report['counts']['rejected']}")
        print(f"Hierarchy updated: {hierarchy_path}")
        print(f"Report: {report_path}")
    else:
        from tract.bridge import run_bridge_analysis
        from tract.config import PHASE1D_ARTIFACTS_PATH, TRAINING_DIR

        run_bridge_analysis(
            artifacts_path=PHASE1D_ARTIFACTS_PATH,
            hub_links_path=TRAINING_DIR / "hub_links_by_framework.json",
            hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
            output_dir=Path(args.output_dir),
            top_k=args.top_k,
            skip_descriptions=args.skip_descriptions,
        )
```

Add `"bridge": _cmd_bridge` to the handlers dict in `main()`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_bridge_cli.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: All existing tests pass + new tests pass

- [ ] **Step 7: Commit**

```bash
git add tract/bridge/__init__.py tract/cli.py tests/test_bridge_cli.py
git commit -m "feat(bridge): add bridge orchestrator and CLI command"
```

---

### Task 8: Bridge Integration Test

**Files:**
- Create: `tests/test_bridge_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_bridge_integration.py`:

```python
"""End-to-end bridge analysis integration test with synthetic data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

FIXTURE_LINKS = Path(__file__).parent / "fixtures" / "bridge_mini_hub_links.json"
ALL_HUB_IDS = [
    "AI-1", "AI-2", "AI-3", "BOTH-1",
    "TRAD-1", "TRAD-2", "TRAD-3", "TRAD-4", "TRAD-5",
    "UNLINKED-1",
]


@pytest.fixture
def bridge_workspace(tmp_path):
    """Set up a complete bridge analysis workspace."""
    n = len(ALL_HUB_IDS)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, 1024)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    artifacts_path = tmp_path / "artifacts.npz"
    np.savez(
        str(artifacts_path),
        hub_embeddings=emb,
        control_embeddings=np.zeros((5, 1024), dtype=np.float32),
        hub_ids=np.array(ALL_HUB_IDS),
        control_ids=np.array([f"c-{i}" for i in range(5)]),
    )

    hubs = {}
    for hid in ALL_HUB_IDS:
        hubs[hid] = {
            "hub_id": hid, "name": f"Hub {hid}", "parent_id": None,
            "children_ids": [], "depth": 0, "branch_root_id": hid,
            "hierarchy_path": f"Hub {hid}", "is_leaf": True,
            "sibling_hub_ids": [], "related_hub_ids": [],
        }
    hier_data = {
        "hubs": hubs, "roots": sorted(ALL_HUB_IDS),
        "label_space": sorted(ALL_HUB_IDS),
        "fetch_timestamp": "2026-01-01T00:00:00", "data_hash": "test",
        "version": "1.1",
    }
    hier_path = tmp_path / "cre_hierarchy.json"
    hier_path.write_text(json.dumps(hier_data, sort_keys=True, indent=2))

    return {
        "artifacts_path": artifacts_path,
        "hierarchy_path": hier_path,
        "tmp_path": tmp_path,
    }


class TestBridgeEndToEnd:

    def test_full_pipeline(self, bridge_workspace) -> None:
        from tract.bridge import run_bridge_analysis
        from tract.bridge.review import commit_bridges
        from tract.io import load_json

        ws = bridge_workspace
        output_dir = ws["tmp_path"] / "output"

        # Step 1: Generate candidates
        candidates_path = run_bridge_analysis(
            artifacts_path=ws["artifacts_path"],
            hub_links_path=FIXTURE_LINKS,
            hierarchy_path=ws["hierarchy_path"],
            output_dir=output_dir,
            top_k=2,
            skip_descriptions=True,
        )

        data = load_json(candidates_path)
        assert len(data["candidates"]) == 6  # 3 AI × top-2
        assert len(data["negative_controls"]) == 3  # 1 per AI hub

        # Step 2: Simulate expert review
        data["candidates"][0]["status"] = "accepted"
        for c in data["candidates"][1:]:
            c["status"] = "rejected"

        candidates_path.write_text(json.dumps(data, sort_keys=True, indent=2))

        # Step 3: Commit bridges
        report_path = output_dir / "bridge_report.json"
        report = commit_bridges(data, ws["hierarchy_path"], report_path)

        assert report["counts"]["accepted"] == 1
        assert report["counts"]["rejected"] == 5
        assert report_path.exists()

        # Step 4: Verify hierarchy updated
        from tract.hierarchy import CREHierarchy
        hier = CREHierarchy.load(ws["hierarchy_path"])
        accepted = data["candidates"][0]
        ai_id = accepted["ai_hub_id"]
        trad_id = accepted["trad_hub_id"]
        assert trad_id in hier.hubs[ai_id].related_hub_ids
        assert ai_id in hier.hubs[trad_id].related_hub_ids
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_bridge_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_bridge_integration.py
git commit -m "test(bridge): add end-to-end integration test"
```

---

### Task 9: LoRA Merge

**Files:**
- Create: `tract/publish/__init__.py`
- Create: `tract/publish/merge.py`
- Create: `tests/test_publish_merge.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish_merge.py`:

```python
"""Tests for tract.publish.merge — LoRA adapter merge into base model."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _create_merged_output(output_dir: Path) -> None:
    """Create the expected post-merge directory structure."""
    (output_dir / "0_Transformer").mkdir(parents=True)
    (output_dir / "0_Transformer" / "model.safetensors").write_bytes(b"fake-weights")
    (output_dir / "0_Transformer" / "config.json").write_text("{}")
    (output_dir / "1_Pooling").mkdir()
    (output_dir / "1_Pooling" / "config.json").write_text("{}")
    (output_dir / "2_Normalize").mkdir()
    (output_dir / "modules.json").write_text("[]")
    (output_dir / "sentence_bert_config.json").write_text("{}")
    (output_dir / "config_sentence_transformers.json").write_text("{}")


def _make_mock_model(fake_save_dir: Path | None = None) -> MagicMock:
    """Create a mock SentenceTransformer with encode() returning unit vectors."""
    mock_peft_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_peft_model

    mock_transformer_module = MagicMock()
    mock_transformer_module.auto_model = mock_peft_model

    mock_model = MagicMock()
    mock_model.__getitem__ = MagicMock(return_value=mock_transformer_module)

    fake_emb = np.ones((3, 1024), dtype=np.float32)
    fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)
    mock_model.encode.return_value = fake_emb

    if fake_save_dir is not None:
        def fake_save(path: str) -> None:
            _create_merged_output(Path(path))
        mock_model.save = fake_save

    return mock_model


class TestValidateMergedOutput:

    def test_rejects_leftover_adapter(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        _create_merged_output(output_dir)
        (output_dir / "0_Transformer" / "adapter_config.json").write_text("{}")
        with pytest.raises(RuntimeError, match="adapter_config.json"):
            validate_merged_output(output_dir)

    def test_rejects_missing_weights(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        output_dir.mkdir(parents=True)
        (output_dir / "0_Transformer").mkdir()
        (output_dir / "modules.json").write_text("[]")
        with pytest.raises(RuntimeError, match="model.safetensors"):
            validate_merged_output(output_dir)

    def test_accepts_clean_output(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        _create_merged_output(output_dir)
        validate_merged_output(output_dir)


class TestMergeLoraAdapters:

    def test_calls_merge_and_unload(self, tmp_path, mocker) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = tmp_path / "input"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        mocker.patch("tract.publish.merge.SentenceTransformer", return_value=mock_model)

        merge_lora_adapters(model_dir, output_dir)

        mock_model[0].auto_model.merge_and_unload.assert_called_once()
        assert mock_model.encode.call_count == 2  # pre-merge + post-merge

    def test_output_directory_created(self, tmp_path, mocker) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = tmp_path / "input"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        mocker.patch("tract.publish.merge.SentenceTransformer", return_value=mock_model)

        result = merge_lora_adapters(model_dir, output_dir)
        assert result == output_dir
        assert (output_dir / "0_Transformer" / "model.safetensors").exists()

    def test_fails_on_cosine_mismatch(self, tmp_path, mocker) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = tmp_path / "input"
        model_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        pre_emb = np.ones((3, 1024), dtype=np.float32)
        pre_emb /= np.linalg.norm(pre_emb, axis=1, keepdims=True)
        post_emb = np.random.default_rng(42).standard_normal((3, 1024)).astype(np.float32)
        post_emb /= np.linalg.norm(post_emb, axis=1, keepdims=True)
        mock_model.encode.side_effect = [pre_emb, post_emb]
        mocker.patch("tract.publish.merge.SentenceTransformer", return_value=mock_model)

        with pytest.raises(RuntimeError, match="Merge verification failed"):
            merge_lora_adapters(model_dir, output_dir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_publish_merge.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement merge module**

Create `tract/publish/__init__.py`:

```python
"""TRACT HuggingFace publication — model merge, bundling, and upload."""
```

Create `tract/publish/merge.py`:

```python
"""Merge LoRA adapters into base model for standalone distribution."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MERGE_VERIFICATION_TEXTS = [
    "Implement access controls for AI model training pipelines",
    "Data encryption at rest using AES-256",
    "Regularly audit AI system outputs for bias and fairness",
]
MERGE_COSINE_THRESHOLD = 0.9999


def validate_merged_output(output_dir: Path) -> None:
    """Validate that a merged model directory is correctly structured.

    Raises RuntimeError if adapter artifacts remain or weights are missing.
    """
    adapter_path = output_dir / "0_Transformer" / "adapter_config.json"
    if adapter_path.exists():
        raise RuntimeError(
            f"adapter_config.json still present after merge: {adapter_path}. "
            "Merge did not fully integrate LoRA weights."
        )

    weights_path = output_dir / "0_Transformer" / "model.safetensors"
    if not weights_path.exists():
        raise RuntimeError(
            f"model.safetensors not found in {output_dir / '0_Transformer'}. "
            "Merge may have failed."
        )


def merge_lora_adapters(
    model_dir: Path,
    output_dir: Path,
) -> Path:
    """Merge LoRA adapters into base model weights.

    Loads via SentenceTransformer (which auto-detects PEFT),
    captures pre-merge embeddings for verification,
    merges the adapter into the base weights, verifies
    cosine similarity > 0.9999, and saves the full
    SentenceTransformer directory structure.

    Args:
        model_dir: Path to SentenceTransformer directory with PEFT adapter overlay.
        output_dir: Path for merged output.

    Returns:
        output_dir path.

    Raises:
        RuntimeError: If merge verification fails (cosine < threshold).
    """
    logger.info("Loading model from %s", model_dir)
    model = SentenceTransformer(str(model_dir))

    logger.info("Computing pre-merge reference embeddings")
    pre_merge_emb = model.encode(
        MERGE_VERIFICATION_TEXTS,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    logger.info("Merging LoRA adapters into base weights")
    model[0].auto_model = model[0].auto_model.merge_and_unload()

    logger.info("Computing post-merge embeddings for verification")
    post_merge_emb = model.encode(
        MERGE_VERIFICATION_TEXTS,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    cosines = np.sum(pre_merge_emb * post_merge_emb, axis=1)
    min_cosine = float(np.min(cosines))
    logger.info("Merge verification: min cosine = %.6f (threshold: %.4f)", min_cosine, MERGE_COSINE_THRESHOLD)

    if min_cosine < MERGE_COSINE_THRESHOLD:
        raise RuntimeError(
            f"Merge verification failed: min cosine {min_cosine:.6f} < {MERGE_COSINE_THRESHOLD}. "
            f"Per-text cosines: {cosines.tolist()}"
        )

    logger.info("Saving merged model to %s", output_dir)
    model.save(str(output_dir))

    validate_merged_output(output_dir)
    logger.info("Merge complete — verified: no adapter artifacts, embeddings match")

    return output_dir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_publish_merge.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/publish/__init__.py tract/publish/merge.py tests/test_publish_merge.py
git commit -m "feat(publish): add SentenceTransformer-aware LoRA merge"
```

---

### Task 10: Inference Data Bundle

**Files:**
- Create: `tract/publish/bundle.py`
- Create: `tests/test_publish_bundle.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish_bundle.py`:

```python
"""Tests for tract.publish.bundle — inference data bundling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _setup_source_files(src: Path) -> dict[str, Path]:
    """Create minimal source files for bundling."""
    paths = {}

    desc_path = src / "hub_descriptions_reviewed.json"
    desc_path.write_text(json.dumps({"hub-1": "A hub"}, indent=2))
    paths["hub_descriptions"] = desc_path

    hier_data = {
        "hubs": {"hub-1": {"hub_id": "hub-1", "name": "Hub 1", "related_hub_ids": []}},
        "roots": ["hub-1"], "label_space": ["hub-1"],
        "fetch_timestamp": "2026-01-01", "data_hash": "test", "version": "1.1",
    }
    hier_path = src / "cre_hierarchy.json"
    hier_path.write_text(json.dumps(hier_data, indent=2))
    paths["hierarchy"] = hier_path

    cal_path = src / "calibration.json"
    cal_path.write_text(json.dumps({"t_deploy": 0.074, "ood_threshold": 0.568}))
    paths["calibration"] = cal_path

    artifacts_path = src / "deployment_artifacts.npz"
    hub_ids = np.array(["hub-1"])
    hub_emb = np.ones((1, 1024), dtype=np.float32)
    hub_emb /= np.linalg.norm(hub_emb)
    np.savez(str(artifacts_path), hub_embeddings=hub_emb, hub_ids=hub_ids,
             control_embeddings=np.zeros((1, 1024)), control_ids=np.array(["c-1"]))
    paths["artifacts"] = artifacts_path

    report_path = src / "bridge_report.json"
    report_path.write_text(json.dumps({"counts": {"accepted": 0}}))
    paths["bridge_report"] = report_path

    return paths


class TestBundleInferenceData:

    def test_all_files_copied(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        staging = tmp_path / "staging"
        staging.mkdir()

        bundle_inference_data(staging, **paths)

        assert (staging / "hub_descriptions.json").exists()
        assert (staging / "cre_hierarchy.json").exists()
        assert (staging / "calibration.json").exists()
        assert (staging / "hub_ids.json").exists()
        assert (staging / "bridge_report.json").exists()

    def test_hub_ids_extracted_from_npz(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        staging = tmp_path / "staging"
        staging.mkdir()

        bundle_inference_data(staging, **paths)

        hub_ids = json.loads((staging / "hub_ids.json").read_text())
        assert hub_ids == ["hub-1"]

    def test_hub_embeddings_bundled(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        staging = tmp_path / "staging"
        staging.mkdir()

        bundle_inference_data(staging, **paths)
        assert (staging / "hub_embeddings.npy").exists()

    def test_missing_file_raises(self, tmp_path) -> None:
        from tract.publish.bundle import bundle_inference_data
        src = tmp_path / "src"
        src.mkdir()
        paths = _setup_source_files(src)
        paths["hub_descriptions"].unlink()
        staging = tmp_path / "staging"
        staging.mkdir()

        with pytest.raises(FileNotFoundError):
            bundle_inference_data(staging, **paths)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_publish_bundle.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement bundle module**

Create `tract/publish/bundle.py`:

```python
"""Bundle inference data alongside the merged model for HuggingFace."""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def bundle_inference_data(
    staging_dir: Path,
    *,
    hub_descriptions: Path,
    hierarchy: Path,
    calibration: Path,
    artifacts: Path,
    bridge_report: Path,
) -> None:
    """Copy and validate inference data files into the staging directory.

    Args:
        staging_dir: Target directory (must exist).
        hub_descriptions: Path to hub_descriptions_reviewed.json.
        hierarchy: Path to cre_hierarchy.json (post-bridge).
        calibration: Path to calibration.json.
        artifacts: Path to deployment_artifacts.npz (hub_ids + hub_embeddings extracted).
        bridge_report: Path to bridge_report.json.

    Raises:
        FileNotFoundError: If any source file is missing.
    """
    copies: list[tuple[Path, str]] = [
        (hub_descriptions, "hub_descriptions.json"),
        (hierarchy, "cre_hierarchy.json"),
        (calibration, "calibration.json"),
        (bridge_report, "bridge_report.json"),
    ]

    for src, dest_name in copies:
        if not src.exists():
            raise FileNotFoundError(f"Required file not found: {src}")
        shutil.copy2(src, staging_dir / dest_name)
        logger.info("Copied %s -> %s", src.name, dest_name)

    if not artifacts.exists():
        raise FileNotFoundError(f"Artifacts not found: {artifacts}")

    data = np.load(str(artifacts), allow_pickle=False)
    hub_ids = list(data["hub_ids"])
    hub_embeddings = data["hub_embeddings"]

    hub_ids_path = staging_dir / "hub_ids.json"
    with open(hub_ids_path, "w", encoding="utf-8") as f:
        json.dump(hub_ids, f, indent=2)
        f.write("\n")
    logger.info("Extracted %d hub IDs to hub_ids.json", len(hub_ids))

    emb_path = staging_dir / "hub_embeddings.npy"
    np.save(str(emb_path), hub_embeddings)
    logger.info(
        "Saved hub embeddings (%s) to hub_embeddings.npy",
        hub_embeddings.shape,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_publish_bundle.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/publish/bundle.py tests/test_publish_bundle.py
git commit -m "feat(publish): add inference data bundling"
```

---

### Task 11: Model Card Generation

**Files:**
- Create: `tract/publish/model_card.py`
- Create: `tests/test_publish_model_card.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish_model_card.py`:

```python
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

SAMPLE_CALIBRATION = {
    "t_deploy": 0.074,
    "ood_threshold": 0.568,
    "conformal_quantile": 0.997,
}

SAMPLE_ECE = {"ece": 0.079, "ece_ci": {"ci_low": 0.049, "ci_high": 0.111}}

SAMPLE_BRIDGE = {"counts": {"accepted": 5, "rejected": 58, "total": 63}}


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
        from tract.publish.model_card import generate_model_card
        generate_model_card(
            tmp_path, fold_results=SAMPLE_FOLD_RESULTS,
            calibration=SAMPLE_CALIBRATION, ece_data=SAMPLE_ECE,
            bridge_summary=SAMPLE_BRIDGE, gpu_hours=2.5,
        )
        content = (tmp_path / "README.md").read_text()
        assert "MIT" in content

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_publish_model_card.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement model card generator**

Create `tract/publish/model_card.py`:

```python
"""Generate AIBOM-compliant model card README.md for HuggingFace."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_model_card(
    staging_dir: Path,
    *,
    fold_results: list[dict],
    calibration: dict,
    ece_data: dict,
    bridge_summary: dict,
    gpu_hours: float,
) -> None:
    """Generate README.md model card in the staging directory.

    Args:
        staging_dir: Directory to write README.md.
        fold_results: List of dicts with fold, hit1, zs_hit1, n, hit_any.
        calibration: Dict with t_deploy, ood_threshold, conformal_quantile.
        ece_data: Dict with ece, ece_ci.ci_low, ece_ci.ci_high.
        bridge_summary: Dict with counts.accepted, counts.rejected, counts.total.
        gpu_hours: Actual GPU training hours for environmental impact.
    """
    lofo_rows = ""
    total_n = 0
    for fold in fold_results:
        delta = fold["hit1"] - fold["zs_hit1"]
        lofo_rows += (
            f"| {fold['fold']} | {fold['hit1']:.3f} | {fold['zs_hit1']:.3f} "
            f"| {delta:+.3f} | {fold.get('hit_any', 'N/A')} | {fold['n']} |\n"
        )
        total_n += fold["n"]

    micro_hit1 = sum(f["hit1"] * f["n"] for f in fold_results) / total_n
    micro_zs = sum(f["zs_hit1"] * f["n"] for f in fold_results) / total_n
    micro_delta = micro_hit1 - micro_zs
    hit_any_folds = [f for f in fold_results if isinstance(f.get("hit_any"), (int, float))]
    if hit_any_folds:
        micro_hit_any = sum(f["hit_any"] * f["n"] for f in hit_any_folds) / sum(f["n"] for f in hit_any_folds)
        hit_any_str = f"**{micro_hit_any:.3f}**"
    else:
        hit_any_str = "—"
    lofo_rows += (
        f"| **Micro average** | **{micro_hit1:.3f}** | **{micro_zs:.3f}** "
        f"| **{micro_delta:+.3f}** | {hit_any_str} | **{total_n}** |\n"
    )

    t_val = calibration.get("t_deploy", 0.074)
    ood_val = calibration.get("ood_threshold", 0.568)
    conformal_val = calibration.get("conformal_quantile", 0.997)
    ece_val = ece_data.get("ece", 0.079)
    ece_ci = ece_data.get("ece_ci", {})
    ece_low = ece_ci.get("ci_low", 0.049)
    ece_high = ece_ci.get("ci_high", 0.111)

    bridge_counts = bridge_summary.get("counts", {})
    n_accepted = bridge_counts.get("accepted", 0)
    n_rejected = bridge_counts.get("rejected", 0)
    n_total = bridge_counts.get("total", 0)

    card = f"""---
language: en
license: mit
tags:
  - security
  - compliance
  - cre
  - sentence-transformers
  - bi-encoder
library_name: sentence-transformers
pipeline_tag: sentence-similarity
---

# TRACT: Transitive Reconciliation and Assignment of CRE Taxonomies

## Model Description

TRACT maps security framework control text to [OpenCRE](https://opencre.org) hub positions via a fine-tuned bi-encoder. It implements the assignment paradigm: `g(control_text) → CRE_position` — each control is independently mapped to the CRE ontology, not compared pairwise.

- **Label space:** 522 CRE hubs (400 leaf hubs as classification targets)
- **Input:** Free-text security control description
- **Output:** Ranked list of CRE hub predictions with calibrated confidence scores

## Architecture

- **Base model:** BAAI/bge-large-en-v1.5 (335M parameters)
- **Fine-tuning:** LoRA rank=16, alpha=32, dropout=0.1, target modules: query/key/value
- **Training:** MNRL contrastive loss with text-aware batch sampling, 20 epochs, batch size=32, lr=5e-4, seed=42
- **Training data:** 4,237 framework-to-hub links → 4,061 training pairs from 22 OpenCRE-linked frameworks

## Evaluation (LOFO Cross-Validation)

Leave-one-framework-out cross-validation with hub firewall (no information leakage from held-out framework into hub representations):

| Fold | hit@1 | Zero-shot | Delta | hit@any | n |
|---|---|---|---|---|---|
{lofo_rows}
Bootstrap confidence intervals (10,000 resamples, 95% CI) are available in the per-fold summary files.

## Calibration

- **Temperature scaling:** T={t_val:.4f}
- **ECE:** {ece_val:.3f}, 95% CI [{ece_low:.3f}, {ece_high:.3f}]
- **OOD threshold:** {ood_val:.3f} (96.7% separation rate)
- **Conformal coverage:** quantile={conformal_val:.4f}

## Limitations

- ATLAS fold shows near-zero improvement (+0.006) — hub disambiguation between closely related ATLAS techniques is the primary failure mode
- ECE={ece_val:.3f} indicates imperfect calibration; confidence scores are ordinal rankings, not true probabilities
- 35% of controls map to multiple hubs — predictions are multi-label by design, hit@1 alone understates performance
- Calibrated on 420 traditional framework holdout items; accuracy on AI-specific text may differ
- DeBERTa-v3-NLI completely fails for this task (hit@1=0.000) — NLI is not semantic similarity

## Ethical Considerations

- Not a replacement for expert judgment in compliance decisions
- Model predictions require human review before use in security assessments
- Active learning rounds used expert-reviewed predictions, not autonomous deployment

## Environmental Impact

- **Training:** H100 GPU via RunPod, {gpu_hours:.1f} GPU-hours
- **Deployment:** Runs on NVIDIA Jetson Orin AGX (edge device, ~30W TDP)

## Bridge Analysis Summary

Bridge analysis identified conceptual overlaps between AI-specific and traditional CRE hubs using hub embedding similarity (top-3 per AI-only hub, expert-reviewed).

- **Candidates evaluated:** {n_total}
- **Accepted bridges:** {n_accepted}
- **Rejected:** {n_rejected}

Full bridge evidence and review decisions are in `bridge_report.json`.

## Usage

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import json

model = SentenceTransformer("rockCO78/tract-cre-assignment")

with open("hub_ids.json") as f:
    hub_ids = json.load(f)
hub_emb = np.load("hub_embeddings.npy")

query = model.encode(["Ensure AI models are tested for adversarial robustness"], normalize_embeddings=True)
similarities = query @ hub_emb.T
top_k = np.argsort(similarities[0])[-5:][::-1]
for idx in top_k:
    print(f"{{hub_ids[idx]}}: {{similarities[0][idx]:.3f}}")
```

See `predict.py` for a complete standalone inference script with calibration.

## Citation

```bibtex
@software{{tract2026,
  title = {{TRACT: Transitive Reconciliation and Assignment of CRE Taxonomies}},
  author = {{Rock}},
  year = {{2026}},
  url = {{https://github.com/rockcyber/TRACT}}
}}
```

## License

MIT License for model weights and code. The base model (BAAI/bge-large-en-v1.5) is also MIT licensed.

Bundled data files (CRE hierarchy, hub descriptions, bridge report) are CC0 1.0 Universal.
"""

    readme_path = staging_dir / "README.md"
    readme_path.write_text(card, encoding="utf-8")
    logger.info("Wrote model card to %s", readme_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_publish_model_card.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/publish/model_card.py tests/test_publish_model_card.py
git commit -m "feat(publish): add AIBOM-compliant model card generator"
```

---

### Task 12: Standalone Scripts

**Files:**
- Create: `tract/publish/scripts.py`
- Create: `tests/test_publish_scripts.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish_scripts.py`:

```python
"""Tests for tract.publish.scripts — standalone predict.py and train.py."""
from __future__ import annotations

from pathlib import Path

import pytest


class TestWritePredictScript:

    def test_creates_file(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        assert (tmp_path / "predict.py").exists()

    def test_contains_imports(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "sentence_transformers" in content
        assert "numpy" in content

    def test_contains_sanitize(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "sanitize_text" in content
        assert "html.unescape" in content  # full pipeline, not simplified 2-step

    def test_contains_temperature_scaling(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "temperature" in content.lower() or "softmax" in content.lower()

    def test_contains_ood_flag(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "ood" in content.lower()

    def test_is_valid_python(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        compile(content, "predict.py", "exec")

    def test_no_tract_import(self, tmp_path) -> None:
        from tract.publish.scripts import write_predict_script
        write_predict_script(tmp_path)
        content = (tmp_path / "predict.py").read_text()
        assert "from tract" not in content
        assert "import tract" not in content


class TestWriteTrainScript:

    def test_creates_file(self, tmp_path) -> None:
        from tract.publish.scripts import write_train_script
        write_train_script(tmp_path)
        assert (tmp_path / "train.py").exists()

    def test_contains_reproduction_note(self, tmp_path) -> None:
        from tract.publish.scripts import write_train_script
        write_train_script(tmp_path)
        content = (tmp_path / "train.py").read_text()
        assert "TRACT" in content
        assert "clone" in content.lower() or "repository" in content.lower()

    def test_contains_hyperparameters(self, tmp_path) -> None:
        from tract.publish.scripts import write_train_script
        write_train_script(tmp_path)
        content = (tmp_path / "train.py").read_text()
        assert "seed" in content.lower()
        assert "lora" in content.lower() or "LoRA" in content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_publish_scripts.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement scripts module**

Create `tract/publish/scripts.py`:

```python
"""Generate standalone predict.py and train.py for HuggingFace repository."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PREDICT_SCRIPT = '''\
"""Standalone inference script for TRACT CRE hub assignment.

Dependencies: sentence-transformers, torch, numpy
No TRACT package required — all inference logic is inlined.

Usage:
    python predict.py "Ensure AI models are tested for bias"
    python predict.py --file controls.txt --top-k 10
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


def sanitize_text(text: str) -> str:
    """Full sanitization pipeline matching training-time preprocessing.

    Steps: null bytes → NFC → zero-width chars → HTML unescape+strip →
    PDF ligatures → broken hyphenation → whitespace collapse → strip.
    Must match tract/sanitize.py exactly to avoid train/inference skew.
    """
    import html
    import re

    text = text.replace("\\x00", " ")
    text = unicodedata.normalize("NFC", text)
    text = re.sub("[\\u200b\\u200c\\u200d\\ufeff]", "", text)
    text = re.sub(r"</?[a-zA-Z][^>]*>", "", html.unescape(text))
    for lig, repl in [("\\ufb04", "ffl"), ("\\ufb03", "ffi"), ("\\ufb00", "ff"), ("\\ufb01", "fi"), ("\\ufb02", "fl")]:
        text = text.replace(lig, repl)
    text = re.sub(r"(\\w)-\\n(\\w)", r"\\1\\2", text)
    text = re.sub(r"\\s+", " ", text)
    return text.strip()


def softmax(x):
    """Numerically stable softmax."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def predict(
    texts: list[str],
    model_dir: str = ".",
    top_k: int = 5,
) -> list[list[dict]]:
    """Predict CRE hub assignments for input texts.

    Args:
        texts: List of control text strings.
        model_dir: Path to this repository (contains model + bundled data).
        top_k: Number of top predictions to return.

    Returns:
        List of prediction lists, one per input text.
    """
    base = Path(model_dir)
    model = SentenceTransformer(str(base))

    with open(base / "calibration.json") as f:
        cal = json.load(f)
    with open(base / "hub_ids.json") as f:
        hub_ids = json.load(f)
    with open(base / "cre_hierarchy.json") as f:
        hierarchy = json.load(f)

    hub_emb = np.load(str(base / "hub_embeddings.npy"))
    temperature = cal["t_deploy"]
    ood_threshold = cal["ood_threshold"]

    cleaned = [sanitize_text(t) for t in texts]
    query_emb = model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
    similarities = query_emb @ hub_emb.T

    calibrated = softmax(similarities / temperature)

    results = []
    for i in range(len(texts)):
        sims = similarities[i]
        confs = calibrated[i]
        max_sim = float(np.max(sims))
        is_ood = max_sim < ood_threshold

        top_indices = np.argsort(confs)[-top_k:][::-1]
        preds = []
        for idx in top_indices:
            hub_id = hub_ids[idx]
            hub_info = hierarchy.get("hubs", {}).get(hub_id, {})
            preds.append({
                "hub_id": hub_id,
                "hub_name": hub_info.get("name", hub_id),
                "hierarchy_path": hub_info.get("hierarchy_path", ""),
                "raw_similarity": round(float(sims[idx]), 4),
                "calibrated_confidence": round(float(confs[idx]), 4),
                "is_ood": is_ood,
            })
        results.append(preds)
    return results


def main():
    parser = argparse.ArgumentParser(description="TRACT CRE hub assignment")
    parser.add_argument("text", nargs="?", help="Control text to assign")
    parser.add_argument("--file", help="File with one control per line")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions")
    parser.add_argument("--model-dir", default=".", help="Path to model directory")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            texts = [line.strip() for line in f if line.strip()]
    elif args.text:
        texts = [args.text]
    else:
        parser.print_help()
        sys.exit(1)

    results = predict(texts, model_dir=args.model_dir, top_k=args.top_k)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for i, preds in enumerate(results):
            if len(texts) > 1:
                print(f"\\n--- Control {i+1}: {texts[i][:80]} ---")
            for p in preds:
                ood = " [OOD]" if p["is_ood"] else ""
                print(f"  {p['hub_id']} ({p['calibrated_confidence']:.3f}){ood} {p['hub_name']}")


if __name__ == "__main__":
    main()
'''

TRAIN_SCRIPT = '''\
"""TRACT model reproduction guide.

Full reproduction requires cloning the TRACT repository, which contains
custom training procedures: text-aware batch sampling, joint temperature-scaled
contrastive loss, LOFO cross-validation with hub firewall, and active learning.

This script documents the exact configuration used for training.
"""

# ── Reproduction Steps ──────────────────────────────────────────────────
#
# 1. Clone the TRACT repository:
#    git clone https://github.com/rockcyber/TRACT.git
#    cd TRACT
#    pip install -e ".[train]"
#
# 2. Fetch training data:
#    python -m tract.cli prepare  # Fetches OpenCRE data, parses frameworks
#
# 3. Run training with the exact configuration below:
#    python scripts/phase1b/train.py \\
#        --base-model BAAI/bge-large-en-v1.5 \\
#        --lora-rank 16 --lora-alpha 32 --lora-dropout 0.1 \\
#        --target-modules query key value \\
#        --batch-size 32 --lr 5e-4 --epochs 20 \\
#        --warmup-ratio 0.1 --weight-decay 0.01 \\
#        --hard-negatives 3 --sampling-temperature 2.0 \\
#        --max-seq-length 512 --seed 42 \\
#        --hub-rep-format path+name \\
#        --training-data joint-tempscaled
#
# 4. Run LOFO evaluation:
#    python scripts/phase1b/evaluate_lofo.py
#
# 5. Run calibration + deployment:
#    python scripts/phase1c/calibrate.py
#    python scripts/phase1c/deploy.py

# ── Key Training Details ────────────────────────────────────────────────
#
# Base model: BAAI/bge-large-en-v1.5 (335M params, 1024-dim embeddings)
# LoRA: rank=16, alpha=32, dropout=0.1, targets=query/key/value
# Loss: MNRL (Multiple Negatives Ranking Loss) with contrastive objective
# Batch sampling: Text-aware — controls grouped by text similarity
# Training data: 4,237 framework-to-hub links → 4,061 pairs, 22 frameworks
# Seed: 42 (all randomness: torch, numpy, random)
#
# ── Pinned Requirements ─────────────────────────────────────────────────
#
# sentence-transformers>=3.0.0
# torch>=2.1.0
# peft>=0.7.0
# numpy>=1.24.0
# scipy>=1.11.0
# wandb>=0.16.0
#
# See requirements.txt in the TRACT repository for exact pinned versions.
'''


def write_predict_script(staging_dir: Path) -> None:
    """Write standalone predict.py to staging directory."""
    path = staging_dir / "predict.py"
    path.write_text(PREDICT_SCRIPT, encoding="utf-8")
    logger.info("Wrote predict.py to %s", path)


def write_train_script(staging_dir: Path) -> None:
    """Write train.py reproduction guide to staging directory."""
    path = staging_dir / "train.py"
    path.write_text(TRAIN_SCRIPT, encoding="utf-8")
    logger.info("Wrote train.py to %s", path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_publish_scripts.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/publish/scripts.py tests/test_publish_scripts.py
git commit -m "feat(publish): add standalone predict.py and train.py scripts"
```

---

### Task 13: Security Scan

**Files:**
- Create: `tract/publish/security.py`
- Create: `tests/test_publish_security.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish_security.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_publish_security.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement security scan module**

Create `tract/publish/security.py`:

```python
"""Context-aware security scan for HuggingFace upload staging directory."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from tract.config import HF_SCAN_EXTENSIONS, HF_SECRET_PATTERNS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SecurityFinding:
    file_path: str
    line_number: int
    pattern_name: str
    matched_text: str


def _scan_file_contents(file_path: Path, findings: list[SecurityFinding]) -> None:
    """Scan a text file against all secret patterns."""
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return

    for line_num, line in enumerate(content.splitlines(), 1):
        for pattern in HF_SECRET_PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append(SecurityFinding(
                    file_path=str(file_path),
                    line_number=line_num,
                    pattern_name=pattern.pattern,
                    matched_text=match.group()[:50],
                ))


def _scan_bridge_report_notes(file_path: Path, findings: list[SecurityFinding]) -> None:
    """Scan reviewer_notes fields in bridge_report.json for PII."""
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    for i, candidate in enumerate(data.get("candidates", [])):
        notes = candidate.get("reviewer_notes", "")
        if not notes:
            continue
        for pattern in HF_SECRET_PATTERNS:
            match = pattern.search(notes)
            if match:
                findings.append(SecurityFinding(
                    file_path=str(file_path),
                    line_number=i,
                    pattern_name=f"reviewer_notes: {pattern.pattern}",
                    matched_text=match.group()[:50],
                ))


def scan_for_secrets(staging_dir: Path) -> list[SecurityFinding]:
    """Scan staging directory for secrets before HuggingFace upload.

    Scans:
    - .py, .md, .txt, .yaml, .yml files against all secret patterns
    - bridge_report.json reviewer_notes fields for PII
    - Structural: no .git/ directory, no adapter_config.json

    Returns:
        List of findings. Any non-empty list = hard failure.
    """
    findings: list[SecurityFinding] = []

    if (staging_dir / ".git").exists():
        findings.append(SecurityFinding(
            file_path=str(staging_dir / ".git"),
            line_number=0,
            pattern_name="structural",
            matched_text=".git directory present in staging area",
        ))

    if (staging_dir / "adapter_config.json").exists():
        findings.append(SecurityFinding(
            file_path=str(staging_dir / "adapter_config.json"),
            line_number=0,
            pattern_name="structural",
            matched_text="adapter_config.json present — merge incomplete",
        ))

    for path in staging_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix in HF_SCAN_EXTENSIONS:
            _scan_file_contents(path, findings)

    bridge_report = staging_dir / "bridge_report.json"
    if bridge_report.exists():
        _scan_bridge_report_notes(bridge_report, findings)

    if findings:
        logger.error("Security scan found %d issues:", len(findings))
        for f in findings:
            logger.error("  %s:%d — %s", f.file_path, f.line_number, f.pattern_name)
    else:
        logger.info("Security scan passed — no secrets detected")

    return findings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_publish_security.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add tract/publish/security.py tests/test_publish_security.py
git commit -m "feat(publish): add context-aware security scan for HF upload"
```

---

### Task 14: Publish Orchestrator + CLI Command

**Files:**
- Modify: `tract/publish/__init__.py`
- Modify: `tract/cli.py`
- Create: `tests/test_publish_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_publish_cli.py`:

```python
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
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo"])
        assert args.command == "publish-hf"
        assert args.repo_id == "test/repo"

    def test_dry_run_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--dry-run"])
        assert args.dry_run is True

    def test_skip_upload_flag(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--skip-upload"])
        assert args.skip_upload is True

    def test_gpu_hours_param(self) -> None:
        from tract.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["publish-hf", "--repo-id", "test/repo", "--gpu-hours", "2.5"])
        assert args.gpu_hours == 2.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_publish_cli.py -v`
Expected: FAIL — `check_publication_gate` not found, `publish-hf` subcommand not in CLI

- [ ] **Step 3: Implement publish orchestrator**

Replace `tract/publish/__init__.py`:

```python
"""TRACT HuggingFace publication — model merge, bundling, and upload."""
from __future__ import annotations

import logging
from pathlib import Path

from tract.io import load_json

logger = logging.getLogger(__name__)


def check_publication_gate(
    bridge_report_path: Path,
    hierarchy_path: Path | None = None,
) -> None:
    """Verify bridge analysis is complete before publication.

    Raises ValueError if:
    - bridge_report.json does not exist
    - Any candidate has status 'pending'
    - Accepted bridges exist but hierarchy not updated (version != "1.1")
    """
    if not bridge_report_path.exists():
        raise ValueError(
            f"bridge_report.json not found at {bridge_report_path}. "
            "Run 'tract bridge' and 'tract bridge --commit' first."
        )

    report = load_json(bridge_report_path)
    pending = [
        c for c in report.get("candidates", [])
        if c.get("status") == "pending"
    ]
    if pending:
        raise ValueError(
            f"{len(pending)} candidates still have 'pending' status in "
            f"{bridge_report_path}. Review all candidates before publishing."
        )

    accepted = [
        c for c in report.get("candidates", [])
        if c.get("status") == "accepted"
    ]
    if accepted and hierarchy_path:
        hier = load_json(hierarchy_path)
        if hier.get("version") != "1.1":
            raise ValueError(
                f"Bridge report has {len(accepted)} accepted bridges but "
                f"hierarchy version is '{hier.get('version')}', not '1.1'. "
                "Run 'tract bridge --commit' to update the hierarchy."
            )
        for bridge in accepted:
            ai_id = bridge["ai_hub_id"]
            trad_id = bridge["trad_hub_id"]
            ai_related = hier.get("hubs", {}).get(ai_id, {}).get("related_hub_ids", [])
            if trad_id not in ai_related:
                raise ValueError(
                    f"Accepted bridge {ai_id}↔{trad_id} not found in "
                    f"hierarchy related_hub_ids. Run 'tract bridge --commit'."
                )


def publish_to_huggingface(
    *,
    repo_id: str,
    staging_dir: Path,
    model_dir: Path,
    artifacts_path: Path,
    hierarchy_path: Path,
    hub_descriptions_path: Path,
    calibration_path: Path,
    ece_gate_path: Path,
    bridge_report_path: Path,
    fold_results: list[dict],
    gpu_hours: float,
    dry_run: bool = False,
    skip_upload: bool = False,
) -> None:
    """Full HuggingFace publication pipeline.

    Steps: gate check → merge → bundle → model card → scripts → security scan → upload.
    """
    import shutil

    from tract.publish.bundle import bundle_inference_data
    from tract.publish.merge import merge_lora_adapters
    from tract.publish.model_card import generate_model_card
    from tract.publish.scripts import write_predict_script, write_train_script
    from tract.publish.security import scan_for_secrets

    check_publication_gate(bridge_report_path, hierarchy_path=hierarchy_path)
    logger.info("Publication gate passed")

    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)

    print("Step 1/7: Merging LoRA adapters...")
    merge_lora_adapters(model_dir, staging_dir)

    print("Step 2/7: Bundling inference data...")
    calibration = load_json(calibration_path)
    ece_data = load_json(ece_gate_path)
    bridge_summary = load_json(bridge_report_path)

    bundle_inference_data(
        staging_dir,
        hub_descriptions=hub_descriptions_path,
        hierarchy=hierarchy_path,
        calibration=calibration_path,
        artifacts=artifacts_path,
        bridge_report=bridge_report_path,
    )

    print("Step 3/7: Generating model card...")
    generate_model_card(
        staging_dir,
        fold_results=fold_results,
        calibration=calibration,
        ece_data=ece_data,
        bridge_summary=bridge_summary,
        gpu_hours=gpu_hours,
    )

    print("Step 4/7: Writing standalone scripts...")
    write_predict_script(staging_dir)
    write_train_script(staging_dir)

    print("Step 5/7: AIBOM validation...")
    _validate_aibom(staging_dir)

    print("Step 6/7: Running security scan...")
    findings = scan_for_secrets(staging_dir)
    if findings:
        for f in findings:
            print(f"  ALERT: {f.file_path}:{f.line_number} — {f.pattern_name}")
        raise ValueError(
            f"Security scan found {len(findings)} issues. Fix and re-run."
        )
    print("  Security scan passed")

    if dry_run:
        print(f"\nDry run complete. Staging directory: {staging_dir}")
        print("Run without --dry-run to upload.")
        return

    if skip_upload:
        print(f"\nBuild complete. Staging directory: {staging_dir}")
        print("Run without --skip-upload to upload.")
        return

    print("Step 7/7: Uploading to HuggingFace...")
    _upload_to_hub(repo_id, staging_dir)
    print(f"\nPublished to https://huggingface.co/{repo_id}")


AIBOM_REPO = "https://github.com/GenAI-Security-Project/aibom-generator.git"
AIBOM_COMMIT_SHA = "main"  # TODO: Pin to specific SHA after verifying a known-good commit


def _validate_aibom(staging_dir: Path) -> None:
    """Run AIBOM validator against the generated model card.

    Clones aibom-generator at a pinned commit to a temp dir, runs it on a
    COPY of README.md (not the staging dir), and reports the score.
    Non-blocking: logs a warning if the tool is unavailable or broken.
    """
    import shutil
    import subprocess
    import tempfile

    readme = staging_dir / "README.md"
    if not readme.exists():
        logger.warning("No README.md found in staging dir — skipping AIBOM validation")
        return

    try:
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "clone", "--depth=1", "--branch", AIBOM_COMMIT_SHA,
                 AIBOM_REPO, tmp],
                check=True, capture_output=True, timeout=60,
            )
            readme_copy = Path(tmp) / "README_to_validate.md"
            shutil.copy2(readme, readme_copy)
            result = subprocess.run(
                ["python", "-m", "aibom_generator", str(readme_copy)],
                capture_output=True, text=True, timeout=120, cwd=tmp,
            )
            print(f"  AIBOM output:\n{result.stdout}")
            if result.returncode != 0:
                logger.warning("AIBOM validation returned non-zero: %s", result.stderr)
    except Exception as e:
        logger.warning("AIBOM validation skipped — tool unavailable: %s", e)
        print(f"  AIBOM validation skipped (tool unavailable: {e})")


def _upload_to_hub(repo_id: str, staging_dir: Path) -> None:
    """Upload staging directory to HuggingFace Hub."""
    import subprocess

    from huggingface_hub import HfApi

    token = subprocess.check_output(
        ["pass", "huggingface/token"], text=True
    ).strip()

    try:
        api = HfApi(token=token)
        api.upload_folder(
            folder_path=str(staging_dir),
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info("Uploaded to %s", repo_id)
    finally:
        del token
```

- [ ] **Step 4: Add publish-hf subcommand to CLI**

In `tract/cli.py`, add the import at the top:

```python
from tract.config import (
    BRIDGE_OUTPUT_DIR,
    BRIDGE_TOP_K,
    HF_DEFAULT_REPO_ID,
    HF_STAGING_DIR,
    # ... existing imports ...
)
```

Add the subparser (after the bridge subparser):

```python
    # ── publish-hf ──────────────────────────────────────────────
    p_publish = subparsers.add_parser(
        "publish-hf",
        help="Publish model to HuggingFace Hub",
        epilog=(
            "Examples:\n"
            "  tract publish-hf --repo-id rockCO78/tract-cre-assignment --dry-run\n"
            "  tract publish-hf --repo-id rockCO78/tract-cre-assignment\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_publish.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    p_publish.add_argument("--staging-dir", default=str(HF_STAGING_DIR), help="Local build dir")
    p_publish.add_argument("--dry-run", action="store_true", help="Build + scan, no upload")
    p_publish.add_argument("--skip-upload", action="store_true", help="Build + scan only")
    p_publish.add_argument("--gpu-hours", type=float, default=0.0, help="GPU training hours for model card")
```

Add the handler:

```python
def _cmd_publish_hf(args: argparse.Namespace) -> None:
    from tract.config import (
        PHASE1B_CORRECTED_METRICS_PATH,
        PHASE1B_TEXTAWARE_RESULTS_DIR,
        PHASE1C_ECE_GATE_PATH,
        PHASE1D_ARTIFACTS_PATH,
        PHASE1D_CALIBRATION_PATH,
    )
    from tract.publish import publish_to_huggingface
    from tract.io import load_json

    model_dir = PHASE1D_DEPLOYMENT_MODEL_DIR / "model" / "model"
    bridge_report = BRIDGE_OUTPUT_DIR / "bridge_report.json"

    fold_results = _load_fold_results(
        PHASE1B_TEXTAWARE_RESULTS_DIR, PHASE1B_CORRECTED_METRICS_PATH,
    )

    publish_to_huggingface(
        repo_id=args.repo_id,
        staging_dir=Path(args.staging_dir),
        model_dir=model_dir,
        artifacts_path=PHASE1D_ARTIFACTS_PATH,
        hierarchy_path=PROCESSED_DIR / "cre_hierarchy.json",
        hub_descriptions_path=PROCESSED_DIR / "hub_descriptions_reviewed.json",
        calibration_path=PHASE1D_CALIBRATION_PATH,
        ece_gate_path=PHASE1C_ECE_GATE_PATH,
        bridge_report_path=bridge_report,
        fold_results=fold_results,
        gpu_hours=args.gpu_hours,
        dry_run=args.dry_run,
        skip_upload=args.skip_upload,
    )


def _load_fold_results(
    textaware_dir: Path,
    corrected_path: Path,
    zero_shot_path: Path | None = None,
) -> list[dict]:
    """Load LOFO fold results from Phase 1B artifacts.

    IMPORTANT: eval count comes from len(predictions.json), NOT n_pairs
    (n_pairs is the training pair count ~3900, not eval items 43-63).
    corrected_metrics must come from the TEXTAWARE experiment (same model),
    not from phase1b_primary (different model, different corpus).
    """
    from tract.io import load_json

    if zero_shot_path and zero_shot_path.exists():
        zs_data = load_json(zero_shot_path)
        zs_baselines = {
            fw: m.get("hit_at_1", 0)
            for fw, m in zs_data.get("per_framework", {}).items()
        }
    else:
        zs_baselines = {
            "MITRE ATLAS": 0.273,
            "NIST AI 100-2": 0.107,
            "OWASP AI Exchange": 0.619,
            "OWASP Top10 for LLM": 0.333,
            "OWASP Top10 for ML": 0.429,
        }
        logger.warning("Zero-shot baselines loaded from hardcoded fallback")

    fold_names = {
        "MITRE_ATLAS": "MITRE ATLAS",
        "NIST_AI_100-2": "NIST AI 100-2",
        "OWASP_AI_Exchange": "OWASP AI Exchange",
        "OWASP_Top10_for_LLM": "OWASP Top10 for LLM",
        "OWASP_Top10_for_ML": "OWASP Top10 for ML",
    }

    if not corrected_path.exists():
        raise FileNotFoundError(
            f"Corrected metrics not found: {corrected_path}. "
            "Run 'python scripts/phase1b/rescore_predictions.py' first."
        )
    corrected = load_json(corrected_path)
    corrected_folds = corrected.get("per_fold", {})

    results = []
    for file_key, display_name in fold_names.items():
        fold_dir = textaware_dir / f"fold_{file_key}"
        predictions_path = fold_dir / "predictions.json"
        summary_path = textaware_dir / f"fold_{file_key}_summary.json"

        if not predictions_path.exists():
            logger.warning("Predictions not found: %s", predictions_path)
            continue

        predictions = load_json(predictions_path)
        n_eval = len(predictions)

        summary = load_json(summary_path) if summary_path.exists() else {}
        metrics = summary.get("metrics", {})

        corrected_fold = corrected_folds.get(display_name, {})
        if not corrected_fold:
            raise ValueError(
                f"No corrected metrics for fold '{display_name}'. "
                f"Available keys: {list(corrected_folds.keys())}"
            )
        hit_any = corrected_fold.get("hit_at_1", metrics.get("hit_at_1", 0))

        results.append({
            "fold": display_name,
            "hit1": metrics.get("hit_at_1", 0),
            "zs_hit1": zs_baselines.get(display_name, 0),
            "n": n_eval,
            "hit_any": hit_any,
        })

    return results
```

Add `"publish-hf": _cmd_publish_hf` to the handlers dict in `main()`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_publish_cli.py -v`
Expected: ALL PASS

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add tract/publish/__init__.py tract/cli.py tests/test_publish_cli.py
git commit -m "feat(publish): add publish orchestrator and CLI command"
```

---

### Task 15: Publish Integration Test

**Files:**
- Create: `tests/test_publish_integration.py`

- [ ] **Step 1: Write the integration test**

Create `tests/test_publish_integration.py`:

```python
"""End-to-end publish pipeline integration test (dry-run, no real model)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


def _setup_publish_workspace(tmp_path: Path) -> dict[str, Path]:
    """Create all files needed for a dry-run publish."""
    ws = {}

    bridge_report = {
        "counts": {"total": 2, "accepted": 1, "rejected": 1},
        "candidates": [
            {"ai_hub_id": "AI-1", "trad_hub_id": "T-1", "status": "accepted",
             "cosine_similarity": 0.7, "reviewer_notes": ""},
            {"ai_hub_id": "AI-2", "trad_hub_id": "T-2", "status": "rejected",
             "cosine_similarity": 0.3, "reviewer_notes": "too weak"},
        ],
        "similarity_stats": {"mean": 0.5},
    }
    ws["bridge_report"] = tmp_path / "bridge_report.json"
    ws["bridge_report"].write_text(json.dumps(bridge_report))

    hier = {
        "hubs": {
            "AI-1": {"hub_id": "AI-1", "name": "AI Hub 1", "parent_id": None,
                     "children_ids": [], "depth": 0, "branch_root_id": "AI-1",
                     "hierarchy_path": "AI Hub 1", "is_leaf": True,
                     "sibling_hub_ids": [], "related_hub_ids": ["T-1"]},
            "T-1": {"hub_id": "T-1", "name": "Trad Hub 1", "parent_id": None,
                    "children_ids": [], "depth": 0, "branch_root_id": "T-1",
                    "hierarchy_path": "Trad Hub 1", "is_leaf": True,
                    "sibling_hub_ids": [], "related_hub_ids": ["AI-1"]},
        },
        "roots": ["AI-1", "T-1"], "label_space": ["AI-1", "T-1"],
        "fetch_timestamp": "2026-01-01", "data_hash": "test", "version": "1.1",
    }
    ws["hierarchy"] = tmp_path / "cre_hierarchy.json"
    ws["hierarchy"].write_text(json.dumps(hier, sort_keys=True, indent=2))

    ws["hub_descriptions"] = tmp_path / "hub_descriptions.json"
    ws["hub_descriptions"].write_text(json.dumps({"AI-1": "AI hub desc"}))

    ws["calibration"] = tmp_path / "calibration.json"
    ws["calibration"].write_text(json.dumps({
        "t_deploy": 0.074, "ood_threshold": 0.568, "conformal_quantile": 0.997,
    }))

    ws["ece_gate"] = tmp_path / "ece_gate.json"
    ws["ece_gate"].write_text(json.dumps({
        "ece": 0.079, "ece_ci": {"ci_low": 0.049, "ci_high": 0.111},
    }))

    hub_emb = np.ones((2, 1024), dtype=np.float32)
    hub_emb /= np.linalg.norm(hub_emb, axis=1, keepdims=True)
    ws["artifacts"] = tmp_path / "artifacts.npz"
    np.savez(str(ws["artifacts"]),
             hub_embeddings=hub_emb, hub_ids=np.array(["AI-1", "T-1"]),
             control_embeddings=np.zeros((1, 1024)), control_ids=np.array(["c-1"]))

    ws["model_dir"] = tmp_path / "model"
    ws["model_dir"].mkdir()

    ws["staging_dir"] = tmp_path / "staging"

    return ws


class TestPublishDryRun:

    def test_dry_run_creates_staging(self, tmp_path, mocker) -> None:
        from tract.publish import publish_to_huggingface

        ws = _setup_publish_workspace(tmp_path)

        def _fake_merge(model_dir, output_dir):
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "0_Transformer").mkdir()
            (output_dir / "0_Transformer" / "model.safetensors").write_bytes(b"fake")
            (output_dir / "modules.json").write_text("[]")
            return output_dir

        mocker.patch("tract.publish.merge.merge_lora_adapters", side_effect=_fake_merge)

        fold_results = [
            {"fold": "Test Fold", "hit1": 0.5, "zs_hit1": 0.3, "n": 10, "hit_any": 0.6},
        ]

        publish_to_huggingface(
            repo_id="test/repo",
            staging_dir=ws["staging_dir"],
            model_dir=ws["model_dir"],
            artifacts_path=ws["artifacts"],
            hierarchy_path=ws["hierarchy"],
            hub_descriptions_path=ws["hub_descriptions"],
            calibration_path=ws["calibration"],
            ece_gate_path=ws["ece_gate"],
            bridge_report_path=ws["bridge_report"],
            fold_results=fold_results,
            gpu_hours=1.0,
            dry_run=True,
        )

        assert ws["staging_dir"].exists()
        assert (ws["staging_dir"] / "README.md").exists()
        assert (ws["staging_dir"] / "predict.py").exists()
        assert (ws["staging_dir"] / "train.py").exists()
        assert (ws["staging_dir"] / "hub_ids.json").exists()
        assert (ws["staging_dir"] / "calibration.json").exists()

    def test_gate_blocks_without_report(self, tmp_path, mocker) -> None:
        from tract.publish import publish_to_huggingface

        ws = _setup_publish_workspace(tmp_path)
        ws["bridge_report"].unlink()

        with pytest.raises(ValueError, match="bridge_report.json"):
            publish_to_huggingface(
                repo_id="test/repo",
                staging_dir=ws["staging_dir"],
                model_dir=ws["model_dir"],
                artifacts_path=ws["artifacts"],
                hierarchy_path=ws["hierarchy"],
                hub_descriptions_path=ws["hub_descriptions"],
                calibration_path=ws["calibration"],
                ece_gate_path=ws["ece_gate"],
                bridge_report_path=ws["bridge_report"],
                fold_results=[],
                gpu_hours=0,
                dry_run=True,
            )
```

- [ ] **Step 2: Run integration test**

Run: `python -m pytest tests/test_publish_integration.py -v`
Expected: ALL PASS

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -q`
Expected: All tests pass (553 existing + ~80 new)

- [ ] **Step 4: Commit**

```bash
git add tests/test_publish_integration.py
git commit -m "test(publish): add end-to-end publish integration test"
```
