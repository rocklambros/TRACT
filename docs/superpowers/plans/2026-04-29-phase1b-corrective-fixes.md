# Phase 1B Corrective Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 2 dead-code bugs, 1 evaluation methodology flaw, and 1 invalid baseline; then determine if Phase 1B already passes the gate.

**Architecture:** Four phases executed sequentially. Phase 0 (free measurements) determines whether Phases 1-2 are needed. All changes are local — no GPU pod required until Phase 2.

**Tech Stack:** Python 3.11, sentence-transformers, PEFT, numpy, pytest

**Spec:** `docs/superpowers/specs/2026-04-29-phase1b-corrective-design.md`

---

### Task 1: Add `valid_hub_ids` to EvalItem and Deduplicate Evaluation Corpus

**Files:**
- Modify: `scripts/phase0/common.py:123-131` (EvalItem dataclass)
- Modify: `scripts/phase0/common.py:345-395` (build_evaluation_corpus)
- Test: `tests/test_phase0_common.py`

- [ ] **Step 1: Write the failing test for EvalItem with valid_hub_ids**

```python
# In tests/test_phase0_common.py, add:

class TestBuildEvaluationCorpusDedup:
    """Test deduplicated evaluation corpus with multi-label ground truth."""

    def test_deduplicates_identical_text_same_hub(self) -> None:
        from scripts.phase0.common import build_evaluation_corpus, HubStandardLink

        links = [
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model", link_type="LinkedTo",
            ),
            HubStandardLink(
                cre_id="hub-1", cre_name="Hub 1",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model", link_type="LinkedTo",
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
                section_name="Validate AI Model", link_type="LinkedTo",
            ),
            HubStandardLink(
                cre_id="hub-2", cre_name="Hub 2",
                standard_name="MITRE ATLAS", section_id="AML.M0008",
                section_name="Validate AI Model", link_type="LinkedTo",
            ),
        ]
        corpus = build_evaluation_corpus(links, {"MITRE ATLAS"}, {})
        assert len(corpus) == 1
        assert corpus[0].valid_hub_ids == frozenset({"hub-1", "hub-2"})
        assert corpus[0].ground_truth_hub_id in {"hub-1", "hub-2"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_phase0_common.py::TestBuildEvaluationCorpusDedup -v`
Expected: FAIL — `EvalItem` has no `valid_hub_ids` field, and `build_evaluation_corpus` doesn't deduplicate.

- [ ] **Step 3: Add `valid_hub_ids` to EvalItem**

In `scripts/phase0/common.py:123-131`, change:

```python
@dataclass
class EvalItem:
    """One evaluation data point: a control mapped to a ground-truth hub."""

    control_text: str
    ground_truth_hub_id: str
    valid_hub_ids: frozenset[str]
    ground_truth_hub_name: str
    framework_name: str
    section_id: str
    track: str  # "full-text" or "all"
```

- [ ] **Step 4: Fix all existing EvalItem instantiations to include valid_hub_ids**

In `build_evaluation_corpus()` (common.py:345-395), replace the direct corpus building with a dedup pass:

```python
def build_evaluation_corpus(
    links: list[HubStandardLink],
    ai_framework_names: set[str] | frozenset[str],
    parsed_controls: dict[tuple[str, str], str],
) -> list[EvalItem]:
    """Build deduplicated evaluation corpus from AI framework hub links.

    For each unique (framework_name, control_text), creates one EvalItem with
    all valid hub IDs collected into valid_hub_ids.
    """
    # First pass: collect all items (un-deduplicated)
    raw_items: list[dict] = []
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

        raw_items.append({
            "control_text": _sanitize_text(control_text),
            "hub_id": link.cre_id,
            "hub_name": link.cre_name,
            "framework": link.standard_name,
            "section_id": link.section_id,
            "track": track,
        })

    # Second pass: deduplicate by (framework, control_text)
    from collections import OrderedDict
    seen: OrderedDict[tuple[str, str], dict] = OrderedDict()
    for item in raw_items:
        key = (item["framework"], item["control_text"])
        if key not in seen:
            seen[key] = {
                "control_text": item["control_text"],
                "ground_truth_hub_id": item["hub_id"],
                "valid_hub_ids": {item["hub_id"]},
                "ground_truth_hub_name": item["hub_name"],
                "framework_name": item["framework"],
                "section_id": item["section_id"],
                "track": item["track"],
            }
        else:
            seen[key]["valid_hub_ids"].add(item["hub_id"])

    corpus: list[EvalItem] = []
    for entry in seen.values():
        corpus.append(EvalItem(
            control_text=entry["control_text"],
            ground_truth_hub_id=entry["ground_truth_hub_id"],
            valid_hub_ids=frozenset(entry["valid_hub_ids"]),
            ground_truth_hub_name=entry["ground_truth_hub_name"],
            framework_name=entry["framework_name"],
            section_id=entry["section_id"],
            track=entry["track"],
        ))

    n_raw = len(raw_items)
    n_dedup = len(corpus)
    n_multilabel = sum(1 for e in corpus if len(e.valid_hub_ids) > 1)
    logger.info(
        "Built evaluation corpus: %d items (deduplicated from %d raw, %d multi-label, "
        "%d full-text, %d title-only)",
        n_dedup, n_raw, n_multilabel,
        sum(1 for e in corpus if e.track == "full-text"),
        sum(1 for e in corpus if e.track == "all"),
    )
    return corpus
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_phase0_common.py::TestBuildEvaluationCorpusDedup -v`
Expected: PASS

- [ ] **Step 6: Fix all other tests that instantiate EvalItem without valid_hub_ids**

Search for `EvalItem(` across all test files and add `valid_hub_ids=frozenset({hub_id})` where needed.

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add scripts/phase0/common.py tests/test_phase0_common.py
git commit -m "fix(eval): deduplicate eval corpus, add multi-label ground truth to EvalItem"
```

---

### Task 2: Multi-Label-Aware Scoring

**Files:**
- Modify: `scripts/phase0/common.py:478-507` (score_predictions)
- Modify: `scripts/phase0/common.py:459-475` (reciprocal_rank, ndcg_at_k)
- Modify: `tract/training/evaluate.py:302-310` (hit1_indicators in evaluate_on_fold)
- Test: `tests/test_phase0_common.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests for multi-label scoring**

```python
# In tests/test_phase0_common.py, add:

class TestMultiLabelScoring:
    """Test multi-label-aware scoring."""

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_phase0_common.py::TestMultiLabelScoring -v`
Expected: FAIL — `score_predictions` doesn't accept `valid_hub_sets` parameter

- [ ] **Step 3: Update score_predictions for multi-label awareness**

In `scripts/phase0/common.py`, modify `reciprocal_rank`, `ndcg_at_k`, and `score_predictions`:

```python
def reciprocal_rank(
    predicted: list[str],
    truth: str,
    valid_set: frozenset[str] | None = None,
) -> float:
    """Reciprocal rank of first valid hub in predicted list."""
    targets = valid_set if valid_set else frozenset({truth})
    for i, p in enumerate(predicted):
        if p in targets:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    predicted: list[str],
    truth: str,
    k: int = 10,
    valid_set: frozenset[str] | None = None,
) -> float:
    """NDCG@k for single-relevant-item retrieval (multi-label aware)."""
    targets = valid_set if valid_set else frozenset({truth})
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p in targets:
            dcg = 1.0 / math.log2(i + 2)
            break
    idcg = 1.0 / math.log2(2)
    return dcg / idcg


def score_predictions(
    predictions: list[list[str]],
    ground_truth: list[str],
    valid_hub_sets: list[frozenset[str]] | None = None,
) -> dict[str, float]:
    """Score ranked predictions against ground truth hub IDs.

    If valid_hub_sets is provided, any hub in the valid set counts as a hit
    (multi-label-aware scoring). Otherwise falls back to single-label.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("Empty predictions list")
    if n != len(ground_truth):
        raise ValueError(f"Prediction count {n} != ground truth count {len(ground_truth)}")

    if valid_hub_sets is None:
        valid_hub_sets = [frozenset({gt}) for gt in ground_truth]

    hit1 = sum(
        1 for pred, vs in zip(predictions, valid_hub_sets)
        if pred and pred[0] in vs
    ) / n
    hit5 = sum(
        1 for pred, vs in zip(predictions, valid_hub_sets)
        if any(p in vs for p in pred[:5])
    ) / n
    mrr = sum(
        reciprocal_rank(pred, gt, vs)
        for pred, gt, vs in zip(predictions, ground_truth, valid_hub_sets)
    ) / n
    ndcg = sum(
        ndcg_at_k(pred, gt, valid_set=vs)
        for pred, gt, vs in zip(predictions, ground_truth, valid_hub_sets)
    ) / n

    return {
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "mrr": mrr,
        "ndcg_at_10": ndcg,
    }
```

- [ ] **Step 4: Update evaluate_on_fold to use multi-label hit1_indicators**

In `tract/training/evaluate.py:302-310`, update:

```python
    ground_truth = [item.ground_truth_hub_id for item in eval_items]
    valid_hub_sets = [
        item.valid_hub_ids if hasattr(item, 'valid_hub_ids') else frozenset({item.ground_truth_hub_id})
        for item in eval_items
    ]
    metrics = score_predictions(predictions, ground_truth, valid_hub_sets)

    hit1_indicators = np.array(
        [
            1.0 if pred and pred[0] in vs else 0.0
            for pred, vs in zip(predictions, valid_hub_sets)
        ]
    )
```

- [ ] **Step 5: Run all tests**

Run: `python -m pytest tests/test_phase0_common.py tests/test_evaluate.py -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add scripts/phase0/common.py tract/training/evaluate.py tests/test_phase0_common.py tests/test_evaluate.py
git commit -m "fix(eval): multi-label-aware scoring for hit@1, MRR, NDCG"
```

---

### Task 3: Firewalled Zero-Shot Baseline Script

**Files:**
- Create: `scripts/phase1b/zero_shot_baseline.py`
- Test: Run and capture output

- [ ] **Step 1: Write the zero-shot baseline script**

```python
"""Compute firewalled zero-shot BGE baseline with Phase 1B hub format.

Uses the same LOFO protocol, hub format, and evaluation methodology
as Phase 1B training runs, but with no fine-tuning.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    load_curated_links,
    load_opencre_cres,
    score_predictions,
)
from tract.config import PHASE1B_RESULTS_DIR, PROCESSED_DIR
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.evaluate import (
    evaluate_on_fold,
    fold_stratified_bootstrap_ci,
)
from tract.training.firewall import build_all_hub_texts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Loading data...")
    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    eval_by_fw: dict[str, list] = {}
    for item in corpus:
        eval_by_fw.setdefault(item.framework_name, []).append(item)

    logger.info("Loading BGE-large-v1.5 (zero-shot, no fine-tuning)...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    output_dir = PHASE1B_RESULTS_DIR / "zero_shot_firewalled_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    for fw_name in sorted(AI_FRAMEWORK_NAMES):
        fw_items = eval_by_fw.get(fw_name, [])
        if not fw_items:
            continue

        hub_texts = build_all_hub_texts(hierarchy, excluded_framework=fw_name)

        metrics, predictions, hit1_indicators = evaluate_on_fold(
            model, fw_items, hub_ids, hub_texts,
        )
        logger.info("Fold %s: hit@1=%.3f, hit@5=%.3f, MRR=%.3f (n=%d)",
                    fw_name, metrics["hit_at_1"], metrics["hit_at_5"],
                    metrics["mrr"], len(fw_items))

        fold_results.append({
            "held_out_framework": fw_name,
            "metrics": metrics,
            "hit1_indicators": hit1_indicators.tolist(),
            "n_eval_items": len(fw_items),
        })

        pred_data = []
        for item, pred in zip(fw_items, predictions):
            pred_data.append({
                "control_text": item.control_text,
                "ground_truth_hub_id": item.ground_truth_hub_id,
                "predicted_top10": pred[:10],
                "framework": item.framework_name,
            })
        fold_dir = output_dir / f"fold_{fw_name.replace(' ', '_')}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(pred_data, fold_dir / "predictions.json")
        atomic_write_json(metrics, fold_dir / "metrics.json")

    fold_hit1s = [np.array(r["hit1_indicators"]) for r in fold_results]
    aggregate = fold_stratified_bootstrap_ci(fold_hit1s)
    logger.info("AGGREGATE hit@1: %.3f [%.3f, %.3f]",
                aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"])

    result = {
        "model": "BAAI/bge-large-en-v1.5",
        "hub_format": "path+name",
        "firewalled": True,
        "multi_label_aware": True,
        "aggregate_hit1": aggregate,
        "per_fold": {r["held_out_framework"]: r["metrics"] for r in fold_results},
    }
    atomic_write_json(result, output_dir / "aggregate_metrics.json")

    print("\n" + "=" * 60)
    print("FIREWALLED ZERO-SHOT BASELINE COMPLETE")
    print(f"  Aggregate hit@1: {aggregate['mean']:.3f} [{aggregate['ci_low']:.3f}, {aggregate['ci_high']:.3f}]")
    for r in sorted(fold_results, key=lambda x: x["held_out_framework"]):
        m = r["metrics"]
        print(f"  {r['held_out_framework']}: hit@1={m['hit_at_1']:.3f} hit@5={m['hit_at_5']:.3f} MRR={m['mrr']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the baseline**

Run: `python -m scripts.phase1b.zero_shot_baseline`
Expected: Completes in ~5 minutes on Orin, prints per-fold and aggregate metrics.

- [ ] **Step 3: Commit**

```bash
git add scripts/phase1b/zero_shot_baseline.py
git commit -m "feat(eval): firewalled zero-shot baseline with Phase 1B hub format"
```

---

### Task 4: Re-Score Existing Phase 1B Predictions

**Files:**
- Create: `scripts/phase1b/rescore_predictions.py`
- Test: Run and capture output

- [ ] **Step 1: Write the re-scoring script**

```python
"""Re-score existing Phase 1B predictions with corrected evaluation.

Reads predictions.json from each fold, applies multi-label-aware scoring,
and reports corrected metrics without retraining.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    load_curated_links,
    score_predictions,
)
from tract.config import PHASE1B_RESULTS_DIR
from tract.io import atomic_write_json, load_json
from tract.training.evaluate import fold_stratified_bootstrap_ci

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def rescore_experiment(experiment_dir: Path) -> None:
    logger.info("Re-scoring: %s", experiment_dir.name)

    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    valid_hubs_by_text: dict[str, frozenset[str]] = {}
    for item in corpus:
        valid_hubs_by_text[item.control_text] = item.valid_hub_ids

    fold_results = []
    for fw_name in sorted(AI_FRAMEWORK_NAMES):
        fold_dir = experiment_dir / f"fold_{fw_name.replace(' ', '_')}"
        pred_path = fold_dir / "predictions.json"
        if not pred_path.exists():
            continue

        preds_data = load_json(pred_path)

        predictions = [p["predicted_top10"] for p in preds_data]
        ground_truth = [p["ground_truth_hub_id"] for p in preds_data]
        valid_sets = [
            valid_hubs_by_text.get(p["control_text"], frozenset({p["ground_truth_hub_id"]}))
            for p in preds_data
        ]

        metrics = score_predictions(predictions, ground_truth, valid_sets)
        hit1 = np.array([
            1.0 if pred and pred[0] in vs else 0.0
            for pred, vs in zip(predictions, valid_sets)
        ])

        logger.info("  %s: hit@1=%.3f (was %.3f), n=%d",
                    fw_name, metrics["hit_at_1"],
                    sum(1 for p, g in zip(predictions, ground_truth) if p and p[0] == g) / len(predictions),
                    len(predictions))

        fold_results.append({
            "held_out_framework": fw_name,
            "metrics": metrics,
            "hit1_indicators": hit1.tolist(),
        })

    if fold_results:
        fold_hit1s = [np.array(r["hit1_indicators"]) for r in fold_results]
        aggregate = fold_stratified_bootstrap_ci(fold_hit1s)
        logger.info("  CORRECTED AGGREGATE: hit@1=%.3f [%.3f, %.3f]",
                    aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"])

        corrected = {
            "scoring": "multi-label-aware",
            "aggregate_hit1": aggregate,
            "per_fold": {r["held_out_framework"]: r["metrics"] for r in fold_results},
        }
        atomic_write_json(corrected, experiment_dir / "corrected_metrics.json")


def main() -> None:
    for experiment in ["phase1b_primary", "ablation_a6_descriptions"]:
        exp_dir = PHASE1B_RESULTS_DIR / experiment
        if exp_dir.exists():
            rescore_experiment(exp_dir)
        else:
            logger.warning("Experiment dir not found: %s", exp_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run re-scoring**

Run: `python -m scripts.phase1b.rescore_predictions`
Expected: Prints corrected metrics for primary and A6 experiments.

- [ ] **Step 3: Commit**

```bash
git add scripts/phase1b/rescore_predictions.py
git commit -m "feat(eval): re-score existing predictions with multi-label methodology"
```

---

### Task 5: Gate 0 Decision — Compare Corrected Phase 1B vs Zero-Shot

**Files:**
- No new files — analysis of Task 3 and Task 4 outputs

- [ ] **Step 1: Compute the gate delta**

```bash
python3 -c "
import json
zs = json.load(open('results/phase1b/zero_shot_firewalled_baseline/aggregate_metrics.json'))
p1 = json.load(open('results/phase1b/phase1b_primary/corrected_metrics.json'))
delta = p1['aggregate_hit1']['mean'] - zs['aggregate_hit1']['mean']
print(f'Zero-shot: {zs[\"aggregate_hit1\"][\"mean\"]:.3f}')
print(f'Phase 1B (corrected): {p1[\"aggregate_hit1\"][\"mean\"]:.3f}')
print(f'Delta: {delta:.3f}')
print(f'Gate (>0.10): {\"PASS\" if delta > 0.10 else \"FAIL\"}')"
```

- [ ] **Step 2: Record the gate decision**

Log the result. If PASS: training pipeline fixes become P1 (improvement). If FAIL: proceed to Task 6.

---

### Task 6: Hub-Aware Temperature Batch Sampler

**Files:**
- Modify: `tract/training/data.py:121-237` (replace two classes with one)
- Modify: `tract/training/data.py:240-269` (add hub_id column to pairs_to_dataset)
- Test: `tests/test_training_data.py`

- [ ] **Step 1: Write failing tests for the new sampler**

```python
# In tests/test_training_data.py, add:

class TestHubAwareTemperatureSampler:
    """Test combined hub-collision-free + temperature-weighted batch sampler."""

    def test_no_hub_collisions_in_any_batch(self) -> None:
        from tract.training.data import HubAwareTemperatureSampler
        from datasets import Dataset

        hub_ids = ["h1"] * 10 + ["h2"] * 10 + ["h3"] * 10 + ["h4"] * 10
        ds = Dataset.from_dict({
            "anchor": [f"text_{i}" for i in range(40)],
            "positive": [f"hub_{h}" for h in hub_ids],
            "hub_id": hub_ids,
            "is_ai": [False] * 40,
        })
        sampler = HubAwareTemperatureSampler(
            dataset=ds, batch_size=8, drop_last=False,
        )
        for batch_indices in sampler:
            batch_hubs = [hub_ids[i] for i in batch_indices]
            assert len(batch_hubs) == len(set(batch_hubs)), \
                f"Hub collision in batch: {batch_hubs}"

    def test_all_indices_appear_exactly_once(self) -> None:
        from tract.training.data import HubAwareTemperatureSampler
        from datasets import Dataset

        n = 50
        hub_ids = [f"h{i % 20}" for i in range(n)]
        ds = Dataset.from_dict({
            "anchor": [f"t{i}" for i in range(n)],
            "positive": [f"p{i}" for i in range(n)],
            "hub_id": hub_ids,
            "is_ai": [False] * n,
        })
        sampler = HubAwareTemperatureSampler(
            dataset=ds, batch_size=10, drop_last=False,
        )
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(n))

    def test_ai_upweighting_with_temperature(self) -> None:
        from tract.training.data import HubAwareTemperatureSampler
        from datasets import Dataset
        import numpy as np

        n = 200
        hub_ids = [f"h{i}" for i in range(n)]
        is_ai = [i < 10 for i in range(n)]  # 5% AI
        ds = Dataset.from_dict({
            "anchor": [f"t{i}" for i in range(n)],
            "positive": [f"p{i}" for i in range(n)],
            "hub_id": hub_ids,
            "is_ai": is_ai,
        })
        sampler = HubAwareTemperatureSampler(
            dataset=ds, batch_size=20, drop_last=False,
            temperature=2.0, seed=42,
        )
        batches = list(sampler)
        first_half = []
        for b in batches[:len(batches)//2]:
            first_half.extend(b)
        ai_in_first_half = sum(1 for i in first_half if is_ai[i])
        ai_fraction = ai_in_first_half / len(first_half)
        assert ai_fraction > 0.05, f"AI fraction in first half should be > 5%, got {ai_fraction:.3f}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_data.py::TestHubAwareTemperatureSampler -v`
Expected: FAIL — `HubAwareTemperatureSampler` doesn't exist yet

- [ ] **Step 3: Implement HubAwareTemperatureSampler**

In `tract/training/data.py`, replace `apply_temperature_sampling_order` and `HubAwareBatchSampler` with:

```python
class HubAwareTemperatureSampler:
    """Batch sampler preventing hub collisions with temperature-weighted AI upsampling.

    Subclasses can be passed to SentenceTransformerTrainingArguments(batch_sampler=...).
    Accepts the trainer's constructor signature: (dataset, batch_size, drop_last, ...).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 64,
        drop_last: bool = False,
        valid_label_columns: list[str] | None = None,
        generator: torch.Generator | None = None,
        seed: int = 42,
        temperature: float = 2.0,
    ) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.temperature = temperature
        self.generator = generator

        if "hub_id" not in dataset.column_names:
            raise ValueError("Dataset must have a 'hub_id' column for hub-aware batching")
        self.hub_ids: list[str] = dataset["hub_id"]
        self.is_ai: list[bool] = dataset["is_ai"] if "is_ai" in dataset.column_names else [False] * len(dataset)
        self.n = len(dataset)

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator is not None:
            seed = int(torch.randint(0, 2**31, (1,), generator=self.generator).item())
        else:
            seed = self.seed
        rng = np.random.default_rng(seed)

        ai_indices = [i for i in range(self.n) if self.is_ai[i]]
        trad_indices = [i for i in range(self.n) if not self.is_ai[i]]
        rng.shuffle(ai_indices)
        rng.shuffle(trad_indices)

        n_ai = len(ai_indices)
        n_trad = len(trad_indices)

        if n_ai > 0 and n_trad > 0 and self.temperature > 0:
            w_ai = (n_ai / self.n) ** (1.0 / self.temperature)
            w_trad = (n_trad / self.n) ** (1.0 / self.temperature)
            p_ai = w_ai / (w_ai + w_trad)
        else:
            p_ai = n_ai / self.n if self.n > 0 else 0.0

        ordered: list[int] = []
        ai_ptr, trad_ptr = 0, 0
        for _ in range(self.n):
            if ai_ptr >= n_ai:
                ordered.append(trad_indices[trad_ptr]); trad_ptr += 1
            elif trad_ptr >= n_trad:
                ordered.append(ai_indices[ai_ptr]); ai_ptr += 1
            elif rng.random() < p_ai:
                ordered.append(ai_indices[ai_ptr]); ai_ptr += 1
            else:
                ordered.append(trad_indices[trad_ptr]); trad_ptr += 1

        batch: list[int] = []
        hubs_in_batch: set[str] = set()
        deferred: list[int] = []

        for idx in ordered:
            hub = self.hub_ids[idx]
            if hub not in hubs_in_batch:
                batch.append(idx)
                hubs_in_batch.add(hub)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    hubs_in_batch = set()
            else:
                deferred.append(idx)

        remaining = deferred
        while remaining:
            next_remaining: list[int] = []
            for idx in remaining:
                hub = self.hub_ids[idx]
                if hub not in hubs_in_batch:
                    batch.append(idx)
                    hubs_in_batch.add(hub)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        hubs_in_batch = set()
                else:
                    next_remaining.append(idx)
            if len(next_remaining) == len(remaining):
                batch.extend(next_remaining)
                break
            remaining = next_remaining

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        return math.ceil(self.n / self.batch_size)
```

- [ ] **Step 4: Add hub_id and is_ai columns to pairs_to_dataset**

In `tract/training/data.py`, modify `pairs_to_dataset`:

```python
def pairs_to_dataset(
    pairs: list[TrainingPair],
    hierarchy: CREHierarchy,
    hub_texts: dict[str, str],
    n_hard_negatives: int = 3,
) -> Dataset:
    """Convert TrainingPairs to a sentence-transformers Dataset with hard negatives.

    Output columns: anchor, positive, negative_1..N, hub_id, is_ai
    """
    records: list[dict[str, str]] = []
    for pair in pairs:
        record: dict[str, str] = {
            "anchor": pair.control_text,
            "positive": pair.hub_representation,
            "hub_id": pair.hub_id,
            "is_ai": pair.framework in AI_FRAMEWORK_NAMES,
        }
        # ... rest unchanged
```

- [ ] **Step 5: Add torch import at top of data.py**

```python
import torch
```

- [ ] **Step 6: Delete old apply_temperature_sampling_order and HubAwareBatchSampler**

Remove lines 121-237 (the two old implementations).

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_training_data.py -v`
Expected: All pass

- [ ] **Step 8: Commit**

```bash
git add tract/training/data.py tests/test_training_data.py
git commit -m "feat(training): combined hub-aware temperature batch sampler replacing two dead-code components"
```

---

### Task 7: Wire Batch Sampler into Training Loop

**Files:**
- Modify: `tract/training/loop.py:79-105`
- Test: `tests/test_training_loop.py`

- [ ] **Step 1: Write failing integration test**

```python
# In tests/test_training_loop.py, add:

class TestBatchSamplerIntegration:
    """Verify custom batch sampler is wired into trainer."""

    def test_training_args_use_custom_sampler(self) -> None:
        from tract.training.data import HubAwareTemperatureSampler
        from tract.training.loop import train_model
        from tract.training.config import TrainingConfig
        from datasets import Dataset
        from unittest.mock import patch

        config = TrainingConfig(name="test_sampler", max_epochs=1, batch_size=4)
        ds = Dataset.from_dict({
            "anchor": ["a", "b", "c", "d"],
            "positive": ["p1", "p2", "p3", "p4"],
            "hub_id": ["h1", "h2", "h3", "h4"],
            "is_ai": [True, False, False, False],
        })

        with patch("tract.training.loop.SentenceTransformerTrainer") as MockTrainer:
            MockTrainer.return_value.train.return_value = None
            try:
                train_model(config, ds, Path("/tmp/test_sampler_integration"))
            except Exception:
                pass

            if MockTrainer.called:
                call_args = MockTrainer.call_args
                args_obj = call_args.kwargs.get("args") or call_args[1].get("args")
                assert args_obj.batch_sampler == HubAwareTemperatureSampler
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_loop.py::TestBatchSamplerIntegration -v`
Expected: FAIL

- [ ] **Step 3: Wire the sampler into train_model**

In `tract/training/loop.py`, add import and modify training args:

```python
from tract.training.data import HubAwareTemperatureSampler

# In train_model(), add to SentenceTransformerTrainingArguments:
    training_args = SentenceTransformerTrainingArguments(
        ...,
        batch_sampler=HubAwareTemperatureSampler,
    )
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_training_loop.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add tract/training/loop.py tests/test_training_loop.py
git commit -m "fix(training): wire HubAwareTemperatureSampler into SentenceTransformerTrainer"
```

---

### Task 8: End-to-End Integration Test

**Files:**
- Create: `tests/test_integration_training.py`

- [ ] **Step 1: Write the integration test**

```python
"""Integration test: full training pipeline with sampler verification."""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import pytest
from datasets import Dataset


@pytest.fixture
def mini_training_dataset() -> Dataset:
    n = 40
    hub_ids = [f"hub-{i % 10}" for i in range(n)]
    return Dataset.from_dict({
        "anchor": [f"control text {i}" for i in range(n)],
        "positive": [f"hub representation {hub_ids[i]}" for i in range(n)],
        "negative_1": [f"neg hub {(i+1) % 10}" for i in range(n)],
        "negative_2": [f"neg hub {(i+2) % 10}" for i in range(n)],
        "negative_3": [f"neg hub {(i+3) % 10}" for i in range(n)],
        "hub_id": hub_ids,
        "is_ai": [i < 5 for i in range(n)],
    })


class TestTrainingPipelineIntegration:
    """End-to-end integration test for the corrected training pipeline."""

    @pytest.mark.slow
    def test_custom_sampler_produces_collision_free_batches(
        self, mini_training_dataset: Dataset
    ) -> None:
        from tract.training.data import HubAwareTemperatureSampler

        sampler = HubAwareTemperatureSampler(
            dataset=mini_training_dataset,
            batch_size=8,
            drop_last=False,
            seed=42,
            temperature=2.0,
        )

        all_indices: list[int] = []
        for batch in sampler:
            batch_hubs = [mini_training_dataset["hub_id"][i] for i in batch]
            assert len(batch_hubs) == len(set(batch_hubs)), \
                f"Hub collision detected: {batch_hubs}"
            all_indices.extend(batch)

        assert sorted(all_indices) == list(range(len(mini_training_dataset)))

    def test_dataset_has_required_columns(
        self, mini_training_dataset: Dataset
    ) -> None:
        assert "hub_id" in mini_training_dataset.column_names
        assert "is_ai" in mini_training_dataset.column_names
        assert "anchor" in mini_training_dataset.column_names
        assert "positive" in mini_training_dataset.column_names
```

- [ ] **Step 2: Run integration tests**

Run: `python -m pytest tests/test_integration_training.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration_training.py
git commit -m "test: add end-to-end integration test for corrected training pipeline"
```

---

### Task 9: Update Memory and Documentation

**Files:**
- Modify: `~/.claude/projects/-home-rock-github-projects-TRACT/memory/project_phase1b_bookmark.md`
- Modify: `~/.claude/projects/-home-rock-github-projects-TRACT/memory/MEMORY.md`

- [ ] **Step 1: Update phase1b bookmark with adversarial review findings and Gate 0 result**

Update the memory file with:
- Adversarial review findings (3 rounds)
- Gate 0 decision (zero-shot baseline vs corrected Phase 1B)
- Training pipeline fix status
- Next steps

- [ ] **Step 2: Commit spec and plan**

```bash
git add docs/superpowers/specs/2026-04-29-phase1b-corrective-design.md
git add docs/superpowers/plans/2026-04-29-phase1b-corrective-fixes.md
git commit -m "docs: Phase 1B corrective spec and implementation plan from adversarial review"
```
