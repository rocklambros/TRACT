# Phase 1B Corrective Design Spec: Pipeline Fixes and Evaluation Methodology

**Date:** 2026-04-29
**Amends:** `2026-04-29-phase1b-design.md`
**Motivation:** 3-round adversarial review of Phase 1B primary + A6 ablation results identified 2 dead-code bugs, 1 evaluation methodology flaw, and 1 invalid baseline comparison. This spec defines the corrective changes.

---

## 0. Summary of Findings

Phase 1B primary run (hit@1=0.437) and A6 ablation (hit@1=0.365) were executed on a pipeline with two designed-but-disconnected components and measured with a flawed evaluation methodology.

| # | Finding | Severity | Impact |
|---|---------|----------|--------|
| F1 | `HubAwareBatchSampler` defined but never wired into trainer | HIGH | ~2% false-negative contamination per batch (5 false-negative pairs / 255 total negatives) |
| F2 | `apply_temperature_sampling_order()` defined but never called; AND incompatible with trainer's shuffle | HIGH | AI examples at 3.2% of batches (design intent: 15.5%) |
| F3 | Multi-label eval items create guaranteed misses | HIGH | 13 texts map to multiple hubs; 18 guaranteed misses; corrected hit@1 = 0.508 |
| F4 | No apples-to-apples zero-shot baseline exists | HIGH | Phase 0 used richer hub reps (up to 78 linked standard names per hub) |
| F5 | True duplicate eval items inflate variance | MEDIUM | 34 duplicate items; hub 364-516 has 11 items (5.6% of corpus) |

### Validated Magnitudes (3-Round Review)

The initial audit (4 parallel agents) inflated some findings. Corrected after 3 sequential review rounds:

| Claim | Initial Audit | Verified |
|-------|--------------|----------|
| Guaranteed misses | 58 | **18** |
| Max achievable hit@1 | 0.706 | **0.909** |
| Corrected hit@1 | 0.534 | **0.508** |
| False-negative contamination rate | ~16% | **~2%** (hard negatives dilute; 255 negatives/anchor, not 63) |
| Batch sampler fix complexity | 5-10 lines | **~80 lines** (interface incompatible with trainer API) |
| Phase 0 firewalled? | "No" | **Yes** (via `build_lofo_folds()`) |

---

## 1. Fix F1+F2: Combined Hub-Aware Temperature Batch Sampler

### 1.1 Problem

`HubAwareBatchSampler` (data.py:173) prevents two examples sharing the same target hub from appearing in the same MNRL batch (eliminating false negatives). `apply_temperature_sampling_order()` (data.py:121) upweights AI examples from 3.2% to ~15.5% of batch positions.

Neither is connected to the training loop. `SentenceTransformerTrainer` creates its own `DataLoader` with `DefaultBatchSampler(RandomSampler(...))`. The custom sampler's interface (`hub_ids, batch_size, seed, drop_last`) is incompatible with the trainer's expected signature (`dataset, batch_size, drop_last, valid_label_columns, generator, seed`).

Additionally, `apply_temperature_sampling_order()` pre-orders the dataset, but the trainer's `RandomSampler` immediately reshuffles — making pre-ordering useless even if called.

### 1.2 Solution

Replace both components with a single `HubAwareTemperatureSampler` that:
1. Subclasses `DefaultBatchSampler` from sentence-transformers
2. Accepts the trainer's constructor signature
3. Reads hub IDs from a `hub_id` column in the dataset
4. Prevents hub collisions in batches (F1 fix)
5. Upweights AI-domain examples via temperature-scaled class selection (F2 fix)
6. Supports per-epoch re-seeding via the `generator` parameter

### 1.3 Integration Path

```python
# In pairs_to_dataset(): add hub_id column to dataset
record["hub_id"] = pair.hub_id

# In train_model(): pass sampler class via training args
training_args = SentenceTransformerTrainingArguments(
    ...,
    batch_sampler=HubAwareTemperatureSampler,
)
```

The trainer calls `batch_sampler(dataset, batch_size, drop_last, ...)` internally. The sampler extracts `hub_id` from the dataset column and `is_ai` from the hub_id or a separate column.

### 1.4 Algorithm

```
For each epoch:
  1. Classify all indices as AI or traditional (from dataset column)
  2. Compute temperature-weighted class probabilities:
     p_ai = (n_ai/n)^(1/T) / ((n_ai/n)^(1/T) + (n_trad/n)^(1/T))
  3. Shuffle indices within each class
  4. Build batches greedily:
     a. Draw next class (AI with prob p_ai, trad with prob 1-p_ai)
     b. From chosen class, take next unplaced index
     c. If index's hub_id already in current batch, defer to next batch
     d. If batch full (batch_size), yield batch, reset hub set
  5. Process deferred indices in subsequent batches
```

### 1.5 Cleanup

- Delete `apply_temperature_sampling_order()` from data.py (wrong abstraction)
- Delete original `HubAwareBatchSampler` class (replaced)
- Rename config `training_data` default from `"joint-tempscaled"` to `"joint"` for new runs (old results retain old name in metadata)

---

## 2. Fix F3: Multi-Label-Aware Evaluation

### 2.1 Problem

`build_evaluation_corpus()` creates one `EvalItem` per link. When control "Differential privacy" maps to 4 CRE hubs, it creates 4 eval items with identical `control_text` but different `ground_truth_hub_id`. The model produces ONE ranking for identical input. At most 1 of 4 gets hit@1=1; the other 3 are guaranteed misses.

13 unique texts create 41 eval items with 18 guaranteed structural misses. Additionally, 34 true duplicates (same text + same hub) inflate variance.

### 2.2 Solution: Deduplicate + Multi-Label Ground Truth

Two changes to `build_evaluation_corpus()`:

1. **Deduplicate** to unique `(framework_name, control_text)` keys. Keep one `EvalItem` per unique query text per framework.

2. **Multi-label ground truth**: Store `valid_hub_ids: set[str]` on each `EvalItem` containing ALL hubs that the control text maps to.

Changes to scoring:
- `hit@1`: `pred[0] in item.valid_hub_ids` (any valid hub = hit)
- `MRR`: reciprocal rank of FIRST valid hub in prediction list
- `NDCG@10`: DCG computed at first valid hub position

### 2.3 EvalItem Schema Change

```python
@dataclass
class EvalItem:
    control_text: str
    ground_truth_hub_id: str        # primary (first encountered)
    valid_hub_ids: frozenset[str]   # ALL valid hubs for this text
    ground_truth_hub_name: str
    framework_name: str
    section_id: str
    track: str
```

### 2.4 Backward Compatibility

Existing `predictions.json` files contain top-10 predictions keyed by `ground_truth_hub_id`. These can be re-scored with corrected methodology by building the `valid_hub_ids` mapping from the curated links and applying multi-label-aware scoring to the stored predictions.

---

## 3. Fix F4: Firewalled Zero-Shot Baseline

### 3.1 Problem

The gate threshold (hit@1 > 0.516) is anchored to Phase 0's best BGE baseline (0.416). Phase 0 used hub representations with up to 78 linked standard names per hub (`"{path} | {hub_name}: {linked names}"`). Phase 1B uses only `"{path} | {hub_name}"`. The Phase 0 baseline had strictly richer hub representations.

Phase 0 WAS LOFO-firewalled (confirmed), but the hub representation difference makes the comparison invalid.

### 3.2 Solution

Compute a new zero-shot baseline using:
- Same model: `BAAI/bge-large-en-v1.5` (no fine-tuning)
- Same hub format: `"{hierarchy_path} | {hub_name}"` (Phase 1B format)
- Same LOFO protocol: 5 folds, firewalled hub texts
- Same evaluation: corrected multi-label-aware methodology (Fix F3)

This baseline replaces the Phase 0 reference for gate comparison.

### 3.3 Gate Revision

```
Old gate:  trained_hit@1 > 0.416 + 0.10 = 0.516
New gate:  trained_hit@1 > firewalled_zero_shot + 0.10
           (measured with corrected multi-label evaluation)
```

---

## 4. Decision Gates

### Gate 0: Free Measurements

Before any code changes to the training pipeline:

1. Compute firewalled zero-shot baseline (5 min, local Orin)
2. Fix evaluation methodology (multi-label + dedup)
3. Re-score existing Phase 1B predictions with corrected eval
4. Score zero-shot with corrected eval

**Decision:** `corrected_phase1b (≈0.508) - corrected_zero_shot > 0.10`?

| Outcome | Action |
|---------|--------|
| Delta > 0.15 | Gate passes comfortably. Training pipeline fixes become improvement (P1), not blocking. |
| 0.10 < Delta < 0.15 | Marginal pass. Pipeline fixes important for confidence. |
| Delta < 0.10 | Gate fails even with corrected eval. Pipeline fixes are blocking. |

### Gate 1: Corrected Primary

After training pipeline fixes, retrain and evaluate:

**Decision:** `corrected_retrained - corrected_zero_shot > 0.10` with CI_low > 0?

---

## 5. What Remains Valid from Original Spec

| Component | Status |
|-----------|--------|
| Model architecture (BGE-large + LoRA rank 16) | Valid |
| Loss function (MNRL) | Valid |
| Hard negative mining (siblings/cousins) | Valid (verified: 255 negatives/anchor, hard negatives ARE used) |
| Hub representation firewall | Valid (battle-tested through 3 iterations) |
| LOFO protocol (5 AI frameworks) | Valid |
| Fold-stratified bootstrap CIs | Valid |
| A6 ablation result (descriptions hurt) | Directionally valid (relative comparison unaffected by bugs) |
| Ablation suite design (A1-A10) | Valid (deferred until corrected primary passes gate) |

---

## 6. A6 Ablation Status

A6 (descriptions) ran on the same buggy pipeline as the primary. Both had identical bugs (no batch sampler, no temperature sampling). The **relative** comparison is valid: descriptions hurt hit@1 by -16% (0.437 → 0.365). The specificity-sensitivity tradeoff interpretation stands.

A6 does NOT need to be re-run. The absolute numbers may shift under a corrected pipeline, but the direction is robust. A6 is closed as a negative result.

---

## 7. Files Modified

| File | Change |
|------|--------|
| `tract/training/data.py` | Replace `HubAwareBatchSampler` + `apply_temperature_sampling_order()` with `HubAwareTemperatureSampler`; add `hub_id` column to `pairs_to_dataset()` |
| `tract/training/loop.py` | Pass `batch_sampler=HubAwareTemperatureSampler` in training args |
| `scripts/phase0/common.py` | Add `valid_hub_ids` to `EvalItem`; update `build_evaluation_corpus()` for dedup; update `score_predictions()` for multi-label |
| `tract/training/evaluate.py` | Update `evaluate_on_fold()` to use multi-label scoring; update `hit1_indicators` |
| `tests/test_firewall.py` | No changes needed |
| `tests/test_data.py` | New tests for `HubAwareTemperatureSampler` |
| `tests/test_evaluate.py` | New tests for multi-label scoring |
| New: `scripts/phase1b/zero_shot_baseline.py` | Firewalled zero-shot baseline script |
| New: `scripts/phase1b/rescore_predictions.py` | Re-score existing predictions with corrected eval |
