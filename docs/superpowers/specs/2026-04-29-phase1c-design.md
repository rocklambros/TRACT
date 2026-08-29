# Phase 1C Design Spec: Guardrails, Active Learning & Crosswalk DB

**PRD Sections:** 6.6 (Guardrails), 6.7 (Active Learning), 6.8 (Crosswalk DB)  
**Predecessor:** Phase 1B Gate 1 CLEAN PASS — hit@1=0.531, delta=+0.132 over zero-shot  
**Review History:** 2 full adversarial reviews (5 rounds each), 26 design changes accepted  
**Date:** 2026-04-29

---

## 1. Scope

Phase 1C bridges the trained contrastive model (Phase 1B) to a usable crosswalk database. Three capabilities:

1. **Calibration pipeline** — Transform raw cosine similarities into calibrated probabilities with conformal coverage guarantees
2. **Active learning loop** — Assign 552 unmapped AI controls to CRE hubs via model prediction + expert review (2-3 rounds)
3. **Crosswalk database** — Persist all assignments (model, expert, training data) in a queryable SQLite store

### What's NOT in scope
- Per-hub thresholds (PRD 6.6 Model Integrity — deferred: only ~3-5 hubs have ≥5 eval items)
- Transfer learning A/B test (PRD 6.6 Model Integrity — deferred: requires full-retrain ablation, out of scope for Phase 1C)
- Multi-hub disagreement detector (PRD 6.6 Output Integrity — partially addressed by conformal prediction sets; explicit detector deferred to Phase 1D)
- Paraphrase robustness probes (PRD 6.6 Adversarial — Phase 1D)
- Hub proposal pipeline for OOD items (Phase 1D)
- Cross-framework relationship matrix (PRD 6.8 — derivable from crosswalk DB schema via SQL joins; explicit computation deferred to Phase 1D CLI)
- Web interface (Phase 2)

---

## 2. Architecture

### 2.1 New Modules

```
tract/
  calibration/
    __init__.py
    temperature.py      # Temperature scaling (diagnostic + production)
    conformal.py        # Conformal prediction sets
    ood.py              # OOD detection (max cosine threshold)
    diagnostics.py      # ECE, KS-test, coverage metrics
  active_learning/
    __init__.py
    deploy.py           # Deployment model training + inference
    review.py           # Review JSON generation + ingestion
    canary.py           # Canary item management
    stopping.py         # AL stopping criteria
  crosswalk/
    __init__.py
    schema.py           # SQLite schema + migrations
    store.py            # CRUD operations (atomic transactions)
    export.py           # JSON/CSV export
    snapshot.py         # Pre-round snapshots
```

### 2.2 Modified Modules

| Module | Changes |
|--------|---------|
| `tract/training/calibrate.py` | **Replace entirely** — current stub uses single-label NLL; new multi-label math lives in `tract/calibration/temperature.py` |
| `tract/training/data_quality.py` | Add `QualityTier.AL = "AL"` to the enum |
| `tract/training/data.py` | Add `"AL": 3` to `TIER_PRIORITY` dict (line 74) |
| `tract/training/evaluate.py` | Add `extract_similarity_matrix()` — returns full sim matrix + hub_ids + ground_truth. Note: `EvalItem.valid_hub_ids` already supports multi-label (frozenset) |
| `tract/training/orchestrate.py` | Add `train_deployment_model()` wrapper |

### 2.3 Key Design Principle: Diagnostic + Production Calibration

**Not** "two-stage calibration." T_lofo is diagnostic-only; T_deploy is the single production parameter.

| Parameter | Fitted on | Used for |
|-----------|-----------|----------|
| T_lofo | Pooled LOFO similarities (147 items, sqrt(n)-weighted NLL) | LOFO characterization, ECE reporting, sanity checks |
| T_deploy | 420 held-out traditional links (deployment model output) | All downstream: threshold, confidence, conformal, OOD |

**Why two temperatures exist at all:** T_lofo characterizes the LOFO evaluation regime — useful for comparing Phase 1B vs future training runs. T_deploy is the only temperature used in production inference. They are not interchangeable; the deployment model was trained on different data than any single LOFO fold model.

---

## 3. Similarity Extraction (T0)

### 3.1 Problem

Phase 1B `predictions.json` stores only ranked hub IDs — no cosine similarity scores. Calibration requires full similarity matrices.

### 3.2 Solution

Re-run inference on the 5 saved LOFO fold models to extract similarity matrices.

**New function:** `extract_similarity_matrix()` in `tract/training/evaluate.py`

```python
def extract_similarity_matrix(
    model: SentenceTransformer,
    eval_items: list[EvalItem],
    hub_ids: list[str],         # canonical sorted order
    hub_embs: NDArray,          # shape (n_hubs, dim), same order as hub_ids
) -> dict:
    """Extract full similarity matrix for calibration.

    Returns:
        {
            "sims": ndarray (n_eval, n_hubs),  # cosine similarities
            "hub_ids": list[str],               # canonical sorted hub IDs
            "gt_json": list[str],               # JSON-encoded list-of-lists for multi-label
            "frameworks": list[str],            # source framework per item
        }
    """
```

### 3.3 Multi-Label Ground Truth Encoding

8.8% of 147 eval items map to multiple valid CRE hubs. NPZ format doesn't support ragged arrays.

**Solution:** Store as JSON strings: `gt_json = [json.dumps(sorted(item.valid_hub_ids)) for item in eval_items]`. Parse at calibration time.

### 3.4 Model Reload

**New function:** `load_fold_model()` in `tract/training/deploy.py`

```python
def load_fold_model(fold_path: Path) -> SentenceTransformer:
    """Load a saved LOFO fold model with LoRA adapters.

    Verifies:
    - Model loads without error
    - Embedding dimensionality matches expected (1024 for BGE-large)
    - Smoke test: encode one text, check output shape
    """
```

### 3.5 Output Format

Per fold: `results/phase1c/similarities/fold_{name}.npz`
```
sims:       float32 (n_eval, 522)    # cosine similarities
hub_ids:    str array (522,)          # canonical sorted
gt_json:    str array (n_eval,)       # JSON-encoded multi-label ground truth
frameworks: str array (n_eval,)       # source framework
```

Hub ordering: `sorted(hierarchy.hubs.keys())` — same canonical order used in Phase 1B orchestrate.py. All 522 hubs present in every fold (firewall changes text, not hub set).

---

## 4. Calibration Pipeline

### 4.1 Multi-Label NLL

```python
def multi_label_nll(
    probs: ndarray,            # (n, n_hubs) calibrated probabilities
    valid_hub_indices: list[list[int]],  # per-item list of valid hub column indices
) -> float:
    """NLL = -log(sum(P(hub) for hub in valid_hubs)) per item."""
    nll = 0.0
    for i, valid_indices in enumerate(valid_hub_indices):
        p_valid = sum(probs[i, j] for j in valid_indices)
        nll -= np.log(p_valid + 1e-10)
    return nll / len(valid_hub_indices)
```

### 4.2 T_lofo Fitting (Diagnostic)

**Input:** 5 fold similarity NPZ files  
**Weighting:** sqrt(n) per fold (meta-analysis standard for combining estimates of different precision)

| Fold | n | Sample weight | Equal weight | sqrt(n) weight |
|------|---|---------------|--------------|----------------|
| OWASP-X | 63 | 43% | 20% | 33% |
| ATLAS | 43 | 29% | 20% | 27% |
| NIST | 28 | 19% | 20% | 22% |
| ML-10 | 7 | 5% | 20% | 11% |
| LLM-10 | 6 | 4% | 20% | 10% |

sqrt(n) dampens OWASP-X dominance (43% → 33%) without giving LLM-10 (n=6) disproportionate influence (4% → 10% instead of 20%).

**Grid search:** 200 points log-spaced in [0.01, 5.0]. Log-spacing gives better resolution at low T where NLL changes rapidly.

**Output:** T_lofo scalar, diagnostic ECE, per-fold NLL breakdown.

### 4.3 T_deploy Fitting (Production)

**Input:** Deployment model similarities on 420 traditional holdout links  
**Method:** Same multi-label NLL grid search (though traditional links are almost always single-label)

**Domain mismatch diagnostic:** After fitting, compute KS-test between:
- Max cosine similarity distribution of 420 traditional holdout items
- Max cosine similarity distribution of 147 LOFO AI items (run through deployment model)

If KS p < 0.01: log WARNING with effect size. This is an acknowledged risk — traditional controls are shorter (mean 200 chars) and more formulaic than AI controls (mean 705 chars). After AL round 1 reviews, refit T_deploy on combined traditional holdout + reviewed AI items.

### 4.4 ECE Gate

**ECE < 0.10** with 5 equal-width bins, bootstrap 95% CI (1000 resamples).

Previous design used 10 bins and ECE < 0.05 — relaxed because:
- 147 items / 10 bins = ~15 per bin (too sparse for reliable calibration)
- 420 holdout / 10 bins = 42 per bin (adequate but tight for CI)
- 5 bins gives ≥29 items per bin in LOFO, ≥84 in holdout

Computed for both T_lofo (on LOFO data, diagnostic) and T_deploy (on holdout, gate).

### 4.5 Conformal Prediction

**Key principle:** Conformal quantile must use the same temperature as inference. Since inference uses T_deploy, conformal quantile uses T_deploy.

**Data flow:**
1. Run 147 LOFO eval items through the **deployment model** (not fold models)
2. Compute P_deploy = softmax(sim / T_deploy) for each item
3. Nonconformity score: `1 - sum(P_deploy(hub) for hub in valid_hubs)`
4. Quantile: `q = quantile(scores, ceil((n+1)*(1-alpha))/n)` at alpha=0.10
5. Prediction set at inference: `{hub : P_deploy(hub) >= 1-q}`

**Known conservative bias:** The deployment model has seen LOFO texts as training anchors. Its similarity scores on LOFO items are inflated relative to genuinely unseen texts. This makes the conformal quantile higher than warranted → prediction sets are larger than necessary for new controls. Conservative, not dangerous. Refit conformal quantile after AL round 1 on reviewed items for tighter sets.

**Coverage guarantee:** Empirical only (not mathematical). LOFO models ≠ deployment model violates exchangeability assumption of conformal prediction. Validate post-hoc on AL round 1 accepted items.

### 4.6 Global Threshold

**Optimization target:** F1 on multi-label hub assignment

- TP: predicted hub ∈ valid_hubs
- FP: predicted hub ∉ valid_hubs  
- FN: no predicted hub ∈ valid_hubs

Grid search over 200 threshold values in [0.001, 0.999]. Fitted on 420 holdout with T_deploy.

No per-hub thresholds — only ~3-5 hubs have ≥5 eval items. Global threshold only.

### 4.7 OOD Detection

**Method:** 5th percentile of max cosine similarity across all in-distribution items (420 holdout + 147 LOFO through deployment model).

**Validation:** 30 synthetic non-security texts from diverse domains (cooking, sports, astronomy, biology, literature, etc.). Curated manually — no security-adjacent terms.

**Gate:** ≥90% of synthetic OOD texts below the threshold. Items below threshold at inference → flagged as "no good hub match," routed to hub proposal pipeline (Phase 1D).

**Fixture:** `tests/fixtures/ood_synthetic_texts.json` — 30 items, version-controlled.

---

## 5. Deployment Model

### 5.1 Training Data

All tiered_links with no framework excluded (i.e., `build_training_pairs(tiered_links, hub_texts, excluded_framework=None)`), minus 440 randomly-sampled traditional links (holdout). This is NOT a concatenation of the 5 LOFO fold training sets — it is the full pool with no LOFO firewall.

**Holdout selection:** Simple random sampling from all ~3,932 traditional TieredLinks, seeded (config.seed). No stratification — 440 items is sufficient for T_deploy fitting regardless of framework distribution. Post-hoc check: verify no single framework contributes >50% of holdout.

**Explicit partition of 440:**
- 420 for T_deploy fitting + ECE + OOD threshold
- 20 for canaries (traditional domain)

### 5.2 Training

**New function:** `train_deployment_model()` in `tract/training/orchestrate.py`

```python
def train_deployment_model(
    config: TrainingConfig,
    tiered_links: list[TieredLink],
    holdout_links: list[TieredLink],
    hierarchy: CREHierarchy,
    output_dir: Path,
) -> SentenceTransformer:
    """Train deployment model on all folds minus holdout.

    1. Remove holdout from tiered_links
    2. build_training_pairs(remaining, hub_texts, excluded_framework=None)
    3. pairs_to_dataset()
    4. train_model()
    """
```

Same hyperparameters as Phase 1B: 20 epochs, batch=32, lr=5e-4, LoRA rank=16. Uses `train_model()` from `loop.py` to inherit CUDA determinism flags.

### 5.3 Inference

Run the deployment model on:
1. **552 unmapped AI controls** — for active learning
2. **420 traditional holdout** — for T_deploy fitting
3. **147 LOFO eval items** — for conformal quantile (with T_deploy) and domain mismatch diagnostic
4. **20 traditional canaries** — for holdout canary baseline
5. **30 OOD synthetic texts** — for OOD threshold validation

Save full similarity matrices + calibrated probabilities.

### 5.4 Unmapped AI Controls (552 items)

**Source:** All controls from the 5 AI-specific frameworks that have zero OpenCRE links:
- CSA AICM (243 controls)
- OWASP Agentic Top 10 (10 controls)
- MITRE ATLAS techniques/mitigations without CRE links
- NIST AI 600-1 controls without CRE links
- EU AI Act articles without CRE links

**Identification:** Any parsed control whose framework is in the AI tier AND has no entry in `hub_links_training.jsonl` (no TieredLink exists). Load from `data/processed/{framework}_controls.json`.

**Preprocessing:** Same `sanitize_text()` pipeline as training data. Construct query text as `"{section_id}: {title}. {description}"` — same format used in `build_training_pairs()`.

### 5.5 Holdout-to-EvalItem Adapter

The 420 holdout items are `TieredLink` objects, not `EvalItem` objects. For calibration, construct lightweight eval-compatible dicts:

```python
def holdout_to_eval(link: TieredLink, hierarchy: CREHierarchy) -> dict:
    """Convert a TieredLink to an eval-compatible record.

    Returns dict with keys: control_text, framework, valid_hub_ids (frozenset).
    Valid_hub_ids comes from link.link["cre_id"] — traditional links are single-label.
    """
```

### 5.6 review.json Schema

```json
{
  "round": 1,
  "model_version": "git_sha + data_hash",
  "temperature": 0.42,
  "threshold": 0.65,
  "generated_at": "2026-04-30T10:15:00Z",
  "items": [
    {
      "control_id": "CSA_AICM:AICM-04-01",
      "framework": "CSA AICM",
      "control_text": "Validate AI Model...",
      "predictions": [
        {"hub_id": "547-824", "confidence": 0.91, "in_conformal_set": true},
        {"hub_id": "364-516", "confidence": 0.23, "in_conformal_set": false}
      ],
      "is_ood": false,
      "auto_accept_candidate": true,
      "review": null
    }
  ]
}
```

**Canary items** are interleaved with real items. They are NOT flagged as canaries in the JSON — the reviewer sees them as regular predictions. Canary tracking is in `canary.py` only.

**Review field** (filled by expert):
```json
{
  "status": "accepted | rejected | corrected",
  "corrected_hub_id": null,
  "notes": "",
  "reviewed_at": "2026-04-30T14:30:00Z"
}
```

---

## 6. Active Learning Loop

### 6.1 Canary Design

**Two canary sources:**

1. **20 pre-labeled AI controls** — Drawn from the 552 unmapped pool. Expert independently assigns hub(s) **before** seeing model predictions. Seeded into review.json alongside model predictions. Measures whether the expert rubber-stamps vs. applies judgment.

2. **20 traditional holdout canaries** — From the explicit 20-item holdout partition. Already have ground truth. Measures deployment model's calibration accuracy on traditional domain.

**Why pre-labeled AI canaries instead of LOFO eval items:** The deployment model was trained on all LOFO texts as anchors. Canary accuracy on LOFO items measures memorization, not prediction quality.

**Gate:** AI canary accuracy ≥ 85% AND acceptance rate > 80%.

### 6.2 Round Structure

```
Round N:
  1. Generate review.json:
     - 552 unmapped controls (round 1) or remaining unmapped (round 2+)
     - Model's top-K predictions with calibrated P_deploy scores
     - 20 AI canaries seeded in (same canaries each round — test consistency)
     - 20 traditional canaries (round 1 only — calibration check)
  
  2. Expert review:
     - Batch accept: P_deploy >= threshold AND in conformal set → auto-accept candidate
     - Individual review: items below threshold or outside conformal set
     - For each: accept / reject / correct (provide correct hub)
  
  3. Ingest accepted predictions:
     - Format as TieredLink objects with tier=QualityTier.AL
     - Set link dict: {standard_name, section_name, cre_id, link_type: "active_learning", al_round: N}
     - Route through build_training_pairs() — inherits dedup, text formatting, existing pipeline
     - Store in crosswalk.db with provenance
  
  4. Retrain deployment model:
     - Original training data + accepted AL predictions
     - Same hyperparameters
     - Refit T_deploy on 420 holdout + reviewed AI items
     - Recompute conformal quantile
     - Update OOD threshold
  
  5. Evaluate stopping criteria
```

### 6.3 Stopping Criteria

All three must be met:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Acceptance rate | > 80% | Expert trusts model predictions |
| AI canary accuracy | ≥ 85% | Expert applies genuine judgment |
| Hub diversity | ≥ 50 unique hubs in accepted | Model not concentrating on easy cases |

**Hub diversity:** After each round, report the number of unique hubs covered by accepted predictions. If < 50 (out of 522), the confirmation loop is active — the model is good at well-represented hubs and ignoring sparse ones. In round 2+, prioritize reviewing LOW-confidence predictions for maximum information gain.

### 6.4 Round Budget

2-3 rounds. After round 3, diminishing returns are expected — remaining unmapped controls are genuinely hard cases that may need new CRE hubs (Phase 1D hub proposal).

### 6.5 Confirmation Loop Mitigation

After round 1:
1. Report hub diversity of accepted items
2. Compare confidence distributions: remaining items vs. accepted items
3. If mean confidence of remaining items isn't increasing, retraining isn't helping them
4. Round 2+: prioritize low-confidence review (uncertainty sampling) over batch-accepting high-confidence items

---

## 7. Crosswalk Database

### 7.1 Schema

```sql
CREATE TABLE frameworks (
    id TEXT PRIMARY KEY,        -- e.g., "OWASP_AI_Exchange"
    name TEXT NOT NULL,         -- e.g., "OWASP AI Exchange"  
    version TEXT,               -- document version
    fetch_date TEXT,            -- ISO-8601
    control_count INTEGER
);

CREATE TABLE controls (
    id TEXT PRIMARY KEY,        -- framework_id + ":" + section_id
    framework_id TEXT NOT NULL REFERENCES frameworks(id),
    section_id TEXT NOT NULL,   -- original section/control ID
    title TEXT,
    description TEXT,
    full_text TEXT
);

CREATE TABLE hubs (
    id TEXT PRIMARY KEY,        -- CRE ID (e.g., "CRE:236-712")
    name TEXT NOT NULL,
    path TEXT,                  -- hierarchy path
    parent_id TEXT REFERENCES hubs(id)
);

CREATE TABLE assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    control_id TEXT NOT NULL REFERENCES controls(id),
    hub_id TEXT NOT NULL REFERENCES hubs(id),
    confidence REAL,            -- P_deploy after calibration
    in_conformal_set INTEGER,   -- 1 if hub is in conformal prediction set
    is_ood INTEGER DEFAULT 0,   -- 1 if control flagged as OOD
    provenance TEXT NOT NULL,   -- "training_T1", "training_T3", "active_learning_round_1", etc.
    source_link_id TEXT,        -- OpenCRE link ID for training provenance (NULL for AL predictions)
    model_version TEXT,         -- git SHA + data hash
    review_status TEXT DEFAULT 'pending',  -- pending, accepted, rejected, corrected
    reviewer TEXT,              -- expert identifier
    review_date TEXT,           -- ISO-8601
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_number INTEGER NOT NULL,
    snapshot_date TEXT DEFAULT (datetime('now')),
    db_hash TEXT NOT NULL,      -- SHA-256 of serialized DB state
    description TEXT
);

CREATE INDEX idx_assignments_control ON assignments(control_id);
CREATE INDEX idx_assignments_hub ON assignments(hub_id);
CREATE INDEX idx_assignments_provenance ON assignments(provenance);
CREATE INDEX idx_assignments_review ON assignments(review_status);
```

### 7.2 Population Strategy

**T0 (initial):** Populate hubs, frameworks, controls from hierarchy + parsed framework data. Populate assignments from training TieredLinks (provenance = "training_T1", "training_T1_AI", "training_T3").

**T2 (deployment model):** Add model predictions for 552 unmapped + 440 holdout as assignments with review_status='pending'.

**T3 (per AL round):** Update review_status to 'accepted'/'rejected'/'corrected'. Add corrected assignments with new hub_id. Take snapshot before each round.

### 7.3 Atomic Operations

All writes wrapped in transactions. Snapshot before destructive operations. SQLite WAL mode for concurrent read access.

### 7.4 Export

```python
def export_crosswalk(db_path: Path, format: str = "json") -> Path:
    """Export accepted assignments as crosswalk matrix."""
    # JSON: {framework: {control_id: [{hub_id, confidence, provenance}]}}
    # CSV: control_id, framework, hub_id, confidence, provenance, review_status
```

---

## 8. Execution Flow

### Single Pod Lifecycle

All GPU work on one RunPod H100. Sequential within the pod to avoid multi-pod orchestration complexity.

```
T0: LOCAL (CPU) + GPU POD
  ├─ LOCAL: Populate crosswalk.db (hubs, frameworks, controls, training assignments)
  └─ GPU: Load 5 fold models sequentially, extract similarity matrices (5 × ~30s = ~2.5 min)
           Save fold_*.npz to results/phase1c/similarities/

T1: LOCAL (CPU) + GPU POD
  ├─ LOCAL: Fit T_lofo (diagnostic) — sqrt(n)-weighted multi-label NLL on fold NPZs
  │         Compute diagnostic ECE (5 bins, bootstrap CI)
  │         Log per-fold NLL breakdown
  └─ GPU: Select 440 holdout (seeded random, partition 420+20)
           Train deployment model (~12 min)
           Save to results/phase1c/deployment_model/

T2: GPU POD → LOCAL (CPU)
  ├─ GPU: Inference on 552 + 420 + 147 + 20 canary + 30 OOD texts
  │       Save similarity matrices
  │       Terminate pod
  └─ LOCAL: Fit T_deploy on 420 holdout similarities
            Compute ECE gate (< 0.10, 5 bins, bootstrap CI)
            Compute conformal quantile (alpha=0.10, using T_deploy)
            Compute OOD threshold (5th percentile + synthetic validation)
            KS-test diagnostic (traditional vs AI similarity distributions)
            Compute global threshold (max-F1 on 420 holdout)
            Generate round_1/review.json (552 + 20 AI canaries + 20 trad canaries)
            Populate crosswalk.db with model predictions

T3: HUMAN REVIEW (async)
  └─ Expert reviews via CLI script (batch accept + individual review)

T4: PER-ROUND RETRAIN (if needed)
  ├─ Snapshot crosswalk.db
  ├─ Ingest accepted predictions through build_training_pairs()
  ├─ GPU: Retrain deployment model on expanded data
  ├─ GPU: Re-inference on remaining unmapped controls
  ├─ LOCAL: Refit T_deploy on 420 holdout + reviewed AI items
  ├─ LOCAL: Recompute conformal quantile
  └─ LOCAL: Generate next round review.json
```

**Estimated wall time:** T0-T2: ~20 minutes GPU. T3: async human. T4: ~15 min/round GPU.

---

## 9. Quality Gates

### Gate 2: Calibration (blocks AL)

| Metric | Threshold | Data |
|--------|-----------|------|
| ECE (T_deploy) | < 0.10 | 420 traditional holdout, 5 bins, bootstrap 95% CI |
| Conformal coverage | ≥ 0.90 | 147 LOFO items through deployment model |
| OOD separation | ≥ 90% | 30 synthetic texts below OOD threshold |
| Global threshold F1 | > 0.0 | 420 holdout (sanity check, not a hard gate) |

### Gate 3: AL Stopping (per round)

| Metric | Threshold |
|--------|-----------|
| Expert acceptance rate | > 80% |
| AI canary accuracy | ≥ 85% |
| Hub diversity (unique hubs in accepted) | ≥ 50 |

### Diagnostics (logged, not gated)

- T_lofo ECE (on LOFO data, 5 bins)
- Per-fold NLL with sqrt(n) weights
- |T_lofo - T_deploy| gap — sanity check that LOFO and deployment models are in same regime (expect < 0.5; flag WARNING if larger)
- KS-test p-value for traditional vs AI similarity distributions
- Full-recall coverage (fraction of multi-label items where ALL valid hubs in prediction set)
- Hub coverage after each AL round (accepted_unique_hubs / total_hubs)
- Confidence distribution comparison: remaining vs. accepted items

---

## 10. Open Risks (Acknowledged)

### 10.1 Conservative Conformal Bias
Deployment model saw LOFO texts in training → inflated similarities → conformal quantile too high → larger prediction sets. Cost: more expert review items, not correctness. Mitigated by refitting after AL round 1.

### 10.2 T_deploy Domain Mismatch
Traditional controls (mean 200 chars, formulaic) vs AI controls (mean 705 chars, descriptive) produce different similarity distributions. T_deploy fitted on traditional may not calibrate well for AI. KS-test diagnostic quantifies the gap. Mitigated by refitting after AL round 1 on combined data.

### 10.3 AL Convergence on Sparse Hubs
Even with hub diversity co-criterion, the AL loop may plateau on controls mapping to underrepresented hubs. These are genuinely hard cases that may need new CRE hubs (Phase 1D hub proposal). Not a design flaw — inherent to the data.

### 10.4 Pre-Labeled Canary Effort
Pre-labeling 20 AI controls requires ~30 minutes of expert time before AL begins. This is the minimum cost for genuine quality measurement — without it, canary accuracy measures memorization.

---

## 11. Design Changes Log

### First Adversarial Review (11 changes)

| # | Change | Source |
|---|--------|--------|
| 1 | Multi-label NLL: -log(sum(P(valid))) instead of -log(P(single)) | R1-F1 |
| 2 | Multi-label conformal score: 1-sum(P(valid)) | R1-F1 |
| 3 | Multi-label threshold F1: TP if ANY valid hub hit | R1-F1 |
| 4 | Two-stage calibration (T_lofo + T_deploy) | R2-F1 |
| 5 | 440 traditional holdout for T_deploy | R2-F1 |
| 6 | Conformal is empirical, not mathematical guarantee | R2-F2 |
| 7 | ECE relaxed to < 0.10 with 5 bins (from < 0.05, 10 bins) | R3-F2 |
| 8 | Hub ordering canonical sorted (sentinel dropped — hub set is fixed across folds, no missing hubs) | R3-F1 |
| 9 | Canary items with dual stopping criterion | R4-F1 |
| 10 | OOD validated on 30 synthetic texts | R4-F2 |
| 11 | AL predictions routed through build_training_pairs() | R3-F3 |

### Second Adversarial Review (15 changes)

| # | Change | Source |
|---|--------|--------|
| 12 | Conformal quantile uses T_deploy, not T_lofo | R2v2-F1 |
| 13 | KS-test diagnostic for T_deploy domain mismatch | R2v2-F2 |
| 14 | sqrt(n) fold weighting for T_lofo | R2v2-F3 |
| 15 | Simple random holdout sampling (no stratification) | R2v2-F4 |
| 16 | Hub diversity reporting after each AL round | R2v2-F6 |
| 17 | Full-recall coverage diagnostic for multi-label | R2v2-F7 |
| 18 | T_lofo renamed to diagnostic-only; T_deploy is sole production parameter | R1v2-F1 |
| 19 | Multi-label ground truth stored as JSON strings in NPZ | R1v2-F3 |
| 20 | Explicit holdout partition: 420 calibration + 20 canaries | R1v2-F4 |
| 21 | load_fold_model() with smoke test | R3v2-F1 |
| 22 | train_deployment_model() wrapper | R3v2-F2 |
| 23 | QualityTier.AL in data_quality.py + TIER_PRIORITY["AL"]=3 in data.py | R3v2-F3 |
| 24 | Single pod lifecycle (sequential T0→T1→T2) | R3v2-F4 |
| 25 | Pre-label 20 AI controls as canaries (not LOFO eval, not holdout) | R4v2-F1 |
| 26 | Hub diversity co-criterion: unique_hubs ≥ 50 | R4v2-F2 |

### Overturned From First Review

| Original | Overturned to | Reason |
|----------|--------------|--------|
| 20 LOFO eval items as canaries | 20 pre-labeled AI controls | Deployment model saw LOFO texts in training — canary accuracy measures memorization |
| T_lofo + T_deploy as co-equal production parameters | T_lofo diagnostic only | Conformal quantile must use T_deploy → T_lofo has no production use |
| Fold-weighted NLL as key design decision | Low priority (diagnostic only) | Follows from T_lofo demotion |
