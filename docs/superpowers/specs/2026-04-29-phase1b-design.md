# Phase 1B Design Spec: CRE Hub Assignment Model Training Pipeline

**Date:** 2026-04-29 (revised 2026-04-29 post-adversarial review round 3)
**PRD Sections:** 6.4, 6.5, 6.11
**Prerequisites:** Phase 1A complete (hierarchy, descriptions, framework ingestion), Phase 0R baselines complete
**Scope:** Minimum viable experiment (Phase 1B-alpha), data quality pipeline, LoRA contrastive fine-tuning, hub representation firewall, LOFO evaluation harness, ablation suite, post-training diagnostic

---

## 0. Phase 0R Context (Corrected Baselines)

Phase 0R ran 12 experiments on audit-corrected curated data (4,405 links, 197 AI eval items, ~458 hubs). Key findings that shape this spec:

| Finding | Result | Implication |
|---------|--------|-------------|
| LLM ceiling | Sonnet 3-shot hit@1=0.624 | Task is learnable, massive headroom exists |
| Best embedding | BGE-large + paths hit@1=0.416 | Below 0.50 gate → fine-tuning required |
| Path enrichment | +5.1pp hit@1 for BGE | Hub representations MUST include hierarchy paths |
| Description enrichment | -4.8pp BGE, -22.8pp GTE | Descriptions hurt zero-shot embeddings (but see ablation A6 — fine-tuning may change this) |
| Bigger models | 7B models worse than 335M | Model size doesn't help; stick with BGE-large |
| DeBERTa-NLI | hit@1=0.000 | NLI paradigm completely wrong; contrastive is correct |
| Per-framework variance | OWASP-X: 0.891, NIST: 0.444 | Hardest frameworks need most improvement |

**PRD success criteria:** Trained model hit@1 > baseline + 0.10 (i.e., > 0.516), hit@5 > 0.70.
**Realistic targets:** hit@1 ≥ 0.55 (conservative), ≥ 0.60 (stretch, approaching Sonnet at 1000x cheaper inference).

### 0.1 Post-Training Diagnostic: LLM Prediction Comparison

After training completes, compare the fine-tuned model's per-item predictions against Phase 0R Sonnet 3-shot predictions (from `results/phase0/exp6_fewshot_sonnet/`). This is a zero-cost diagnostic, not an experiment.

**Analysis:**
- **Model agrees with Sonnet, both correct:** High-confidence assignments
- **Model agrees with Sonnet, both wrong:** Genuinely hard examples (ambiguous hub assignments)
- **Model and Sonnet disagree:** Identifies the model's specific failure modes vs. LLM reasoning

If Phase 0R did not save per-item Sonnet predictions, re-run exp6 on the 197 AI items (~$18). This is deferred until after training, not on the critical path.

### 0.2 Phase 1B-alpha: Minimum Viable Experiment

Before building the full pipeline, run a minimal training experiment to validate that LoRA + MNRL produces any improvement over zero-shot BGE:

- **Data:** Single fold (hold out ATLAS, n=65). AI-only training data (132 examples).
- **Model:** BGE-large-v1.5 + LoRA rank 16, vanilla MNRL (in-batch negatives only, no hard negatives).
- **No:** temperature-scaled sampling, hub-aware batching, hard negative mining, calibration.
- **Implementation:** ~50 lines of training code using `SentenceTransformerTrainer` + PEFT. Run on Orin in ~10 minutes.
- **Gate:** Any improvement over hit@1=0.416 on the 65 ATLAS eval items → proceed to full pipeline. No improvement → investigate before investing 3-4 weeks in the full pipeline.
- **Estimated effort:** 3 hours implementation, <$0.50 compute.

This is Task 1 in the implementation plan. All subsequent tasks are gated on Phase 1B-alpha passing.

---

## 1. Training Data Pipeline

### 1.1 Data Sources

All training data comes from OpenCRE standard-to-hub links, curated during Phase 0R audit:

| Source | Raw Count | After Quality Filter | Usage |
|--------|-----------|---------------------|-------|
| Traditional framework links (17 frameworks) | 4,208 | ~3,876 usable | Joint training (with temp-scaled AI sampling) |
| AI framework links (5 frameworks, curated) | 197 | 197 | LOFO evaluation + fine-tuning |
| **Total** | 4,405 | ~4,073 | |

AutomaticallyLinkedTo links are expert-quality (deterministic CAPEC→CWE→CRE chain). Treat identically to human LinkedTo with no penalty.

### 1.2 Training Data Quality Pipeline

Programmatic audit of traditional links (run: `scripts/analysis/audit_traditional_links.py`) found 92.1% of traditional links (3,876) are usable as-is. The remaining 7.9% (332 links) are dropped.

**Dropped links:**

| Framework | Links | Issue | Action |
|-----------|-------|-------|--------|
| NIST 800-63 | 79 | 100% bare IDs ("5.1.4.2") | Drop (parsed data also has bare IDs — enrichment impossible) |
| OWASP Proactive Controls | 76 | 100% bare IDs ("C1") | Drop (parsed data also has bare IDs — enrichment impossible) |
| DSOMM + misc | 177 | Short entries ("Process", "Logging") | Drop (<10 chars descriptive text) |

**Quality tier metadata** (tracked per example, NOT used in training weights by default):

| Tier | Description | Count |
|------|-------------|-------|
| T1 | Human LinkedTo with descriptive text (traditional) | ~2,200 |
| T1-AI | Human-curated AI framework links | 197 |
| T3 | AutomaticallyLinkedTo with descriptive text | ~1,676 |
| DROPPED | Bare-IDs, short, or empty | ~332 |

AI framework links (197) are tier T1-AI — curated during Phase 0R audit, all descriptive. Tiers are stored as metadata on each `TrainingPair` for diagnostic purposes. The primary training config treats all non-dropped tiers with uniform weight. If the CAPEC mapping quality check reveals that AutomaticallyLinkedTo links are systematically lower quality, tier-weighted sampling will be added as a training config change.

**CAPEC auto-link quality check (runs parallel with training, not blocking):** Compute BGE zero-shot cosine similarity between each auto-linked control's section_name and its target hub representation for all 1,799 CAPEC entries. Flag bottom 5% (~90 links) for manual review. If auto-link similarity is systematically lower than LinkedTo similarity, introduce tier-weighted sampling (T1 weight > T3 weight). Decision documented in training config metadata.

**Spot-check validation (runs parallel with training, not blocking):** Random sample of 100 post-filter traditional links, stratified across the 14 surviving frameworks (~7 per framework), manually verified for mapping correctness.

**Data hash chain:** SHA-256 hash computed at each pipeline stage: raw links → quality-filtered → training-ready. Hashes stored in output metadata for provenance tracking.

### 1.3 Training Example Format

Each training example is a (query, positive_key) pair for contrastive learning:

```python
@dataclass(frozen=True)
class TrainingPair:
    control_text: str          # section name (or full parsed text if available — see ablation A10)
    hub_id: str                # target CRE hub ID
    hub_representation: str    # hub name + hierarchy path (built by firewall)
    framework: str             # source framework name
    link_type: str             # "LinkedTo" | "AutomaticallyLinkedTo"
    quality_tier: str          # "T1" | "T3" (metadata only, uniform weight by default)
```

### 1.4 Multi-Label Handling

Some controls map to multiple hubs (median=1, max=38). Each (control, hub) pair is a separate training example. During evaluation, the model returns a ranked list of hubs; hit@k checks if the ground-truth hub appears in the top k.

---

## 2. Model Architecture

### 2.1 Base Model

**BGE-large-v1.5** (BAAI/bge-large-en-v1.5, 335M params)

Rationale:
- Highest zero-shot performance with paths (hit@1=0.416, tied with GTE)
- Higher hit@5 than GTE (0.706 vs 0.655) — better retrieval depth
- Proven contrastive fine-tuning architecture (sentence-transformers compatible)
- Fits on consumer GPUs (Orin 61GB, single H100)
- 7B models performed worse (E5-Mistral: 0.320, SFR: 0.142)

### 2.2 Bi-Encoder Architecture

```
Control text  →  [BGE Encoder]  →  control_embedding (1024d)
Hub text      →  [BGE Encoder]  →  hub_embedding (1024d)
                                      ↓
                              cosine_similarity(control, hub)
```

Shared encoder (weight-tied). Both control text and hub representations pass through the same encoder. At inference time, hub embeddings are pre-computed and cached.

### 2.3 Hub Representation (Input to Encoder)

Based on Phase 0R findings, the primary hub representation is:

```
"{hierarchy_path} | {hub_name}"
```

Example: `"Root > Application Security > API Security | API Security"`

**Primary config excludes descriptions.** Phase 0R showed descriptions degrade zero-shot embedding performance (-4.8pp BGE, -22.8pp GTE). However, the adversarial review identified that this was tested only on zero-shot embeddings — a fine-tuned model may learn to extract useful signal from descriptions. Ablation A6 tests `path+name+description` to determine if fine-tuning unlocks description utility.

---

## 3. Training Strategy

### 3.1 Loss Function: MNRL (MultipleNegativesRankingLoss)

**MNRL (InfoNCE variant) with in-batch negatives + hard negative mining.**

Each batch contains (control, positive_hub) pairs. The loss treats all other hubs in the batch as negatives, plus explicitly mined hard negatives from the CRE hierarchy:

```python
# For each (control, positive_hub) pair in a batch:
# - positive: the ground-truth hub
# - in-batch negatives: all other hubs in the batch
# - hard negatives: 1-3 sibling hubs (same parent in CRE tree)
```

Hard negative mining uses the CRE hierarchy: sibling hubs (children of the same parent) are semantically close but distinct — exactly the confusion the model must resolve.

**Why MNRL, not SupCon:** SupCon requires multi-positive batching (multiple controls mapped to the same hub as co-positives). Our data has single-positive per anchor in most cases. MNRL is the correct loss for single-positive + in-batch negatives. SupCon multi-positive batching is tested in ablation A8 for hubs with multiple controls.

### 3.2 Hard Negative Strategy

For each training example with hub H:
1. **Sibling negatives** (primary): other children of H's parent hub. These share the same category but differ in specialization.
2. **Cousin negatives** (secondary): children of H's parent's siblings. One level more distant.
3. **Random negatives**: handled by in-batch sampling.

```python
def mine_hard_negatives(hub_id: str, hierarchy: CREHierarchy, n: int = 3) -> list[str]:
    """Return up to n hard negative hub IDs for contrastive training."""
    siblings = hierarchy.get_siblings(hub_id)
    if len(siblings) >= n:
        return [s.hub_id for s in siblings[:n]]
    cousins = []
    parent = hierarchy.get_parent(hub_id)
    if parent:
        for uncle in hierarchy.get_siblings(parent.hub_id):
            cousins.extend([c.hub_id for c in hierarchy.get_children(uncle.hub_id)])
    all_negatives = [s.hub_id for s in siblings] + cousins
    return all_negatives[:n]
```

**Future improvement:** Phase 0R zero-shot predictions contain a confusion matrix — which hubs BGE actually confuses. Confusion-mined hard negatives (top-k most-confused non-sibling hubs per hub) would target actual error modes, not just structural proximity. This is standard practice (ANCE, ADORE) and could be added as an augmentation to hierarchy negatives after the primary config is validated.

### 3.3 Parameter-Efficient Fine-Tuning (LoRA)

**Full fine-tuning of 335M parameters on 132-187 AI training examples per fold (varies by held-out framework) is catastrophically overparameterized.** The primary training strategy uses LoRA (Low-Rank Adaptation) to reduce trainable parameters to ~2.4M.

**LoRA configuration:**

| Parameter | Value | Notes |
|-----------|-------|-------|
| LoRA rank | 16 | Applied to all 24 transformer layers |
| LoRA alpha | 32 | Scaling factor (alpha/rank = 2) |
| LoRA dropout | 0.1 | Regularization |
| Target modules | `query`, `key`, `value` | BERT-style naming (NOT `q_proj`/`k_proj`/`v_proj` — those are LLaMA naming) |
| Frozen layers | None | LoRA's low rank IS the regularization — no separate layer freezing |
| Trainable params | ~2.4M | 24 layers × 3 matrices × (1024×16 + 16×1024) = 2,359,296 (0.7% of 335M) |

**Rationale:** Standard LoRA practice applies adapters to all layers without freezing — the rank controls capacity. Layer freezing + LoRA is double regularization that confounds ablation A7 (rank sweep). With ~4,000 training examples per fold (joint config) and ~2.4M trainable params, the ratio is ~1.7 examples per parameter. This is a viable regime because the pretrained BGE-large already provides a strong initialization (hit@1=0.416 zero-shot) — LoRA learns a low-rank residual correction, not representations from scratch. Rank 16 is the initial starting point; ablation A7 (rank 4/8/16/32) determines optimal capacity. Full fine-tuning becomes ablation A9.

### 3.4 Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch size | 64 | Larger batch = more in-batch negatives |
| Hard negatives per example | 3 | Sibling hubs from hierarchy |
| Learning rate | 5e-4 | Higher LR for LoRA adapters (standard: 1e-4 to 1e-3) |
| LR scheduler | Warmup + cosine decay | 10% warmup steps |
| Epochs | 10-30 | Early stopping (see below) |
| Max sequence length | 512 | Sufficient for control text + hub paths |
| Gradient accumulation | None (primary) | MNRL negative pool = physical batch only (64). Accumulation does NOT increase negatives — use `CachedMultipleNegativesRankingLoss` if larger pool needed. |
| Weight decay | 0.01 | Standard AdamW |
| Max grad norm | 1.0 | Gradient clipping for training stability |
| Seed | 42 | Deterministic training |
| Mixed precision | fp16 | Speed + memory on H100/Orin |

**Early stopping validation set:** Within each LOFO fold, split the 4 non-held-out AI frameworks 90/10 into train/validation. Early stopping monitors validation loss on the 10% split. This gives ~13-19 validation items per fold — small but sufficient for loss monitoring. The held-out fold is NEVER used for early stopping (that would be leakage).

### 3.5 Training Data Configuration

**Primary config: Joint training with temperature-scaled AI sampling + LoRA**

All quality-filtered traditional links combined with AI-framework links. AI examples are overrepresented via temperature-scaled sampling (NOT naive upsampling — see rationale below).

**Temperature-scaled sampling (class-level formulation):**
- Class-level weight: `w_class = (n_class / n_total) ^ (1/T)` with T=2
- With T=2: P(AI class) = w_AI / (w_AI + w_trad) = (0.0324)^0.5 / ((0.0324)^0.5 + (0.9676)^0.5) = 0.180 / 1.164 = **15.5%** of batches (up from 3.3% natural)
- Sampling procedure: draw class first (AI vs traditional with temperature-weighted probability), then draw example uniformly within class
- **Epoch semantics:** Full pass over all data. Each unique example appears exactly once per epoch. Temperature affects batch composition ordering (which examples are grouped together), not per-epoch coverage.
- **Cross-epoch behavior:** Over 20 epochs, each AI example is seen 20 times vs. each traditional example ~20 times (same coverage). Temperature scaling affects batch composition, not repetition. The benefit is within-batch: AI examples are grouped into ~15.5% of batches rather than diluted across all batches.

**Why not naive upsampling:** Duplicating AI examples 10-20x creates P(duplicate anchor in batch of 64) = 0.82. When MNRL sees duplicate anchors in the same batch, it treats one copy's positive hub as a negative for the other — pushing identical texts apart. This is a well-known false-negative pathology in contrastive learning with in-batch negatives. Temperature-scaled sampling avoids this entirely by adjusting batch composition without creating duplicates.

**Hub-aware batch construction:** No two examples in a batch share the same target hub. This eliminates false negatives where the "correct" hub appears as a negative because another example in the batch maps to it. Requires a custom `HubAwareBatchSampler` (NOT `NoDuplicatesBatchSampler`, which checks ALL columns including hard negatives — far too restrictive).

**HubAwareBatchSampler algorithm:**
1. Shuffle all examples (with fixed seed per epoch)
2. Initialize empty batch, track set of hubs already in batch
3. Iterate through shuffled examples: if example's positive hub not in batch hub set, add to batch
4. If hub collision, skip and try next example; skipped examples go to next batch
5. When batch is full (64), yield batch, reset hub set
6. Class-level temperature weighting is applied to the shuffle ordering (AI examples sorted into early positions with probability proportional to class weight)

With ~458 hubs and batch_size=64, collision rate is ~13.7% — manageable. Diagnostic: log batch utilization (fraction of target batches successfully constructed without constraint violation). If utilization drops below 90%, decrease batch size.

| Config | Training Data | Role | Hypothesis |
|--------|--------------|------|------------|
| **joint-tempscaled** (PRIMARY) | Traditional + AI (T=2 temp sampling) | Primary config | Balanced joint training without false negatives |
| **ai-only** | 132-187 AI links per fold | Ablation (A1) | Minimal baseline — can LoRA learn from AI alone? |
| **joint-flat** | Traditional + AI (uniform sampling) | Ablation (A1) | Does temperature scaling matter vs. flat mixing? |
| **two-stage-transfer** | Stage 1: traditional → Stage 2: AI | Ablation (A1) | Catastrophic anchoring risk vs. staged learning |

Each evaluated via LOFO on the 5 AI frameworks.

---

## 4. Hub Representation Firewall (PRD 6.5)

### 4.1 Core Constraint

When evaluating framework X, hub representations MUST be rebuilt WITHOUT any contribution from X's linked sections. This prevents information leakage in LOFO evaluation.

### 4.2 Implementation

The firewall is a **build step**, not a runtime check:

```python
def build_firewalled_hub_text(
    hub_id: str,
    hierarchy: CREHierarchy,
    links: list[HubStandardLink],
    excluded_framework: str,
) -> str:
    """Build hub representation excluding one framework's contributions.
    
    Returns: "{hierarchy_path} | {hub_name}"
    
    The hierarchy path and hub name come from CRE structure (never from
    framework text), so they're firewall-safe by construction.
    """
    return f"{hierarchy.get_hierarchy_path(hub_id)} | {hierarchy.hubs[hub_id].name}"
```

Since the Phase 0R results showed that our hub representation uses ONLY hierarchy path + hub name (both from CRE structure, not from framework text), the firewall is inherently satisfied. No framework contributes to the hub text.

**However**, if future ablations add linked-standard-names to hub representations, the firewall becomes load-bearing: it must strip the held-out framework's sections before encoding.

### 4.3 Firewall Assertion

Every training/evaluation run includes a programmatic assertion:

```python
def assert_firewall(
    hub_texts: dict[str, str],
    eval_items: list[EvalItem],
    held_out_framework: str,
) -> None:
    """Assert no information leakage from held-out framework into hub representations."""
    for item in eval_items:
        assert item.framework == held_out_framework
        for hub_id, text in hub_texts.items():
            assert item.control_text not in text, \
                f"Firewall breach: control '{item.control_text[:50]}' found in hub {hub_id}"
```

---

## 5. LOFO Evaluation Harness (PRD 6.11)

### 5.1 Cross-Validation Protocol

Leave-one-framework-out on 5 AI frameworks:

| Fold | Held Out | Train On | Eval Items |
|------|----------|----------|------------|
| 1 | MITRE ATLAS | 4 remaining AI + optional traditional | 65 |
| 2 | NIST AI 100-2 | 4 remaining AI + optional traditional | 45 |
| 3 | OWASP AI Exchange | 4 remaining AI + optional traditional | 64 |
| 4 | OWASP Top10 for LLM | 4 remaining AI + optional traditional | 13 |
| 5 | OWASP Top10 for ML | 4 remaining AI + optional traditional | 10 |
| | | **Total eval items** | **197** |

For each fold:
1. Remove all links from the held-out framework from training data
2. Rebuild hub representations (firewall)
3. Train model from scratch (or from traditional-pretrained checkpoint for transfer ablation)
4. Predict ranked hub lists for all held-out framework controls
5. Score predictions against ground truth

### 5.2 Metrics

**Primary (hub assignment quality):**
- **hit@1**: Fraction where top prediction is correct
- **hit@5**: Fraction where correct hub appears in top 5
- **MRR**: Mean reciprocal rank
- **NDCG@10**: Normalized discounted cumulative gain at 10

**Split metrics (diagnostic):**
- **Covered hubs** (~73/458 with any AI training example): metrics on controls whose target hub appears in AI training data
- **Uncovered hubs** (~385/458 with no AI training example): metrics on controls whose target hub has only traditional training data (or none)

**Note on traditional link confound:** In joint-training configs, covered hubs receive training signal from both AI and traditional examples. Covered-hub performance is therefore inflated relative to the model's true generalization ability. The real test of the model is uncovered-hub performance — hubs where only traditional data (or none) exists. Report both splits.

**All metrics computed with:**
- **Fold-stratified micro-average bootstrap CIs** (10,000 resamples): for each bootstrap replicate, resample items with replacement *within each fixed LOFO fold* (preserving fold sizes at 65/45/64/13/10), concatenate the 5 resampled sets into a single pool of 197 items, compute aggregate hit@1 over the full pool. Percentile CI from the distribution of 10,000 aggregate values. This is micro-averaging (each item contributes equally; the 3 large folds dominate by design since they contain 88% of items). Fold-stratified resampling preserves the model-item assignment (each item is scored by the model that excluded its framework).
- **Paired bootstrap** for all comparisons: for each bootstrap replicate, compute per-item deltas (trained minus baseline) on the fold-stratified resample. The pairing is per-item: same item, same fold, different methods. This cancels item-level difficulty and dramatically reduces variance.
- Per-fold metrics for framework-level diagnostics and soft floors (see 5.3)
- **Benjamini-Hochberg FDR control** at q=0.10 for multi-model comparisons (~20 ablation configs, each compared to primary). BH assumes positive regression dependency (PRDS) among test statistics, which holds because all ablations share evaluation data and base architecture. If PRDS is violated for any specific comparison, report BY-adjusted p-values as sensitivity check.

### 5.3 Success Gate

**Aggregate gate with paired stratified bootstrap:**

**Note:** The PRD Section 6.2 defined a 0.50 gate for *skipping training entirely* (zero-shot sufficiency). That gate was already failed (best zero-shot = 0.416). The gate below is the *training success* criterion — a separate threshold.

**Gate:** Paired bootstrap delta (trained minus baseline) on all 197 items has 95% CI lower bound > 0 AND point estimate > 0.10 (i.e., hit@1 > 0.516). The paired test dramatically reduces variance because item-level difficulty cancels out. Note: 0.516 is the minimum success threshold. The practical utility target is 0.55 (conservative) and 0.60 (stretch).

**Per-framework soft floors:**
- MITRE ATLAS (n=65), OWASP AI Exchange (n=64): paired bootstrap delta CI lower bound must be ≥ **-0.05**. With SE ≈ 0.062 at n≥60, this detects framework-specific regressions exceeding ~7pp.
- NIST AI 100-2 (n=45): paired bootstrap delta CI lower bound must be ≥ **-0.10**. With SE ≈ 0.074 at n=45, this detects regressions exceeding ~5pp (catastrophe detector).
- OWASP Top10 for LLM (n=13), OWASP Top10 for ML (n=10): diagnostic only — too small for meaningful statistical tests.

**Per-framework metrics reported for all folds** (point estimate + CI), but only the 3 large folds have enforceable soft floors.

### 5.4 Baseline Comparison

Every trained model is compared against Phase 0R baselines:

| Baseline | hit@1 | hit@5 | MRR | NDCG@10 |
|----------|-------|-------|-----|---------|
| BGE-large + paths (best embedding) | 0.416 | 0.706 | 0.544 | 0.620 |
| Sonnet 3-shot + desc (LLM ceiling) | 0.624 | 0.863 | 0.730 | 0.771 |
| BGE-large plain (without paths) | 0.365 | 0.670 | 0.497 | 0.572 |

Paired stratified bootstrap test (10,000 resamples) on per-item deltas for statistical significance.

---

## 6. Training Pipeline Architecture

### 6.1 Pipeline Stages

```
[Curated Links] → [Quality Filter] → [CAPEC Quality Check] → [LOFO Split]
                                                                    ↓
[Firewall Hub Build] → [Temp-Scaled Sampling] → [Hub-Aware Batching]
                                                         ↓
[Hard Negative Mining] → [LoRA Training] → [Checkpoint]
                                                  ↓
[Evaluation] → [Paired Stratified Bootstrap CIs] → [WandB Log]
                                                  ↓
                              [LLM Prediction Comparison (post-training diagnostic)]
```

### 6.2 Training Loop (Per Fold)

```python
def train_fold(
    fold: LOFOFold,
    config: TrainingConfig,
    hierarchy: CREHierarchy,
) -> FoldResult:
    """Train and evaluate one LOFO fold.
    
    Note: PyTorch training is synchronous (GPU kernels launched from single thread).
    Fold parallelism is achieved via separate processes (each needs own CUDA context),
    not async tasks. The orchestrator uses multiprocessing, not asyncio.
    """
    # 1. Build training pairs (excluding held-out framework)
    train_pairs = build_training_pairs(
        links=fold.train_links,
        hierarchy=hierarchy,
        hub_texts=fold.hub_texts,
    )
    
    # 2. Mine hard negatives
    train_examples = add_hard_negatives(train_pairs, hierarchy, n_negatives=3)
    
    # 3. Initialize model from base (or pretrained checkpoint)
    model = load_base_model(config.base_model, config.checkpoint_path)
    
    # 4. Train with contrastive loss
    model = train_contrastive(
        model=model,
        examples=train_examples,
        config=config,
    )
    
    # 5. Encode all hub representations
    hub_embeddings = encode_hubs(model, fold.hub_texts)
    
    # 6. Predict for held-out controls
    predictions = predict_rankings(model, fold.eval_items, hub_embeddings)
    
    # 7. Score
    metrics = score_predictions(predictions, fold.ground_truth)
    
    return FoldResult(fold=fold, predictions=predictions, metrics=metrics)
```

### 6.3 Experiment Orchestration

Each experiment configuration is a WandB run:

```python
@dataclass
class TrainingConfig:
    name: str                    # e.g., "phase1b_joint_tempscaled_v1"
    base_model: str              # "BAAI/bge-large-en-v1.5"
    training_data: str           # "joint-tempscaled" | "ai-only" | "joint-flat" | "two-stage-transfer"
    checkpoint_path: Path | None # for transfer learning stage 2
    lora_rank: int               # 16 (primary), 0 = full fine-tuning
    lora_alpha: int              # 32
    sampling_temperature: float  # 2.0 (primary), inf = uniform
    control_text_source: str     # "section_name" | "full_parsed" (ablation A10)
    batch_size: int
    learning_rate: float
    max_grad_norm: float         # 1.0
    epochs: int
    hard_negatives: int
    seed: int
    hub_rep_format: str          # "path+name" | "path+name+desc" | "path+name+standards"
    data_hash: str               # SHA-256 of training data
```

Logged to WandB: data hash, git SHA, all hyperparameters, per-fold metrics, aggregate metrics with CIs, training loss curves, GPU utilization, LoRA rank, sampling temperature.

### 6.4 GPU Strategy

Training runs on H100 pods (RunPod) for fast iteration, with local Orin as fallback:

| Resource | Use Case | Est. Time per Fold |
|----------|----------|--------------------|
| H100 80GB | Full training runs, ablation sweep | ~5-10 min |
| Orin 61GB | Debugging, small runs | ~30-60 min |

LOFO = 5 folds × 1 primary config = 5 training runs minimum.
With full ablation suite: 5 folds × ~20 unique configs = ~100 runs. (Ablation table sums to 25 raw configs; deducting overlaps with primary ≈ 20 unique.)
Ablation code is implemented ON DEMAND after primary results — not all upfront. If primary hits 0.60, only A1 and A6 are worth running.
Parallelizable across pods: each fold is independent within a config. LoRA training is substantially faster than full fine-tuning (~3x speedup).
**Estimated budget:** ~100 runs × ~3 min/fold (LoRA on H100) × 5 folds = ~25 GPU-hours for full sweep ≈ $62 at RunPod H100 pricing.

---

## 7. Ablation Suite

### 7.1 Planned Ablations

| # | Ablation | Variable | Configs | Rationale |
|---|----------|----------|---------|-----------|
| A1 | Training data scope | joint-tempscaled vs. ai-only vs. joint-flat vs. two-stage-transfer | 4 | Does temp-scaled sampling help? Does traditional data help at all? |
| A2 | Hard negative count | 0 vs. 1 vs. 3 vs. 5 | 4 | Diminishing returns on hierarchy negatives |
| A3 | Hub rep + standards | path+name+standards (with firewall) | 1 | Firewall-complex — run last |
| A4 | Epochs | 5 vs. 10 vs. 20 vs. 30 | 4 | Overfitting detection |
| A5 | Learning rate | 1e-4 vs. 5e-4 vs. 1e-3 | 3 | LoRA adapters need higher LR than full fine-tuning |
| A6 | Hub rep + descriptions | path+name+desc vs. path+name | 1 | Does fine-tuning unlock description utility? |
| A7 | LoRA rank | 4 vs. 8 vs. 16 vs. 32 | 4 | Adapter capacity sweep |
| A8 | Loss function | MNRL vs. SupCon (multi-positive batching) | 2 | Multi-label exploitation |
| A9 | Full fine-tuning | LoRA vs. full | 1 | Prove LoRA isn't bottleneck |
| A10 | Control text richness | full parsed text (200-2000 chars) vs. section_name (41 chars) | 1 | Training on short titles vs. full descriptions |

### 7.2 Ablation Protocol

1. Run the **primary configuration** first (joint-tempscaled T=2, LoRA rank 16 all layers, 3 hard negatives, path+name hub rep, section_name control text, 20 epochs, 5e-4 LR)
2. If primary passes the aggregate success gate, run ablations one variable at a time
3. Each ablation compared via paired stratified bootstrap test against primary, with Benjamini-Hochberg FDR at q=0.10
4. Report all results including negative results — what didn't work is informative

### 7.3 Ablation Ordering

Architectural choices first, hyperparameter tuning second:

1. **A1** (training data) — most fundamental: does traditional data help?
2. **A6** (descriptions in hub reps) — architectural: untested in fine-tuned setting
3. **A10** (full-text controls) — query-side enrichment opportunity
4. **A2** (hard negatives) — core training signal
5. **A5** (learning rate) — LoRA-specific sensitivity
6. **A7** (LoRA rank) — adapter capacity
7. **A4** (epochs) — overfitting regime
8. **A8** (SupCon) — multi-positive batching for multi-label hubs
9. **A9** (full fine-tuning) — most expensive, confirms LoRA isn't limiting
10. **A3** (standards in hub rep) — most complex due to firewall interaction, run last

A3 firewall note: if standard names are included in hub reps, the firewall must strip the held-out framework's standard names from hub text. This makes A3 the most complex ablation — implement last.

---

## 8. Calibration (Post-Training)

### 8.1 Temperature Scaling

Raw cosine similarities are not probabilities. Calibration is performed INSIDE the LOFO loop: for each fold, the 90/10 AI training split (same split used for early stopping in Section 3.4) provides the calibration validation set (~13-19 items per fold).

```python
def calibrate_temperature(
    similarities: np.ndarray,    # (n_val_samples, n_hubs)
    ground_truth: list[str],     # true hub IDs
) -> float:
    """Find optimal temperature T that minimizes NLL on validation set.
    
    Validation set = 10% of non-held-out AI training data within each LOFO fold.
    Same split used for early stopping — no additional data consumed.
    """
    # Grid search over T in [0.01, 5.0]
    # P(hub_i | control) = exp(sim_i / T) / sum(exp(sim_j / T))
```

**Limitation:** Small calibration sets (13-19 items) may yield noisy temperature estimates. Report calibration temperature per fold and check stability. If unstable, use the median temperature across folds.

### 8.2 Per-Hub Thresholds (Deferred)

Per-hub thresholds for multi-label assignment require sufficient examples per hub. With ~458 hubs and ~4,000 training examples (many hubs have 1-5 examples), per-hub calibration is infeasible in Phase 1B. Deferred to Phase 2 when more labeled data is available.

For Phase 1B, multi-label assignment uses a single global threshold: the similarity value at max-F1 on the calibration validation set. This addresses PRD 6.4's requirement at the global level.

---

## 9. Checkpointing and Reproducibility

### 9.1 Checkpoint Contents

Every checkpoint includes:

```python
@dataclass
class TrainingCheckpoint:
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict
    epoch: int
    step: int
    best_val_loss: float
    config: TrainingConfig
    data_hash: str              # SHA-256 of training data
    git_sha: str
    metrics: dict               # eval metrics at checkpoint time
    random_state: dict          # torch + numpy + python random states
```

### 9.2 Deterministic Training

- Fixed seeds (torch, numpy, python random, CUDA)
- `torch.use_deterministic_algorithms(True)` where supported
- Sorted training data before shuffling (shuffle with fixed seed)
- Byte-identical results on same hardware with same seed

### 9.3 Data Hash Chain

Every pipeline stage computes and records SHA-256 hashes for provenance:

```
hub_links_curated.jsonl  →  hash_raw
        ↓ (quality filter)
hub_links_training.jsonl →  hash_filtered
        ↓ (LOFO split)
fold_{framework}_train.jsonl → hash_fold_train
fold_{framework}_eval.jsonl  → hash_fold_eval
```

Each checkpoint (9.1) records `hash_filtered` and `hash_fold_train`. If the training data hash changes, all dependent checkpoints are invalidated.

---

## 10. Output Artifacts

### 10.1 Per-Experiment

```
results/phase1b/
  {config_name}/
    config.json               # full TrainingConfig
    fold_{framework}/
      model/                  # best checkpoint
      predictions.json        # ranked hub lists per eval item
      metrics.json            # per-fold metrics
    aggregate_metrics.json    # LOFO aggregate with bootstrap CIs
    comparison_vs_baseline.json  # paired deltas vs Phase 0R
```

### 10.2 Final Model

```
models/phase1b/
  best_model/                 # model weights (safetensors format)
  hub_embeddings.npy          # pre-computed hub embeddings (458 × 1024)
  calibration.json            # temperature + per-hub thresholds
  metadata.json               # data hash, git SHA, training config, metrics
```

### 10.3 WandB Tracking

- Project: `tract-phase1b`
- One run per (config, fold) combination
- Summary run per config with aggregate metrics
- Comparison table across all configs and baselines

---

## 11. Module Layout

```
tract/
  training/
    __init__.py
    config.py          # TrainingConfig, hyperparameter constants, LoRA config
    data.py            # TrainingPair generation, hard negative mining, temp-scaled sampling, hub-aware batching
    data_quality.py    # Traditional link filtering, CAPEC quality check, quality tier metadata
    firewall.py        # Hub representation firewall
    loop.py            # LoRA training loop (MNRL loss via SentenceTransformerTrainer, checkpointing, gradient clipping)
    evaluate.py        # LOFO harness, metric computation, fold-stratified micro-average bootstrap CIs
    calibrate.py       # Temperature scaling, global threshold (per-hub deferred)
    orchestrate.py     # Multi-fold, multi-config experiment runner

scripts/
  phase1b/
    train.py           # CLI entrypoint: python -m scripts.phase1b.train
    ablation.py        # Ablation sweep runner
    llm_comparison.py     # Post-training diagnostic: compare predictions vs Phase 0R Sonnet
  analysis/
    audit_traditional_links.py  # Traditional link quality audit (already written)
```

Dependencies on Phase 1A:
- `tract.hierarchy.CREHierarchy` — hub tree, sibling queries, path generation
- `tract.io` — atomic writes, JSON handling
- `tract.config` — project-wide constants
- `tract.descriptions.HubDescriptionSet` — for description ablation (A6)
- `scripts.phase0.common` — LOFO fold builder, metric scoring (shared with Phase 0R)
- `data/processed/controls/*.json` — parsed framework controls (for A10 full-text ablation)

**sentence-transformers API notes:**
- Use `SentenceTransformerTrainer` + `SentenceTransformerTrainingArguments` (modern API), not legacy `model.fit()`.
- MNRL hard negatives passed via dataset columns: `anchor`, `positive`, `negative_1`, `negative_2`, `negative_3`.
- MNRL v5.3.0 has a `hardness_mode` parameter (`in_batch_negatives`, `hard_negatives`, `all_negatives`) for gradient weighting — an unspecified hyperparameter, default to `all_negatives`.
- LoRA via PEFT: `model.add_adapter(lora_config)` works transparently with `model.encode()` at inference (no merge needed for evaluation; merge with `merge_and_unload()` for production deployment).

**Estimated implementation timeline:** ~3-4 weeks for full pipeline (solo developer). Phase 1B-alpha (Section 0.2) is ~3 hours. Implement Phase 1B-alpha first, then build full pipeline incrementally gated on results.

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| 197 AI links too few for contrastive training | LoRA reduces trainable params to ~2.4M (vs. 335M). Joint training with temp-scaled AI sampling adds traditional signal without false negatives. |
| Overfitting on small folds (n=10, n=13) | LoRA + early stopping + paired stratified bootstrap for honest CIs. Soft floors for large folds, small folds diagnostic only. |
| Hub representation firewall breach | Programmatic assertion in every eval run; Phase 0R hub reps already firewall-safe |
| Training instability (contrastive loss collapse) | Gradient clipping (max_grad_norm=1.0), LR warmup, WandB loss curve monitoring |
| Traditional link data noise | Quality pipeline: 92.1% usable, 332 bare-ID/short entries dropped. Spot-check validation (parallel with training). |
| GPU cost overrun | Budget ~100 runs × ~3 min/fold (LoRA) × 5 folds = ~25 GPU-hours for full ablation sweep ≈ $62 RunPod. Ablations implemented on-demand, not all upfront. |
| DeBERTa-NLI failure pattern | Architecture decision already made (bi-encoder, not NLI). No classification heads. |
| Data hash chain breakage | SHA-256 hashes at every pipeline stage, recorded in checkpoints. Hash mismatch invalidates downstream. |

### 12.1 Escalation Path

If the primary configuration (LoRA + joint-tempscaled + MNRL) fails the 0.516 aggregate gate:

1. **Full fine-tuning** (ablation A9) — remove LoRA constraint, accept overfitting risk with aggressive early stopping
2. **LLM-as-labeler augmentation** (~$1,500) — use Sonnet to generate ~2,000 synthetic (control_text, hub_id) pairs from hub descriptions + hierarchy context
3. **Ship retrieve+rerank** — BGE retrieval (top-20) + Sonnet re-ranking (20 candidates). Phase 0R Sonnet hit 0.624 on all 458 hubs; ranking 20 candidates is strictly easier → likely >0.70 hit@1. Deployable immediately, ~$0.006/assignment. Break-even vs. fine-tuning at ~800 assignments. This is the strongest fallback and can be built in parallel as a Phase 1B-zero prototype.
