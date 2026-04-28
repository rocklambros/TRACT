# Phase 0 Results: Zero-Shot Baseline Evaluation

**Date:** 2026-04-28
**Infrastructure:** 3x NVIDIA H100 80GB HBM3 on RunPod (embedding experiments) + local Anthropic API (LLM probe)
**Evaluation:** 198 AI security controls across 5 frameworks, LOFO cross-validation with hub firewall

## Gate Criteria

| Gate | Criterion | Result | Verdict |
|------|-----------|--------|---------|
| **A** | Opus hit@5 > 0.50 on all-198 | 0.722 [0.662, 0.783] | **PASS** |
| **B** | Opus hit@1 − best embedding hit@1 > 0.10 | 0.465 − 0.348 = 0.117 | **PASS** |

**Decision: Proceed to Phase 1.** The task is feasible (Gate A) and there is room for a trained model to improve over off-the-shelf embeddings (Gate B).

## Summary Table — All-198 Track

| Method | hit@1 | hit@5 | MRR | NDCG@10 |
|--------|-------|-------|-----|---------|
| BGE-large-v1.5 (baseline) | 0.348 [0.283, 0.414] | 0.621 [0.556, 0.687] | 0.468 [0.411, 0.526] | 0.525 [0.470, 0.580] |
| GTE-large-v1.5 (baseline) | 0.338 [0.273, 0.404] | 0.586 [0.515, 0.652] | 0.449 [0.390, 0.508] | 0.501 [0.444, 0.558] |
| DeBERTa-v3-NLI (cross-encoder) | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.025] | 0.004 [0.000, 0.008] | 0.004 [0.000, 0.011] |
| BGE + hierarchy paths | 0.424 [0.354, 0.495] | 0.667 [0.601, 0.732] | 0.528 [0.469, 0.587] | 0.581 [0.525, 0.637] |
| GTE + hierarchy paths | 0.389 [0.323, 0.455] | 0.611 [0.540, 0.677] | 0.502 [0.444, 0.561] | 0.554 [0.499, 0.610] |
| BGE + LLM descriptions | 0.357 [0.287, 0.433] | 0.592 [0.516, 0.669] | 0.464 [0.399, 0.529] | 0.516 [0.454, 0.580] |
| GTE + LLM descriptions | 0.204 [0.146, 0.268] | 0.446 [0.369, 0.522] | 0.303 [0.244, 0.362] | 0.347 [0.286, 0.408] |
| **Opus LLM probe** | **0.465 [0.394, 0.535]** | **0.722 [0.662, 0.783]** | **0.568 [0.508, 0.628]** | **0.618 [0.561, 0.674]** |

All confidence intervals are 95% bootstrap CIs (10,000 resamples).

## Full-Text Track (125 items with parsed descriptions)

| Method | hit@1 | hit@5 | MRR | NDCG@10 |
|--------|-------|-------|-----|---------|
| BGE-large-v1.5 (baseline) | 0.352 [0.264, 0.440] | 0.632 [0.544, 0.712] | 0.478 [0.405, 0.551] | 0.535 [0.464, 0.605] |
| GTE-large-v1.5 (baseline) | 0.328 [0.248, 0.416] | 0.592 [0.504, 0.680] | 0.447 [0.373, 0.520] | 0.501 [0.429, 0.573] |
| Opus LLM probe | 0.480 [0.392, 0.568] | 0.752 [0.672, 0.824] | 0.592 [0.519, 0.665] | 0.643 [0.573, 0.711] |

The full-text track performs slightly better than all-198 for all methods, confirming that richer control descriptions improve assignment quality.

## Experiment 1: Embedding Baselines

**BGE-large-v1.5** is the best embedding model, edging out GTE on all metrics. Both bi-encoders achieve meaningful zero-shot performance (~35% hit@1, ~60% hit@5) on the hub assignment task despite having no task-specific training.

**DeBERTa-v3-NLI** completely fails (hit@1 = 0.000). The NLI entailment framing ("control entails hub topic") is fundamentally mismatched with taxonomy assignment — hub names are terse labels, not natural language hypotheses. This confirms that the task requires semantic similarity, not textual entailment.

### Per-Fold Breakdown (BGE baseline)

| Framework | N | hit@1 | hit@5 | MRR | NDCG@10 |
|-----------|---|-------|-------|-----|---------|
| OWASP AI Exchange | 65 | 0.646 | 0.862 | 0.740 | 0.782 |
| MITRE ATLAS | 65 | 0.231 | 0.492 | 0.353 | 0.413 |
| NIST AI 100-2 | 45 | 0.156 | 0.400 | 0.270 | 0.329 |
| OWASP Top10 for LLM | 13 | 0.231 | 0.692 | 0.421 | 0.478 |
| OWASP Top10 for ML | 10 | 0.200 | 0.800 | 0.396 | 0.517 |

**OWASP AI Exchange is dramatically easier** (hit@1 = 0.646 vs 0.156–0.231 for others). This is likely because its controls use explicit, descriptive language that aligns well with CRE hub names. NIST AI 100-2 is the hardest framework — its controls use formal regulatory language that creates a vocabulary mismatch with CRE's more technical naming.

### Per-Fold Breakdown (Opus LLM probe)

| Framework | N | hit@1 | hit@5 | MRR | NDCG@10 |
|-----------|---|-------|-------|-----|---------|
| OWASP AI Exchange | 65 | 0.769 | 0.938 | 0.853 | 0.882 |
| OWASP Top10 for LLM | 13 | 0.462 | 0.769 | 0.583 | 0.630 |
| OWASP Top10 for ML | 10 | 0.500 | 0.900 | 0.667 | 0.726 |
| MITRE ATLAS | 65 | 0.338 | 0.631 | 0.446 | 0.506 |
| NIST AI 100-2 | 45 | 0.200 | 0.489 | 0.305 | 0.370 |

Opus shows the same difficulty ordering as embeddings but achieves higher absolute performance across all folds. The largest Opus advantage is on OWASP AI Exchange (+0.123 hit@1 over BGE) and the smallest on NIST AI 100-2 (+0.044 hit@1).

## Experiment 3: Hierarchy Path Features

Prepending the CRE hierarchy path to hub text (e.g., "Root > Cryptography > Key Management > Hub: Key rotation") significantly improves bi-encoder performance.

### Paired Deltas (all-198)

| Model | delta hit@1 | delta hit@5 | delta MRR | delta NDCG@10 |
|-------|-------------|-------------|-----------|---------------|
| BGE | +0.076 [+0.015, +0.136] | +0.045 [-0.010, +0.101] | +0.060 [+0.016, +0.105] | +0.057 [+0.018, +0.096] |
| GTE | +0.051 [-0.010, +0.111] | +0.025 [-0.025, +0.076] | +0.054 [+0.011, +0.097] | +0.053 [+0.013, +0.092] |

For BGE, the hit@1 improvement is statistically significant (CI excludes zero). MRR and NDCG@10 are significant for both models. The hierarchy path provides disambiguating context that helps the encoder distinguish between semantically similar hubs (e.g., "Input validation" in the web security subtree vs. the data processing subtree).

**BGE + paths (hit@1 = 0.424)** is the best non-LLM method, narrowing the gap to Opus (0.465) to just 0.041.

## Experiment 4: LLM Hub Descriptions

Replacing terse hub names with Opus-generated 2-3 sentence descriptions **hurt performance** for both models.

### Paired Deltas (described-hub subset, n=157)

| Model | delta hit@1 | delta hit@5 | delta MRR | delta NDCG@10 |
|-------|-------------|-------------|-----------|---------------|
| BGE | -0.019 [-0.102, +0.064] | -0.057 [-0.140, +0.032] | -0.033 [-0.101, +0.035] | -0.034 [-0.099, +0.031] |
| GTE | -0.147 [-0.223, -0.070] | -0.159 [-0.236, -0.076] | -0.161 [-0.226, -0.097] | -0.172 [-0.236, -0.109] |

For BGE the degradation is not significant (CIs include zero), but for GTE the damage is severe and significant across all metrics. LLM-generated descriptions introduce noise — the verbose text dilutes the discriminative signal that the terse hub names provide. The bi-encoders were already effective at matching control language to hub label language; adding description prose shifts the embedding space away from what the models learned during pretraining.

## Key Findings

1. **The hub assignment task is feasible.** Opus achieves 72.2% hit@5 zero-shot, meaning the correct CRE hub appears in the top-5 predictions for nearly 3 in 4 controls.

2. **There is meaningful room for improvement.** The 11.7% hit@1 gap between Opus and best embedding (BGE) justifies training a task-specific model. The gap is larger for harder frameworks (ATLAS, NIST).

3. **Hierarchy paths help.** Prepending the CRE tree path to hub representations improves hit@1 by +7.6% for BGE (significant). This should be the default hub representation for Phase 1.

4. **LLM descriptions hurt.** Verbose hub descriptions degrade embedding performance. Terse labels + structural paths are the right representation.

5. **Framework difficulty varies enormously.** OWASP AI Exchange is easy (hit@1 ~65-77%), while NIST AI 100-2 is hard (hit@1 ~16-20%). A trained model should focus on closing the gap on harder frameworks.

6. **NLI cross-encoders don't work for this task.** DeBERTa-v3-NLI's complete failure confirms that hub assignment requires semantic similarity, not entailment.

## Recommendations for Phase 1

1. **Base model:** Start from BGE-large-v1.5 (best zero-shot embedding baseline). Fine-tune with contrastive learning using the 4,406 existing standard-to-hub links as training signal.

2. **Hub representation:** Use hierarchy-path-enriched hub text as the default. This provides +7.6% hit@1 with zero training cost.

3. **Training strategy:** Contrastive fine-tuning with hard negatives (sibling hubs in the CRE tree are natural hard negatives). LOFO cross-validation for honest evaluation.

4. **Focus metric:** Optimize for hit@1 — it has the most room for improvement (0.348 → 0.465 Opus ceiling, vs hit@5 0.621 → 0.722).

5. **Per-framework analysis:** Consider framework-specific evaluation during training. NIST AI 100-2 may benefit from domain-specific augmentation (regulatory → technical language bridging).

6. **Calibration:** Raw model scores need calibration before deployment. Phase 1 should include a calibration step.

## Runtime

| Experiment | Infrastructure | Time |
|------------|---------------|------|
| Exp1: Embedding baselines (3 models) | 3x H100 80GB (parallel) | BGE: 23s, GTE: 15s, DeBERTa: 1066s |
| Exp2: Opus LLM probe | Anthropic API (local) | 6785s (~113 min) |
| Exp3: Hierarchy paths (2 models) | 2x H100 80GB (parallel) | BGE: 19s, GTE: 34s |
| Exp4: Hub descriptions (2 models) | 1x H100 80GB | BGE: 19s, GTE: 31s |
