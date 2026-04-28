# Phase 0: Zero-Shot Baseline Gate — Design Spec

**Goal**: Establish baselines BEFORE committing to model training. Four experiments determine whether the CRE hub assignment task is feasible and whether a trained model can improve over zero-shot approaches.

**Gate criteria** (evaluated on all-198 track):
- (a) LLM probe (Opus) hit@5 > 0.50 → task is feasible
- (b) Best embedding hit@1 at least 0.10 below Opus hit@1 → room for trained model
- Both pass → proceed to Phase 1. Either fails → reassess architecture.

---

## Data Landscape

### Evaluation corpus: 198 AI-specific links

5 AI frameworks have existing CRE links in OpenCRE:

| Framework | CRE Links | Full-Text Matched | Title-Only Fallback |
|---|---|---|---|
| MITRE ATLAS | 65 | 65 | 0 |
| OWASP AI Exchange | 65 | 47 | 18 |
| OWASP LLM Top 10 | 13 | 13 | 0 |
| NIST AI 100-2 | 45 | 0 | 45 (no parser) |
| OWASP ML Top 10 | 10 | 0 | 10 (no parser) |
| **Total** | **198** | **125** | **73** |

"Full-text matched" means the hub link's `section_id` maps (after ID normalization) to a parsed control with a full description. "Title-only" means only the `section_name` from OpenCRE is available.

**ID normalization for OWASP AI Exchange**: OpenCRE uses Hugo URL slugs (`promptinjectioniohandling`), parser uses `UPPERCASE_WITH_UNDERSCORES` (`PROMPT_INJECTION_I/O_HANDLING`). Normalize both to lowercase-no-separators for matching.

### Two evaluation tracks (reported side-by-side)

- **full-text** (125 links): Only controls with parsed descriptions. 3 frameworks.
- **all-198** (198 links): All AI links, using `section_name` as fallback text for 73 unmatched. 5 frameworks.

Gate criteria evaluated on the all-198 track.

### CRE hierarchy

- 522 CRE hubs: 5 roots, 122 parent hubs, 400 leaf hubs
- Depth distribution: 5 at depth 0, 42 at depth 1, 112 at depth 2, 255 at depth 3, 108 at depth 4
- 517 Contains/Is Part Of relationships → full hierarchy paths (e.g., "Technical application security controls > Technical AI security controls > AI engineering controls > Weakening training set backdoors > Train data distortion")

### Hub pilot selection (experiment 4)

71 unique leaf hubs are linked to AI frameworks. Select the top 50 by AI link count for the description pilot. These are the hubs most relevant to the evaluation surface.

---

## LOFO Protocol (shared across all experiments)

Leave-one-framework-out cross-validation on the 5 AI frameworks:

1. Hold out framework X (e.g., MITRE ATLAS with 65 links)
2. Rebuild hub representations using all 4,406 - |X| remaining links
3. Hub text excludes X's contributions: linked standard names from remaining frameworks only
4. Predict hub assignments for X's held-out controls
5. Score predictions against ground truth
6. Repeat for all 5 AI frameworks
7. Aggregate metrics (micro-average across all predictions)

### Metrics

All metrics computed per-fold and aggregated:
- **hit@1**: Fraction where the correct hub is the top prediction
- **hit@5**: Fraction where the correct hub is in the top 5
- **MRR**: Mean reciprocal rank of the correct hub
- **NDCG@10**: Normalized discounted cumulative gain at 10

### Bootstrap confidence intervals

10,000 resamples per metric. Reports mean ± 95% CI. Bootstrap parallelized across CPU cores. For experiment 3 deltas, use paired bootstrap (resample the same indices for baseline and enriched).

---

## Experiment 1: Multi-Model Embedding Baseline

**Goal**: Measure how well off-the-shelf encoders assign controls to CRE hubs via similarity.

### Models

| Model | Type | Params | Dim |
|---|---|---|---|
| BAAI/bge-large-en-v1.5 | Bi-encoder | 335M | 1024 |
| Alibaba-NLP/gte-large-en-v1.5 | Bi-encoder | 434M | 1024 |
| cross-encoder/nli-deberta-v3-large | Cross-encoder (NLI) | 304M | — |

### Bi-encoder pipeline (BGE, GTE)

1. Build hub text: `"{hub_name}: {comma-separated linked standard names}"` (LOFO-firewalled)
2. Embed all hub texts → hub embedding matrix (522 × 1024)
3. Embed held-out control texts → control embeddings
4. Rank hubs by cosine similarity for each control
5. Score against ground truth

### Cross-encoder pipeline (DeBERTa NLI)

1. Build hub text: same as above
2. For each (control, hub) pair: score with NLI entailment probability
3. Rank hubs by entailment score for each control
4. Score against ground truth

Note: 198 × 522 = ~103K forward passes for the cross-encoder.

### Output

`results/phase0/exp1_embedding_baseline.json`:
- Per-model, per-framework, per-track metrics with 95% CIs
- Aggregated comparison table

---

## Experiment 2: LLM Probe (Opus)

**Goal**: Establish a feasibility ceiling. Can a general-purpose reasoner assign controls to CRE hubs?

### Model

Claude Opus 4 via Anthropic API.

### Prompt design

522 hubs won't fit in a single prompt at full detail. Two-stage approach:

**Stage 1 — Branch shortlisting**: For each control, make 5 API calls (one per root branch). Each call provides:
- The control text
- All hubs under that branch with: hub name, full hierarchy path, linked standard names (LOFO-firewalled)
- Ask: "Which of these hubs are relevant to this control? Return up to 20 candidates ranked by relevance."

**Stage 2 — Final ranking**: One API call with:
- The control text
- The ~100 shortlisted candidates from stage 1 (up to 20 per branch)
- Ask: "Rank the top 10 hubs for this control. For each, provide a brief justification."

### Execution

- 198 controls × 6 calls each = ~1,188 API calls
- Async batching with rate limit respect
- Runs from local machine (no GPU needed)
- Concurrent with experiment 3 GPU work

### Cost estimate

~3M input tokens + ~500K output tokens ≈ $50-60.

### Output

`results/phase0/exp2_llm_probe.json`:
- Per-framework, per-track metrics with 95% CIs
- Raw predictions with justifications (for debugging and qualitative analysis)

---

## Experiment 3: Hierarchy Path Features

**Goal**: Test whether CRE hierarchy paths improve embedding retrieval.

### Method

Re-run experiment 1's bi-encoder pipeline (BGE and GTE only) with enriched hub text:

- **Baseline** (from exp 1): `"{hub_name}: {linked standard names}"`
- **Path-enriched**: `"{full_hierarchy_path} | {hub_name}: {linked standard names}"`

Example: `"Technical application security controls > Technical AI security controls > AI engineering controls > Train data distortion | Train data distortion: CAPEC-271, CWE-506"`

Same LOFO protocol, same two tracks. Runs on the same RunPod GPUs after experiment 1 completes.

### Output

`results/phase0/exp3_hierarchy_paths.json`:
- Side-by-side: baseline vs path-enriched, per model, per framework, per track
- Delta in each metric with CI on the delta (paired bootstrap)

---

## Experiment 4: LLM Hub Description Pilot

**Goal**: Test whether LLM-generated descriptions improve embedding hit rates. Pilot 50 hubs before committing to all 400 in Phase 1.

### Hub selection

Top 50 leaf hubs by AI link count (from the 71 AI-linked leaf hubs).

### Description generation

Claude Opus generates 2-3 sentence descriptions. Input per hub:
- Hub name + full hierarchy path
- All linked standard section names (no LOFO firewall — descriptions are static features)
- Sibling hub names (for differentiation)

Output per hub: (a) what the hub covers, (b) what distinguishes it from siblings, (c) scope boundary. Matches PRD Section 6.2 spec — reusable in Phase 1.

Cost: 50 Opus calls ≈ $3-5.

### Evaluation

Re-run best bi-encoder(s) from experiment 1 with a third hub text template:
- **Description-enriched**: `"{hub_name}: {description}. Linked: {standard names}"`

Scored only on the subset of evaluation links mapping to the 50 described hubs. Reports delta vs baseline (exp 1) and vs path-enriched (exp 3).

### Expert review

Descriptions saved to `results/phase0/pilot_hub_descriptions.json` for review. PRD acceptance gate: > 80%.

### Output

`results/phase0/exp4_hub_descriptions.json`:
- Metrics on described-hub subset
- Descriptions for review

---

## RunPod Infrastructure

### Instances

3 GPU pods, largest available (A100 80GB or H100). One per embedding model.

### Deployment

`scripts/phase0/runpod_setup.sh` handles:
1. Provision pods via RunPod API (`pass runpod/api_key`)
2. Install Python dependencies (torch, transformers, sentence-transformers, numpy, scipy)
3. Sync experiment code to pods
4. After completion: pull results, terminate pods

### Execution phases

| Phase | GPU 1 | GPU 2 | GPU 3 | Local |
|---|---|---|---|---|
| A | BGE baseline | GTE baseline | DeBERTa baseline | — |
| B | BGE + paths | GTE + paths | (idle or teardown) | LLM probe (Claude API) |
| C | Best model + descriptions | (teardown) | (teardown) | — |

### Bootstrap parallelization

Each GPU instance runs bootstrap resampling across all available CPU cores using numpy vectorized operations. 10,000 resamples × 5 folds × 4 metrics — embarrassingly parallel.

### Estimated timeline

- Phase A: ~15-20 minutes
- Phase B: ~10 minutes GPU + ~40 minutes LLM probe (concurrent)
- Phase C: ~15 minutes
- Setup/teardown: ~10 minutes
- **Total: ~60-90 minutes wall-clock**

### Cost estimate

- RunPod: 3 GPU-hours × ~$3-4/hr ≈ $10-12
- Claude API (probe): ~$55
- Claude API (descriptions): ~$5
- **Total: ~$70**

---

## File Structure

```
scripts/phase0/
├── common.py                  # Shared: hierarchy builder, LOFO evaluator, corpus loader, bootstrap
├── exp1_embedding_baseline.py # Experiment 1: multi-model embedding comparison
├── exp2_llm_probe.py          # Experiment 2: Opus feasibility ceiling
├── exp3_hierarchy_paths.py    # Experiment 3: path-enriched hub features
├── exp4_hub_descriptions.py   # Experiment 4: description pilot
├── run_summary.py             # Aggregate results, print gate table
└── runpod_setup.sh            # RunPod provisioning + teardown

results/phase0/
├── exp1_embedding_baseline.json
├── exp2_llm_probe.json
├── exp3_hierarchy_paths.json
├── exp4_hub_descriptions.json
├── pilot_hub_descriptions.json
└── summary.json
```

---

## Summary Table (final output)

```
┌─────────────────────────┬─────────┬─────────┬─────────┬─────────┐
│ Method                  │ hit@1   │ hit@5   │ MRR     │ NDCG@10 │
├─────────────────────────┼─────────┼─────────┼─────────┼─────────┤
│ BGE-large (baseline)    │ X ± CI  │ X ± CI  │ X ± CI  │ X ± CI  │
│ GTE-large (baseline)    │ X ± CI  │ X ± CI  │ X ± CI  │ X ± CI  │
│ DeBERTa-v3 (cross-enc)  │ X ± CI  │ X ± CI  │ X ± CI  │ X ± CI  │
│ Best + hierarchy paths  │ X ± CI  │ X ± CI  │ X ± CI  │ X ± CI  │
│ Best + hub descriptions │ X ± CI  │ X ± CI  │ X ± CI  │ X ± CI  │
│ Opus LLM probe          │ X ± CI  │ X ± CI  │ X ± CI  │ X ± CI  │
├─────────────────────────┼─────────┼─────────┼─────────┼─────────┤
│ GATE (a): Opus hit@5    │         │ >0.50?  │         │         │
│ GATE (b): hit@1 gap     │ >0.10?  │         │         │         │
└─────────────────────────┴─────────┴─────────┴─────────┴─────────┘
```

Both full-text (125) and all-198 tracks reported. Gate evaluated on all-198.
