# Phase 3B: Experimental Narrative Notebook — Design Spec

> **For agentic workers:** This is a design spec, not an implementation plan. Use `superpowers:writing-plans` to create the implementation plan from this spec.

**Goal:** Create a publication-quality Jupyter notebook that tells the complete TRACT story — from zero-shot baselines through model training, human review, and a hands-on CLI tutorial — in the voice of a practitioner educating peers.

**Deliverable:** `notebooks/tract_experimental_narrative.ipynb` + `notebooks/nb_helpers.py`

**Target audience:** Security professionals, compliance officers, GRC practitioners, and AI security researchers who are smart but don't know ML. The notebook teaches them enough ML to understand and trust (or question) the crosswalk dataset.

---

## 1. Narrative Voice

**Tone:** Senior security practitioner explaining to peers at a conference. First person where natural. Concrete, honest, no hedging behind jargon. "We tried X, it failed because Y, so we did Z" — not "We employ X to optimize Y."

**Jargon rule:** Every ML concept gets a security-domain analogy BEFORE the technical term appears:

| ML Concept | Introduce As |
|-----------|-------------|
| Embedding | "Converting control text into GPS coordinates — controls about similar topics land near each other on the map" |
| Cosine similarity | "How close two points are on the map. 1.0 = identical, 0.0 = completely unrelated" |
| Contrastive fine-tuning | "We show the model thousands of pairs: 'these two controls are about the same thing, those aren't.' It adjusts its internal map until related controls cluster together" |
| LOFO cross-validation | "To test fairly, we hide an entire framework and ask: can the model still figure out where its controls belong? If yes, it genuinely understands security concepts — it didn't memorize the answer sheet" |
| Bi-encoder | "Two copies of the same model — one reads the control text, one reads the hub description. They each produce coordinates, and we check if they landed near each other" |
| LoRA | "Instead of rewriting the entire model (1.3 billion parameters), we add a small adapter (a few million parameters) that nudges it toward our task. Like teaching a translator a new dialect without rewriting their entire vocabulary" |
| Temperature scaling | "The model's raw scores aren't real probabilities — they're overconfident. Temperature scaling is like recalibrating a thermometer: we adjust the scale until a '70% confident' prediction is actually right ~70% of the time" |
| OOD (out-of-distribution) | "The model saying 'I've never seen anything like this control before.' It's the difference between an uncertain answer and no answer at all" |
| Hit@1, Hit@5, MRR | "Did the model's top guess match the expert? Top 5? And when it was wrong, how far down the list was the right answer?" |

**Section depth pattern (every section):**
1. Why this matters — a concrete problem a security professional would recognize
2. What we tried — the approach, explained plainly
3. What happened — results with honest interpretation
4. What we learned — the takeaway, including dead ends
5. `> **Plain English:**` blockquote — 2-3 sentences a non-technical manager could read

---

## 2. Architecture

### Files

| File | Purpose |
|------|---------|
| `notebooks/tract_experimental_narrative.ipynb` | The notebook (13 sections + 2 appendices, ~168 cells) |
| `notebooks/nb_helpers.py` | Shared utilities: palette, figure numbering, axis styling, Plotly static fallback |

### Data Loading

All data loaded from canonical project paths. No copies, no hardcoded values. One setup cell defines path constants:

```python
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # or manual
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"
PHASE0_DIR = RESULTS_DIR / "phase0"
PHASE1B_DIR = RESULTS_DIR / "phase1b"
PHASE1C_DIR = RESULTS_DIR / "phase1c"
REVIEW_DIR = RESULTS_DIR / "review"
BRIDGE_DIR = RESULTS_DIR / "bridge"
DATASET_DIR = PROJECT_ROOT / "build" / "dataset"
```

**Prerequisite:** Phase 3 review results must be copied from `.worktrees/phase3/results/review/` to `results/review/` on main before the notebook can run. The notebook's setup cell will check for required files and print a clear error message listing missing paths.

### Visualization Strategy

**matplotlib/seaborn (primary, ~30 figures):** Bar charts, heatmaps, forest plots, reliability diagrams, confusion matrices, loss curves. Publication-quality by default. Consistent styling from `nb_helpers.py`.

**Plotly interactive (5 figures):**
1. **Section 1:** CRE hierarchy sunburst — collapsible, shows hub counts per subtree
2. **Section 2:** Framework→hub Sankey — flow showing which frameworks map to which hub clusters
3. **Section 4 or 8:** Embedding space t-SNE scatter — hover shows control text + predicted hub + framework
4. **Section 9:** Error analysis scatter — hover shows misclassified controls with predicted vs actual hub
5. **Section 13:** Journey timeline — interactive walkthrough of approaches tried and their results

Each Plotly figure has a static PNG fallback saved inline so the notebook renders meaningfully as HTML/PDF export.

### Color Palette

Okabe-Ito for categorical data (8 colors, colorblind-safe):
```python
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#999999"]
```

Single-hue sequential ramps for counts (Blues for positive, Oranges for negative). Diverging palette centered at zero for deltas. No rainbow or jet.

### Figure Numbering

`Figure {section}.{n}` format (e.g., Figure 3.1, Figure 8.4). Managed by a counter class in `nb_helpers.py`:

```python
class FigureCounter:
    def __init__(self): self._counts = {}
    def next(self, section: int) -> str:
        self._counts[section] = self._counts.get(section, 0) + 1
        return f"Figure {section}.{self._counts[section]}"
```

### Reproducibility

- `random.seed(42)`, `np.random.seed(42)` in setup cell
- All JSON loads use `sort_keys=True` where order matters
- t-SNE uses `random_state=42`
- No network calls — all data loaded from local files
- Full notebook runs top-to-bottom in < 10 minutes (pre-computed embeddings from deployment_artifacts.npz, no model inference)

---

## 3. Section-by-Section Content Plan

### Section 1: Introduction & Motivation (10 cells, 2 figures)

**Opens with the real problem:** "You're a security architect. Your company is ISO 27001 certified. The board asks: 'Does that cover us for the EU AI Act?' You look at 300 ISO controls and 100 AI Act articles and realize you're staring at thousands of hours of manual comparison."

**Introduces CRE as the solution concept:** A shared coordinate system — like GPS for security. Instead of comparing every framework to every other framework (N² comparisons), you map each framework to the shared coordinate system (N comparisons). Then cross-framework questions become simple lookups.

**Introduces the assignment paradigm:** `g(control_text) → CRE_hub`. NOT pairwise. The distinction matters because it's the core architectural decision that makes the whole system work.

**Figures:**
- Figure 1.1: CRE hierarchy sunburst (Plotly interactive) — shows the 522-hub taxonomy, collapsible by depth
- Figure 1.2: The N² problem vs assignment paradigm — simple diagram showing why assignment scales and pairwise doesn't

**Plain English:** "We built a system that reads any security control and tells you which part of a universal security taxonomy it belongs to. This lets you instantly compare any two frameworks."

### Section 2: Data Landscape (12 cells, 3 figures)

**The training data:** 4,406 expert-created links from OpenCRE mapping controls from 22 frameworks to 522 hubs. Explain what OpenCRE is, what a "link" means, and why expert-curated links are gold.

**The long tail problem:** Most hubs have very few links. Some have dozens. This matters because the model will be better at popular hubs and worse at rare ones.

**Multi-hub mappings:** 35% of controls map to more than one hub. "A control about 'encrypted authentication' maps to both 'Cryptography' and 'Authentication.' This isn't noise — it's the CRE graph structure being honest about controls that span multiple concepts."

**Figures:**
- Figure 2.1: Framework→hub Sankey (Plotly interactive) — which frameworks map to which hub clusters, flow thickness = link count
- Figure 2.2: Hub link distribution (matplotlib bar/histogram) — the long tail, annotated with the 80/20 point
- Figure 2.3: Framework size comparison (matplotlib horizontal bar) — controls per framework, colored by coverage type (GT vs model prediction)

### Section 3: Phase 0 — Can a Pre-Trained Model Do This? (14 cells, 3 figures)

**The question:** "Before training anything, can an off-the-shelf embedding model solve this? We tested multiple models straight out of the box — BGE-large, GTE-large, DeBERTa-v3-NLI, E5-Mistral-7B, SFR-Embedding-2, plus a kNN baseline and few-shot Claude Sonnet."

**The DeBERTa disaster:** Hit@1 = 0.000. Zero. "DeBERTa-v3 is designed for natural language inference — 'does sentence A entail sentence B?' That's a different question from 'do these two texts describe the same security concept?' NLI models classify logical relationships; we need semantic similarity. Wrong tool for the job."

**BGE-large-v1.5 wins zero-shot:** Hit@1 = 0.348. "Not great, but it proves the concept is feasible — an embedding model CAN distinguish security concepts without any security-specific training."

**Hierarchy paths help:** +7.6% when you prepend the hub's position in the CRE tree to its description. "Telling the model that 'Multi-factor Authentication' lives under 'Authentication > Verification methods' gives it structural context that pure text similarity misses."

**The Opus ceiling:** Claude 3.5 Opus as a zero-shot classifier. Hit@1 = 0.465, Hit@5 = 0.722. "An LLM with security knowledge does better, but at $0.60 per control it's not scalable."

**Figures:**
- Figure 3.1: Model comparison bar chart — hit@1 for all 6 models + Opus (matplotlib)
- Figure 3.2: Per-framework radar chart — each model's hit@1 across 5 LOFO folds (matplotlib)
- Figure 3.3: Hierarchy path impact — with/without comparison (matplotlib paired bar)

### Section 4: Picking the Right Base Model (10 cells, 2 figures)

**Per-fold complementarity:** "BGE-large is best overall, but GTE-large does better on ATLAS. No single model dominates everywhere. The interesting question is where each model's strengths and weaknesses lie."

**The selection decision:** BGE-large-v1.5 chosen for fine-tuning because it has the highest aggregate score and the most consistent performance across folds. The per-fold variance matters — a model that's great on 4 folds and terrible on 1 is worse than one that's decent across all 5.

**Figures:**
- Figure 4.1: Per-framework performance matrix heatmap (matplotlib) — models × frameworks, colored by hit@1
- Figure 4.2: Embedding space t-SNE (Plotly interactive) — colored by framework, hover shows control text. Shows how controls cluster even without fine-tuning.

### Section 5: Teaching the Model (Contrastive Fine-Tuning) (12 cells, 3 figures)

**The training approach:** "We take BGE-large and teach it what 'same security concept' means by showing it pairs: controls that map to the same hub should have similar embeddings, controls mapping to different hubs shouldn't."

**Key technical choices, explained plainly:**
- MNRL (Multiple Negatives Ranking Loss): "For each correct pair, every other control in the batch becomes a negative example. Efficient — one batch gives you thousands of comparisons."
- LoRA rank 16: "We're not rewriting the whole model — just adding a small adapter layer."
- Text-aware batching: "We make sure each training batch contains controls from different frameworks and different hubs, so the model sees diverse examples in every step."

**Training dynamics:** Loss curves, what they tell us (convergence in ~15 epochs), signs of overfitting or healthy learning.

**Figures:**
- Figure 5.1: Training loss curves across folds (matplotlib line plot)
- Figure 5.2: Before/after embedding space (matplotlib side-by-side t-SNE) — same controls, showing how fine-tuning reorganizes the space
- Figure 5.3: Negative sampling distribution — what the model actually trains against

### Section 6: What Actually Mattered (Ablation Analysis) (10 cells, 2 figures)

**The ablation approach:** "We turned features off one at a time to see what matters. This is how you separate 'this actually helps' from 'we got lucky.'"

**Key findings:**
- Hierarchy paths: +7.6% hit@1. "Giving the model structural context about where a hub sits in the tree matters a lot."
- Hub descriptions: -2.1% hit@1 on zero-shot. "Surprisingly, adding human-written descriptions actually hurt the zero-shot model. The descriptions use different vocabulary than the controls, confusing the similarity calculation."
- The implications: "Structure > prose for this task."

**Figures:**
- Figure 6.1: Ablation forest plot — paired deltas with 95% CIs (matplotlib). Each ablation factor as a row, delta from baseline with confidence interval.
- Figure 6.2: Interaction heatmap — which combinations help/hurt (matplotlib)

### Section 7: The Hub Firewall — Honest Evaluation (10 cells, 2 figures)

**Why LOFO matters:** "If ATLAS controls are in the training data AND in the evaluation data (via shared hubs), the model has already seen the answer. LOFO removes the entire framework from training — including all of its hub representations — before evaluating."

**The firewall mechanism:** "When evaluating ATLAS, we rebuild hub representations using ONLY non-ATLAS frameworks. If 'Validate AI Model' (ATLAS) was in training, the hub it mapped to has seen ATLAS text. Without the firewall, the model gets free information."

**What happens without the firewall:** Show the inflated metrics vs honest metrics. "Without the firewall, ATLAS hit@1 jumps from 0.279 to ~0.45. That's not the model being smart — that's information leakage making it look smart."

**Figures:**
- Figure 7.1: With-firewall vs without-firewall per-framework comparison (matplotlib grouped bar)
- Figure 7.2: Leakage magnitude per hub — which hubs are most affected by the firewall (matplotlib bar)

### Section 8: Final Results — The Honest Picture (14 cells, 4 figures)

**The headline:** Hit@1 improved from 0.348 (zero-shot) to 0.531 (fine-tuned). "That's a +52% relative improvement. But averages lie — let's look per framework."

**Per-fold deep dive:**
- NIST AI: 0.107 → 0.429 (+0.322) — "The biggest single improvement. The model learned to distinguish AI risk management concepts."
- OWASP AI Exchange: 0.619 → 0.762 (+0.143) — "Already the easiest fold (security-adjacent language close to training data)."
- ATLAS: 0.273 → 0.279 (+0.006) — "Essentially flat. The model didn't learn ATLAS. This is the most important finding in the whole project, and we dedicate Section 9 to understanding why."
- LLM Top 10: 0.333 → 0.333 (+0.000) — "Zero improvement, but n=6. Too small to draw conclusions."
- ML Top 10: 0.429 → 0.714 (+0.285) — "Big improvement, but n=7."

**Comparison with Opus:** "The fine-tuned BGE model (hit@1=0.531) surpasses Opus zero-shot (0.465) at 1/1000th the cost per prediction."

**Bootstrap confidence intervals:** "We don't just report point estimates — we resample 10,000 times to show the uncertainty. With only 147 eval items across 5 folds, some estimates are wide."

**Figures:**
- Figure 8.1: Baseline vs fine-tuned per-framework grouped bar with bootstrap CIs (matplotlib)
- Figure 8.2: Delta waterfall — per-framework improvement from zero-shot to fine-tuned (matplotlib)
- Figure 8.3: Full metrics table — hit@1, hit@5, MRR, NDCG@10 per fold (matplotlib table or heatmap)
- Figure 8.4: Bootstrap CI comparison — fine-tuned vs zero-shot vs Opus, showing overlap (matplotlib forest plot)

### Section 9: Where the Model Gets It Wrong (12 cells, 3 figures)

**ATLAS deep dive:** "The model trades hits 1:1. When it learns one ATLAS control correctly, it forgets another. The net result: zero improvement. Why?"

**Hub disambiguation:** "The model finds the right neighborhood (parent hub) but picks the wrong leaf. 77% of ATLAS misses are unrelated-subtree errors — the model is confused at the top level of the hierarchy, not at the leaf level."

**Attractor hubs:** "Some hubs attract predictions they shouldn't. They're broad enough that many controls are 'close enough' to match."

**Per-control analysis:** Specific ATLAS examples showing what went wrong. "Control: 'Validate AI Model.' Predicted: 'Software Testing.' Actual: 'AI Model Validation.' The model sees 'validate' and 'model' and reaches for the nearest testing hub, missing the AI-specific one."

**Figures:**
- Figure 9.1: Error analysis scatter (Plotly interactive) — t-SNE of misclassified controls, hover shows control text + predicted vs actual hub
- Figure 9.2: Hub confusion matrix — top confused hub pairs (matplotlib heatmap)
- Figure 9.3: Similarity distribution — correct predictions vs errors (matplotlib histogram/KDE)

### Section 10: Confidence Calibration (10 cells, 3 figures)

**The problem:** "The model outputs cosine similarity scores between 0 and 1, but they aren't probabilities. A score of 0.7 doesn't mean '70% chance of being correct.' Without calibration, these numbers are meaningless to a reviewer."

**Temperature scaling:** "We fit a single parameter (T=0.074) that transforms raw scores into calibrated probabilities. After calibration, a '70% confident' prediction is actually right about 70% of the time."

**ECE (Expected Calibration Error):** ECE = 0.079, passing the < 0.10 gate. "If you bin predictions by confidence and check the actual accuracy in each bin, the average gap between predicted and actual is ~8 percentage points. Not perfect, but usable."

**OOD detection:** "Controls where the model's maximum similarity to any hub is below 0.568 are flagged as 'I don't know.' 96.7% of true OOD items (controls with no good hub) are caught by this threshold."

**Figures:**
- Figure 10.1: Reliability diagram — before/after calibration (matplotlib). Diagonal = perfect calibration.
- Figure 10.2: Confidence histogram — distribution of calibrated scores (matplotlib)
- Figure 10.3: OOD separation — in-distribution vs OOD score distributions with threshold line (matplotlib)

### Section 11: Human Review — What the Expert Found (12 cells, 3 figures)

**The review process:** "A cybersecurity domain expert reviewed all 878 model predictions individually. For each one: read the control's full text, evaluate whether the suggested CRE hub is semantically correct, and decide: accept, reassign to a better hub, or reject."

**The results:**
- 680 accepted (77.4%) — "The model got it right."
- 196 reassigned (22.3%) — "The model was in the right neighborhood but the expert found a more precise hub."
- 2 rejected (0.2%) — "The model was completely wrong. Only 2 out of 878."

**Per-framework breakdown tells the real story:**
- CSA AICM: 99% acceptance — "243 controls, the model nailed almost all of them. This framework's language closely matches the CRE vocabulary."
- EU AI Act: 100% acceptance — "100 controls, all correct."
- AIUC-1: 29% acceptance — "The model struggled badly here. AIUC-1 uses unique terminology that doesn't map cleanly to existing CRE hubs."
- CoSAI: 45% acceptance — "Similar story — newer AI governance concepts that the training data didn't cover well."

**Calibration quality:** "We hid 20 ground-truth items among the predictions (the reviewer didn't know which were tests). The expert agreed with the known-correct answer 65% of the time. The 35% disagreement was all reassignments — the expert picked a different-but-valid hub. This tells us the task itself has legitimate ambiguity."

**What this means for trust:** "For traditional security frameworks mapped through OpenCRE ground truth — high trust. For AI frameworks where the model did well (CSA AICM, ATLAS) — trust with verification. For frameworks where the model struggled (AIUC-1, CoSAI) — the expert corrections ARE the value."

**Figures:**
- Figure 11.1: Per-framework acceptance rate bar chart — colored by acceptance rate tier (matplotlib)
- Figure 11.2: Confusion between review decisions — what happened to reassigned items (matplotlib Sankey or alluvial)
- Figure 11.3: Calibration item agreement — agreement vs disagreement breakdown (matplotlib)

### Section 12: Using TRACT — CLI Tutorial (18 cells, 0 figures)

**Transition:** "Enough theory. Let's use the tool."

Uses `!tract ...` shell cells with real output. Three self-contained workflows:

**Workflow A: "I have a control, what hub does it map to?"**
- `tract assign "Implement multi-factor authentication for all privileged accounts"`
- `tract hierarchy --hub <hub_id>` (from the result)
- `tract compare --fw1 nist_800_53 --fw2 iso_27001`

**Workflow B: "I have a new framework to onboard"**
- `tract prepare --format csv --input examples/sample_framework.csv --output /tmp/demo.json`
- `tract validate /tmp/demo.json`
- `tract ingest /tmp/demo.json --dry-run`

**Workflow C: "I want to explore the published crosswalk"**
- `tract export --format jsonl --framework mitre_atlas`
- Loading from HuggingFace: `load_dataset("rockCO78/tract-crosswalk-dataset")`

Each command: markdown cell explaining what and when → shell cell with real output → markdown cell interpreting the output → Plain English blockquote.

**Commands documented but not run** (modify state): `bridge --commit`, `publish-hf`, `publish-dataset`, `import-ground-truth`, `review-import`. Shown with example output.

### Section 13: What We Built and What We Learned (16 cells, 3 figures)

**The journey recap (1 page):** Problem → 6 zero-shot models → DeBERTa failure → BGE-large selection → contrastive fine-tuning → LOFO evaluation → calibration → human review → published dataset. Each step in 1-2 sentences.

**Master results table:** Every approach tried, its hit@1, and the one-line lesson learned. Sortable by performance.

**"Should I trust this crosswalk?"** An honest, nuanced answer:
- Ground truth frameworks (NIST 800-53, ISO 27001, CWE, etc.): "Yes — these are expert-curated by the CRE project maintainers."
- AI frameworks with high acceptance (CSA AICM, ATLAS, EU AI Act): "Yes, with the understanding that 10-20% of assignments were expert-corrected to better hubs."
- AI frameworks with low acceptance (AIUC-1, CoSAI): "Use with caution — these rely heavily on expert corrections rather than model accuracy."
- For compliance decisions: "This is a starting point, not a certification. Use it to identify likely equivalences, then have your compliance team verify the ones that matter."

**Known limitations, concretely:**
1. Training distribution bias (traditional security overrepresented)
2. Single reviewer (no inter-rater reliability)
3. Hub taxonomy gaps for AI-specific concepts
4. Small evaluation folds (LLM Top 10: n=6)

**What would make this better:**
- More training data from AI-specific frameworks
- Multiple reviewers with agreement metrics
- New CRE hubs for AI concepts not well-covered
- Active learning with the reviewer's corrections feeding back into training

**Figures:**
- Figure 13.1: The full journey visualization (Plotly interactive timeline) — approaches tried, color-coded by success/failure, hover for details
- Figure 13.2: Master comparison table — all approaches × all metrics (matplotlib)
- Figure 13.3: Trust level guide — framework × trust tier heatmap (matplotlib)

### Appendix A: Experiment Log (4 cells, 1 figure)

Full table of every experiment run with hyperparameters, metrics, git SHA, and WandB link.

### Appendix B: Visual Style Guide (4 cells, 1 figure)

Palette definitions with swatches, accessibility notes, font choices. Citations for Okabe-Ito palette.

---

## 4. nb_helpers.py Module

Shared utilities to keep notebook code cells short:

```python
# Palette
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#999999"]
SEQUENTIAL_BLUE = "Blues"
SEQUENTIAL_ORANGE = "Oranges"
DIVERGING = "RdBu_r"

# Figure counter
class FigureCounter: ...

# Consistent axis styling
def style_axes(ax, title, xlabel, ylabel, fig_num): ...

# Plotly static fallback
def plotly_with_fallback(fig, fig_num, title, width=900, height=600): ...

# Common data loaders (thin wrappers, no logic)
def load_phase0_summary() -> dict: ...
def load_fold_predictions(run_name: str, fold: str) -> list[dict]: ...
def load_corrected_metrics(run_name: str) -> dict: ...
def load_calibration_data() -> dict: ...
def load_deployment_embeddings() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]: ...
def load_review_metrics() -> dict: ...
```

---

## 5. Dependencies

**Already installed:** plotly 5.24.1, matplotlib 3.10.8, seaborn 0.13.2, ipywidgets 7.8.1, nbformat 5.10.4, wandb 0.25.1, sklearn (for t-SNE)

**Needs install:** `numba` (required by `umap-learn`). Since umap-learn fails without numba, use **sklearn t-SNE** instead. t-SNE is sufficient for 2D projections. If 3D projections are wanted later, install numba+umap-learn.

**No new dependencies required.**

---

## 6. Prerequisites Before Running

1. Phase 3 review files must exist at `results/review/` (copy from `.worktrees/phase3/results/review/` if needed: `review_metrics.json`, `review_export.json`, `hub_reference.json`, `reviewer_guide.md`)
2. Phase 3 dataset staging must exist at `build/dataset/` (copy from `.worktrees/phase3/build/dataset/` if needed: `crosswalk_v1.0.jsonl`, `framework_metadata.json`, `review_metrics.json`)
3. Deployment model at `results/phase1c/deployment_model/`
4. All Phase 0/1B results in `results/phase0/`, `results/phase1b/`

The notebook's setup cell checks all prerequisite paths and prints a clear checklist of what's missing.

---

## 7. Cell Budget

| Section | Cells | Figures | Interactive |
|---------|-------|---------|-------------|
| 1. Introduction & Motivation | 10 | 2 | 1 (sunburst) |
| 2. Data Landscape | 12 | 3 | 1 (Sankey) |
| 3. Phase 0 Baselines | 14 | 3 | 0 |
| 4. Base Model Selection | 10 | 2 | 1 (t-SNE) |
| 5. Contrastive Fine-Tuning | 12 | 3 | 0 |
| 6. Ablation Analysis | 10 | 2 | 0 |
| 7. Hub Firewall | 10 | 2 | 0 |
| 8. Final Results | 14 | 4 | 0 |
| 9. Error Analysis | 12 | 3 | 1 (scatter) |
| 10. Calibration | 10 | 3 | 0 |
| 11. Human Review | 12 | 3 | 0 |
| 12. CLI Tutorial | 18 | 0 | 0 |
| 13. What We Built | 16 | 3 | 1 (timeline) |
| App A: Experiment Log | 4 | 1 | 0 |
| App B: Style Guide | 4 | 1 | 0 |
| **Total** | **~168** | **~35** | **5** |

Markdown-to-code ratio target: ≥ 1.5:1 (~100+ markdown cells, ~65 code cells).

---

## 8. Success Criteria

- ≥ 128 cells (target: ~168)
- ≥ 24 figures (target: ~35)
- ≥ 1.5:1 markdown-to-code ratio
- All cells run top-to-bottom with identical output (seeded, deterministic)
- Full notebook runs in < 10 minutes
- 5 Plotly interactive figures with static PNG fallbacks
- Okabe-Ito palette throughout, no rainbow/jet
- Every ML concept introduced with a plain-language analogy before the technical term
- Plain English blockquote after every section
- CLI tutorial: 3 workflows with real output
- Section 13 gives a nuanced, honest answer to "should I trust this?"
- Practitioner voice throughout — no PhD-speak
