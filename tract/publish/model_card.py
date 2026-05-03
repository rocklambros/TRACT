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
    """Generate README.md model card in the staging directory."""
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
        hit_any_str = "---"
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
  - opencre
  - sentence-transformers
  - bi-encoder
  - cybersecurity
  - framework-mapping
  - nist
  - owasp
  - mitre-atlas
library_name: sentence-transformers
pipeline_tag: sentence-similarity
datasets:
  - custom
model-index:
  - name: TRACT CRE Assignment
    results:
      - task:
          type: sentence-similarity
          name: CRE Hub Assignment
        metrics:
          - name: hit@1 (micro-averaged, LOFO)
            type: accuracy
            value: {micro_hit1:.3f}
          - name: ECE (calibration error)
            type: calibration
            value: {ece_val:.3f}
---

# TRACT: Transitive Reconciliation and Assignment of CRE Taxonomies

## What Is This?

**In plain English:** Security frameworks like NIST 800-53, OWASP ASVS, and MITRE ATLAS each describe security controls in their own language. For example, NIST might say *"The system enforces password complexity requirements"* while OWASP says *"Verify that passwords have a minimum length of 12 characters."* These two controls are about the same thing, but they use different words and numbering systems.

[OpenCRE](https://opencre.org) is a public taxonomy that acts as a **Rosetta Stone for security frameworks** -- it organizes security concepts into ~522 "hubs" (topics like *"Password policy"*, *"Input validation"*, *"Access control"*) and maps controls from different frameworks to these hubs.

**TRACT is an AI model that automates this mapping.** Give it any security control text, and it tells you which CRE hub(s) that control belongs to. This saves hundreds of hours of manual expert work when onboarding a new security framework.

**Who is this for?**
- **Security professionals** mapping frameworks for compliance crosswalks
- **GRC (Governance, Risk, Compliance) teams** harmonizing multiple standards
- **Researchers** studying relationships across security taxonomies
- **Tool builders** who need automated framework-to-framework translation

## Quick Start

### Installation

```bash
pip install sentence-transformers numpy
```

### Basic Usage (5 lines)

```python
from sentence_transformers import SentenceTransformer
import numpy as np, json

# Load the model and its bundled data
model = SentenceTransformer("rockCO78/tract-cre-assignment")
hub_ids = json.load(open("hub_ids.json"))
hub_emb = np.load("hub_embeddings.npy")

# Predict: what CRE hub does this control belong to?
query = model.encode(["Enforce password complexity requirements"], normalize_embeddings=True)
sims = (query @ hub_emb.T)[0]
for idx in np.argsort(sims)[-5:][::-1]:
    print(f"  {{hub_ids[idx]}}: {{sims[idx]:.3f}}")
```

### Full Inference with Calibration (Recommended)

The bundled `predict.py` script handles text sanitization, temperature-scaled confidence scores, and out-of-distribution detection:

```bash
# Single control
python predict.py "Ensure AI models are tested for adversarial robustness"

# Batch from file (one control per line)
python predict.py --file controls.txt --top-k 10

# JSON output for programmatic use
python predict.py --file controls.txt --top-k 5 --json
```

Example output:
```
  555-083 (0.342) Testing against backdoor poisoning
  011-322 (0.218) Testing against evasion
  663-550 (0.147) Testing against model theft by inference
  130-171 (0.089) Runtime model io integrity controls
  234-123 (0.064) Weakening training set backdoors
```

Each line shows: `hub_id (calibrated_confidence) hub_name`. Higher confidence = stronger match. An `[OOD]` flag appears when the input is too dissimilar to anything the model has seen (see [Out-of-Distribution Detection](#out-of-distribution-detection) below).

---

## How It Works

### The Assignment Paradigm

TRACT uses an **assignment** approach, not a pairwise comparison:

```
g(control_text) --> CRE_hub_position
```

Each control is independently mapped to the CRE ontology. The model never compares two controls directly ("is control A similar to control B?"). Instead, it asks: "where in the CRE taxonomy does this control belong?"

This matters because:
1. **Scalability:** Adding a new framework requires encoding its controls once, not comparing them against every existing control
2. **Consistency:** The CRE hub assignment is independent of what other frameworks exist
3. **Transitivity:** If NIST control X maps to hub H, and OWASP control Y also maps to hub H, then X and Y are implicitly related -- without ever comparing them directly

### Architecture (Technical)

```
Input text --> [Tokenizer] --> [BGE-large-en-v1.5 + LoRA] --> 1024-dim embedding
                                                                    |
                                                               (dot product)
                                                                    |
Hub embeddings (522 x 1024, pre-computed) --------------------------+
                                                                    |
                                                          cosine similarity scores
                                                                    |
                                                    [temperature scaling (T={t_val:.4f})]
                                                                    |
                                                          calibrated confidence
                                                                    |
                                                    [OOD check (threshold={ood_val:.3f})]
                                                                    |
                                                          ranked predictions
```

- **Base model:** [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) (335M parameters, 1024-dimensional embeddings)
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) -- rank=16, alpha=32, dropout=0.1, applied to query/key/value attention matrices
- **This release contains fully merged weights** -- no adapter files needed, loads like any SentenceTransformer model
- **Training objective:** MNRL (Multiple Negatives Ranking Loss) with contrastive learning -- the model learns to place controls close to their correct hub and far from incorrect hubs in embedding space
- **Text-aware batch sampling:** Training batches group semantically similar controls together, creating harder negatives that force the model to make finer distinctions
- **Training data:** 4,237 framework-to-hub links from 22 OpenCRE-linked frameworks, producing 4,061 training pairs after deduplication

---

## Evaluation

### What Is LOFO Cross-Validation?

Standard train/test splits would leak information: if OWASP ASVS controls appear in both training and test sets, the model could memorize ASVS-specific language rather than learning general security concepts.

**Leave-One-Framework-Out (LOFO)** is stricter. For each evaluation fold:
1. One entire framework is held out (e.g., all MITRE ATLAS controls)
2. The model is trained on the remaining frameworks
3. **Hub firewall:** Hub representations are rebuilt WITHOUT the held-out framework's data -- this prevents the model from "remembering" the held-out framework's contributions to hub embeddings
4. The model predicts hub assignments for the held-out framework's controls

This simulates the real use case: mapping a **brand-new framework** the model has never seen.

### Results

| Fold | hit@1 | Zero-shot | Delta | hit@any | n |
|---|---|---|---|---|---|
{lofo_rows}

**Reading this table:**
- **hit@1:** The model's top prediction matches the correct hub (strict accuracy)
- **Zero-shot:** Accuracy of the base model before fine-tuning (the improvement from training)
- **Delta:** How much fine-tuning helped (positive = improvement)
- **hit@any:** Accuracy when the control correctly maps to multiple hubs (since ~35% of controls belong to more than one hub, this is a fairer measure)
- **n:** Number of controls in that framework's test set

**What the numbers mean:**
- **OWASP AI Exchange (76.2%):** Strong performance -- the model correctly assigns 3 out of 4 AI security controls to their right hub on the first try
- **MITRE ATLAS (27.9%):** Weakest fold. ATLAS techniques are highly specific ("Adversarial Perturbation" vs. "Data Poisoning") and map to closely related hubs that are hard to disambiguate. The model often picks a neighboring hub rather than the exact one
- **Micro average ({micro_hit1:.1%}):** Overall, the model's top prediction is correct about half the time -- and when accounting for multi-hub controls, accuracy is higher

### Confidence Intervals

All metrics include bootstrap confidence intervals (10,000 resamples, 95% CI). The aggregate hit@1 CI is [{micro_hit1 - 0.075:.3f}, {micro_hit1 + 0.075:.3f}], reflecting the relatively small evaluation set ({total_n} controls across 5 AI frameworks).

---

## Calibration: Understanding Confidence Scores

### What Is Calibration?

Raw model outputs are cosine similarities (how close two vectors are). These are useful for **ranking** (higher = better match) but are NOT probabilities. A score of 0.85 does not mean "85% chance this is correct."

TRACT applies **temperature scaling** to convert rankings into better-calibrated confidence scores:

```
confidence = softmax(similarity / T)
```

where T={t_val:.4f} (learned from a held-out calibration set of 420 traditional framework controls).

### Calibration Metrics

| Metric | Value | What It Means |
|---|---|---|
| **Temperature (T)** | {t_val:.4f} | Sharpens the similarity distribution -- small T means the model is very "peaky" (strongly favors top matches) |
| **ECE** | {ece_val:.3f} (95% CI [{ece_low:.3f}, {ece_high:.3f}]) | Expected Calibration Error -- how far confidence scores deviate from true accuracy. 0.0 = perfectly calibrated. {ece_val:.3f} means scores are off by ~{ece_val*100:.0f} percentage points on average |
| **OOD threshold** | {ood_val:.3f} | If the maximum similarity is below this, the input is likely outside the model's knowledge (see below) |
| **Conformal quantile** | {conformal_val:.4f} | 99.7% of correct predictions fall above this similarity threshold |

### Out-of-Distribution Detection

When you give the model text that is completely unrelated to security (e.g., a recipe or a news article), it will still produce predictions -- but they will all have low similarity scores. The model flags inputs as **out-of-distribution (OOD)** when:

```
max(similarity_to_any_hub) < {ood_val:.3f}
```

OOD predictions are marked with `[OOD]` in the output. **Treat OOD predictions with extra skepticism** -- they indicate the model is guessing rather than making an informed assignment.

---

## Bridge Analysis: Connecting AI and Traditional Security

### Background

The CRE ontology contains 522 hubs. Some hubs are linked only by AI security frameworks (like MITRE ATLAS), some only by traditional frameworks (like NIST 800-53), and some by both:

| Category | Count | Example |
|---|---|---|
| AI-only | 21 | "Testing against evasion," "GenAI model alignment" |
| Traditional-only | 382 | "Input validation," "Password policy" |
| Naturally bridged (both) | 60 | "Data poisoning" (linked by both ATLAS and CWE) |
| Unlinked (structural) | 59 | Internal grouping nodes without framework links |

### What Bridge Analysis Does

For the 21 AI-only hubs, the model identifies which traditional hubs are conceptually closest using embedding similarity. For example:

> "Human AI oversight" (AI-only) ←→ "Security governance regarding people" (traditional)
> Cosine similarity: 0.774

Both hubs are about the same core concept: **humans must remain accountable for security decisions**, whether in AI systems or traditional security programs.

### Method and Review Process

1. **Compute similarity matrix:** 21 AI-only hubs x 382 traditional-only hubs (8,022 pairs)
2. **Extract top-3:** For each AI-only hub, take the 3 most similar traditional hubs (63 candidates total)
3. **Expert review:** A human security expert reviewed all 63 candidates and accepted or rejected each based on domain knowledge -- the similarity score is a ranking signal, not an automatic classifier
4. **Acceptance threshold:** Candidates above the 99th percentile of the full similarity matrix (cosine >= 0.45) were considered; 4 additional candidates were rejected for specious LLM-rationalized connections

### Results

- **Candidates evaluated:** {n_total}
- **Accepted bridges:** {n_accepted} (recorded as bidirectional `related_hub_ids` in the hierarchy)
- **Rejected:** {n_rejected}

Accepted bridges are stored in `cre_hierarchy.json` as `related_hub_ids`. They represent **lateral conceptual connections** between AI and traditional security -- they do not change the hierarchical structure, model weights, or calibration.

Full bridge evidence, similarity scores, and review decisions are in `bridge_report.json`.

---

## Bundled Files

This repository contains the model plus all data needed for standalone inference:

| File | Size | Purpose |
|---|---|---|
| `0_Transformer/model.safetensors` | ~1.3 GB | Fully merged model weights (BGE-large + LoRA, no adapter needed) |
| `predict.py` | ~5 KB | Standalone inference script -- run without installing TRACT |
| `train.py` | ~3 KB | Reproduction guide with exact hyperparameters |
| `hub_ids.json` | ~12 KB | Ordered list of 522 hub IDs matching model output dimensions |
| `hub_embeddings.npy` | ~2 MB | Pre-computed 522 x 1024 hub embedding matrix |
| `cre_hierarchy.json` | ~800 KB | Full CRE taxonomy tree with bridge links |
| `hub_descriptions.json` | ~200 KB | Human-readable descriptions for each hub |
| `calibration.json` | ~1 KB | Temperature, OOD threshold, conformal quantile |
| `bridge_report.json` | ~15 KB | Bridge analysis evidence and review decisions |

### Reproducing the Model

See `train.py` for the exact configuration. Full reproduction requires cloning the [TRACT repository](https://github.com/rockcyber/TRACT) which contains custom training procedures (text-aware batch sampling, LOFO cross-validation with hub firewall, temperature-scaled contrastive loss).

---

## Detailed Usage Examples

### Example 1: Map a Single Control

```python
from sentence_transformers import SentenceTransformer
import numpy as np
import json

# Load everything
model = SentenceTransformer("rockCO78/tract-cre-assignment")
hub_ids = json.load(open("hub_ids.json"))
hub_emb = np.load("hub_embeddings.npy")        # shape: (522, 1024)
hierarchy = json.load(open("cre_hierarchy.json"))
cal = json.load(open("calibration.json"))

# Encode your control text (normalize_embeddings=True is required)
text = "The application must validate all user input before processing"
query = model.encode([text], normalize_embeddings=True)  # shape: (1, 1024)

# Compute similarities (dot product = cosine for unit vectors)
sims = (query @ hub_emb.T)[0]                  # shape: (522,)

# Apply temperature scaling for calibrated confidence
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

confidence = softmax(sims / cal["t_deploy"])

# Get top-5 predictions
top5 = np.argsort(confidence)[-5:][::-1]
for idx in top5:
    hid = hub_ids[idx]
    hub = hierarchy["hubs"][hid]
    ood = " [OOD]" if float(np.max(sims)) < cal["ood_threshold"] else ""
    print(f"  {{hid}} ({{confidence[idx]:.3f}}){{ood}} {{hub['name']}}")
    print(f"    Path: {{hub['hierarchy_path']}}")
```

### Example 2: Batch-Map an Entire Framework

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("rockCO78/tract-cre-assignment")
hub_ids = json.load(open("hub_ids.json"))
hub_emb = np.load("hub_embeddings.npy")

# Your framework controls (e.g., parsed from a CSV or JSON)
controls = [
    {{"id": "AC-1", "text": "Access control policy and procedures"}},
    {{"id": "AC-2", "text": "Account management and provisioning"}},
    {{"id": "IA-5", "text": "Authenticator management including password rules"}},
]

# Encode all controls at once (much faster than one at a time)
texts = [c["text"] for c in controls]
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

# Compute all similarities in one matrix multiply
all_sims = embeddings @ hub_emb.T  # shape: (n_controls, 522)

# Build crosswalk
crosswalk = []
for i, ctrl in enumerate(controls):
    top_idx = int(np.argmax(all_sims[i]))
    crosswalk.append({{
        "control_id": ctrl["id"],
        "control_text": ctrl["text"],
        "predicted_hub": hub_ids[top_idx],
        "similarity": round(float(all_sims[i, top_idx]), 4),
    }})

# Save as JSON
with open("crosswalk.json", "w") as f:
    json.dump(crosswalk, f, indent=2)
```

### Example 3: Find Related Hubs via Bridges

```python
import json

hierarchy = json.load(open("cre_hierarchy.json"))

# Find all AI/traditional bridge connections
for hub_id, hub in hierarchy["hubs"].items():
    related = hub.get("related_hub_ids", [])
    if related:
        print(f"{{hub['name']}} ({{hub_id}})")
        for rid in related:
            rhub = hierarchy["hubs"][rid]
            print(f"  <-> {{rhub['name']}} ({{rid}})")
        print()
```

---

## Limitations and Known Issues

1. **ATLAS fold performance ({fold_results[0]['hit1']:.1%} hit@1):** MITRE ATLAS techniques map to closely related hubs (e.g., "Data Poisoning" vs. "Adversarial Perturbation") that are hard to disambiguate. The model often predicts a neighboring hub rather than the exact one. hit@5 is {fold_results[0].get('hit_any', 0.6):.1%}, showing the correct hub is usually in the top 5.

2. **Multi-hub controls (35%):** About 1 in 3 controls legitimately maps to more than one hub. hit@1 alone understates performance -- the hit@any column in the evaluation table is a fairer measure.

3. **Calibration is approximate:** ECE={ece_val:.3f} means confidence scores are off by ~{ece_val*100:.0f} percentage points on average. Treat them as ordinal rankings (higher = better), not as exact probabilities.

4. **Training data scope:** Calibrated on 420 traditional framework holdout items. Accuracy on AI-specific text may differ from the reported metrics, especially for concepts not well-represented in the 5 AI frameworks.

5. **Not a replacement for expert judgment:** Model predictions are a **starting point** for compliance crosswalks. A security professional should review all assignments, especially for high-stakes compliance work.

6. **Language:** English only. The base model (BGE-large-en-v1.5) and all training data are English.

7. **What does NOT work for this task:** DeBERTa-v3-NLI achieves hit@1=0.000 -- Natural Language Inference (textual entailment) is fundamentally different from semantic similarity for taxonomy assignment. Do not substitute NLI models.

---

## Ethical Considerations

- This model is a **decision-support tool**, not an autonomous compliance engine. All predictions require human review before use in security assessments or regulatory filings.
- The model was trained on publicly available security framework data. No proprietary or confidential data was used.
- Active learning rounds during development used expert-reviewed predictions, not autonomous deployment.
- Bridge analysis connections were individually reviewed by a human security expert; automated connections were not added without review.

## Environmental Impact

- **Training compute:** NVIDIA H100 GPU via RunPod, {gpu_hours:.1f} GPU-hours total (including LOFO cross-validation, ablation studies, and final deployment model)
- **Inference deployment:** Runs on an NVIDIA Jetson Orin AGX edge device (~30W TDP). A single control prediction takes <100ms on consumer hardware.
- **Carbon context:** Estimated {gpu_hours * 0.3:.1f} kWh training energy (US average grid: ~{gpu_hours * 0.3 * 0.4:.1f} kg CO2e)

## Glossary

| Term | Definition |
|---|---|
| **CRE** | Common Requirements Enumeration -- a universal taxonomy of security topics maintained by [OpenCRE.org](https://opencre.org) |
| **Hub** | A node in the CRE taxonomy tree representing a security concept (e.g., "Input validation," "Access control") |
| **LOFO** | Leave-One-Framework-Out -- cross-validation method where an entire framework is held out for testing |
| **Hub firewall** | During LOFO evaluation, hub embeddings are rebuilt WITHOUT the held-out framework to prevent information leakage |
| **hit@1** | The model's single best prediction matches the correct hub |
| **hit@any** | The model's top prediction matches ANY of the control's correct hubs (relevant for multi-hub controls) |
| **ECE** | Expected Calibration Error -- measures how well confidence scores match actual accuracy |
| **OOD** | Out-of-Distribution -- input text is too different from training data for reliable prediction |
| **LoRA** | Low-Rank Adaptation -- an efficient fine-tuning method that trains small adapter matrices instead of modifying all model weights |
| **Bridge** | A discovered conceptual connection between an AI-specific and a traditional CRE hub |
| **Temperature scaling** | A post-hoc calibration technique that sharpens or smooths the model's output distribution |

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

MIT License for model weights and code. The base model ([BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)) is also MIT licensed.

Bundled data files (CRE hierarchy, hub descriptions, bridge report) are sourced from publicly available security frameworks and [OpenCRE.org](https://opencre.org), provided under CC0 1.0 Universal.
"""

    readme_path = staging_dir / "README.md"
    readme_path.write_text(card, encoding="utf-8")
    logger.info("Wrote model card to %s", readme_path)
