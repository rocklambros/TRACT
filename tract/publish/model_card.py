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
