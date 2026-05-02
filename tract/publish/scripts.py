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
