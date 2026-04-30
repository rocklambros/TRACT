"""Phase 1B-alpha: Minimum Viable Experiment.

Single fold (hold out ATLAS, n=65). AI-only training data (132 examples).
BGE-large-v1.5 + LoRA rank 16, vanilla MNRL (in-batch negatives only).
Gate: any improvement over zero-shot hit@1 on the 65 ATLAS eval items.

Usage:
    python -m scripts.phase1b.alpha
    python -m scripts.phase1b.alpha --epochs 20
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    load_curated_links,
    load_opencre_cres,
    score_predictions,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HELD_OUT: str = "MITRE ATLAS"
OUTPUT_DIR: Path = Path("models/phase1b_alpha")
SEED: int = 42


def evaluate_model(
    model: SentenceTransformer,
    eval_items: list,
    hub_ids: list[str],
    hub_texts: dict[str, str],
) -> dict[str, float]:
    """Encode hubs + eval items, compute hit@k/MRR/NDCG."""
    hub_texts_ordered = [hub_texts[hid] for hid in hub_ids]
    hub_embs = model.encode(
        hub_texts_ordered, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False, batch_size=128,
    )

    predictions: list[list[str]] = []
    for item in eval_items:
        q_emb = model.encode(
            [item.control_text], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        sims = q_emb @ hub_embs.T
        ranked = np.argsort(sims[0])[::-1]
        predictions.append([hub_ids[i] for i in ranked])

    ground_truth = [item.ground_truth_hub_id for item in eval_items]
    return score_predictions(predictions, ground_truth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1B-alpha MVE")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── Load data using Phase 0 infrastructure ──
    logger.info("Loading data...")
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})
    folds = build_lofo_folds(tree, links, corpus, AI_FRAMEWORK_NAMES, template="path")
    atlas_fold = next(f for f in folds if f.held_out_framework == HELD_OUT)
    logger.info("ATLAS fold: %d eval items, %d hubs", len(atlas_fold.eval_items), len(atlas_fold.hub_ids))

    # ── Build training pairs (AI-only, excluding ATLAS) ──
    train_pairs: list[dict[str, str]] = []
    for link in links:
        if link.standard_name not in AI_FRAMEWORK_NAMES:
            continue
        if link.standard_name == HELD_OUT:
            continue
        control_text = link.section_name or link.section_id
        if not control_text or len(control_text) < 3:
            continue
        hub_text = atlas_fold.hub_texts.get(link.cre_id)
        if not hub_text:
            continue
        train_pairs.append({"anchor": control_text, "positive": hub_text})

    logger.info("Training pairs: %d (from 4 AI frameworks, excluding %s)", len(train_pairs), HELD_OUT)
    train_dataset = Dataset.from_list(train_pairs)

    # ── Load base model ──
    logger.info("Loading BGE-large-v1.5...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    # ── Zero-shot baseline ──
    logger.info("Computing zero-shot baseline on ATLAS fold...")
    baseline = evaluate_model(model, atlas_fold.eval_items, atlas_fold.hub_ids, atlas_fold.hub_texts)
    logger.info("Zero-shot baseline: hit@1=%.3f, hit@5=%.3f, MRR=%.3f, NDCG@10=%.3f",
                baseline["hit_at_1"], baseline["hit_at_5"], baseline["mrr"], baseline["ndcg_at_10"])

    # ── Apply LoRA ──
    logger.info("Applying LoRA (rank=%d)...", args.lora_rank)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %s / %s (%.2f%%)", f"{trainable:,}", f"{total:,}", 100 * trainable / total)

    # ── Train ──
    loss = MultipleNegativesRankingLoss(model)
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )

    logger.info("Training %d epochs, batch_size=%d, lr=%s...", args.epochs, args.batch_size, args.lr)
    start = time.time()
    trainer.train()
    elapsed = time.time() - start
    logger.info("Training completed in %.1fs", elapsed)

    # ── Evaluate fine-tuned model ──
    logger.info("Evaluating fine-tuned model on ATLAS fold...")
    finetuned = evaluate_model(model, atlas_fold.eval_items, atlas_fold.hub_ids, atlas_fold.hub_texts)
    logger.info("Fine-tuned:    hit@1=%.3f, hit@5=%.3f, MRR=%.3f, NDCG@10=%.3f",
                finetuned["hit_at_1"], finetuned["hit_at_5"], finetuned["mrr"], finetuned["ndcg_at_10"])

    # ── Gate decision ──
    delta = finetuned["hit_at_1"] - baseline["hit_at_1"]
    logger.info("=" * 60)
    logger.info("PHASE 1B-ALPHA RESULTS")
    logger.info("  Zero-shot hit@1: %.3f", baseline["hit_at_1"])
    logger.info("  Fine-tuned hit@1: %.3f", finetuned["hit_at_1"])
    logger.info("  Delta:            %+.3f", delta)
    if delta > 0:
        logger.info("  GATE: PASS — improvement detected. Proceed to full pipeline.")
    else:
        logger.info("  GATE: FAIL — no improvement. Investigate before proceeding.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
