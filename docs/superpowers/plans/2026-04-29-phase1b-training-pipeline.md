# Phase 1B Implementation Plan: CRE Hub Assignment Model Training Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a BGE-large-v1.5 bi-encoder with LoRA to assign security controls to CRE hubs. The model must beat zero-shot embedding baselines (hit@1 > 0.516) on a 5-fold LOFO evaluation over 197 AI-framework controls mapped to ~458 CRE hubs.

**Architecture:** LoRA rank-16 adapters on all 24 BGE-large transformer layers (~2.4M trainable params). Contrastive learning via MNRL with in-batch + hierarchy-mined hard negatives. Temperature-scaled sampling balances AI/traditional training data. Hub-aware batch construction prevents false negatives. LOFO cross-validation with fold-stratified micro-average bootstrap CIs for honest evaluation. Calibration via temperature scaling inside the LOFO loop.

**Tech Stack:** Python 3.12, sentence-transformers 5.3.0, PEFT (LoRA), PyTorch, Pydantic v2, WandB, pytest, mypy --strict

**Design Spec:** `docs/superpowers/specs/2026-04-29-phase1b-design.md` (714 lines). Read Sections 0.2, 3.3-3.5, 5.1-5.3 before starting.

**Critical Constraint:** Task 1 (Phase 1B-alpha) is a 3-hour gate. All subsequent tasks are blocked until alpha shows any improvement over zero-shot hit@1 on a single ATLAS fold.

---

## File Structure

| File | Responsibility | Action |
|------|---------------|--------|
| `requirements.txt` | Add ML dependencies (torch, sentence-transformers, peft, etc.) | MODIFY |
| `scripts/phase1b/__init__.py` | Package marker | CREATE |
| `scripts/phase1b/alpha.py` | Phase 1B-alpha MVE (~80 lines) | CREATE |
| `tract/config.py` | Add Phase 1B constants | MODIFY (append) |
| `tract/training/__init__.py` | Package marker | CREATE |
| `tract/training/config.py` | TrainingConfig dataclass, LoRA config, hyperparameter constants | CREATE |
| `tract/training/data_quality.py` | Traditional link filtering, quality tier assignment, data hash chain | CREATE |
| `tract/training/data.py` | TrainingPair, hard negative mining, temp-scaled sampling, HubAwareBatchSampler, Dataset conversion | CREATE |
| `tract/training/firewall.py` | Hub representation firewall (build + assert) | CREATE |
| `tract/training/loop.py` | LoRA training loop (SentenceTransformerTrainer, checkpointing, early stopping) | CREATE |
| `tract/training/evaluate.py` | LOFO harness, fold-stratified bootstrap CIs, paired bootstrap, BH FDR, soft floors | CREATE |
| `tract/training/calibrate.py` | Temperature scaling, global threshold | CREATE |
| `tract/training/orchestrate.py` | Multi-fold, multi-config experiment runner (multiprocessing) | CREATE |
| `scripts/phase1b/train.py` | CLI entrypoint for training | CREATE |
| `scripts/phase1b/ablation.py` | Ablation sweep runner with config generation | CREATE |
| `scripts/phase1b/llm_comparison.py` | Post-training diagnostic vs Phase 0R Sonnet predictions | CREATE |
| `tests/test_data_quality.py` | Quality pipeline tests | CREATE |
| `tests/test_training_data.py` | TrainingPair, hard negatives, sampling, batching tests | CREATE |
| `tests/test_firewall.py` | Firewall assertion tests | CREATE |
| `tests/test_training_loop.py` | Training loop smoke tests (CPU, 1 step) | CREATE |
| `tests/test_evaluate.py` | Evaluation harness tests (fold-stratified bootstrap, BH FDR) | CREATE |
| `tests/test_calibrate.py` | Calibration tests | CREATE |

---

### Task 1: Phase 1B-alpha — Minimum Viable Experiment

**This task gates everything. Do not start Task 2 until the gate decision is made.**

**What:** Train BGE-large + LoRA on 132 AI-only examples, evaluate on 65 held-out ATLAS items. Compare to zero-shot baseline. Any improvement → proceed. No improvement → investigate before building the full pipeline.

**Time estimate:** 3 hours. **Compute cost:** <$0.50 (runs locally on Orin).

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/phase1b/__init__.py`, `scripts/phase1b/alpha.py`

- [ ] **Step 1: Add ML dependencies to requirements.txt**

Add after the `# ML (Phase 2+)` section. The existing comment is wrong — rename it:

```python
# requirements.txt changes:

# ML (Phase 2+)     ← rename to:
# ML

# existing:
# numpy==2.0.2
# safetensors==0.7.0
# wandb==0.25.1

# ADD these:
torch>=2.2.0
transformers>=4.40.0
sentence-transformers==5.3.0
peft>=0.11.0
datasets>=2.19.0
accelerate>=0.30.0
```

**Orin/Jetson note:** PyTorch on Jetson must be installed from NVIDIA's JetPack repository. If `torch` is already installed via JetPack, skip its pip install. Install the rest normally: `pip install transformers sentence-transformers peft datasets accelerate`.

- [ ] **Step 2: Install and verify dependencies**

```bash
pip install transformers sentence-transformers==5.3.0 peft datasets accelerate
python -c "
import torch
import sentence_transformers
import peft
print(f'torch={torch.__version__}, CUDA={torch.cuda.is_available()}')
print(f'sentence-transformers={sentence_transformers.__version__}')
print(f'peft={peft.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')
"
```

Expected: CUDA available, GPU recognized (Orin or H100).

- [ ] **Step 3: Create scripts/phase1b/ package**

```bash
mkdir -p scripts/phase1b
touch scripts/phase1b/__init__.py
```

- [ ] **Step 4: Write the alpha training script**

Create `scripts/phase1b/alpha.py`:

```python
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
```

**Key design choices in this script:**
- Reuses Phase 0's `common.py` infrastructure for data loading, LOFO fold building, and scoring — no new infrastructure needed
- Computes zero-shot baseline on the *same* ATLAS fold with the *same* hub representations for a fair comparison
- Uses `model[0].auto_model = get_peft_model(...)` which is the standard way to apply PEFT to sentence-transformers models (the `[0]` accesses the Transformer module inside the SentenceTransformer wrapper)
- `target_modules=["query", "key", "value"]` matches BERT/BGE attention layer naming (not LLaMA-style `q_proj`/`k_proj`/`v_proj`)
- No hard negatives, no temperature sampling, no hub-aware batching — all stripped to bare minimum
- `save_strategy="no"` and `report_to="none"` — no checkpoints, no WandB for the alpha

- [ ] **Step 5: Run Phase 1B-alpha**

```bash
python -m scripts.phase1b.alpha 2>&1 | tee results/phase1b_alpha.log
```

Expected runtime on Orin: ~10-30 minutes (132 examples × 10 epochs with LoRA on 61GB VRAM).

If `CUDA out of memory`: reduce batch size to 16: `python -m scripts.phase1b.alpha --batch-size 16`

- [ ] **Step 6: Gate decision**

Read the log output. The gate is simple:
- **Delta > 0** → PASS. Proceed to Task 2.
- **Delta ≤ 0** → FAIL. Before proceeding:
  1. Try `--epochs 20` and `--epochs 30` (underfitting?)
  2. Try `--lr 1e-3` (LoRA adapters often need higher LR)
  3. Try `--lora-rank 32` (more capacity)
  4. If still no improvement after 3 variations, stop and reassess the approach before investing in the full pipeline.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt scripts/phase1b/__init__.py scripts/phase1b/alpha.py
git commit -m "feat: Phase 1B-alpha MVE — LoRA+MNRL on single ATLAS fold"
```

---

### Task 2: Add Phase 1B constants to config.py

**Prerequisite:** Task 1 gate PASSED.

**Files:**
- Modify: `tract/config.py:139`

- [ ] **Step 1: Add Phase 1B constants**

Append to the end of `tract/config.py`:

```python
# ── Phase 1B: Model Training ──────────────────────────────────────────

PHASE1B_BASE_MODEL: Final[str] = "BAAI/bge-large-en-v1.5"
PHASE1B_EMBEDDING_DIM: Final[int] = 1024

PHASE1B_LORA_RANK: Final[int] = 16
PHASE1B_LORA_ALPHA: Final[int] = 32
PHASE1B_LORA_DROPOUT: Final[float] = 0.1
PHASE1B_LORA_TARGET_MODULES: Final[list[str]] = ["query", "key", "value"]

PHASE1B_BATCH_SIZE: Final[int] = 64
PHASE1B_LEARNING_RATE: Final[float] = 5e-4
PHASE1B_WARMUP_RATIO: Final[float] = 0.1
PHASE1B_WEIGHT_DECAY: Final[float] = 0.01
PHASE1B_MAX_GRAD_NORM: Final[float] = 1.0
PHASE1B_MAX_EPOCHS: Final[int] = 20
PHASE1B_MAX_SEQ_LENGTH: Final[int] = 512
PHASE1B_SEED: Final[int] = 42

PHASE1B_HARD_NEGATIVES: Final[int] = 3
PHASE1B_SAMPLING_TEMPERATURE: Final[float] = 2.0
PHASE1B_MIN_CONTROL_TEXT_LENGTH: Final[int] = 10

PHASE1B_BOOTSTRAP_N_RESAMPLES: Final[int] = 10_000
PHASE1B_BOOTSTRAP_SEED: Final[int] = 42
PHASE1B_BOOTSTRAP_CI_LEVEL: Final[float] = 0.95

PHASE1B_BH_FDR_Q: Final[float] = 0.10

PHASE1B_GATE_HIT1_DELTA: Final[float] = 0.10
PHASE1B_GATE_HIT1_MIN: Final[float] = 0.516
PHASE1B_GATE_HIT5_MIN: Final[float] = 0.70

PHASE1B_SOFT_FLOOR_LARGE: Final[float] = -0.05
PHASE1B_SOFT_FLOOR_NIST: Final[float] = -0.10

PHASE1B_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase1b"
PHASE1B_MODELS_DIR: Final[Path] = MODELS_DIR / "phase1b"

PHASE1B_WANDB_PROJECT: Final[str] = "tract-phase1b"

PHASE1B_DROPPED_FRAMEWORKS: Final[frozenset[str]] = frozenset({
    "nist_800_63",
    "owasp_proactive_controls",
})
PHASE1B_MIN_SECTION_TEXT_LENGTH: Final[int] = 10
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from tract.config import PHASE1B_BASE_MODEL, PHASE1B_LORA_RANK, PHASE1B_RESULTS_DIR; print(PHASE1B_BASE_MODEL, PHASE1B_LORA_RANK, PHASE1B_RESULTS_DIR)"
```

Expected: `BAAI/bge-large-en-v1.5 16 /home/rock/github_projects/TRACT/results/phase1b`

- [ ] **Step 3: Commit**

```bash
git add tract/config.py
git commit -m "feat: add Phase 1B training constants to config"
```

---

### Task 3: Training data quality pipeline

**What:** Filter traditional links by quality, assign tier metadata, compute data hash chain. This produces `hub_links_training.jsonl` (~4,073 usable links from 4,405 curated).

**Files:**
- Create: `tract/training/__init__.py`, `tract/training/data_quality.py`
- Create: `tests/test_data_quality.py`

**Reference:** Design spec Section 1.2 (quality pipeline), `scripts/analysis/audit_traditional_links.py` (existing audit).

- [ ] **Step 1: Create tract/training/ package**

```bash
mkdir -p tract/training
touch tract/training/__init__.py
```

- [ ] **Step 2: Write failing tests**

Create `tests/test_data_quality.py`:

```python
"""Tests for training data quality pipeline."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tract.training.data_quality import (
    QualityTier,
    TieredLink,
    assign_quality_tier,
    compute_data_hash,
    filter_training_links,
)


class TestQualityTierAssignment:
    """Test quality tier assignment logic."""

    def test_human_linked_traditional_is_t1(self) -> None:
        link = {
            "cre_id": "760-764", "cre_name": "Injection protection",
            "standard_name": "OWASP Top 10 2021", "link_type": "LinkedTo",
            "section_id": "A03", "section_name": "Injection",
            "framework_id": "owasp_top10_2021",
        }
        assert assign_quality_tier(link) == QualityTier.T1

    def test_auto_linked_is_t3(self) -> None:
        link = {
            "cre_id": "760-764", "cre_name": "Injection protection",
            "standard_name": "CAPEC", "link_type": "AutomaticallyLinkedTo",
            "section_id": "CAPEC-66", "section_name": "SQL Injection",
            "framework_id": "capec",
        }
        assert assign_quality_tier(link) == QualityTier.T3

    def test_ai_framework_is_t1_ai(self) -> None:
        link = {
            "cre_id": "123-456", "cre_name": "Test",
            "standard_name": "MITRE ATLAS", "link_type": "LinkedTo",
            "section_id": "AML.T0001", "section_name": "Adversarial Perturbation",
            "framework_id": "mitre_atlas",
        }
        assert assign_quality_tier(link) == QualityTier.T1_AI

    def test_bare_id_short_text_is_dropped(self) -> None:
        link = {
            "cre_id": "111-222", "cre_name": "Test",
            "standard_name": "NIST 800-63", "link_type": "LinkedTo",
            "section_id": "5.1.4.2", "section_name": "5.1.4.2",
            "framework_id": "nist_800_63",
        }
        assert assign_quality_tier(link) == QualityTier.DROPPED

    def test_short_section_name_is_dropped(self) -> None:
        link = {
            "cre_id": "333-444", "cre_name": "Test",
            "standard_name": "DSOMM", "link_type": "LinkedTo",
            "section_id": "D1", "section_name": "Process",
            "framework_id": "dsomm",
        }
        assert assign_quality_tier(link) == QualityTier.DROPPED


class TestFilterTrainingLinks:
    """Test end-to-end link filtering."""

    def test_filters_dropped_links(self) -> None:
        links = [
            {"cre_id": "1", "cre_name": "A", "standard_name": "ASVS",
             "link_type": "LinkedTo", "section_id": "V1", "section_name": "Architecture Assessment",
             "framework_id": "asvs"},
            {"cre_id": "2", "cre_name": "B", "standard_name": "NIST 800-63",
             "link_type": "LinkedTo", "section_id": "5.1.4.2", "section_name": "5.1.4.2",
             "framework_id": "nist_800_63"},
        ]
        result = filter_training_links(links)
        assert len(result) == 1
        assert result[0].link["cre_id"] == "1"
        assert result[0].tier == QualityTier.T1

    def test_preserves_all_ai_links(self) -> None:
        links = [
            {"cre_id": "1", "cre_name": "A", "standard_name": "MITRE ATLAS",
             "link_type": "LinkedTo", "section_id": "AML.T0001",
             "section_name": "Adversarial Perturbation", "framework_id": "mitre_atlas"},
        ]
        result = filter_training_links(links)
        assert len(result) == 1
        assert result[0].tier == QualityTier.T1_AI


class TestDataHash:
    """Test deterministic data hashing."""

    def test_hash_is_deterministic(self) -> None:
        data = [{"a": 1}, {"b": 2}]
        h1 = compute_data_hash(data)
        h2 = compute_data_hash(data)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_changes_with_data(self) -> None:
        h1 = compute_data_hash([{"a": 1}])
        h2 = compute_data_hash([{"a": 2}])
        assert h1 != h2
```

Run: `python -m pytest tests/test_data_quality.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement data_quality.py**

Create `tract/training/data_quality.py`:

```python
"""Training data quality pipeline.

Filters curated hub links by quality, assigns tier metadata,
and computes data hash chain for provenance tracking.

Quality tiers:
  T1     — Human LinkedTo with descriptive text (traditional frameworks)
  T1-AI  — Human-curated AI framework links
  T3     — AutomaticallyLinkedTo with descriptive text
  DROPPED — Bare-ID, short text, or from dropped frameworks
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tract.config import (
    PHASE1B_DROPPED_FRAMEWORKS,
    PHASE1B_MIN_SECTION_TEXT_LENGTH,
    TRAINING_DIR,
)

logger = logging.getLogger(__name__)

AI_FRAMEWORK_NAMES: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS",
    "NIST AI 100-2",
    "OWASP AI Exchange",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
})

CURATED_PATH: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"
TRAINING_OUTPUT_PATH: Final[Path] = TRAINING_DIR / "hub_links_training.jsonl"


class QualityTier(enum.Enum):
    T1 = "T1"
    T1_AI = "T1-AI"
    T3 = "T3"
    DROPPED = "DROPPED"


@dataclass(frozen=True)
class TieredLink:
    link: dict[str, str]
    tier: QualityTier


def _has_descriptive_text(link: dict[str, str]) -> bool:
    """Check if section_name has enough descriptive content."""
    section = link.get("section_name", "")
    return len(section) >= PHASE1B_MIN_SECTION_TEXT_LENGTH


def assign_quality_tier(link: dict[str, str]) -> QualityTier:
    """Assign a quality tier to a single hub link."""
    framework_id = link.get("framework_id", "")
    standard_name = link.get("standard_name", "")
    link_type = link.get("link_type", "")

    if framework_id in PHASE1B_DROPPED_FRAMEWORKS:
        return QualityTier.DROPPED

    if not _has_descriptive_text(link):
        return QualityTier.DROPPED

    if standard_name in AI_FRAMEWORK_NAMES:
        return QualityTier.T1_AI

    if link_type == "AutomaticallyLinkedTo":
        return QualityTier.T3

    return QualityTier.T1


def filter_training_links(links: list[dict[str, str]]) -> list[TieredLink]:
    """Filter links by quality and assign tier metadata.

    Returns non-DROPPED links with their tier assignment.
    """
    result: list[TieredLink] = []
    tier_counts: dict[QualityTier, int] = {t: 0 for t in QualityTier}

    for link in links:
        tier = assign_quality_tier(link)
        tier_counts[tier] += 1
        if tier != QualityTier.DROPPED:
            result.append(TieredLink(link=link, tier=tier))

    for tier, count in tier_counts.items():
        logger.info("Quality tier %s: %d links", tier.value, count)

    return result


def compute_data_hash(data: list[dict]) -> str:
    """Compute deterministic SHA-256 hash of structured data."""
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_and_filter_curated_links(
    path: Path | None = None,
) -> tuple[list[TieredLink], str]:
    """Load curated links, filter by quality, return with data hash.

    Returns:
        Tuple of (filtered links with tiers, SHA-256 hash of raw data).
    """
    p = path or CURATED_PATH
    raw_links: list[dict[str, str]] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_links.append(json.loads(line))

    raw_hash = compute_data_hash(raw_links)
    logger.info("Loaded %d curated links (hash=%s)", len(raw_links), raw_hash[:16])

    filtered = filter_training_links(raw_links)
    logger.info("After quality filter: %d usable links (dropped %d)",
                len(filtered), len(raw_links) - len(filtered))

    return filtered, raw_hash


def save_training_links(
    links: list[TieredLink],
    raw_hash: str,
    path: Path | None = None,
) -> str:
    """Save filtered training links to JSONL with tier metadata.

    Returns SHA-256 hash of the output data.
    """
    p = path or TRAINING_OUTPUT_PATH
    output_records: list[dict] = []
    for tiered in links:
        record = dict(tiered.link)
        record["quality_tier"] = tiered.tier.value
        output_records.append(record)

    output_hash = compute_data_hash(output_records)

    import tempfile, os
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=f".{p.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in output_records:
                f.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Saved %d training links to %s (hash=%s, raw_hash=%s)",
                len(output_records), p.name, output_hash[:16], raw_hash[:16])
    return output_hash
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/test_data_quality.py -v
```

Expected: All pass.

- [ ] **Step 5: Run quality pipeline on real data**

```bash
python -c "
from tract.training.data_quality import load_and_filter_curated_links, save_training_links
filtered, raw_hash = load_and_filter_curated_links()
output_hash = save_training_links(filtered, raw_hash)
print(f'Raw: 4405, Filtered: {len(filtered)}, Hash: {output_hash[:16]}')
# Verify tier counts
from collections import Counter
from tract.training.data_quality import QualityTier
counts = Counter(t.tier for t in filtered)
for tier in QualityTier:
    print(f'  {tier.value}: {counts.get(tier, 0)}')
"
```

Expected: ~4,073 usable links. T1 ~2,200, T1-AI 197, T3 ~1,676. DROPPED ~332.

- [ ] **Step 6: Commit**

```bash
git add tract/training/__init__.py tract/training/data_quality.py tests/test_data_quality.py
git commit -m "feat: training data quality pipeline with tier assignment and hash chain"
```

---

### Task 4: Hub representation firewall

**What:** Build firewalled hub text representations and assert no information leakage. The primary hub rep is `"{hierarchy_path} | {hub_name}"` — inherently firewall-safe because neither component comes from framework text.

**Files:**
- Create: `tract/training/firewall.py`, `tests/test_firewall.py`

**Reference:** Design spec Section 4 (firewall), `tract/hierarchy.py` (CREHierarchy API).

- [ ] **Step 1: Write failing tests**

Create `tests/test_firewall.py`:

```python
"""Tests for hub representation firewall."""
from __future__ import annotations

import pytest

from tests.test_hierarchy import _build_test_hierarchy  # reuse fixture builder


class TestBuildFirewalledHubText:
    """Test firewalled hub text construction."""

    def test_returns_path_pipe_name(self) -> None:
        from tract.training.firewall import build_firewalled_hub_text
        hierarchy = _build_test_hierarchy()
        hub_id = list(hierarchy.hubs.keys())[0]
        text = build_firewalled_hub_text(hub_id, hierarchy)
        assert " | " in text
        assert hierarchy.hubs[hub_id].name in text
        assert hierarchy.hubs[hub_id].hierarchy_path in text

    def test_excluded_framework_does_not_affect_primary_rep(self) -> None:
        from tract.training.firewall import build_firewalled_hub_text
        hierarchy = _build_test_hierarchy()
        hub_id = list(hierarchy.hubs.keys())[0]
        text_a = build_firewalled_hub_text(hub_id, hierarchy, excluded_framework="MITRE ATLAS")
        text_b = build_firewalled_hub_text(hub_id, hierarchy, excluded_framework="OWASP AI Exchange")
        assert text_a == text_b  # primary rep doesn't use framework text


class TestBuildAllHubTexts:
    """Test bulk hub text construction."""

    def test_builds_text_for_all_hubs(self) -> None:
        from tract.training.firewall import build_all_hub_texts
        hierarchy = _build_test_hierarchy()
        texts = build_all_hub_texts(hierarchy)
        assert len(texts) == len(hierarchy.hubs)
        for hub_id, text in texts.items():
            assert hub_id in hierarchy.hubs
            assert len(text) > 0


class TestFirewallAssertion:
    """Test firewall breach detection."""

    def test_passes_when_no_leakage(self) -> None:
        from tract.training.firewall import assert_firewall
        hub_texts = {"hub-1": "Root > Security | Security", "hub-2": "Root > Privacy | Privacy"}
        # Mock eval items with control text that doesn't appear in hub texts
        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw
        items = [MockItem("SQL Injection attacks", "ATLAS")]
        assert_firewall(hub_texts, items, "ATLAS")  # should not raise

    def test_fails_when_control_text_in_hub_text(self) -> None:
        from tract.training.firewall import assert_firewall
        hub_texts = {"hub-1": "Security: SQL Injection attacks"}
        class MockItem:
            def __init__(self, text: str, fw: str) -> None:
                self.control_text = text
                self.framework = fw
        items = [MockItem("SQL Injection attacks", "ATLAS")]
        with pytest.raises(AssertionError, match="Firewall breach"):
            assert_firewall(hub_texts, items, "ATLAS")
```

**Note:** The test imports `_build_test_hierarchy` from existing hierarchy tests. If that fixture isn't directly importable, extract the hierarchy fixture to `tests/conftest.py` or build a local one. Check `tests/test_hierarchy.py` for the exact fixture builder function.

Run: `python -m pytest tests/test_firewall.py -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Implement firewall.py**

Create `tract/training/firewall.py`:

```python
"""Hub representation firewall for LOFO evaluation.

Ensures no information from the held-out framework leaks into hub
representations during evaluation. The primary representation
("{hierarchy_path} | {hub_name}") is inherently firewall-safe because
both components come from CRE structure, not framework text.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from tract.hierarchy import CREHierarchy

logger = logging.getLogger(__name__)


class HasControlText(Protocol):
    control_text: str
    framework: str


def build_firewalled_hub_text(
    hub_id: str,
    hierarchy: CREHierarchy,
    excluded_framework: str | None = None,
    include_description: bool = False,
    descriptions: dict[str, str] | None = None,
    include_standards: bool = False,
    standard_sections: dict[str, list[str]] | None = None,
) -> str:
    """Build a single hub's text representation with firewall.

    Primary format: "{hierarchy_path} | {hub_name}"

    If include_description=True (ablation A6): appends description.
    If include_standards=True (ablation A3): appends standard names,
    excluding the held-out framework.
    """
    node = hierarchy.hubs[hub_id]
    text = f"{node.hierarchy_path} | {node.name}"

    if include_description and descriptions and hub_id in descriptions:
        text = f"{text}: {descriptions[hub_id]}"

    if include_standards and standard_sections and hub_id in standard_sections:
        sections = [s for s in standard_sections[hub_id]
                    if excluded_framework is None or excluded_framework not in s]
        if sections:
            text = f"{text}. Standards: {', '.join(sorted(sections))}"

    return text


def build_all_hub_texts(
    hierarchy: CREHierarchy,
    excluded_framework: str | None = None,
    include_description: bool = False,
    descriptions: dict[str, str] | None = None,
    include_standards: bool = False,
    standard_sections: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Build text representations for all hubs with firewall applied."""
    texts: dict[str, str] = {}
    for hub_id in hierarchy.hubs:
        texts[hub_id] = build_firewalled_hub_text(
            hub_id, hierarchy, excluded_framework,
            include_description, descriptions,
            include_standards, standard_sections,
        )
    return texts


def assert_firewall(
    hub_texts: dict[str, str],
    eval_items: list[Any],
    held_out_framework: str,
) -> None:
    """Assert no information leakage from held-out framework into hub representations.

    Raises AssertionError if any eval item's control text appears verbatim
    in any hub text. This is a conservative check — substring matching
    catches the most obvious leakage patterns.
    """
    for item in eval_items:
        control_text = item.control_text
        if len(control_text) < 5:
            continue
        for hub_id, text in hub_texts.items():
            if control_text in text:
                raise AssertionError(
                    f"Firewall breach: control '{control_text[:50]}' "
                    f"(framework={held_out_framework}) found in hub {hub_id} text"
                )
    logger.info("Firewall assertion passed: %d items checked against %d hubs",
                len(eval_items), len(hub_texts))
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_firewall.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tract/training/firewall.py tests/test_firewall.py
git commit -m "feat: hub representation firewall with assertion check"
```

---

### Task 5: Training data generation

**What:** Build `TrainingPair` objects from filtered links. Mine hard negatives from CRE hierarchy. Implement temperature-scaled class-level sampling and `HubAwareBatchSampler`. Convert to sentence-transformers `Dataset` format.

**Files:**
- Create: `tract/training/data.py`, `tests/test_training_data.py`

**Reference:** Design spec Sections 1.3 (format), 3.2 (hard negatives), 3.4-3.5 (sampling + batching).

- [ ] **Step 1: Write failing tests**

Create `tests/test_training_data.py`:

```python
"""Tests for training data generation."""
from __future__ import annotations

import pytest
import numpy as np

from tract.training.data import (
    TrainingPair,
    mine_hard_negatives,
    build_training_pairs,
    apply_temperature_sampling_order,
    HubAwareBatchSampler,
)


class TestTrainingPair:
    """Test TrainingPair construction."""

    def test_frozen_dataclass(self) -> None:
        pair = TrainingPair(
            control_text="SQL Injection",
            hub_id="760-764",
            hub_representation="Root > AppSec | Injection protection",
            framework="OWASP Top 10 2021",
            link_type="LinkedTo",
            quality_tier="T1",
        )
        assert pair.control_text == "SQL Injection"
        with pytest.raises(AttributeError):
            pair.control_text = "changed"  # type: ignore[misc]


class TestHardNegativeMining:
    """Test hierarchy-based hard negative mining."""

    def test_returns_sibling_hub_ids(self) -> None:
        from tests.test_hierarchy import _build_test_hierarchy
        hierarchy = _build_test_hierarchy()
        leaf_ids = hierarchy.leaf_hub_ids()
        if len(leaf_ids) < 2:
            pytest.skip("Need at least 2 leaf hubs for sibling test")
        hub_id = leaf_ids[0]
        negatives = mine_hard_negatives(hub_id, hierarchy, n=3)
        assert isinstance(negatives, list)
        assert hub_id not in negatives
        for neg_id in negatives:
            assert neg_id in hierarchy.hubs

    def test_returns_at_most_n(self) -> None:
        from tests.test_hierarchy import _build_test_hierarchy
        hierarchy = _build_test_hierarchy()
        hub_id = list(hierarchy.hubs.keys())[0]
        negatives = mine_hard_negatives(hub_id, hierarchy, n=2)
        assert len(negatives) <= 2

    def test_no_duplicates(self) -> None:
        from tests.test_hierarchy import _build_test_hierarchy
        hierarchy = _build_test_hierarchy()
        hub_id = list(hierarchy.hubs.keys())[0]
        negatives = mine_hard_negatives(hub_id, hierarchy, n=5)
        assert len(negatives) == len(set(negatives))


class TestTemperatureScaledSampling:
    """Test temperature-scaled class ordering."""

    def test_ai_fraction_increases_with_temperature(self) -> None:
        is_ai = [True] * 20 + [False] * 200
        indices_t1 = apply_temperature_sampling_order(is_ai, temperature=1.0, seed=42)
        indices_t2 = apply_temperature_sampling_order(is_ai, temperature=2.0, seed=42)
        # With T=2, first 64 items should have more AI than with T=1
        ai_in_first_batch_t1 = sum(1 for i in indices_t1[:64] if is_ai[i])
        ai_in_first_batch_t2 = sum(1 for i in indices_t2[:64] if is_ai[i])
        assert ai_in_first_batch_t2 >= ai_in_first_batch_t1

    def test_all_items_appear_exactly_once(self) -> None:
        is_ai = [True] * 10 + [False] * 50
        indices = apply_temperature_sampling_order(is_ai, temperature=2.0, seed=42)
        assert sorted(indices) == list(range(len(is_ai)))


class TestHubAwareBatchSampler:
    """Test hub-aware batch construction."""

    def test_no_duplicate_hubs_in_batch(self) -> None:
        hub_ids = ["h1", "h2", "h3", "h1", "h2", "h4", "h5", "h6"]
        sampler = HubAwareBatchSampler(hub_ids, batch_size=4, seed=42)
        for batch in sampler:
            batch_hubs = [hub_ids[i] for i in batch]
            assert len(batch_hubs) == len(set(batch_hubs)), \
                f"Duplicate hubs in batch: {batch_hubs}"

    def test_all_items_covered(self) -> None:
        hub_ids = ["h1", "h2", "h3", "h1", "h2", "h4"]
        sampler = HubAwareBatchSampler(hub_ids, batch_size=3, seed=42)
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(len(hub_ids)))

    def test_deterministic_with_same_seed(self) -> None:
        hub_ids = ["h1", "h2", "h3", "h4", "h5"] * 3
        batches1 = list(HubAwareBatchSampler(hub_ids, batch_size=4, seed=42))
        batches2 = list(HubAwareBatchSampler(hub_ids, batch_size=4, seed=42))
        assert batches1 == batches2
```

Run: `python -m pytest tests/test_training_data.py -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Implement data.py**

Create `tract/training/data.py`:

```python
"""Training data generation for contrastive fine-tuning.

Handles:
- TrainingPair construction from filtered hub links
- Hard negative mining from CRE hierarchy (siblings, then cousins)
- Temperature-scaled class-level sampling (AI vs traditional)
- HubAwareBatchSampler preventing positive-hub collisions
- Conversion to sentence-transformers Dataset format
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import numpy as np
from datasets import Dataset

from tract.hierarchy import CREHierarchy
from tract.training.data_quality import QualityTier, TieredLink

logger = logging.getLogger(__name__)

AI_FRAMEWORK_NAMES: frozenset[str] = frozenset({
    "MITRE ATLAS", "NIST AI 100-2", "OWASP AI Exchange",
    "OWASP Top10 for LLM", "OWASP Top10 for ML",
})


@dataclass(frozen=True)
class TrainingPair:
    control_text: str
    hub_id: str
    hub_representation: str
    framework: str
    link_type: str
    quality_tier: str


def mine_hard_negatives(
    hub_id: str,
    hierarchy: CREHierarchy,
    n: int = 3,
) -> list[str]:
    """Return up to n hard negative hub IDs from hierarchy structure.

    Priority: siblings first (same parent), then cousins (parent's siblings' children).
    """
    siblings = [s.hub_id for s in hierarchy.get_siblings(hub_id)]
    if len(siblings) >= n:
        return siblings[:n]

    cousins: list[str] = []
    parent = hierarchy.get_parent(hub_id)
    if parent:
        for uncle in hierarchy.get_siblings(parent.hub_id):
            for child in hierarchy.get_children(uncle.hub_id):
                if child.hub_id != hub_id and child.hub_id not in siblings:
                    cousins.append(child.hub_id)

    all_negatives = siblings + cousins
    seen: set[str] = set()
    deduped: list[str] = []
    for neg_id in all_negatives:
        if neg_id not in seen:
            seen.add(neg_id)
            deduped.append(neg_id)
    return deduped[:n]


def build_training_pairs(
    tiered_links: list[TieredLink],
    hub_texts: dict[str, str],
    excluded_framework: str | None = None,
) -> list[TrainingPair]:
    """Build TrainingPair objects from filtered links.

    Args:
        tiered_links: Quality-filtered links with tier metadata.
        hub_texts: Firewalled hub text representations.
        excluded_framework: Framework to exclude (the LOFO held-out framework).
    """
    pairs: list[TrainingPair] = []
    skipped = 0

    for tiered in tiered_links:
        link = tiered.link
        standard_name = link.get("standard_name", "")

        if excluded_framework and standard_name == excluded_framework:
            continue

        control_text = link.get("section_name") or link.get("section_id", "")
        if not control_text or len(control_text) < 3:
            skipped += 1
            continue

        hub_id = link["cre_id"]
        hub_rep = hub_texts.get(hub_id)
        if not hub_rep:
            skipped += 1
            continue

        pairs.append(TrainingPair(
            control_text=control_text,
            hub_id=hub_id,
            hub_representation=hub_rep,
            framework=standard_name,
            link_type=link.get("link_type", ""),
            quality_tier=tiered.tier.value,
        ))

    if skipped:
        logger.info("Skipped %d links (empty text or missing hub)", skipped)
    logger.info("Built %d training pairs (excluded=%s)", len(pairs), excluded_framework)
    return pairs


def apply_temperature_sampling_order(
    is_ai: list[bool],
    temperature: float = 2.0,
    seed: int = 42,
) -> list[int]:
    """Reorder indices so AI examples cluster into early batches.

    Class-level formulation: w_class = (n_class / n_total) ^ (1/T)
    With T=2: P(AI class) ≈ 15.5% of positions (up from ~3.3% natural).
    All items appear exactly once (no duplication).
    """
    rng = np.random.default_rng(seed)
    n = len(is_ai)
    ai_indices = [i for i, v in enumerate(is_ai) if v]
    trad_indices = [i for i, v in enumerate(is_ai) if not v]

    n_ai = len(ai_indices)
    n_trad = len(trad_indices)

    if n_ai == 0 or n_trad == 0 or temperature <= 0:
        indices = list(range(n))
        rng.shuffle(indices)
        return indices

    w_ai = (n_ai / n) ** (1.0 / temperature)
    w_trad = (n_trad / n) ** (1.0 / temperature)
    p_ai = w_ai / (w_ai + w_trad)

    rng.shuffle(ai_indices)
    rng.shuffle(trad_indices)

    result: list[int] = []
    ai_ptr = 0
    trad_ptr = 0

    for _ in range(n):
        if ai_ptr >= n_ai:
            result.append(trad_indices[trad_ptr])
            trad_ptr += 1
        elif trad_ptr >= n_trad:
            result.append(ai_indices[ai_ptr])
            ai_ptr += 1
        elif rng.random() < p_ai:
            result.append(ai_indices[ai_ptr])
            ai_ptr += 1
        else:
            result.append(trad_indices[trad_ptr])
            trad_ptr += 1

    return result


class HubAwareBatchSampler:
    """Batch sampler that prevents positive-hub collisions.

    No two examples in a batch share the same target hub. This eliminates
    false negatives in MNRL where a "negative" hub is actually correct
    for another example in the batch.
    """

    def __init__(
        self,
        hub_ids: list[str],
        batch_size: int = 64,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        self.hub_ids = hub_ids
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed)
        indices = list(range(len(self.hub_ids)))
        rng.shuffle(indices)

        batch: list[int] = []
        hubs_in_batch: set[str] = set()
        deferred: list[int] = []

        for idx in indices:
            hub = self.hub_ids[idx]
            if hub not in hubs_in_batch:
                batch.append(idx)
                hubs_in_batch.add(hub)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    hubs_in_batch = set()
            else:
                deferred.append(idx)

        remaining = deferred
        while remaining:
            next_remaining: list[int] = []
            for idx in remaining:
                hub = self.hub_ids[idx]
                if hub not in hubs_in_batch:
                    batch.append(idx)
                    hubs_in_batch.add(hub)
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        hubs_in_batch = set()
                else:
                    next_remaining.append(idx)
            if len(next_remaining) == len(remaining):
                batch.extend(next_remaining)
                break
            remaining = next_remaining

        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.hub_ids) / self.batch_size)


def pairs_to_dataset(
    pairs: list[TrainingPair],
    hierarchy: CREHierarchy,
    hub_texts: dict[str, str],
    n_hard_negatives: int = 3,
) -> Dataset:
    """Convert TrainingPairs to a sentence-transformers Dataset with hard negatives.

    Output columns: anchor, positive, negative_1, negative_2, negative_3
    """
    records: list[dict[str, str]] = []
    for pair in pairs:
        record: dict[str, str] = {
            "anchor": pair.control_text,
            "positive": pair.hub_representation,
        }
        negatives = mine_hard_negatives(pair.hub_id, hierarchy, n=n_hard_negatives)
        for i, neg_id in enumerate(negatives):
            neg_text = hub_texts.get(neg_id, "")
            if neg_text:
                record[f"negative_{i + 1}"] = neg_text

        for i in range(len(negatives), n_hard_negatives):
            record[f"negative_{i + 1}"] = ""

        records.append(record)

    ds = Dataset.from_list(records)
    logger.info("Built dataset: %d examples, columns=%s", len(ds), ds.column_names)
    return ds
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_training_data.py -v
```

- [ ] **Step 4: Integration test with real data**

```bash
python -c "
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.firewall import build_all_hub_texts
from tract.training.data import build_training_pairs, pairs_to_dataset, apply_temperature_sampling_order
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.config import PROCESSED_DIR
from collections import Counter

hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / 'cre_hierarchy.json'))
hub_texts = build_all_hub_texts(hierarchy)
filtered, raw_hash = load_and_filter_curated_links()

# Build pairs excluding ATLAS
pairs = build_training_pairs(filtered, hub_texts, excluded_framework='MITRE ATLAS')
print(f'Training pairs (excl ATLAS): {len(pairs)}')

# Check AI fraction
ai_count = sum(1 for p in pairs if p.framework in {
    'NIST AI 100-2', 'OWASP AI Exchange', 'OWASP Top10 for LLM', 'OWASP Top10 for ML'
})
print(f'AI pairs: {ai_count} ({100*ai_count/len(pairs):.1f}%)')

# Test temperature sampling
is_ai = [p.framework in {'NIST AI 100-2', 'OWASP AI Exchange', 'OWASP Top10 for LLM', 'OWASP Top10 for ML'} for p in pairs]
order = apply_temperature_sampling_order(is_ai, temperature=2.0, seed=42)
ai_in_first_64 = sum(1 for i in order[:64] if is_ai[i])
print(f'AI in first batch of 64: {ai_in_first_64} (expected ~10)')

# Convert to dataset
ds = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=3)
print(f'Dataset: {ds}')
print(f'Sample: {ds[0]}')
"
```

- [ ] **Step 5: Commit**

```bash
git add tract/training/data.py tests/test_training_data.py
git commit -m "feat: training data pipeline with hard negatives and hub-aware batching"
```

---

### Task 6: LoRA training loop

**What:** Implement the core training loop using `SentenceTransformerTrainer` with PEFT LoRA, MNRL loss, early stopping, and checkpointing.

**Files:**
- Create: `tract/training/config.py`, `tract/training/loop.py`
- Create: `tests/test_training_loop.py`

**Reference:** Design spec Sections 2.3 (LoRA config), 3.1 (MNRL), 3.4 (hyperparameters), 6.2 (training loop), 9.1 (checkpointing).

- [ ] **Step 1: Write TrainingConfig**

Create `tract/training/config.py`:

```python
"""Training configuration and experiment metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tract.config import (
    PHASE1B_BASE_MODEL,
    PHASE1B_BATCH_SIZE,
    PHASE1B_HARD_NEGATIVES,
    PHASE1B_LEARNING_RATE,
    PHASE1B_LORA_ALPHA,
    PHASE1B_LORA_DROPOUT,
    PHASE1B_LORA_RANK,
    PHASE1B_LORA_TARGET_MODULES,
    PHASE1B_MAX_EPOCHS,
    PHASE1B_MAX_GRAD_NORM,
    PHASE1B_MAX_SEQ_LENGTH,
    PHASE1B_SAMPLING_TEMPERATURE,
    PHASE1B_SEED,
    PHASE1B_WARMUP_RATIO,
    PHASE1B_WEIGHT_DECAY,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Full configuration for one training experiment."""

    name: str
    base_model: str = PHASE1B_BASE_MODEL
    training_data: str = "joint-tempscaled"
    checkpoint_path: Path | None = None

    lora_rank: int = PHASE1B_LORA_RANK
    lora_alpha: int = PHASE1B_LORA_ALPHA
    lora_dropout: float = PHASE1B_LORA_DROPOUT
    lora_target_modules: list[str] = field(default_factory=lambda: list(PHASE1B_LORA_TARGET_MODULES))

    sampling_temperature: float = PHASE1B_SAMPLING_TEMPERATURE
    control_text_source: str = "section_name"

    batch_size: int = PHASE1B_BATCH_SIZE
    learning_rate: float = PHASE1B_LEARNING_RATE
    warmup_ratio: float = PHASE1B_WARMUP_RATIO
    weight_decay: float = PHASE1B_WEIGHT_DECAY
    max_grad_norm: float = PHASE1B_MAX_GRAD_NORM
    max_epochs: int = PHASE1B_MAX_EPOCHS
    max_seq_length: int = PHASE1B_MAX_SEQ_LENGTH
    hard_negatives: int = PHASE1B_HARD_NEGATIVES
    seed: int = PHASE1B_SEED

    hub_rep_format: str = "path+name"
    data_hash: str = ""

    def to_dict(self) -> dict:
        """Serialize for WandB/JSON logging."""
        d = {
            "name": self.name,
            "base_model": self.base_model,
            "training_data": self.training_data,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "sampling_temperature": self.sampling_temperature,
            "control_text_source": self.control_text_source,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "max_epochs": self.max_epochs,
            "max_seq_length": self.max_seq_length,
            "hard_negatives": self.hard_negatives,
            "seed": self.seed,
            "hub_rep_format": self.hub_rep_format,
            "data_hash": self.data_hash,
        }
        return d
```

- [ ] **Step 2: Write failing tests for training loop**

Create `tests/test_training_loop.py`:

```python
"""Tests for the LoRA training loop.

These tests run on CPU with minimal data to verify the training
pipeline wiring. They do NOT test model quality.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from tract.training.config import TrainingConfig


class TestLoadBaseModel:
    """Test model loading and LoRA application."""

    @pytest.mark.slow
    def test_loads_bge_with_lora(self) -> None:
        from tract.training.loop import load_model_with_lora
        config = TrainingConfig(name="test", lora_rank=4, max_seq_length=64)
        model = load_model_with_lora(config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable < total
        assert trainable > 0

    @pytest.mark.slow
    def test_full_finetune_when_rank_zero(self) -> None:
        from tract.training.loop import load_model_with_lora
        config = TrainingConfig(name="test", lora_rank=0, max_seq_length=64)
        model = load_model_with_lora(config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total


class TestTrainStep:
    """Test that training runs without error on synthetic data."""

    @pytest.mark.slow
    def test_single_epoch_smoke(self) -> None:
        from datasets import Dataset
        from tract.training.loop import train_model

        config = TrainingConfig(
            name="smoke-test",
            lora_rank=4,
            batch_size=2,
            max_epochs=1,
            max_seq_length=32,
            learning_rate=1e-4,
        )
        train_data = Dataset.from_list([
            {"anchor": "SQL injection attack", "positive": "Root > Security | Injection"},
            {"anchor": "Cross-site scripting", "positive": "Root > Security | XSS"},
            {"anchor": "Buffer overflow", "positive": "Root > Security | Memory Safety"},
            {"anchor": "Broken authentication", "positive": "Root > Security | Auth"},
        ])

        with tempfile.TemporaryDirectory() as tmpdir:
            model = train_model(config, train_data, output_dir=Path(tmpdir))
            assert model is not None
```

Run: `python -m pytest tests/test_training_loop.py -v -m "not slow"`
(No slow tests run — that's OK. Run with `--run-slow` when you have GPU time.)

- [ ] **Step 3: Implement loop.py**

Create `tract/training/loop.py`:

```python
"""LoRA training loop for contrastive fine-tuning.

Uses SentenceTransformerTrainer (modern API, not legacy model.fit())
with PEFT LoRA adapters and MNRL loss.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

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

from tract.training.config import TrainingConfig

logger = logging.getLogger(__name__)


def load_model_with_lora(config: TrainingConfig) -> SentenceTransformer:
    """Load base model and optionally apply LoRA adapters.

    If config.lora_rank == 0, returns the base model for full fine-tuning.
    """
    model = SentenceTransformer(config.base_model)
    model.max_seq_length = config.max_seq_length

    if config.lora_rank > 0:
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        model[0].auto_model = get_peft_model(model[0].auto_model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info("LoRA applied: %s trainable / %s total (%.2f%%)",
                    f"{trainable:,}", f"{total:,}", 100 * trainable / total)
    else:
        logger.info("Full fine-tuning (no LoRA)")

    return model


def train_model(
    config: TrainingConfig,
    train_dataset: Dataset,
    output_dir: Path,
    eval_dataset: Dataset | None = None,
) -> SentenceTransformer:
    """Train the model with MNRL and return the trained model.

    Args:
        config: Training hyperparameters.
        train_dataset: Dataset with columns: anchor, positive, [negative_1, ...].
        output_dir: Directory for checkpoints and logs.
        eval_dataset: Optional validation set for early stopping.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    model = load_model_with_lora(config)
    loss = MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.max_epochs,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        fp16=torch.cuda.is_available(),
        seed=config.seed,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=eval_dataset is not None,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        metric_for_best_model="eval_loss" if eval_dataset is not None else None,
        greater_is_better=False if eval_dataset is not None else None,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    logger.info("Starting training: %d examples, %d epochs, batch=%d, lr=%s",
                len(train_dataset), config.max_epochs, config.batch_size, config.learning_rate)
    trainer.train()
    logger.info("Training complete")

    return model


def save_checkpoint(
    model: SentenceTransformer,
    config: TrainingConfig,
    metrics: dict[str, Any],
    output_dir: Path,
    git_sha: str = "unknown",
) -> Path:
    """Save model checkpoint with full metadata for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save(str(output_dir / "model"))

    metadata = {
        "config": config.to_dict(),
        "metrics": metrics,
        "git_sha": git_sha,
        "torch_seed": config.seed,
    }

    meta_path = output_dir / "metadata.json"
    fd, tmp = tempfile.mkstemp(dir=output_dir, prefix=".metadata.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(metadata, f, sort_keys=True, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, meta_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

    logger.info("Saved checkpoint to %s", output_dir)
    return output_dir
```

- [ ] **Step 4: Smoke test on Orin (GPU)**

```bash
python -c "
from datasets import Dataset
from pathlib import Path
from tract.training.config import TrainingConfig
from tract.training.loop import train_model
import tempfile

config = TrainingConfig(name='smoke', lora_rank=4, batch_size=2, max_epochs=1, max_seq_length=32)
ds = Dataset.from_list([
    {'anchor': 'SQL injection', 'positive': 'Injection protection'},
    {'anchor': 'XSS attack', 'positive': 'Cross-site scripting defense'},
    {'anchor': 'Buffer overflow', 'positive': 'Memory safety'},
    {'anchor': 'Auth bypass', 'positive': 'Authentication control'},
])
with tempfile.TemporaryDirectory() as td:
    model = train_model(config, ds, Path(td))
    emb = model.encode(['test input'], normalize_embeddings=True)
    print(f'Output shape: {emb.shape}')  # (1, 1024)
    print('Smoke test PASSED')
"
```

- [ ] **Step 5: Commit**

```bash
git add tract/training/config.py tract/training/loop.py tests/test_training_loop.py
git commit -m "feat: LoRA training loop with SentenceTransformerTrainer and checkpointing"
```

---

### Task 7: LOFO evaluation harness

**What:** Implement the evaluation pipeline: fold-stratified micro-average bootstrap CIs, paired bootstrap for comparisons, Benjamini-Hochberg FDR control, per-framework soft floors, and covered/uncovered hub split metrics.

**Files:**
- Create: `tract/training/evaluate.py`, `tests/test_evaluate.py`

**Reference:** Design spec Section 5 (evaluation), especially 5.2 (metrics), 5.3 (gate), 5.4 (baselines). Also `scripts/phase0/common.py:456-686` for scoring functions to reuse.

- [ ] **Step 1: Write failing tests**

Create `tests/test_evaluate.py`:

```python
"""Tests for LOFO evaluation harness."""
from __future__ import annotations

import numpy as np
import pytest


class TestFoldStratifiedBootstrap:
    """Test fold-stratified micro-average bootstrap CIs."""

    def test_preserves_fold_sizes(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci
        fold_hit1s = [
            np.array([1.0, 0.0, 1.0, 1.0, 0.0]),  # fold 1: n=5
            np.array([0.0, 1.0, 1.0]),                # fold 2: n=3
        ]
        result = fold_stratified_bootstrap_ci(fold_hit1s, n_resamples=1000, seed=42)
        assert "mean" in result
        assert "ci_low" in result
        assert "ci_high" in result
        assert result["ci_low"] <= result["mean"] <= result["ci_high"]

    def test_perfect_scores_give_tight_ci(self) -> None:
        from tract.training.evaluate import fold_stratified_bootstrap_ci
        fold_hit1s = [np.ones(10), np.ones(5)]
        result = fold_stratified_bootstrap_ci(fold_hit1s, n_resamples=1000, seed=42)
        assert result["mean"] == 1.0
        assert result["ci_low"] == 1.0


class TestPairedBootstrapDelta:
    """Test paired bootstrap for method comparison."""

    def test_positive_delta_detected(self) -> None:
        from tract.training.evaluate import paired_bootstrap_delta
        fold_a = [np.array([0.0, 0.0, 0.0, 0.0, 0.0])]
        fold_b = [np.array([1.0, 1.0, 1.0, 1.0, 1.0])]
        result = paired_bootstrap_delta(fold_a, fold_b, n_resamples=1000, seed=42)
        assert result["delta_mean"] > 0
        assert result["ci_low"] > 0

    def test_no_difference_includes_zero(self) -> None:
        from tract.training.evaluate import paired_bootstrap_delta
        rng = np.random.default_rng(99)
        data = rng.binomial(1, 0.5, size=50).astype(float)
        fold_a = [data]
        fold_b = [data]
        result = paired_bootstrap_delta(fold_a, fold_b, n_resamples=1000, seed=42)
        assert result["delta_mean"] == 0.0
        assert result["ci_low"] == 0.0
        assert result["ci_high"] == 0.0


class TestBenjaminiHochberg:
    """Test BH FDR control."""

    def test_corrects_multiple_pvalues(self) -> None:
        from tract.training.evaluate import benjamini_hochberg
        p_values = [0.01, 0.03, 0.05, 0.20, 0.50]
        rejected, adjusted = benjamini_hochberg(p_values, q=0.10)
        assert isinstance(rejected, list)
        assert len(rejected) == len(p_values)
        assert all(isinstance(r, bool) for r in rejected)

    def test_no_rejections_for_high_pvalues(self) -> None:
        from tract.training.evaluate import benjamini_hochberg
        p_values = [0.50, 0.60, 0.70, 0.80, 0.90]
        rejected, adjusted = benjamini_hochberg(p_values, q=0.10)
        assert not any(rejected)


class TestSoftFloorCheck:
    """Test per-framework soft floor enforcement."""

    def test_passes_when_above_floor(self) -> None:
        from tract.training.evaluate import check_soft_floors
        per_fold_deltas = {
            "MITRE ATLAS": {"delta_mean": 0.05, "ci_low": -0.02, "ci_high": 0.12},
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 0

    def test_detects_violation(self) -> None:
        from tract.training.evaluate import check_soft_floors
        per_fold_deltas = {
            "MITRE ATLAS": {"delta_mean": -0.10, "ci_low": -0.15, "ci_high": -0.05},
        }
        violations = check_soft_floors(per_fold_deltas)
        assert len(violations) == 1
        assert "MITRE ATLAS" in violations


class TestEvaluateModel:
    """Test end-to-end evaluation pipeline."""

    def test_returns_ranked_predictions(self) -> None:
        from tract.training.evaluate import rank_hubs_by_similarity
        query_emb = np.array([[0.5, 0.5, 0.0]])
        hub_embs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.4, 0.0],
        ])
        hub_ids = ["h1", "h2", "h3"]
        ranked = rank_hubs_by_similarity(query_emb[0], hub_embs, hub_ids)
        assert ranked[0] == "h3"  # most similar
        assert len(ranked) == 3
```

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL (module not found)

- [ ] **Step 2: Implement evaluate.py**

Create `tract/training/evaluate.py`:

```python
"""LOFO evaluation harness with statistical testing.

Implements:
- Fold-stratified micro-average bootstrap CIs (10K resamples)
- Paired bootstrap for method comparisons
- Benjamini-Hochberg FDR control at q=0.10
- Per-framework soft floor checks
- Covered/uncovered hub split metrics
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from scripts.phase0.common import (
    ndcg_at_k,
    reciprocal_rank,
    score_predictions,
)
from tract.config import (
    PHASE1B_BOOTSTRAP_CI_LEVEL,
    PHASE1B_BOOTSTRAP_N_RESAMPLES,
    PHASE1B_BOOTSTRAP_SEED,
    PHASE1B_SOFT_FLOOR_LARGE,
    PHASE1B_SOFT_FLOOR_NIST,
)

logger = logging.getLogger(__name__)

LARGE_FOLDS: dict[str, float] = {
    "MITRE ATLAS": PHASE1B_SOFT_FLOOR_LARGE,
    "OWASP AI Exchange": PHASE1B_SOFT_FLOOR_LARGE,
    "NIST AI 100-2": PHASE1B_SOFT_FLOOR_NIST,
}


def rank_hubs_by_similarity(
    query_emb: np.ndarray,
    hub_embs: np.ndarray,
    hub_ids: list[str],
) -> list[str]:
    """Rank hub IDs by cosine similarity to query embedding."""
    sims = query_emb @ hub_embs.T
    ranked_indices = np.argsort(sims)[::-1]
    return [hub_ids[i] for i in ranked_indices]


def evaluate_on_fold(
    model: SentenceTransformer,
    eval_items: list[Any],
    hub_ids: list[str],
    hub_texts: dict[str, str],
) -> tuple[dict[str, float], list[list[str]], list[float]]:
    """Evaluate a model on one LOFO fold.

    Returns:
        Tuple of (metrics dict, per-item predictions, per-item hit@1 indicators).
    """
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
        ranked = rank_hubs_by_similarity(q_emb[0], hub_embs, hub_ids)
        predictions.append(ranked)

    ground_truth = [item.ground_truth_hub_id for item in eval_items]
    metrics = score_predictions(predictions, ground_truth)

    hit1_indicators = [
        1.0 if pred and pred[0] == gt else 0.0
        for pred, gt in zip(predictions, ground_truth)
    ]

    return metrics, predictions, hit1_indicators


def fold_stratified_bootstrap_ci(
    fold_values: list[np.ndarray],
    n_resamples: int = PHASE1B_BOOTSTRAP_N_RESAMPLES,
    ci_level: float = PHASE1B_BOOTSTRAP_CI_LEVEL,
    seed: int = PHASE1B_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute fold-stratified micro-average bootstrap CI.

    For each bootstrap replicate:
    1. Resample items with replacement WITHIN each fold (preserving fold sizes)
    2. Concatenate resampled folds into a single pool
    3. Compute aggregate metric over the full pool

    This preserves the model-item assignment: each item is scored by the
    model that excluded its framework.
    """
    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples)

    for b in range(n_resamples):
        resampled_values: list[float] = []
        for fold_vals in fold_values:
            n = len(fold_vals)
            indices = rng.integers(0, n, size=n)
            resampled_values.extend(fold_vals[indices])
        boot_means[b] = np.mean(resampled_values)

    all_values = np.concatenate(fold_values)
    alpha = (1 - ci_level) / 2

    return {
        "mean": float(np.mean(all_values)),
        "ci_low": float(np.percentile(boot_means, 100 * alpha)),
        "ci_high": float(np.percentile(boot_means, 100 * (1 - alpha))),
        "n_total": int(len(all_values)),
    }


def paired_bootstrap_delta(
    fold_values_a: list[np.ndarray],
    fold_values_b: list[np.ndarray],
    n_resamples: int = PHASE1B_BOOTSTRAP_N_RESAMPLES,
    ci_level: float = PHASE1B_BOOTSTRAP_CI_LEVEL,
    seed: int = PHASE1B_BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute paired bootstrap CI for the difference (B - A).

    Per-item deltas within each fold, fold-stratified resampling.
    Pairing cancels item-level difficulty for reduced variance.
    """
    rng = np.random.default_rng(seed)
    boot_deltas = np.empty(n_resamples)

    for b in range(n_resamples):
        resampled_deltas: list[float] = []
        for va, vb in zip(fold_values_a, fold_values_b):
            n = len(va)
            indices = rng.integers(0, n, size=n)
            resampled_deltas.extend((vb[indices] - va[indices]))
        boot_deltas[b] = np.mean(resampled_deltas)

    all_a = np.concatenate(fold_values_a)
    all_b = np.concatenate(fold_values_b)
    alpha = (1 - ci_level) / 2

    p_value = float(np.mean(boot_deltas <= 0))

    return {
        "delta_mean": float(np.mean(all_b - all_a)),
        "ci_low": float(np.percentile(boot_deltas, 100 * alpha)),
        "ci_high": float(np.percentile(boot_deltas, 100 * (1 - alpha))),
        "p_value": p_value,
    }


def benjamini_hochberg(
    p_values: list[float],
    q: float = 0.10,
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg FDR control.

    Returns (rejected, adjusted_p_values).
    """
    m = len(p_values)
    if m == 0:
        return [], []

    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.empty(m)
    adjusted[-1] = sorted_p[-1]
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * m / (i + 1))

    adjusted = np.minimum(adjusted, 1.0)

    result_adjusted = np.empty(m)
    result_adjusted[sorted_indices] = adjusted

    rejected = [p <= q for p in result_adjusted]

    return rejected, result_adjusted.tolist()


def check_soft_floors(
    per_fold_deltas: dict[str, dict[str, float]],
) -> dict[str, str]:
    """Check per-framework soft floor constraints.

    Returns dict of framework -> violation message for any that fail.
    Only enforced for large folds (ATLAS n=65, OWASP-X n=64, NIST n=45).
    """
    violations: dict[str, str] = {}

    for framework, floor in LARGE_FOLDS.items():
        if framework not in per_fold_deltas:
            continue
        delta_info = per_fold_deltas[framework]
        ci_low = delta_info.get("ci_low", 0.0)
        if ci_low < floor:
            violations[framework] = (
                f"Soft floor violation: CI low={ci_low:.3f} < floor={floor:.3f}"
            )
            logger.warning("SOFT FLOOR VIOLATION: %s — %s", framework, violations[framework])

    return violations


def compute_covered_uncovered_split(
    eval_items: list[Any],
    predictions: list[list[str]],
    training_hub_ids: set[str],
) -> dict[str, dict[str, float]]:
    """Split metrics by covered (hub in training) vs uncovered hubs."""
    covered_preds: list[list[str]] = []
    covered_gt: list[str] = []
    uncovered_preds: list[list[str]] = []
    uncovered_gt: list[str] = []

    for item, pred in zip(eval_items, predictions):
        if item.ground_truth_hub_id in training_hub_ids:
            covered_preds.append(pred)
            covered_gt.append(item.ground_truth_hub_id)
        else:
            uncovered_preds.append(pred)
            uncovered_gt.append(item.ground_truth_hub_id)

    result: dict[str, dict[str, float]] = {}
    if covered_preds:
        result["covered"] = score_predictions(covered_preds, covered_gt)
        result["covered"]["n"] = len(covered_preds)
    if uncovered_preds:
        result["uncovered"] = score_predictions(uncovered_preds, uncovered_gt)
        result["uncovered"]["n"] = len(uncovered_preds)

    return result
```

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_evaluate.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tract/training/evaluate.py tests/test_evaluate.py
git commit -m "feat: LOFO evaluation harness with fold-stratified bootstrap and BH FDR"
```

---

### Task 8: Calibration

**What:** Temperature scaling of cosine similarities to calibrated probabilities. Global threshold at max-F1 for multi-label assignment.

**Files:**
- Create: `tract/training/calibrate.py`, `tests/test_calibrate.py`

**Reference:** Design spec Section 8 (calibration).

- [ ] **Step 1: Write failing tests**

Create `tests/test_calibrate.py`:

```python
"""Tests for temperature scaling calibration."""
from __future__ import annotations

import numpy as np
import pytest


class TestTemperatureScaling:
    """Test optimal temperature finding."""

    def test_finds_reasonable_temperature(self) -> None:
        from tract.training.calibrate import find_optimal_temperature
        rng = np.random.default_rng(42)
        n_samples, n_hubs = 20, 50
        similarities = rng.uniform(-0.5, 0.5, size=(n_samples, n_hubs))
        ground_truth_indices = rng.integers(0, n_hubs, size=n_samples)
        for i in range(n_samples):
            similarities[i, ground_truth_indices[i]] += 0.5

        temp = find_optimal_temperature(similarities, ground_truth_indices)
        assert 0.01 <= temp <= 5.0

    def test_calibrated_probabilities_sum_to_one(self) -> None:
        from tract.training.calibrate import calibrate_similarities
        sims = np.array([[0.3, 0.7, 0.1], [0.9, 0.2, 0.4]])
        probs = calibrate_similarities(sims, temperature=1.0)
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0], atol=1e-6)


class TestGlobalThreshold:
    """Test global threshold at max-F1."""

    def test_threshold_in_valid_range(self) -> None:
        from tract.training.calibrate import find_global_threshold
        rng = np.random.default_rng(42)
        n_samples, n_hubs = 30, 10
        similarities = rng.uniform(0, 1, size=(n_samples, n_hubs))
        ground_truth_indices = rng.integers(0, n_hubs, size=n_samples)
        for i in range(n_samples):
            similarities[i, ground_truth_indices[i]] += 0.3

        threshold = find_global_threshold(similarities, ground_truth_indices, temperature=1.0)
        assert 0.0 < threshold < 1.0
```

- [ ] **Step 2: Implement calibrate.py**

Create `tract/training/calibrate.py`:

```python
"""Temperature scaling calibration for cosine similarities.

Calibration is performed INSIDE the LOFO loop using the 10% AI
validation split (same split used for early stopping).
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.special import softmax

logger = logging.getLogger(__name__)


def calibrate_similarities(
    similarities: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Convert cosine similarities to calibrated probabilities.

    P(hub_i | control) = exp(sim_i / T) / sum(exp(sim_j / T))
    """
    scaled = similarities / temperature
    return softmax(scaled, axis=1)


def _negative_log_likelihood(
    similarities: np.ndarray,
    ground_truth_indices: np.ndarray,
    temperature: float,
) -> float:
    """Compute NLL for a given temperature."""
    probs = calibrate_similarities(similarities, temperature)
    log_probs = np.log(probs[np.arange(len(ground_truth_indices)), ground_truth_indices] + 1e-10)
    return -float(np.mean(log_probs))


def find_optimal_temperature(
    similarities: np.ndarray,
    ground_truth_indices: np.ndarray,
    t_min: float = 0.01,
    t_max: float = 5.0,
    n_grid: int = 200,
) -> float:
    """Find temperature T that minimizes NLL on validation set via grid search."""
    temperatures = np.linspace(t_min, t_max, n_grid)
    best_t = 1.0
    best_nll = float("inf")

    for t in temperatures:
        nll = _negative_log_likelihood(similarities, ground_truth_indices, t)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    logger.info("Optimal temperature: %.4f (NLL=%.4f, searched %d values in [%.2f, %.2f])",
                best_t, best_nll, n_grid, t_min, t_max)
    return best_t


def find_global_threshold(
    similarities: np.ndarray,
    ground_truth_indices: np.ndarray,
    temperature: float,
    n_thresholds: int = 200,
) -> float:
    """Find global threshold at max-F1 for multi-label assignment."""
    probs = calibrate_similarities(similarities, temperature)

    thresholds = np.linspace(0.001, 0.999, n_thresholds)
    best_f1 = 0.0
    best_threshold = 0.5

    for t in thresholds:
        tp = 0
        fp = 0
        fn = 0
        for i in range(len(ground_truth_indices)):
            gt_idx = ground_truth_indices[i]
            for j in range(probs.shape[1]):
                predicted = probs[i, j] >= t
                is_true = j == gt_idx
                if predicted and is_true:
                    tp += 1
                elif predicted and not is_true:
                    fp += 1
                elif not predicted and is_true:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(t)

    logger.info("Global threshold: %.4f (F1=%.4f)", best_threshold, best_f1)
    return best_threshold
```

**Note:** Add `scipy` to `requirements.txt` if not already present.

- [ ] **Step 3: Run tests**

```bash
python -m pytest tests/test_calibrate.py -v
```

- [ ] **Step 4: Commit**

```bash
git add tract/training/calibrate.py tests/test_calibrate.py
git commit -m "feat: temperature scaling calibration with max-F1 global threshold"
```

---

### Task 9: Orchestration + CLI

**What:** Multi-fold, multi-config experiment runner that ties together all pipeline stages. CLI entrypoint for launching training runs.

**Files:**
- Create: `tract/training/orchestrate.py`, `scripts/phase1b/train.py`

**Reference:** Design spec Sections 6.2-6.3 (orchestration), 10 (output artifacts). `scripts/phase0/common.py:719-861` for WandB patterns.

- [ ] **Step 1: Implement orchestrate.py**

Create `tract/training/orchestrate.py`:

```python
"""Multi-fold, multi-config experiment runner.

Orchestrates the full Phase 1B pipeline:
1. Load and filter training data
2. For each LOFO fold:
   a. Build firewalled hub texts
   b. Generate training pairs with hard negatives
   c. Split AI data 90/10 for early stopping + calibration
   d. Train LoRA model
   e. Evaluate on held-out framework
   f. Calibrate on validation split
3. Aggregate across folds with fold-stratified bootstrap CIs
4. Compare against Phase 0R baselines with paired bootstrap
5. Log everything to WandB

Note: Fold parallelism uses multiprocessing (each fold needs
its own CUDA context), not asyncio.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts.phase0.common import (
    AI_FRAMEWORK_NAMES,
    EvalItem,
    LOFOFold,
    build_evaluation_corpus,
    build_hierarchy,
    build_lofo_folds,
    load_curated_links,
    load_opencre_cres,
    score_predictions,
)
from tract.config import (
    PHASE1B_MODELS_DIR,
    PHASE1B_RESULTS_DIR,
    PHASE1B_WANDB_PROJECT,
    PROCESSED_DIR,
)
from tract.hierarchy import CREHierarchy
from tract.io import atomic_write_json, load_json
from tract.training.calibrate import find_global_threshold, find_optimal_temperature
from tract.training.config import TrainingConfig
from tract.training.data import (
    build_training_pairs,
    pairs_to_dataset,
)
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.evaluate import (
    check_soft_floors,
    compute_covered_uncovered_split,
    evaluate_on_fold,
    fold_stratified_bootstrap_ci,
    paired_bootstrap_delta,
)
from tract.training.firewall import assert_firewall, build_all_hub_texts
from tract.training.loop import save_checkpoint, train_model

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _split_ai_validation(
    eval_items_by_framework: dict[str, list],
    held_out: str,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> tuple[list, list]:
    """Split non-held-out AI framework items 90/10 for early stopping + calibration.

    Returns (train_ai_items, val_ai_items).
    """
    rng = np.random.default_rng(seed)
    train_items: list = []
    val_items: list = []

    for fw_name, items in eval_items_by_framework.items():
        if fw_name == held_out:
            continue
        n_val = max(1, int(len(items) * val_fraction))
        indices = rng.permutation(len(items))
        val_indices = set(indices[:n_val])
        for i, item in enumerate(items):
            if i in val_indices:
                val_items.append(item)
            else:
                train_items.append(item)

    return train_items, val_items


def run_single_fold(
    config: TrainingConfig,
    held_out_framework: str,
    tiered_links: list,
    hierarchy: CREHierarchy,
    eval_items: list,
    hub_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Train and evaluate one LOFO fold. Returns fold result dict."""
    logger.info("=== FOLD: %s ===", held_out_framework)
    fold_start = time.time()

    # Build firewalled hub texts
    include_desc = config.hub_rep_format == "path+name+desc"
    descriptions = None
    if include_desc:
        desc_data = load_json(PROCESSED_DIR / "hub_descriptions_reviewed.json")
        descriptions = {
            hid: d["description"]
            for hid, d in desc_data.get("descriptions", {}).items()
        }

    hub_texts = build_all_hub_texts(
        hierarchy,
        excluded_framework=held_out_framework,
        include_description=include_desc,
        descriptions=descriptions,
    )

    # Firewall assertion
    assert_firewall(hub_texts, eval_items, held_out_framework)

    # Build training pairs
    pairs = build_training_pairs(tiered_links, hub_texts, excluded_framework=held_out_framework)

    # Convert to dataset with hard negatives
    dataset = pairs_to_dataset(pairs, hierarchy, hub_texts, n_hard_negatives=config.hard_negatives)

    # Train
    fold_output = output_dir / f"fold_{held_out_framework.replace(' ', '_')}"
    fold_output.mkdir(parents=True, exist_ok=True)

    model = train_model(config, dataset, fold_output)

    # Evaluate
    metrics, predictions, hit1_indicators = evaluate_on_fold(
        model, eval_items, hub_ids, hub_texts,
    )
    logger.info("Fold %s: hit@1=%.3f, hit@5=%.3f, MRR=%.3f, NDCG@10=%.3f",
                held_out_framework, metrics["hit_at_1"], metrics["hit_at_5"],
                metrics["mrr"], metrics["ndcg_at_10"])

    # Save checkpoint
    save_checkpoint(model, config, metrics, fold_output / "model", _get_git_sha())

    # Save predictions
    pred_data = []
    for item, pred in zip(eval_items, predictions):
        pred_data.append({
            "control_text": item.control_text,
            "ground_truth_hub_id": item.ground_truth_hub_id,
            "predicted_top10": pred[:10],
            "framework": item.framework_name,
        })
    atomic_write_json(pred_data, fold_output / "predictions.json")
    atomic_write_json(metrics, fold_output / "metrics.json")

    elapsed = time.time() - fold_start
    logger.info("Fold %s complete in %.1fs", held_out_framework, elapsed)

    return {
        "held_out_framework": held_out_framework,
        "metrics": metrics,
        "predictions": predictions,
        "hit1_indicators": hit1_indicators,
        "n_eval_items": len(eval_items),
        "n_training_pairs": len(pairs),
        "elapsed_s": elapsed,
    }


def run_experiment(config: TrainingConfig) -> dict[str, Any]:
    """Run a full LOFO experiment with the given config."""
    logger.info("Starting experiment: %s", config.name)
    exp_start = time.time()

    # Load data
    tiered_links, raw_hash = load_and_filter_curated_links()

    # Load hierarchy (Pydantic production model)
    hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / "cre_hierarchy.json"))
    hub_ids = sorted(hierarchy.hubs.keys())

    # Build evaluation corpus using Phase 0 infrastructure
    cres = load_opencre_cres()
    tree = build_hierarchy(cres)
    links = load_curated_links()
    corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})

    # Group eval items by framework
    eval_by_fw: dict[str, list] = {}
    for item in corpus:
        eval_by_fw.setdefault(item.framework_name, []).append(item)

    # Run each fold
    output_dir = PHASE1B_RESULTS_DIR / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[dict[str, Any]] = []
    for fw_name in sorted(AI_FRAMEWORK_NAMES):
        fw_items = eval_by_fw.get(fw_name, [])
        if not fw_items:
            logger.warning("No eval items for %s, skipping", fw_name)
            continue

        result = run_single_fold(
            config=config,
            held_out_framework=fw_name,
            tiered_links=tiered_links,
            hierarchy=hierarchy,
            eval_items=fw_items,
            hub_ids=hub_ids,
            output_dir=output_dir,
        )
        fold_results.append(result)

    # Aggregate with fold-stratified bootstrap
    fold_hit1s = [np.array(r["hit1_indicators"]) for r in fold_results]
    aggregate = fold_stratified_bootstrap_ci(fold_hit1s)
    logger.info("AGGREGATE hit@1: %.3f [%.3f, %.3f]",
                aggregate["mean"], aggregate["ci_low"], aggregate["ci_high"])

    # Save aggregate results
    experiment_result = {
        "config": config.to_dict(),
        "aggregate_hit1": aggregate,
        "per_fold": {r["held_out_framework"]: r["metrics"] for r in fold_results},
        "raw_hash": raw_hash,
        "git_sha": _get_git_sha(),
        "total_elapsed_s": time.time() - exp_start,
    }
    atomic_write_json(experiment_result, output_dir / "aggregate_metrics.json")

    return experiment_result
```

- [ ] **Step 2: Write scripts/phase1b/train.py CLI**

Create `scripts/phase1b/train.py`:

```python
"""CLI entrypoint for Phase 1B training.

Usage:
    python -m scripts.phase1b.train
    python -m scripts.phase1b.train --name my_experiment --lora-rank 32
    python -m scripts.phase1b.train --training-data ai-only --epochs 30
"""
from __future__ import annotations

import argparse
import logging
import sys

from tract.training.config import TrainingConfig
from tract.training.orchestrate import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1B model training")
    parser.add_argument("--name", type=str, default="phase1b_primary",
                        help="Experiment name (used for output dir and WandB)")
    parser.add_argument("--training-data", type=str, default="joint-tempscaled",
                        choices=["joint-tempscaled", "ai-only", "joint-flat", "two-stage-transfer"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hard-negatives", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--hub-rep", type=str, default="path+name",
                        choices=["path+name", "path+name+desc", "path+name+standards"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = TrainingConfig(
        name=args.name,
        training_data=args.training_data,
        lora_rank=args.lora_rank,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        hard_negatives=args.hard_negatives,
        sampling_temperature=args.temperature,
        hub_rep_format=args.hub_rep,
        seed=args.seed,
    )

    logger.info("Config: %s", config.to_dict())
    result = run_experiment(config)

    agg = result["aggregate_hit1"]
    logger.info("=" * 60)
    logger.info("EXPERIMENT COMPLETE: %s", config.name)
    logger.info("  Aggregate hit@1: %.3f [%.3f, %.3f]",
                agg["mean"], agg["ci_low"], agg["ci_high"])

    gate_pass = agg["ci_low"] > 0 and agg["mean"] > 0.516
    if gate_pass:
        logger.info("  GATE: PASS (hit@1 > 0.516, CI excludes 0)")
    else:
        logger.info("  GATE: FAIL (hit@1=%.3f, CI_low=%.3f)", agg["mean"], agg["ci_low"])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: End-to-end smoke test (1 fold, 1 epoch)**

```bash
python -c "
from tract.training.config import TrainingConfig
from tract.training.orchestrate import run_single_fold
from tract.training.data_quality import load_and_filter_curated_links
from tract.training.firewall import build_all_hub_texts
from tract.hierarchy import CREHierarchy
from tract.io import load_json
from tract.config import PROCESSED_DIR
from scripts.phase0.common import AI_FRAMEWORK_NAMES, build_evaluation_corpus, load_curated_links, load_opencre_cres, build_hierarchy
from pathlib import Path
import tempfile

# Minimal config for smoke test
config = TrainingConfig(name='smoke', lora_rank=4, batch_size=8, max_epochs=1, max_seq_length=64)

# Load data
tiered_links, _ = load_and_filter_curated_links()
hierarchy = CREHierarchy.model_validate(load_json(PROCESSED_DIR / 'cre_hierarchy.json'))
hub_ids = sorted(hierarchy.hubs.keys())

cres = load_opencre_cres()
tree = build_hierarchy(cres)
links = load_curated_links()
corpus = build_evaluation_corpus(links, AI_FRAMEWORK_NAMES, {})
atlas_items = [i for i in corpus if i.framework_name == 'MITRE ATLAS']

with tempfile.TemporaryDirectory() as td:
    result = run_single_fold(config, 'MITRE ATLAS', tiered_links, hierarchy, atlas_items, hub_ids, Path(td))
    print(f'Smoke test PASSED: hit@1={result[\"metrics\"][\"hit_at_1\"]:.3f}')
"
```

- [ ] **Step 4: Commit**

```bash
git add tract/training/orchestrate.py scripts/phase1b/train.py
git commit -m "feat: LOFO experiment orchestrator with CLI entrypoint"
```

---

### Task 10: Run primary training experiment

**This is the real training run — not a code task.**

- [ ] **Step 1: Run primary config on H100 (or Orin)**

```bash
python -m scripts.phase1b.train \
    --name phase1b_joint_tempscaled_v1 \
    --training-data joint-tempscaled \
    --lora-rank 16 \
    --batch-size 64 \
    --lr 5e-4 \
    --epochs 20 \
    --hard-negatives 3 \
    --temperature 2.0 \
    2>&1 | tee results/phase1b/primary_run.log
```

**Estimated time:** 5 folds × ~5-10 min/fold (H100) = 25-50 min total. On Orin: 5 × ~30-60 min = 2.5-5 hours.

- [ ] **Step 2: Check gate results**

Read `results/phase1b/phase1b_joint_tempscaled_v1/aggregate_metrics.json`:
- `aggregate_hit1.mean` > 0.516?
- `aggregate_hit1.ci_low` > 0? (paired delta vs baseline)
- Per-framework soft floors pass?

- [ ] **Step 3: If gate passes, save best model**

```bash
cp -r results/phase1b/phase1b_joint_tempscaled_v1 models/phase1b/best_model
```

- [ ] **Step 4: If gate fails, try escalation path**

1. Full fine-tuning: `--lora-rank 0`
2. Different LR: `--lr 1e-3` or `--lr 1e-4`
3. More epochs: `--epochs 30`
4. If still failing: consider retrieve+rerank fallback (design spec Section 12.1)

---

### Task 11: Ablation suite

**Prerequisite:** Primary config passes the gate.

**What:** Implement and run ablations one variable at a time, each compared to primary via paired bootstrap + BH FDR.

**Files:**
- Create: `scripts/phase1b/ablation.py`

**Reference:** Design spec Section 7 (ablation suite).

- [ ] **Step 1: Write ablation.py**

Create `scripts/phase1b/ablation.py`:

```python
"""Ablation sweep runner for Phase 1B.

Generates configs for each ablation variable, runs experiments,
and compares against primary via paired bootstrap + BH FDR.

Usage:
    python -m scripts.phase1b.ablation --ablation A1
    python -m scripts.phase1b.ablation --all
    python -m scripts.phase1b.ablation --list
"""
from __future__ import annotations

import argparse
import logging

from tract.training.config import TrainingConfig
from tract.training.orchestrate import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_ablation_configs(ablation_id: str) -> list[TrainingConfig]:
    """Generate configs for a specific ablation."""
    configs: dict[str, list[TrainingConfig]] = {
        "A1": [
            TrainingConfig(name="ablation_A1_ai_only", training_data="ai-only"),
            TrainingConfig(name="ablation_A1_joint_flat", training_data="joint-flat", sampling_temperature=float("inf")),
            TrainingConfig(name="ablation_A1_two_stage", training_data="two-stage-transfer"),
        ],
        "A2": [
            TrainingConfig(name="ablation_A2_neg0", hard_negatives=0),
            TrainingConfig(name="ablation_A2_neg1", hard_negatives=1),
            TrainingConfig(name="ablation_A2_neg5", hard_negatives=5),
        ],
        "A3": [
            TrainingConfig(name="ablation_A3_standards", hub_rep_format="path+name+standards"),
        ],
        "A4": [
            TrainingConfig(name="ablation_A4_ep5", max_epochs=5),
            TrainingConfig(name="ablation_A4_ep10", max_epochs=10),
            TrainingConfig(name="ablation_A4_ep30", max_epochs=30),
        ],
        "A5": [
            TrainingConfig(name="ablation_A5_lr1e4", learning_rate=1e-4),
            TrainingConfig(name="ablation_A5_lr1e3", learning_rate=1e-3),
        ],
        "A6": [
            TrainingConfig(name="ablation_A6_descriptions", hub_rep_format="path+name+desc"),
        ],
        "A7": [
            TrainingConfig(name="ablation_A7_rank4", lora_rank=4, lora_alpha=8),
            TrainingConfig(name="ablation_A7_rank8", lora_rank=8, lora_alpha=16),
            TrainingConfig(name="ablation_A7_rank32", lora_rank=32, lora_alpha=64),
        ],
        "A8": [
            TrainingConfig(name="ablation_A8_supcon"),
        ],
        "A9": [
            TrainingConfig(name="ablation_A9_full_ft", lora_rank=0),
        ],
        "A10": [
            TrainingConfig(name="ablation_A10_full_text", control_text_source="full_parsed"),
        ],
    }

    if ablation_id not in configs:
        raise ValueError(f"Unknown ablation: {ablation_id}. Valid: {sorted(configs.keys())}")

    return configs[ablation_id]


ABLATION_ORDER: list[str] = ["A1", "A6", "A10", "A2", "A5", "A7", "A4", "A8", "A9", "A3"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1B ablation sweep")
    parser.add_argument("--ablation", type=str, help="Run a specific ablation (A1-A10)")
    parser.add_argument("--all", action="store_true", help="Run all ablations in order")
    parser.add_argument("--list", action="store_true", help="List available ablations")
    args = parser.parse_args()

    if args.list:
        for aid in ABLATION_ORDER:
            configs = get_ablation_configs(aid)
            names = [c.name for c in configs]
            print(f"  {aid}: {', '.join(names)}")
        return

    ablations_to_run = ABLATION_ORDER if args.all else [args.ablation]

    for aid in ablations_to_run:
        configs = get_ablation_configs(aid)
        for config in configs:
            logger.info("Running ablation %s: %s", aid, config.name)
            run_experiment(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run ablations on-demand (only if primary hits 0.60, prioritize A1 and A6)**

```bash
python -m scripts.phase1b.ablation --ablation A1
python -m scripts.phase1b.ablation --ablation A6
```

- [ ] **Step 3: Commit**

```bash
git add scripts/phase1b/ablation.py
git commit -m "feat: ablation sweep runner with config generation"
```

---

### Task 12: LLM prediction comparison diagnostic

**What:** Compare fine-tuned model's per-item predictions against Phase 0R Sonnet 3-shot predictions. Zero-cost diagnostic (no API calls unless Sonnet predictions need re-running).

**Files:**
- Create: `scripts/phase1b/llm_comparison.py`

**Reference:** Design spec Section 0.1 (LLM comparison).

- [ ] **Step 1: Write llm_comparison.py**

Create `scripts/phase1b/llm_comparison.py`:

```python
"""Post-training diagnostic: compare model predictions vs Phase 0R Sonnet.

Categorizes each of the 197 eval items into:
- Both correct: High-confidence assignments
- Both wrong: Genuinely hard examples
- Model correct, Sonnet wrong: Model learned something Sonnet missed
- Model wrong, Sonnet correct: Model's specific failure modes

Usage:
    python -m scripts.phase1b.llm_comparison --model-results results/phase1b/phase1b_primary/
    python -m scripts.phase1b.llm_comparison --model-results results/phase1b/phase1b_primary/ --sonnet-results results/phase0/exp6_fewshot_sonnet/
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from tract.io import load_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model_predictions(results_dir: Path) -> dict[str, dict]:
    """Load per-item predictions from all fold directories."""
    predictions: dict[str, dict] = {}
    for fold_dir in sorted(results_dir.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold_"):
            continue
        pred_file = fold_dir / "predictions.json"
        if not pred_file.exists():
            continue
        preds = load_json(pred_file)
        for item in preds:
            key = f"{item['framework']}:{item.get('control_text', '')[:50]}"
            predictions[key] = item
    return predictions


def compare_predictions(
    model_preds: dict[str, dict],
    sonnet_preds: dict[str, dict],
) -> dict[str, list[dict]]:
    """Compare model vs Sonnet predictions per item."""
    categories: dict[str, list[dict]] = {
        "both_correct": [],
        "both_wrong": [],
        "model_only_correct": [],
        "sonnet_only_correct": [],
    }

    for key, model_item in model_preds.items():
        gt = model_item["ground_truth_hub_id"]
        model_top1 = model_item["predicted_top10"][0] if model_item["predicted_top10"] else None

        sonnet_item = sonnet_preds.get(key)
        if sonnet_item is None:
            continue
        sonnet_top1 = sonnet_item.get("predicted_top10", [None])[0]

        model_correct = model_top1 == gt
        sonnet_correct = sonnet_top1 == gt

        entry = {
            "key": key,
            "ground_truth": gt,
            "model_prediction": model_top1,
            "sonnet_prediction": sonnet_top1,
        }

        if model_correct and sonnet_correct:
            categories["both_correct"].append(entry)
        elif not model_correct and not sonnet_correct:
            categories["both_wrong"].append(entry)
        elif model_correct and not sonnet_correct:
            categories["model_only_correct"].append(entry)
        else:
            categories["sonnet_only_correct"].append(entry)

    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Model vs Sonnet prediction comparison")
    parser.add_argument("--model-results", type=Path, required=True)
    parser.add_argument("--sonnet-results", type=Path, default=None)
    args = parser.parse_args()

    model_preds = load_model_predictions(args.model_results)
    logger.info("Loaded %d model predictions", len(model_preds))

    if args.sonnet_results and args.sonnet_results.exists():
        sonnet_preds = load_model_predictions(args.sonnet_results)
    else:
        logger.warning("No Sonnet predictions provided. Re-run Phase 0R exp6 if needed (~$18).")
        return

    categories = compare_predictions(model_preds, sonnet_preds)

    total = sum(len(v) for v in categories.values())
    logger.info("=" * 60)
    logger.info("MODEL vs SONNET COMPARISON (%d items)", total)
    for cat, items in categories.items():
        logger.info("  %s: %d (%.1f%%)", cat, len(items), 100 * len(items) / total if total else 0)
    logger.info("=" * 60)

    output_path = args.model_results / "llm_comparison.json"
    from tract.io import atomic_write_json
    atomic_write_json(
        {cat: items for cat, items in categories.items()},
        output_path,
    )
    logger.info("Saved comparison to %s", output_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run comparison (after primary training completes)**

```bash
python -m scripts.phase1b.llm_comparison \
    --model-results results/phase1b/phase1b_joint_tempscaled_v1/ \
    --sonnet-results results/phase0/exp6_fewshot_sonnet/
```

- [ ] **Step 3: Commit**

```bash
git add scripts/phase1b/llm_comparison.py
git commit -m "feat: LLM prediction comparison diagnostic"
```

---

## Dependency Graph

```
Task 1 (alpha MVE) ─── GATE ───┐
                                │
Task 2 (config constants) ◄─────┤
Task 3 (data quality) ◄────────┤
Task 4 (firewall) ◄────────────┤
                                │
Task 5 (training data) ◄── Tasks 3, 4
Task 6 (training loop) ◄── Task 2
Task 7 (evaluation) ◄── Task 6
Task 8 (calibration) ◄── Task 7
                                │
Task 9 (orchestration) ◄── Tasks 5, 6, 7, 8
Task 10 (primary run) ◄── Task 9
Task 11 (ablations) ◄── Task 10 (on-demand)
Task 12 (LLM comparison) ◄── Task 10
```

Tasks 2-4 can be implemented in parallel after the alpha gate passes. Tasks 5 and 6 depend on their respective prerequisites but are independent of each other. Everything converges at Task 9 (orchestration).

---

## Pre-Flight Checklist

Before starting Task 1:
- [ ] Confirm CUDA is available on the target machine
- [ ] Confirm sufficient disk space for BGE-large model weights (~1.3GB)
- [ ] Confirm `data/training/hub_links_curated.jsonl` exists (4,405 lines)
- [ ] Confirm `data/raw/opencre/opencre_all_cres.json` exists
- [ ] Confirm `data/processed/cre_hierarchy.json` exists
- [ ] Read design spec Sections 0.2 and 3.3 before writing any code
