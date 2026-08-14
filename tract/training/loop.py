"""LoRA training loop for contrastive fine-tuning.

Uses SentenceTransformerTrainer (modern API, not legacy model.fit())
with PEFT LoRA adapters and MNRL loss.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers

from tract.training.config import TrainingConfig
from tract.training.data import HubAwareTemperatureSampler

logger = logging.getLogger(__name__)

# Probes for the save/reload check in save_checkpoint. Mirrors the pattern in
# tract/publish/merge.py. Content is arbitrary; only the embeddings matter.
CHECKPOINT_PROBE_TEXTS = [
    "Implement access controls for AI model training pipelines",
    "Data encryption at rest using AES-256",
    "Regularly audit AI system outputs for bias and fairness",
]
# Same bar the merge path uses for "these are the same model". Float32 reload
# noise sits many orders of magnitude below this; a lost adapter lands near 0.4.
CHECKPOINT_COSINE_THRESHOLD = 0.999


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
        # Do NOT assign to model[0].auto_model. sentence-transformers 5.7 made
        # auto_model a read-only property aliasing `.model`, and
        # nn.Module.__setattr__ accepts an assignment to a property name without
        # raising -- it files the value into _modules, which property reads never
        # consult. Training still worked (get_peft_model mutates the module tree
        # in place) but save_pretrained ran against the unwrapped backbone and
        # wrote peft-mangled keys with no adapter_config.json, so the checkpoint
        # reloaded with randomly initialised q/k/v. Measured cosine on reload was
        # 0.38. add_adapter is the supported surface and exists on
        # PeftAdapterMixin in every sentence-transformers version we pin.
        model.add_adapter(lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        # A silent no-op here is the exact failure this migration fixes, so make
        # "no adapter attached" and "base model not frozen" both loud.
        if trainable == 0:
            raise RuntimeError(
                "LoRA adapter did not attach: no trainable parameters after "
                f"add_adapter with target_modules={config.lora_target_modules}. "
                "Check that the target module names match this architecture."
            )
        if trainable >= total:
            raise RuntimeError(
                f"LoRA adapter attached but the base model was not frozen: "
                f"{trainable:,} trainable of {total:,} total. This would train "
                "every weight rather than the adapter."
            )
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    model = load_model_with_lora(config)
    loss = MultipleNegativesRankingLoss(model)

    use_custom_sampler = "hub_id" in train_dataset.column_names

    if use_custom_sampler:
        anchor_keys = (
            list(train_dataset["anchor_key"])
            if "anchor_key" in train_dataset.column_names
            else None
        )
        HubAwareTemperatureSampler.set_metadata(
            hub_ids=list(train_dataset["hub_id"]),
            is_ai=list(train_dataset["is_ai"]),
            anchor_keys=anchor_keys,
        )
        meta_cols = [c for c in ["hub_id", "is_ai", "anchor_key"]
                     if c in train_dataset.column_names]
        train_dataset = train_dataset.remove_columns(meta_cols)

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
        batch_sampler=HubAwareTemperatureSampler if use_custom_sampler else BatchSamplers.BATCH_SAMPLER,
    )

    try:
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
    finally:
        if use_custom_sampler:
            HubAwareTemperatureSampler.clear_metadata()

    return model


def verify_checkpoint_roundtrip(
    model: SentenceTransformer,
    saved_dir: Path,
    threshold: float = CHECKPOINT_COSINE_THRESHOLD,
) -> float:
    """Prove the saved directory reproduces the in-memory model's embeddings.

    Folds evaluate the live model and then save it, so an artifact that does not
    match the evaluated model reports a healthy score for something that was
    never written to disk. That is exactly how a lost LoRA adapter stayed
    invisible. Reload what was just written and compare.

    Returns the minimum per-probe cosine. Raises RuntimeError below threshold.
    """
    reference = model.encode(
        CHECKPOINT_PROBE_TEXTS, normalize_embeddings=True, show_progress_bar=False,
    )
    reloaded = SentenceTransformer(str(saved_dir))
    reloaded.max_seq_length = model.max_seq_length
    actual = reloaded.encode(
        CHECKPOINT_PROBE_TEXTS, normalize_embeddings=True, show_progress_bar=False,
    )

    cosines = np.sum(reference * actual, axis=1)
    min_cosine = float(np.min(cosines))
    if min_cosine < threshold:
        raise RuntimeError(
            f"Checkpoint at {saved_dir} does not reproduce the in-memory model: "
            f"min cosine {min_cosine:.6f} < {threshold}. Per-probe cosines: "
            f"{cosines.tolist()}. The evaluated model and the saved artifact are "
            "not the same model, so any metric measured against the live model "
            "does not describe this checkpoint."
        )
    logger.info("Checkpoint round-trip verified: min cosine = %.6f", min_cosine)
    return min_cosine


def save_checkpoint(
    model: SentenceTransformer,
    config: TrainingConfig,
    metrics: dict[str, Any],
    output_dir: Path,
    git_sha: str = "unknown",
    verify_roundtrip: bool = True,
) -> Path:
    """Save model checkpoint with full metadata for reproducibility.

    Args:
        verify_roundtrip: Reload the saved model and assert it reproduces the
            in-memory embeddings. Costs one model load. Leave enabled for any
            run whose metrics will be reported.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = output_dir / "model"
    model.save(str(model_dir))

    if verify_roundtrip:
        verify_checkpoint_roundtrip(model, model_dir)

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
