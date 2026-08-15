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


def _assert_adapter_learned(model: Any, config: TrainingConfig) -> None:
    """Fail if training ran but the adapter never moved.

    PEFT initialises every lora_B to zeros, so the adapter is the identity
    until a gradient reaches it. If lora_B is still all zeros after training,
    the run produced the base model wearing an adapter that does nothing,
    and every downstream metric would describe the base model while the
    record claimed a fine-tune. Loss going down does not rule this out --
    with a frozen backbone and no gradient path there would be no loss curve
    at all, but a partially connected graph can still produce one.

    This matters most with gradient checkpointing: a frozen backbone plus the
    default reentrant implementation means the checkpointed segment sees no
    input requiring grad, and torch drops the gradient silently rather than
    raising. use_reentrant=False avoids that; this asserts it worked.
    """
    import torch

    lora_b = [
        (name, param) for name, param in model.named_parameters()
        if "lora_B" in name
    ]
    if not lora_b:
        if config.lora_rank == 0:
            return          # full fine-tune arm has no adapter to check
        raise RuntimeError(
            "No lora_B parameters found after training, so the adapter is not "
            "attached to the model that was trained."
        )

    moved = [name for name, param in lora_b if torch.any(param != 0).item()]
    if not moved:
        raise RuntimeError(
            f"All {len(lora_b)} lora_B tensors are still zero after training. "
            f"PEFT initialises them to zero, so the adapter is the identity "
            f"and this run produced the base model. No gradient reached the "
            f"adapter"
            + (
                "; gradient_checkpointing is on, which drops gradients "
                "silently when the checkpointed segment has no input "
                "requiring grad."
                if config.gradient_checkpointing else "."
            )
        )
    logger.info(
        "Adapter learned: %d/%d lora_B tensors are non-zero", len(moved), len(lora_b),
    )


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
            strata=(
                list(train_dataset["branch"])
                if "branch" in train_dataset.column_names else None
            ),
            temperature=config.sampling_temperature,
            strata_temperature=config.branch_balance_temperature,
        )
        meta_cols = [c for c in ["hub_id", "is_ai", "anchor_key", "branch"]
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
        gradient_checkpointing=config.gradient_checkpointing,
        # LoRA freezes the backbone, so with the default reentrant
        # implementation the checkpointed segments see no input requiring
        # grad and torch silently produces no gradients for the adapter.
        gradient_checkpointing_kwargs=(
            {"use_reentrant": False} if config.gradient_checkpointing else None
        ),
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
        _assert_adapter_learned(model, config)
    finally:
        if use_custom_sampler:
            HubAwareTemperatureSampler.clear_metadata()

    return model


def _reload_saved_model(
    saved_dir: Path, reference_model: SentenceTransformer
) -> SentenceTransformer:
    """Load a saved checkpoint the way a consumer would.

    A LoRA checkpoint is adapter-only. SentenceTransformer.save writes
    adapter_config.json and adapter_model.safetensors but no base config and
    no base weights, because the base is named inside adapter_config rather
    than copied. SentenceTransformer(saved_dir) therefore fails with
    "Unrecognized model ... should have a model_type key": it is asking a
    directory of adapter weights to describe an architecture.

    So reload the way the adapter itself says to -- base model first, then the
    adapter on top. A full fine-tune has no adapter_config and loads directly.
    """
    adapter_config = saved_dir / "adapter_config.json"
    if not adapter_config.is_file():
        return SentenceTransformer(str(saved_dir))

    with open(adapter_config, encoding="utf-8") as handle:
        base_name = json.load(handle).get("base_model_name_or_path")
    if not base_name:
        raise RuntimeError(
            f"{adapter_config} names no base_model_name_or_path, so the "
            f"adapter cannot be reattached to anything and the checkpoint "
            f"cannot be loaded."
        )

    reloaded = SentenceTransformer(base_name)
    backbone = reloaded[0]
    module = getattr(backbone, "auto_model", None) or backbone.model
    from peft import PeftModel

    # Shared with the publish path: sentence-transformers 5.7 made auto_model
    # a read-only property, and a plain assignment is silently discarded.
    from tract.publish.merge import _set_backbone

    merged = PeftModel.from_pretrained(module, str(saved_dir)).merge_and_unload()
    _set_backbone(backbone, merged)
    reloaded.max_seq_length = reference_model.max_seq_length
    return reloaded


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
    reloaded = _reload_saved_model(saved_dir, model)
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
