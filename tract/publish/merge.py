"""Merge LoRA adapters into base model for standalone distribution."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from tract.training.checkpoint import assert_loadable_checkpoint

logger = logging.getLogger(__name__)

MERGE_VERIFICATION_TEXTS = [
    "Implement access controls for AI model training pipelines",
    "Data encryption at rest using AES-256",
    "Regularly audit AI system outputs for bias and fairness",
]
MERGE_COSINE_THRESHOLD = 0.9999


def _set_backbone(module: object, new_model: object) -> None:
    """Replace the transformer backbone inside a SentenceTransformer module.

    sentence-transformers 5.7 turned ``auto_model`` into a read-only property
    aliasing ``.model``. ``nn.Module.__setattr__`` accepts an assignment to a
    property name without raising -- it files the value into ``_modules``, which
    property reads never consult -- so a plain assignment is silently discarded
    and the caller keeps operating on the old backbone. Write to whichever
    attribute is actually the storage, then prove the swap took effect.
    """
    attr = (
        "model"
        if isinstance(getattr(type(module), "auto_model", None), property)
        else "auto_model"
    )
    setattr(module, attr, new_model)
    if getattr(module, "auto_model") is not new_model:
        raise RuntimeError(
            f"Could not replace the transformer backbone: assigned to '{attr}' but "
            f"auto_model still reads back as {type(getattr(module, 'auto_model')).__name__}. "
            "The sentence-transformers internal layout has changed; the merge "
            "would otherwise silently publish an unmerged model."
        )


def validate_merged_output(output_dir: Path) -> None:
    """Validate that a merged model directory is correctly structured.

    Supports both flat layout (model.safetensors at root) and
    subdirectory layout (0_Transformer/model.safetensors).

    Raises RuntimeError if adapter artifacts remain or weights are missing.
    """
    for adapter_loc in [output_dir / "adapter_config.json",
                        output_dir / "0_Transformer" / "adapter_config.json"]:
        if adapter_loc.exists():
            raise RuntimeError(
                f"adapter_config.json still present after merge: {adapter_loc}. "
                "Merge did not fully integrate LoRA weights."
            )

    weights_found = (
        (output_dir / "model.safetensors").exists()
        or (output_dir / "0_Transformer" / "model.safetensors").exists()
    )
    if not weights_found:
        raise RuntimeError(
            f"model.safetensors not found in {output_dir}. "
            "Merge may have failed."
        )


def merge_lora_adapters(
    model_dir: Path,
    output_dir: Path,
) -> Path:
    """Merge LoRA adapters into base model weights.

    Loads via SentenceTransformer (which auto-detects PEFT),
    captures pre-merge embeddings for verification,
    merges the adapter into the base weights, verifies
    cosine similarity > 0.9999, and saves the full
    SentenceTransformer directory structure.

    Args:
        model_dir: Path to SentenceTransformer directory with PEFT adapter overlay.
        output_dir: Path for merged output.

    Returns:
        output_dir path.

    Raises:
        RuntimeError: If merge verification fails (cosine < threshold).
    """
    logger.info("Loading model from %s", model_dir)
    # An adapter-only checkpoint with no base config raises a transformers error
    # about an unrecognised model, which reads as a corrupt download rather than
    # a checkpoint written by an older TRACT. Name the real problem first.
    assert_loadable_checkpoint(model_dir)
    model = SentenceTransformer(str(model_dir))

    inner = model[0].auto_model
    adapter_config = model_dir / "adapter_config.json"
    if not hasattr(inner, "merge_and_unload") and not adapter_config.exists():
        raise RuntimeError(
            f"No PEFT adapter found at {model_dir}. "
            "Expected adapter_config.json + adapter_model.safetensors."
        )

    logger.info("Computing pre-merge reference embeddings")
    pre_merge_emb = model.encode(
        MERGE_VERIFICATION_TEXTS,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    logger.info("Merging LoRA adapters into base weights")
    if hasattr(inner, "merge_and_unload"):
        merged = inner.merge_and_unload()
    else:
        # sentence-transformers 5.7 injects the adapter straight into the
        # backbone instead of wrapping it in a PeftModel, so the loaded model
        # embeds correctly but exposes no merge_and_unload. Rebuild the wrapper
        # from base + adapter and let peft's own merge path do the arithmetic,
        # rather than hand-rolling layer surgery on the live module tree.
        logger.info("Backbone carries an injected adapter; rebuilding a PeftModel to merge")
        import json

        from peft import PeftModel as _PeftModel
        from transformers import AutoModel
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base = AutoModel.from_pretrained(config["base_model_name_or_path"])
        merged = _PeftModel.from_pretrained(base, str(model_dir)).merge_and_unload()

    if hasattr(merged, "peft_config"):
        delattr(merged, "peft_config")
    _set_backbone(model[0], merged)

    logger.info("Computing post-merge embeddings for verification")
    post_merge_emb = model.encode(
        MERGE_VERIFICATION_TEXTS,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    cosines = np.sum(pre_merge_emb * post_merge_emb, axis=1)
    min_cosine = float(np.min(cosines))
    logger.info("Merge verification: min cosine = %.6f (threshold: %.4f)", min_cosine, MERGE_COSINE_THRESHOLD)

    if min_cosine < MERGE_COSINE_THRESHOLD:
        raise RuntimeError(
            f"Merge verification failed: min cosine {min_cosine:.6f} < {MERGE_COSINE_THRESHOLD}. "
            f"Per-text cosines: {cosines.tolist()}"
        )

    logger.info("Saving merged model to %s", output_dir)
    model.save(str(output_dir))

    validate_merged_output(output_dir)
    logger.info("Merge complete — verified: no adapter artifacts, embeddings match")

    return output_dir
