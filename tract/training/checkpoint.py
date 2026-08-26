"""Write LoRA checkpoints that a plain ``SentenceTransformer(path)`` can load.

WHY THIS MODULE EXISTS

``transformers.PreTrainedModel.save_pretrained`` deliberately skips
``config.json`` when the model carries an injected PEFT adapter
(``modeling_utils.py`` in 4.57.6 guards the config write with
``if not _hf_peft_config_loaded``). The reasoning upstream is that the adapter
names its own base model, so copying the base config would be redundant.

sentence-transformers 5.7 disagrees with that reasoning. ``Transformer.__init__``
resolves the *model* through the adapter -- ``_load_config`` calls
``find_adapter_config_file`` first and hands back a ``PeftConfig`` -- but it then
calls ``AutoProcessor.from_pretrained`` against the same directory. AutoProcessor
walks its own resolution chain (processor config, image-processor config, video
processor, feature extractor, then ``processor_class`` inside
``tokenizer_config.json``) and falls through every step for an ordinary BERT
tokenizer. The last resort is ``AutoConfig.from_pretrained(directory)``, which
finds no ``config.json``, receives an empty config dict and raises
``Unrecognized model in <dir>``.

The result is a checkpoint whose every weight is correct and which no consumer
can open. ``tract.active_learning.model_io.load_fold_model`` and
``tract.publish.merge.merge_lora_adapters`` both load checkpoints with a plain
``SentenceTransformer(dir)``, so both hit it.

Writing the backbone's own config next to the adapter costs a few hundred bytes
and makes the directory self-describing. It stays correct if upstream later
drops the AutoProcessor fallback, because the config written is the base config
the adapter already points at.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# transformers.utils.CONFIG_NAME and peft's adapter config filename. Spelled out
# rather than imported so this module stays importable without the ML stack,
# which is what lets its logic be tested off a GPU host.
HF_CONFIG_NAME: Final[str] = "config.json"
ADAPTER_CONFIG_NAME: Final[str] = "adapter_config.json"


def save_sentence_transformer(
    model: SentenceTransformer, output_dir: Path
) -> Path:
    """Save ``model`` so that ``SentenceTransformer(output_dir)`` reloads it.

    Use this instead of ``model.save()`` for anything that will be read back.
    A bare ``model.save()`` on an adapter-carrying model writes a directory that
    raises ``Unrecognized model`` on reload; see the module docstring.

    Returns:
        ``output_dir``.

    Raises:
        RuntimeError: If the base config is missing and cannot be recovered from
            the live model, so the checkpoint would be unreadable.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(output_dir))
    _write_base_config_if_missing(model, output_dir)
    return output_dir


def _write_base_config_if_missing(model: SentenceTransformer, output_dir: Path) -> None:
    """Fill in the ``config.json`` that an adapter-only save leaves out."""
    config_path = output_dir / HF_CONFIG_NAME
    if config_path.is_file():
        # Full fine-tune: transformers already wrote the config, and it wrote it
        # after moving any misplaced generation parameters out into
        # generation_config.json. Rewriting from the live config would put those
        # back and produce a pair of files that contradict each other.
        return

    backbone = model[0]
    inner: Any = getattr(backbone, "auto_model", None)
    if inner is None:
        raise RuntimeError(
            f"Checkpoint at {output_dir} has no {HF_CONFIG_NAME} and the first "
            f"module of the model, a {type(backbone).__name__}, exposes no "
            "auto_model to recover one from. The directory cannot be loaded by "
            "SentenceTransformer, so refusing to leave it on disk unmarked."
        )

    config: Any = getattr(inner, "config", None)
    if config is None:
        raise RuntimeError(
            f"Checkpoint at {output_dir} has no {HF_CONFIG_NAME} and the "
            f"backbone, a {type(inner).__name__}, carries no config attribute to "
            "recover one from."
        )

    config.save_pretrained(str(output_dir))
    if not config_path.is_file():
        raise RuntimeError(
            f"Wrote the backbone config to {output_dir} but {HF_CONFIG_NAME} is "
            f"still absent, so the checkpoint stays unreadable. "
            f"{type(config).__name__}.save_pretrained did not produce the file "
            "this code depends on."
        )
    logger.info(
        "Wrote the base %s next to the adapter in %s", HF_CONFIG_NAME, output_dir,
    )


def repair_adapter_only_checkpoint(
    checkpoint_dir: Path, base_config_path: Path, base_model_id: str
) -> bool:
    """Write the missing base config into an adapter-only checkpoint.

    The second half of ``assert_loadable_checkpoint``'s error message, made
    executable. Deliberately a file copy and not a model round-trip: the 98
    checkpoints this exists for hold correct weights, and re-serialising them
    would rewrite artifacts to fix a directory listing. It also means the
    repair runs on a machine that never allocates a model.

    Args:
        checkpoint_dir: An adapter-only checkpoint directory.
        base_config_path: A ``config.json`` fetched for ``base_model_id``.
        base_model_id: The repo id ``base_config_path`` was fetched for. This
            is the identity the adapter is checked against, NOT
            ``_name_or_path`` inside the config: that field records wherever
            the config was last saved from, and BAAI published
            ``bge-large-en-v1.5`` with a path on their own build machine in it.
            Ninety-five of the 98 checkpoints are that backbone, so a guard
            reading the config's own field refuses the whole real workload.

    Returns:
        True if a config was written, False if the checkpoint already had one.

    Raises:
        ValueError: If ``checkpoint_dir`` is not an adapter checkpoint, or if
            the adapter names a backbone other than ``base_model_id``. A
            checkpoint that opens with the wrong backbone's config is worse
            than one that refuses to open, and 3 of the 98 are Qwen among 95
            BGE.
    """
    adapter_path = checkpoint_dir / ADAPTER_CONFIG_NAME
    if not adapter_path.is_file():
        raise ValueError(
            f"{checkpoint_dir} has no {ADAPTER_CONFIG_NAME}, so it is not an "
            "adapter checkpoint and there is nothing here to repair."
        )

    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    declared_base = adapter.get("base_model_name_or_path")
    if declared_base and declared_base != base_model_id:
        raise ValueError(
            f"{checkpoint_dir} names {declared_base} as its backbone but the "
            f"config supplied was fetched for {base_model_id}. Writing it "
            "would produce a checkpoint that loads and is wrong."
        )

    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))

    config_path = checkpoint_dir / HF_CONFIG_NAME
    if config_path.is_file():
        return False

    # Atomic: a half-written config.json is a checkpoint that opens and then
    # fails deep inside transformers, which is harder to diagnose than one
    # that never opened.
    tmp = config_path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(base_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(config_path)
    logger.info("Repaired %s with the base config from %s",
                checkpoint_dir, base_config_path)
    return True


def assert_loadable_checkpoint(checkpoint_dir: Path) -> None:
    """Raise unless ``checkpoint_dir`` carries enough config to be loaded.

    Reader-side guard. A checkpoint written before ``save_sentence_transformer``
    existed is adapter-only and raises a 200-line transformers traceback about
    an unrecognised model. This turns that into one sentence naming the fix.

    Raises:
        RuntimeError: If the directory has no usable base config.
    """
    if (checkpoint_dir / HF_CONFIG_NAME).is_file():
        return

    if (checkpoint_dir / ADAPTER_CONFIG_NAME).is_file():
        raise RuntimeError(
            f"{checkpoint_dir} is an adapter-only checkpoint with no "
            f"{HF_CONFIG_NAME}, so sentence-transformers cannot resolve a "
            "processor for it and loading raises 'Unrecognized model'. It was "
            "written before tract.training.checkpoint.save_sentence_transformer "
            "existed. Re-save it, or copy the base model's config.json into the "
            "directory."
        )

    raise RuntimeError(
        f"{checkpoint_dir} holds neither {HF_CONFIG_NAME} nor "
        f"{ADAPTER_CONFIG_NAME}, so it is not a model checkpoint."
    )
