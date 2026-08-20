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
