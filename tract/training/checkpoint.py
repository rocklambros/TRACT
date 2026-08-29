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

WHY THE INERTNESS GUARD LIVES HERE TOO

``assert_checkpoint_is_inert`` runs before any loader hands a checkpoint
directory to sentence-transformers. It sits next to the writer because nothing
in this module imports the ML stack at runtime -- ``SentenceTransformer`` is
referenced only under ``TYPE_CHECKING``, so importing this module costs 0.0018s
against 9.03s for ``import sentence_transformers``. That is what makes the
guard importable on a machine that never allocates a model, and what lets its
logic be tested in CI where neither the phase0 extra nor results/ is present.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
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

# ── Checkpoint inertness: the directory is untrusted input ────────────
# A checkpoint is reachable input, not our own build output: fold models come
# back off rented third-party GPU pods and the deployment model comes off the
# Hub. sentence-transformers picks the module CLASSES it instantiates out of
# the checkpoint's own modules.json, and ``_load_module_class_from_ref`` takes
# the ``get_class_from_dynamic_module`` branch whenever
# ``trust_remote_code or os.path.exists(model_name_or_path)`` holds -- a local
# directory satisfies the second half on its own. A modules.json naming
# ``evil_module.EvilTransformer`` therefore copies evil_module.py out of the
# checkpoint and imports it, running its module-level code, with
# ``trust_remote_code=False`` set: that flag only ever governed code fetched
# from the Hub, never local module wiring. Refs already inside the
# sentence_transformers namespace short-circuit to a plain importlib import
# that can only resolve inside site-packages, so the namespace is the guard.
#
# The trailing dot is load-bearing. It anchors the prefix at a package
# boundary, so ``sentence_transformers_evil.Transformer`` does not pass. And it
# stays a prefix rather than a list of class names because our own checkpoints
# carry six spellings across three module-path families: the legacy
# ``sentence_transformers.models.*`` on 59 of them, and the 5.x
# ``sentence_transformers.base.modules.*`` /
# ``sentence_transformers.sentence_transformer.modules.*`` on the other 60. A
# literal allow-list would refuse half our own artifacts and break again on the
# next upstream rename.
TRUSTED_MODULE_NAMESPACE: Final[str] = "sentence_transformers."

MODULES_CONFIG_NAME: Final[str] = "modules.json"

# The keys by which a transformers config hands the loader a Python symbol to
# import, and the two configs transformers reads them out of.
CUSTOM_CODE_CONFIG_KEYS: Final[tuple[str, ...]] = ("auto_map", "custom_pipelines")
CUSTOM_CODE_CONFIG_NAMES: Final[tuple[str, ...]] = (
    HF_CONFIG_NAME, "tokenizer_config.json",
)

# The escape hatch, deliberately empty and deliberately present.
#
# auto_map in a tokenizer_config.json is not inherently hostile: a model with a
# genuinely custom tokenizer declares it there because that is where
# transformers looks. Refusing it costs nothing on this fleet -- 0 of 119
# tokenizer_config.json under results/ and build/ carry the key, and neither
# pinned backbone needs it -- so the check stays on by default.
#
# But if the pinned model is ever swapped for one that DOES need it, this guard
# refuses the shipped model with text that reads like an attack report, at
# which point the fastest path for whoever is on call is to delete the check.
# Naming the file here instead is a narrower change, it survives review, and it
# leaves a record of which model needed the exemption and when. A guard with no
# legitimate escape does not get respected, it gets removed.
CUSTOM_CODE_EXEMPT_CONFIG_NAMES: Final[frozenset[str]] = frozenset()

# torch.load on a pickle is arbitrary execution, so a weight file in a pickle
# container is refused rather than reasoned about.
PICKLE_WEIGHT_SUFFIXES: Final[tuple[str, ...]] = (
    ".bin", ".ckpt", ".pkl", ".pickle", ".pt", ".pth",
)

# HuggingFace Trainer writes its optimizer, scheduler, RNG and scaler state as
# pickles into checkpoint-NNNN subdirectories. Every one of the 380 pickles
# under results/ lives in one, and no loader in this package opens them: the
# deployment model is read from its model/model/ subtree, which holds none.
# Scanning them anyway would refuse results/phase1c/deployment_model -- the
# default `tract assign` directory -- over files nothing reads.
TRAINER_STATE_DIR_PREFIX: Final[str] = "checkpoint-"


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


def _reject_custom_code_config(config_path: Path) -> None:
    """Refuse a transformers config that names Python for the loader to import."""
    try:
        config: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {config_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(
            f"Expected a JSON object in {config_path}, got {type(config).__name__}"
        )
    for key in CUSTOM_CODE_CONFIG_KEYS:
        if config.get(key):
            raise ValueError(
                f"Refusing to load model with custom code ({key}) in "
                f"{config_path}. If this is a model whose tokenizer legitimately "
                f"needs {key}, add {config_path.name!r} to "
                f"CUSTOM_CODE_EXEMPT_CONFIG_NAMES rather than removing this "
                f"check -- an allowlist is reviewable, a deleted guard is not."
            )


def _reject_untrusted_modules(modules_path: Path) -> None:
    """Refuse a modules.json that wires in a class outside the ST namespace.

    Also refuses a module ``path`` that leaves the directory holding the
    modules.json, since sentence-transformers joins it and hands the result to
    ``Module.load``.
    """
    try:
        modules: Any = json.loads(modules_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {modules_path}: {exc}") from exc
    if not isinstance(modules, list):
        raise ValueError(
            f"Expected a JSON list in {modules_path}, got {type(modules).__name__}"
        )

    root = modules_path.parent.resolve()
    for entry in modules:
        if not isinstance(entry, dict):
            raise ValueError(
                f"Expected JSON objects in {modules_path}, got {type(entry).__name__}"
            )
        module_type = entry.get("type")
        if not isinstance(module_type, str) or not module_type.startswith(
            TRUSTED_MODULE_NAMESPACE
        ):
            raise ValueError(
                f"Refusing to load model with custom code: {modules_path} names "
                f"module type {module_type!r}, outside the "
                f"{TRUSTED_MODULE_NAMESPACE!r} namespace. sentence-transformers "
                f"may import that name from a .py file inside {root} and execute "
                f"it at import time, which trust_remote_code=False does not stop."
            )
        module_path = entry.get("path", "")
        if not isinstance(module_path, str):
            raise ValueError(
                f"Expected a string 'path' for module {module_type} in "
                f"{modules_path}, got {type(module_path).__name__}"
            )
        if module_path and not (root / module_path).resolve().is_relative_to(root):
            raise ValueError(
                f"Refusing to load model: {modules_path} aims module "
                f"{module_type} at {module_path!r}, which escapes {root}"
            )


def _walk_loadable_tree(root: Path) -> Iterator[tuple[Path, list[str]]]:
    """Yield (directory, sorted filenames) for everything a loader would read.

    Prunes the Trainer's checkpoint-NNNN state directories; see
    TRAINER_STATE_DIR_PREFIX. Sorted so that a directory with more than one
    problem always reports the same one.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            name for name in dirnames
            if not name.startswith(TRAINER_STATE_DIR_PREFIX)
        )
        yield Path(dirpath), sorted(filenames)


def assert_checkpoint_is_inert(checkpoint_dir: Path) -> None:
    """Raise unless loading ``checkpoint_dir`` can execute no attacker code.

    Runs before ``SentenceTransformer`` is handed the path, because by the time
    it has read modules.json the code has already run. Config-bearing files are
    inspected ahead of weights so a directory carrying only a hostile
    config.json is reported as custom code rather than as an incomplete
    checkpoint.

    A missing config.json is not an error: an adapter-only LoRA checkpoint
    names its backbone in adapter_config.json instead, and config.json is
    absent from all five phase1b_primary fold checkpoints on disk. Loadability
    is ``assert_loadable_checkpoint``'s question, not this one's.

    WHAT THIS DOES NOT PROVE. It is a CONTENT check and not an attestation.
    Nothing here re-derives a digest the pod could not have produced, so it
    cannot tell a tampered adapter from an honest one: no fold checkpoint has a
    recorded hash, because the rented pod is what produced it. What it does is
    narrow a hostile checkpoint to changing the NUMBERS instead of running
    CODE. Stated because the competing implementation this one was chosen over
    said so and this one did not, and a guard that is silent about its range
    gets trusted past it.

    Args:
        checkpoint_dir: Directory a loader is about to open. May be the
            sentence-transformers root itself or a parent holding it.

    Raises:
        FileNotFoundError: If ``checkpoint_dir`` is not a directory.
        ValueError: If a config declares auto_map or custom_pipelines, if a
            modules.json is malformed, names a module type outside the
            sentence_transformers namespace, or aims a module outside its own
            directory, or if a pickle-format weight file is present.
    """
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {checkpoint_dir}")

    modules_vetted = 0
    for directory, filenames in _walk_loadable_tree(checkpoint_dir):
        for name in filenames:
            path = directory / name
            if (name in CUSTOM_CODE_CONFIG_NAMES
                    and name not in CUSTOM_CODE_EXEMPT_CONFIG_NAMES):
                _reject_custom_code_config(path)
            elif name == MODULES_CONFIG_NAME:
                _reject_untrusted_modules(path)
                modules_vetted += 1
            elif path.suffix in PICKLE_WEIGHT_SUFFIXES:
                raise ValueError(
                    f"Refusing to load model: {path} is a pickle-format weight "
                    f"file and torch.load on a pickle executes whatever it "
                    f"contains. Re-save the checkpoint as safetensors."
                )

    logger.debug(
        "Vetted %s as inert: %d %s validated against %s",
        checkpoint_dir, modules_vetted, MODULES_CONFIG_NAME,
        TRUSTED_MODULE_NAMESPACE,
    )
