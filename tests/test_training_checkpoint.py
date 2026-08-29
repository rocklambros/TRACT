"""Tests for tract.training.checkpoint.

These drive the checkpoint-completion logic with stand-in objects rather than a
real SentenceTransformer, so they run anywhere. The end-to-end proof that a
completed checkpoint actually reloads lives in tests/test_training_loop.py and
needs the pinned training stack.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tract.training.checkpoint import (
    ADAPTER_CONFIG_NAME,
    CUSTOM_CODE_EXEMPT_CONFIG_NAMES,
    HF_CONFIG_NAME,
    assert_checkpoint_is_inert,
    assert_loadable_checkpoint,
    save_sentence_transformer,
)


class _FakeConfig:
    """Stands in for a transformers PretrainedConfig."""

    def __init__(self, payload: dict[str, Any], writes_file: bool = True) -> None:
        self.payload = payload
        self.writes_file = writes_file

    def save_pretrained(self, output_path: str) -> None:
        if not self.writes_file:
            return
        target = Path(output_path) / HF_CONFIG_NAME
        target.write_text(json.dumps(self.payload, sort_keys=True), encoding="utf-8")


class _FakeBackbone:
    """Stands in for the transformer backbone held by ST module 0."""

    def __init__(self, config: _FakeConfig | None) -> None:
        self.config = config


class _FakeModule:
    """Stands in for a sentence_transformers Transformer module."""

    def __init__(self, backbone: _FakeBackbone | None) -> None:
        if backbone is not None:
            self.auto_model = backbone


class _FakeModel:
    """Stands in for a SentenceTransformer.

    ``save`` writes whichever files the scenario needs, mirroring what
    transformers does for an adapter-carrying model versus a plain one.
    """

    def __init__(self, module: _FakeModule, files: dict[str, str]) -> None:
        self._module = module
        self.files = files
        self.save_calls: list[str] = []

    def __getitem__(self, index: int) -> _FakeModule:
        if index != 0:
            raise IndexError(index)
        return self._module

    def save(self, output_path: str) -> None:
        self.save_calls.append(output_path)
        for name, body in self.files.items():
            (Path(output_path) / name).write_text(body, encoding="utf-8")


def _adapter_only_model(model_type: str = "bert") -> _FakeModel:
    """A model whose save leaves out config.json, as an adapter save does."""
    config = _FakeConfig({"model_type": model_type, "architectures": ["BertModel"]})
    module = _FakeModule(_FakeBackbone(config))
    return _FakeModel(module, {
        ADAPTER_CONFIG_NAME: json.dumps({"base_model_name_or_path": "BAAI/bge-large-en-v1.5"}),
        "tokenizer_config.json": "{}",
    })


class TestSaveSentenceTransformer:

    def test_completes_an_adapter_only_save(self, tmp_path: Path) -> None:
        """The whole point: the saved directory ends up self-describing."""
        model = _adapter_only_model()
        out = tmp_path / "checkpoint"

        result = save_sentence_transformer(model, out)  # type: ignore[arg-type]

        assert result == out
        written = json.loads((out / HF_CONFIG_NAME).read_text(encoding="utf-8"))
        assert written["model_type"] == "bert"
        # The adapter must still be there. A "fix" that merged or dropped it
        # would satisfy the config assertion above and destroy the checkpoint.
        assert (out / ADAPTER_CONFIG_NAME).is_file()

    def test_creates_the_output_directory(self, tmp_path: Path) -> None:
        model = _adapter_only_model()
        out = tmp_path / "nested" / "checkpoint"

        save_sentence_transformer(model, out)  # type: ignore[arg-type]

        assert out.is_dir()
        assert model.save_calls == [str(out)]

    def test_leaves_an_existing_config_untouched(self, tmp_path: Path) -> None:
        """A full fine-tune's config is written by transformers, not by us.

        Rewriting it from the live config would undo the generation-parameter
        migration transformers performs on the way out.
        """
        config = _FakeConfig({"model_type": "from-the-live-model"})
        module = _FakeModule(_FakeBackbone(config))
        model = _FakeModel(module, {
            HF_CONFIG_NAME: json.dumps({"model_type": "from-save-pretrained"}),
        })
        out = tmp_path / "checkpoint"

        save_sentence_transformer(model, out)  # type: ignore[arg-type]

        written = json.loads((out / HF_CONFIG_NAME).read_text(encoding="utf-8"))
        assert written["model_type"] == "from-save-pretrained"

    def test_raises_when_the_module_exposes_no_backbone(self, tmp_path: Path) -> None:
        model = _FakeModel(_FakeModule(None), {ADAPTER_CONFIG_NAME: "{}"})

        with pytest.raises(RuntimeError, match="exposes no auto_model"):
            save_sentence_transformer(model, tmp_path / "checkpoint")  # type: ignore[arg-type]

    def test_raises_when_the_backbone_carries_no_config(self, tmp_path: Path) -> None:
        model = _FakeModel(_FakeModule(_FakeBackbone(None)), {ADAPTER_CONFIG_NAME: "{}"})

        with pytest.raises(RuntimeError, match="carries no config attribute"):
            save_sentence_transformer(model, tmp_path / "checkpoint")  # type: ignore[arg-type]

    def test_raises_when_the_config_write_produces_nothing(self, tmp_path: Path) -> None:
        """save_pretrained returning quietly without writing must not pass."""
        config = _FakeConfig({"model_type": "bert"}, writes_file=False)
        model = _FakeModel(
            _FakeModule(_FakeBackbone(config)), {ADAPTER_CONFIG_NAME: "{}"},
        )

        with pytest.raises(RuntimeError, match="still absent"):
            save_sentence_transformer(model, tmp_path / "checkpoint")  # type: ignore[arg-type]


class TestAssertLoadableCheckpoint:

    def test_accepts_a_directory_with_a_base_config(self, tmp_path: Path) -> None:
        (tmp_path / HF_CONFIG_NAME).write_text('{"model_type": "bert"}', encoding="utf-8")

        assert_loadable_checkpoint(tmp_path)

    def test_accepts_a_completed_adapter_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / HF_CONFIG_NAME).write_text('{"model_type": "bert"}', encoding="utf-8")
        (tmp_path / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")

        assert_loadable_checkpoint(tmp_path)

    def test_rejects_an_adapter_only_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")

        with pytest.raises(RuntimeError, match="adapter-only checkpoint"):
            assert_loadable_checkpoint(tmp_path)

    def test_rejects_a_directory_that_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        (tmp_path / "modules.json").write_text("[]", encoding="utf-8")

        with pytest.raises(RuntimeError, match="not a model checkpoint"):
            assert_loadable_checkpoint(tmp_path)

    def test_a_directory_named_config_json_is_not_a_config(self, tmp_path: Path) -> None:
        """is_file, not exists: a directory of that name must not satisfy the guard."""
        (tmp_path / HF_CONFIG_NAME).mkdir()
        (tmp_path / ADAPTER_CONFIG_NAME).write_text("{}", encoding="utf-8")

        with pytest.raises(RuntimeError, match="adapter-only checkpoint"):
            assert_loadable_checkpoint(tmp_path)


class TestRepairAdapterOnlyCheckpoint:
    """D2(b), answered 2026-08-26: make the 98 adapter-only checkpoints loadable.

    They were written before save_sentence_transformer existed, so every one
    carries correct weights that no consumer can open. The repair is the second
    half of assert_loadable_checkpoint's own error message: copy the base
    model's config.json in beside the adapter. It is a file operation, which
    matters because it must run on a machine that never allocates a model.

    The guard that earns its place here is the base-model match. These are 98
    artifacts whose provenance nobody has audited, and 95 of them name
    BAAI/bge-large-en-v1.5 while 3 name Qwen/Qwen3-Embedding-0.6B. Writing the
    wrong backbone's config produces a checkpoint that opens and is wrong,
    which is worse than one that refuses to open.

    The match is against the repo id the caller fetched the config FOR, never
    against `_name_or_path` inside the config. See the build-path test below:
    that field records wherever the config was last saved from, which for a
    published model is a path on the publisher's build machine.
    """

    BGE = "BAAI/bge-large-en-v1.5"
    QWEN = "Qwen/Qwen3-Embedding-0.6B"

    def _adapter_dir(self, tmp_path: Path, base_model: str) -> Path:
        d = tmp_path / "checkpoint-1234"
        d.mkdir()
        (d / ADAPTER_CONFIG_NAME).write_text(
            json.dumps({"base_model_name_or_path": base_model, "peft_type": "LORA"}),
            encoding="utf-8",
        )
        return d

    def _base_config(
        self, tmp_path: Path, name_or_path: str, model_type: str = "bert"
    ) -> Path:
        p = tmp_path / "base_config.json"
        p.write_text(
            json.dumps({"_name_or_path": name_or_path, "model_type": model_type}),
            encoding="utf-8",
        )
        return p

    def test_the_repaired_checkpoint_becomes_loadable(self, tmp_path: Path) -> None:
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        cfg = self._base_config(tmp_path, self.BGE)

        with pytest.raises(RuntimeError):
            assert_loadable_checkpoint(d)

        assert repair_adapter_only_checkpoint(d, cfg, self.BGE) is True
        assert_loadable_checkpoint(d)

    def test_the_written_config_is_the_base_config(self, tmp_path: Path) -> None:
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        cfg = self._base_config(tmp_path, self.BGE)

        repair_adapter_only_checkpoint(d, cfg, self.BGE)

        written = json.loads((d / HF_CONFIG_NAME).read_text(encoding="utf-8"))
        assert written["model_type"] == "bert"

    def test_a_publisher_build_path_is_not_treated_as_a_mismatch(
        self, tmp_path: Path
    ) -> None:
        """BAAI shipped bge-large-en-v1.5 with a build-machine path in the config.

        `_name_or_path` reads
        `/root/.cache/torch/sentence_transformers/BAAI_bge-large-en/` in the
        published file. Ninety-five of the 98 checkpoints are this backbone, so
        a guard reading that field refuses the entire real workload.
        """
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        cfg = self._base_config(
            tmp_path, "/root/.cache/torch/sentence_transformers/BAAI_bge-large-en/"
        )

        assert repair_adapter_only_checkpoint(d, cfg, self.BGE) is True
        assert_loadable_checkpoint(d)

    def test_a_mismatched_backbone_is_refused(self, tmp_path: Path) -> None:
        """A Qwen checkpoint must not receive the config fetched for BGE."""
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.QWEN)
        cfg = self._base_config(tmp_path, self.BGE)

        with pytest.raises(ValueError, match="names Qwen/Qwen3-Embedding-0.6B"):
            repair_adapter_only_checkpoint(d, cfg, self.BGE)

        assert not (d / HF_CONFIG_NAME).exists()

    def test_an_already_complete_checkpoint_is_left_alone(self, tmp_path: Path) -> None:
        """Idempotent, and it must not rewrite a config transformers wrote."""
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = self._adapter_dir(tmp_path, self.BGE)
        original = json.dumps({"model_type": "bert", "written_by": "transformers"})
        (d / HF_CONFIG_NAME).write_text(original, encoding="utf-8")
        cfg = self._base_config(tmp_path, self.BGE)

        assert repair_adapter_only_checkpoint(d, cfg, self.BGE) is False
        assert (d / HF_CONFIG_NAME).read_text(encoding="utf-8") == original

    def test_a_directory_that_is_not_a_checkpoint_raises(self, tmp_path: Path) -> None:
        from tract.training.checkpoint import repair_adapter_only_checkpoint

        d = tmp_path / "not-a-checkpoint"
        d.mkdir()
        cfg = self._base_config(tmp_path, self.BGE)

        with pytest.raises(ValueError, match="no adapter_config.json"):
            repair_adapter_only_checkpoint(d, cfg, self.BGE)


class TestAssertCheckpointIsInert:
    """The checkpoint directory is untrusted input, so this guard is load-bearing.

    These live here rather than in tests/test_active_learning_model_io.py on
    purpose. That module is behind ``pytest.importorskip("sentence_transformers")``
    and its positive cases skip when results/ is absent, so in CI a validator
    that rejected every checkpoint in existence would show green. This module
    imports nothing from the ML stack, so these run everywhere, and the
    acceptance cases below fail loudly if the predicate is ever tightened into
    one that locks out our own artifacts.
    """

    LEGACY_TYPES = (
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    )
    # What results/phase1b/c2r_A1_prose_sw_bge/* actually ships. Sixty of the
    # 119 modules.json in the tree use these; a class-name allow-list would
    # have refused every one of them.
    ST5_TYPES = (
        "sentence_transformers.base.modules.transformer.Transformer",
        "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
        "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
    )
    MODULE_PATHS = ("", "1_Pooling", "2_Normalize")

    def _write_modules(self, model_dir: Path, entries: list[dict[str, Any]]) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "modules.json").write_text(json.dumps(entries), encoding="utf-8")

    def _st_root(
        self, model_dir: Path, types: tuple[str, ...] = LEGACY_TYPES
    ) -> Path:
        """Build the on-disk shape of a real sentence-transformers root."""
        self._write_modules(model_dir, [
            {"idx": i, "name": str(i), "path": path, "type": type_}
            for i, (path, type_) in enumerate(zip(self.MODULE_PATHS, types))
        ])
        for path in self.MODULE_PATHS:
            if path:
                sub = model_dir / path
                sub.mkdir(exist_ok=True)
                (sub / HF_CONFIG_NAME).write_text(
                    json.dumps({"word_embedding_dimension": 1024}), encoding="utf-8",
                )
        (model_dir / "adapter_model.safetensors").write_bytes(b"")
        return model_dir

    def _plant_payload(self, model_dir: Path, marker: Path) -> None:
        """The .py that sentence-transformers would import and run."""
        (model_dir / "evil_module.py").write_text(
            "import pathlib\n"
            f"pathlib.Path({str(marker)!r}).write_text('pwned')\n"
            "class EvilTransformer:\n    pass\n",
            encoding="utf-8",
        )

    # ── acceptance: these fail if the guard is ever over-tightened ────

    def test_accepts_a_legacy_sentence_transformers_root(self, tmp_path: Path) -> None:
        model_dir = self._st_root(tmp_path / "model")
        (model_dir / HF_CONFIG_NAME).write_text(
            json.dumps({"model_type": "bert"}), encoding="utf-8")

        assert_checkpoint_is_inert(model_dir)

    def test_accepts_the_5x_module_path_spellings(self, tmp_path: Path) -> None:
        """Half our checkpoints name their modules the 5.x way."""
        model_dir = self._st_root(tmp_path / "model", types=self.ST5_TYPES)

        assert_checkpoint_is_inert(model_dir)

    def test_accepts_an_adapter_only_checkpoint(self, tmp_path: Path) -> None:
        """config.json is OPTIONAL: it is absent from all five phase1b_primary folds."""
        model_dir = self._st_root(tmp_path / "model")
        (model_dir / ADAPTER_CONFIG_NAME).write_text(
            json.dumps({"base_model_name_or_path": "BAAI/bge-large-en-v1.5"}),
            encoding="utf-8",
        )

        assert not (model_dir / HF_CONFIG_NAME).is_file()
        assert_checkpoint_is_inert(model_dir)

    def test_accepts_a_root_whose_sibling_holds_trainer_pickles(
        self, tmp_path: Path
    ) -> None:
        """The production shape of results/phase1c/deployment_model.

        Four checkpoint-NNNN directories, 20 pickles, none of which any loader
        opens: the model is read from model/model/. Without the prefix prune
        this refuses the DEFAULT `tract assign` directory every single time.
        """
        root = tmp_path / "deployment_model"
        self._st_root(root / "model" / "model")
        trainer_state = root / "checkpoint-2044"
        trainer_state.mkdir(parents=True)
        (trainer_state / "optimizer.pt").write_bytes(b"trainer state, never loaded")
        (trainer_state / "rng_state.pth").write_bytes(b"")

        assert_checkpoint_is_inert(root)

    def test_accepts_a_published_repo_carrying_example_scripts(
        self, tmp_path: Path
    ) -> None:
        """build/hf_repo ships predict.py and train.py, and `tract assign`
        downloads that layout. A blanket "no .py in the model dir" rule would
        break the CLI's primary path 100% of the time.
        """
        model_dir = self._st_root(tmp_path / "hf_repo")
        (model_dir / "predict.py").write_text("# usage example\n", encoding="utf-8")
        (model_dir / "train.py").write_text("# usage example\n", encoding="utf-8")
        (model_dir / "model.safetensors").write_bytes(b"")

        assert_checkpoint_is_inert(model_dir)

    # ── rejection: the vector that is actually live ─────────────────

    def test_rejects_a_module_type_outside_the_namespace(self, tmp_path: Path) -> None:
        """trust_remote_code=False does not stop this.

        _load_module_class_from_ref takes the dynamic-module branch on
        ``trust_remote_code or os.path.exists(path)``, and a local directory
        satisfies the second half by itself, so the .py runs at import time.
        """
        model_dir = tmp_path / "pod_model"
        marker = tmp_path / "executed"
        self._write_modules(model_dir, [
            {"idx": 0, "name": "0", "path": "", "type": "evil_module.EvilTransformer"},
        ])
        self._plant_payload(model_dir, marker)

        with pytest.raises(ValueError, match="custom code"):
            assert_checkpoint_is_inert(model_dir)
        assert not marker.exists(), "payload ran despite validation"

    def test_the_namespace_prefix_is_anchored_at_a_package_boundary(
        self, tmp_path: Path
    ) -> None:
        """A bare startswith on "sentence_transformers" without the dot would
        wave through a top-level package named sentence_transformers_evil.
        """
        model_dir = tmp_path / "lookalike"
        self._write_modules(model_dir, [
            {"idx": 0, "name": "0", "path": "",
             "type": "sentence_transformers_evil.Transformer"},
        ])

        with pytest.raises(ValueError, match="custom code"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_a_module_path_escaping_its_own_directory(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "escape"
        self._write_modules(model_dir, [
            {"idx": 0, "name": "0", "path": "../../elsewhere",
             "type": "sentence_transformers.models.Pooling"},
        ])

        with pytest.raises(ValueError, match="escapes"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_auto_map_in_the_top_level_config(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "auto_map"
        model_dir.mkdir()
        (model_dir / HF_CONFIG_NAME).write_text(
            json.dumps({"auto_map": {"AutoModel": "evil--repo.modeling.Evil"}}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="custom code"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_custom_pipelines_in_the_top_level_config(
        self, tmp_path: Path
    ) -> None:
        model_dir = tmp_path / "pipelines"
        model_dir.mkdir()
        (model_dir / HF_CONFIG_NAME).write_text(
            json.dumps({"custom_pipelines": {"x": {"impl": "evil.Pipe"}}}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="custom code"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_auto_map_in_the_tokenizer_config(self, tmp_path: Path) -> None:
        model_dir = self._st_root(tmp_path / "model")
        (model_dir / "tokenizer_config.json").write_text(
            json.dumps({"auto_map": {"AutoTokenizer": ["evil.Tok", None]}}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="custom code"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_auto_map_in_a_nested_module_config(self, tmp_path: Path) -> None:
        """1_Pooling/config.json is read too; the walk is not top-level only."""
        model_dir = self._st_root(tmp_path / "model")
        (model_dir / "1_Pooling" / HF_CONFIG_NAME).write_text(
            json.dumps({"auto_map": {"AutoModel": "evil.Evil"}}), encoding="utf-8")

        with pytest.raises(ValueError, match="custom code"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_a_pickle_weight_file_beside_the_model(self, tmp_path: Path) -> None:
        """No */model/model directory in this repo carries one, so a pod-returned
        checkpoint that suddenly does is not a shape we have ever loaded.
        """
        model_dir = self._st_root(tmp_path / "model")
        (model_dir / "pytorch_model.bin").write_bytes(b"\x80\x04")

        with pytest.raises(ValueError, match="pickle-format"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_malformed_modules_json(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "bad"
        model_dir.mkdir()
        (model_dir / "modules.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed JSON"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_modules_json_that_is_not_a_list(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "notalist"
        model_dir.mkdir()
        (model_dir / "modules.json").write_text(
            json.dumps({"0": "sentence_transformers.models.Transformer"}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="Expected a JSON list"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_a_modules_entry_that_is_not_an_object(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "scalars"
        self._write_modules(
            model_dir, ["sentence_transformers.models.Transformer"],  # type: ignore[list-item]
        )

        with pytest.raises(ValueError, match="Expected JSON objects"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_a_non_string_module_path(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "badpath"
        self._write_modules(model_dir, [
            {"idx": 0, "name": "0", "path": 0,
             "type": "sentence_transformers.models.Transformer"},
        ])

        with pytest.raises(ValueError, match="string 'path'"):
            assert_checkpoint_is_inert(model_dir)

    def test_rejects_malformed_json_in_a_config(self, tmp_path: Path) -> None:
        model_dir = tmp_path / "badcfg"
        model_dir.mkdir()
        (model_dir / HF_CONFIG_NAME).write_text("{", encoding="utf-8")

        with pytest.raises(ValueError, match="Malformed JSON"):
            assert_checkpoint_is_inert(model_dir)

    def test_raises_file_not_found_for_a_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            assert_checkpoint_is_inert(tmp_path / "nope")

    def test_a_file_is_not_a_checkpoint_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "model"
        target.write_text("", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            assert_checkpoint_is_inert(target)


class TestTheGuardRefusesNothingWeShip:
    """The measurement that keeps this guard from being deleted under pressure.

    A guard that rejects our own artifacts does not get fixed at 3am, it gets
    disabled, and then it protects nothing. Every real checkpoint on this disk
    must pass. Skipped rather than failed where results/ is absent, because a
    fresh clone has no fold output and that is not a defect.
    """

    REPO = Path(__file__).resolve().parents[1]

    def _dirs(self, pattern: str) -> list[Path]:
        return sorted(p for p in self.REPO.glob(pattern) if p.is_dir())

    def test_every_real_fold_checkpoint_passes(self) -> None:
        found = self._dirs("results/**/model/model")
        if not found:
            pytest.skip("no fold checkpoints on disk")
        for d in found:
            assert_checkpoint_is_inert(d)

    def test_the_deployment_model_passes_at_every_layout_depth(self) -> None:
        """find_st_model_root probes three candidates and may return the parent.

        The parent holds four checkpoint-NNNN dirs of Trainer pickles. If the
        prune ever goes, this is the test that fails instead of `tract assign`.
        """
        root = self.REPO / "results" / "phase1c" / "deployment_model"
        if not root.is_dir():
            pytest.skip("no deployment model on disk")
        for cand in (root, root / "model", root / "model" / "model"):
            if cand.is_dir():
                assert_checkpoint_is_inert(cand)

    def test_the_exempt_list_is_empty_by_default(self) -> None:
        """The escape exists so it is never taken by deleting the guard.

        Empty is the shipped state: 0 of 119 tokenizer_config.json on this
        fleet carry auto_map. This asserts nobody widened it casually -- adding
        a name here should be a deliberate, reviewed act with a model behind it.
        """
        assert CUSTOM_CODE_EXEMPT_CONFIG_NAMES == frozenset()

    def test_an_exempted_config_is_skipped(self, tmp_path: Path) -> None:
        """And when a name IS added, the check actually honours it."""
        import tract.training.checkpoint as ck

        d = tmp_path / "m"
        d.mkdir()
        (d / "modules.json").write_text(json.dumps([
            {"idx": 0, "name": "0", "path": "",
             "type": "sentence_transformers.models.Transformer"}]), encoding="utf-8")
        (d / "tokenizer_config.json").write_text(
            json.dumps({"auto_map": {"AutoTokenizer": ["x.Tok", None]}}),
            encoding="utf-8")
        with pytest.raises(ValueError, match="custom code"):
            assert_checkpoint_is_inert(d)

        monkey = frozenset({"tokenizer_config.json"})
        original = ck.CUSTOM_CODE_EXEMPT_CONFIG_NAMES
        ck.CUSTOM_CODE_EXEMPT_CONFIG_NAMES = monkey  # type: ignore[misc]
        try:
            assert_checkpoint_is_inert(d)
        finally:
            ck.CUSTOM_CODE_EXEMPT_CONFIG_NAMES = original  # type: ignore[misc]

    def test_a_trainer_state_dir_passed_directly_is_still_vetted(
        self, tmp_path: Path,
    ) -> None:
        """os.walk yields the root before the prune applies, so it is vetted.

        The prune skips checkpoint-* found BENEATH the target. Passing one as
        the target itself is a different question, and the answer must stay
        'still checked' -- otherwise the skip becomes a way to smuggle a pickle
        past the guard by naming the directory carefully.
        """
        trainer = tmp_path / "checkpoint-2044"
        trainer.mkdir()
        (trainer / "optimizer.pt").write_bytes(b"\x80\x04\x95")
        with pytest.raises(ValueError):
            assert_checkpoint_is_inert(trainer)
