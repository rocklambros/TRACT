"""Tests for the publish path: the LoRA adapter merge, and the figures the
model card is allowed to print once that merge has produced something to ship.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# These need the optional `phase0` extra (and matplotlib for the notebook
# helpers), which the default CI test job does not install. Skip visibly
# rather than failing collection; run them with
# `pip install -e '.[phase0]' matplotlib`.
pytest.importorskip("sentence_transformers", reason="needs the phase0 extra")


# Deliberately unlike any campaign-1 figure. If a rendered card ever shows
# 76.2%, 27.9% or 0.531 while these are the inputs, the number came from a
# literal in the generator rather than from the run being published.
CARD_FOLD_RESULTS: list[dict[str, Any]] = [
    {"fold": "Framework A", "hit1": 0.588, "zs_hit1": 0.412, "n": 51},
    {"fold": "Framework B", "hit1": 0.311, "zs_hit1": 0.208, "n": 45},
]

CARD_CALIBRATION: dict[str, float] = {
    "t_deploy": 0.074,
    "ood_threshold": 0.568,
    "conformal_quantile": 0.997,
}

CARD_ECE: dict[str, Any] = {"ece": 0.079, "ece_ci": {"ci_low": 0.049, "ci_high": 0.111}}

CARD_BRIDGE: dict[str, Any] = {
    "counts": {"accepted": 5, "rejected": 58, "total": 63},
    # Required: the card refuses to publish unmeasured classification counts.
    "hub_classification": {
        "ai_only": 78, "trad_only": 380, "naturally_bridged": 0, "unlinked": 64,
    },
}

# PRD.md:380-402 withdrew the campaign-1 Gate 1 headline. These are the strings
# that headline left in the card's prose: the two per-fold percentages, the
# aggregate hit@1 and its delta, and the two verbal restatements of them.
WITHDRAWN_CAMPAIGN1_LITERALS: tuple[str, ...] = (
    "76.2%",
    "27.9%",
    "0.531",
    "+0.132",
    "3 out of 4",
    "about half the time",
)

# The fabricated hit@5: `fold_results[0].get("hit_any", 0.6)` rendered this
# whenever the field was missing, indistinguishable from a measurement.
FABRICATED_HIT_ANY_RENDERING: str = "60.0%"


def _with_indicators(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach the per-item hit@1 indicators the bootstrap CI needs.

    Derived from each fold's own hit1 and n so the fixture agrees with itself;
    the card refuses to build without them rather than inventing a band.
    """
    out: list[dict[str, Any]] = []
    for fold in folds:
        copied = dict(fold)
        hits = round(copied["hit1"] * copied["n"])
        copied["hit1_indicators"] = [1.0] * hits + [0.0] * (copied["n"] - hits)
        out.append(copied)
    return out


def _render_card(tmp_path: Path, folds: list[dict[str, Any]]) -> str:
    from tract.publish.model_card import generate_model_card

    tmp_path.mkdir(parents=True, exist_ok=True)
    generate_model_card(
        tmp_path,
        fold_results=_with_indicators(folds),
        calibration=CARD_CALIBRATION,
        ece_data=CARD_ECE,
        bridge_summary=CARD_BRIDGE,
        gpu_hours=2.5,
    )
    return (tmp_path / "README.md").read_text(encoding="utf-8")


def _interpretation(card: str) -> str:
    """The block of prose that reads the results table for the reader.

    Isolated from the table itself, whose row order legitimately follows the
    caller's list, so an order-sensitivity assertion tests the interpretation
    rather than the transcription.
    """
    start = card.index("**What the numbers mean:**")
    return card[start:card.index("### Confidence Intervals", start)]


def _create_merged_output(output_dir: Path) -> None:
    """Create the expected post-merge directory structure."""
    (output_dir / "0_Transformer").mkdir(parents=True)
    (output_dir / "0_Transformer" / "model.safetensors").write_bytes(b"fake-weights")
    (output_dir / "0_Transformer" / "config.json").write_text("{}")
    (output_dir / "1_Pooling").mkdir()
    (output_dir / "1_Pooling" / "config.json").write_text("{}")
    (output_dir / "2_Normalize").mkdir()
    (output_dir / "modules.json").write_text("[]")
    (output_dir / "sentence_bert_config.json").write_text("{}")
    (output_dir / "config_sentence_transformers.json").write_text("{}")


def _make_mock_model(fake_save_dir: Path | None = None) -> MagicMock:
    """Create a mock SentenceTransformer with encode() returning unit vectors."""
    mock_peft_model = MagicMock()
    mock_peft_model.merge_and_unload.return_value = mock_peft_model

    mock_transformer_module = MagicMock()
    mock_transformer_module.auto_model = mock_peft_model

    mock_model = MagicMock()
    mock_model.__getitem__ = MagicMock(return_value=mock_transformer_module)

    fake_emb = np.ones((3, 1024), dtype=np.float32)
    fake_emb /= np.linalg.norm(fake_emb, axis=1, keepdims=True)
    mock_model.encode.return_value = fake_emb

    if fake_save_dir is not None:
        def fake_save(path: str) -> None:
            _create_merged_output(Path(path))
        mock_model.save = fake_save

    return mock_model


def _create_checkpoint_dir(model_dir: Path) -> Path:
    """Create the input side of a merge: an adapter beside its base config.

    An empty directory is not a checkpoint. merge_lora_adapters refuses one
    before it reaches sentence-transformers, so the fixture has to carry the two
    files that make a directory loadable.
    """
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text('{"model_type": "bert"}')
    (model_dir / "adapter_config.json").write_text(
        '{"base_model_name_or_path": "BAAI/bge-large-en-v1.5"}'
    )
    return model_dir


class TestValidateMergedOutput:

    def test_rejects_leftover_adapter(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        _create_merged_output(output_dir)
        (output_dir / "0_Transformer" / "adapter_config.json").write_text("{}")
        with pytest.raises(RuntimeError, match="adapter_config.json"):
            validate_merged_output(output_dir)

    def test_rejects_missing_weights(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        output_dir.mkdir(parents=True)
        (output_dir / "0_Transformer").mkdir()
        (output_dir / "modules.json").write_text("[]")
        with pytest.raises(RuntimeError, match="model.safetensors"):
            validate_merged_output(output_dir)

    def test_accepts_clean_output(self, tmp_path) -> None:
        from tract.publish.merge import validate_merged_output
        output_dir = tmp_path / "model"
        _create_merged_output(output_dir)
        validate_merged_output(output_dir)


class TestMergeLoraAdapters:

    def test_calls_merge_and_unload(self, tmp_path) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = _create_checkpoint_dir(tmp_path / "input")
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        with patch("tract.publish.merge.SentenceTransformer", return_value=mock_model):
            merge_lora_adapters(model_dir, output_dir)

        mock_model[0].auto_model.merge_and_unload.assert_called_once()
        assert mock_model.encode.call_count == 2  # pre-merge + post-merge

    def test_output_directory_created(self, tmp_path) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = _create_checkpoint_dir(tmp_path / "input")
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        with patch("tract.publish.merge.SentenceTransformer", return_value=mock_model):
            result = merge_lora_adapters(model_dir, output_dir)
        assert result == output_dir
        assert (output_dir / "0_Transformer" / "model.safetensors").exists()

    def test_fails_on_cosine_mismatch(self, tmp_path) -> None:
        from tract.publish.merge import merge_lora_adapters

        model_dir = _create_checkpoint_dir(tmp_path / "input")
        output_dir = tmp_path / "output"

        mock_model = _make_mock_model(fake_save_dir=output_dir)
        pre_emb = np.ones((3, 1024), dtype=np.float32)
        pre_emb /= np.linalg.norm(pre_emb, axis=1, keepdims=True)
        post_emb = np.random.default_rng(42).standard_normal((3, 1024)).astype(np.float32)
        post_emb /= np.linalg.norm(post_emb, axis=1, keepdims=True)
        mock_model.encode.side_effect = [pre_emb, post_emb]

        with patch("tract.publish.merge.SentenceTransformer", return_value=mock_model):
            with pytest.raises(RuntimeError, match="Merge verification failed"):
                merge_lora_adapters(model_dir, output_dir)

    def test_refuses_an_adapter_only_checkpoint(self, tmp_path) -> None:
        """Checkpoints written before the config fix must be named, not decoded.

        Publishing is where a months-old fold checkpoint gets picked up, so this
        is the path most likely to meet one.
        """
        from tract.publish.merge import merge_lora_adapters

        model_dir = tmp_path / "input"
        model_dir.mkdir()
        (model_dir / "adapter_config.json").write_text("{}")

        mock_model = _make_mock_model(fake_save_dir=tmp_path / "output")
        with patch("tract.publish.merge.SentenceTransformer", return_value=mock_model):
            with pytest.raises(RuntimeError, match="adapter-only checkpoint"):
                merge_lora_adapters(model_dir, tmp_path / "output")


# The rest of this file runs on mocks. This class alone builds a real adapter,
# which needs `peft` on top of the phase0 extra and downloads a small model.
# Without the guard the whole local suite fails here on a machine that
# deliberately has no training stack, which is every developer machine under
# the standing rule that inference and training run on RunPod.
@pytest.mark.skipif(
    importlib.util.find_spec("peft") is None,
    reason="needs the phase0 extra: peft",
)
class TestMergeRealAdapter:
    """End-to-end merge against a real adapter-only checkpoint.

    Every other test in this file mocks SentenceTransformer, so none of them can
    see that sentence-transformers 5.7 made `auto_model` a read-only property.
    Under 5.7 the loaded model carries an injected adapter rather than a
    PeftModel wrapper, so merge_and_unload is absent, and the assignment the old
    code used to recover from that was silently discarded -- publishing would
    have raised AttributeError. Only a real model exercises that path.
    """

    SMALL_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def _adapter_checkpoint(self, path: Path):
        import torch
        from peft import LoraConfig, TaskType
        from sentence_transformers import SentenceTransformer

        from tract.training.checkpoint import save_sentence_transformer

        torch.manual_seed(42)
        model = SentenceTransformer(self.SMALL_MODEL)
        model.max_seq_length = 64
        model.add_adapter(LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0,
            target_modules=["query", "key", "value"],
            task_type=TaskType.FEATURE_EXTRACTION,
        ))
        # lora_B initialises to zero, making the adapter an identity map. Perturb
        # it so a dropped adapter is actually observable in the embeddings.
        generator = torch.Generator().manual_seed(0)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                with torch.no_grad():
                    param.copy_(torch.randn(param.shape, generator=generator) * 0.05)
        # The save path under test, not a bare model.save: a bare save omits
        # config.json and the checkpoint cannot be reloaded at all.
        save_sentence_transformer(model, path)
        return model

    @pytest.mark.slow
    def test_merges_adapter_only_checkpoint(self, tmp_path) -> None:
        from sentence_transformers import SentenceTransformer

        from tract.publish.merge import MERGE_VERIFICATION_TEXTS, merge_lora_adapters

        model_dir = tmp_path / "checkpoint"
        output_dir = tmp_path / "merged"
        model = self._adapter_checkpoint(model_dir)
        assert (model_dir / "adapter_config.json").exists(), "expected an adapter-only checkpoint"

        before = model.encode(
            MERGE_VERIFICATION_TEXTS, normalize_embeddings=True, show_progress_bar=False,
        )

        merge_lora_adapters(model_dir, output_dir)

        # validate_merged_output already asserts these, but state them here so a
        # regression names the artifact rather than a helper.
        assert not (output_dir / "adapter_config.json").exists()
        assert (output_dir / "model.safetensors").exists()

        merged = SentenceTransformer(str(output_dir))
        merged.max_seq_length = 64
        after = merged.encode(
            MERGE_VERIFICATION_TEXTS, normalize_embeddings=True, show_progress_bar=False,
        )
        cosines = np.sum(before * after, axis=1)
        assert float(np.min(cosines)) >= 0.999, (
            f"merged model does not reproduce the adapter model: {cosines.tolist()}"
        )


class TestModelCardPublishesOnlyMeasuredFigures:
    """The card is the artifact a reader trusts; every figure in it must be one.

    Two defects motivate this class. `fold_results[0].get("hit_any", 0.6)`
    published "hit@5 is 60.0%" whenever the field was missing -- a number no run
    ever produced, presented in the same sentence as measured ones. Separately,
    the interpretive prose under the results table quoted 76.2%, 27.9% and "about
    half the time" as literals, so the card kept asserting the campaign-1 Gate 1
    headline that PRD.md:380-402 withdrew, no matter which run was being
    published.
    """

    def test_omits_the_hit5_figure_when_hit_any_is_absent(self, tmp_path) -> None:
        card = _render_card(tmp_path, CARD_FOLD_RESULTS)
        assert FABRICATED_HIT_ANY_RENDERING not in card, (
            "the card published a hit@any figure that no fold measured"
        )
        assert "hit@5" not in card, (
            "the card names a hit@5 measurement it never receives; hit@any is a "
            "top-1 measure and must not be relabelled as a top-5 one"
        )

    def test_quotes_hit_any_under_its_own_name_when_measured(self, tmp_path) -> None:
        folds = [dict(f) for f in CARD_FOLD_RESULTS]
        folds[0]["hit_any"] = 0.706
        folds[1]["hit_any"] = 0.422
        card = _render_card(tmp_path, folds)
        assert "42.2%" in card, (
            "the weakest fold's measured hit@any is absent from the limitations"
        )
        assert "hit@5" not in card
        assert FABRICATED_HIT_ANY_RENDERING not in card

    def test_no_withdrawn_campaign1_literal_survives(self, tmp_path) -> None:
        card = _render_card(tmp_path, CARD_FOLD_RESULTS)
        for literal in WITHDRAWN_CAMPAIGN1_LITERALS:
            assert literal not in card, (
                f"withdrawn campaign-1 figure {literal!r} is still rendered as a "
                f"current measurement; the folds passed in carry none of it"
            )

    def test_interpretive_prose_follows_the_measurements(self, tmp_path) -> None:
        """The strongest and weakest folds are named from the data.

        With Framework A at 0.588 and Framework B at 0.311 there is exactly one
        right answer, and it is not the one the old literals gave.
        """
        card = _render_card(tmp_path, CARD_FOLD_RESULTS)
        assert "58.8%" in card
        assert "31.1%" in card
        strongest = card.index("Framework A (58.8% hit@1)")
        weakest = card.index("Framework B (31.1% hit@1)")
        assert strongest < weakest, (
            "the strongest fold must be introduced before the weakest"
        )

    def test_prose_does_not_depend_on_fold_order(self, tmp_path) -> None:
        """`fold_results[0]` was assumed to be MITRE ATLAS and named as such.

        Reordering the caller's list is enough to move one fold's number under
        another fold's name, so the rendered interpretation is compared across
        both orderings.
        """
        forward = _render_card(tmp_path / "forward", list(CARD_FOLD_RESULTS))
        reversed_ = _render_card(tmp_path / "reversed", list(reversed(CARD_FOLD_RESULTS)))

        assert _interpretation(forward) == _interpretation(reversed_)
        assert "MITRE ATLAS" not in _interpretation(forward), (
            "a fold name no caller supplied appears in the interpretation"
        )

    def test_tied_folds_still_render_deterministically(self, tmp_path) -> None:
        """Sorting on hit@1 alone would leave a tie in the caller's order.

        Two folds that scored identically would then swap places between runs
        and the card would stop being byte-identical on re-generation, so the
        tie breaks on fold name.
        """
        tied = [dict(fold) for fold in CARD_FOLD_RESULTS]
        tied[1]["hit1"] = tied[0]["hit1"]

        forward = _render_card(tmp_path / "forward", tied)
        reversed_ = _render_card(tmp_path / "reversed", list(reversed(tied)))
        assert _interpretation(forward) == _interpretation(reversed_)

    def test_bridge_counts_are_never_defaulted_to_zero(self, tmp_path) -> None:
        """`counts.get("accepted", 0)` published "Accepted bridges: 0" silently.

        Zero accepted bridges is a publishable research result, so it must not
        also be what a missing input looks like.
        """
        from tract.publish.model_card import generate_model_card

        with pytest.raises(ValueError, match="accepted"):
            generate_model_card(
                tmp_path,
                fold_results=_with_indicators(CARD_FOLD_RESULTS),
                calibration=CARD_CALIBRATION,
                ece_data=CARD_ECE,
                bridge_summary={"counts": {"rejected": 58, "total": 63}},
                gpu_hours=2.5,
            )
