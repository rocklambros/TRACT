"""The encoders this project can fine-tune, pinned and described.

BGE-large was chosen in 2023 and is the weakest option now available: 335M
parameters and a 512-token ceiling that is architectural, not configurable
(BertModel, absolute position embeddings, a 512-row matrix). Everything else
here is larger, longer-context, or both, and all of them are native
sentence-transformers models with the same Transformer/Pooling/Normalize
structure, so swapping is a config change rather than a rewrite.

Every entry carries a revision. A model checkpoint loaded into the training
process is a dependency, and QC.1 requires dependencies to be pinned; the
serving path already pins TRACT_MODEL_PINNED_REVISION with a sha256, while
the training path fetched whatever `main` pointed at on the day. That made
the number a run produced untraceable to the weights that produced it.

Only model types the pinned transformers supports natively are listed.
Alibaba-NLP/gte-large-en-v1.5 is excluded despite being a good fit by hidden
size: it declares model_type "new" and requires trust_remote_code=True, which
executes code from the repository on every training pod.

Owner: TRACT. See PRD Section 6.4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class EncoderSpec:
    """One fine-tunable encoder, pinned to a commit.

    hidden_size and num_layers are carried because peak activation memory
    scales with both, and the memory pre-flight was calibrated on BGE-large's
    shape. Without them a guard tuned on a 1024-wide 24-layer model gives
    false confidence about a 2560-wide 36-layer one.
    """

    revision: str
    max_seq_length: int
    hidden_size: int
    num_layers: int
    model_type: str
    # Attention projections LoRA attaches to. PEFT raises when none match,
    # but only after the encoder has downloaded and the zero-shot pass has
    # run, so the wrong names cost a fold's setup on every pod.
    lora_target_modules: tuple[str, ...]
    note: str = ""

    @property
    def activation_cost_ratio(self) -> float:
        """Peak activation memory per token-slot, relative to BGE-large.

        Activations scale with width times depth, which is the first-order
        term for a transformer at fixed batch and sequence length.
        """
        return (self.hidden_size * self.num_layers) / (1024 * 24)


_BERT_ATTENTION: Final[tuple[str, ...]] = ("query", "key", "value")
_QWEN_ATTENTION: Final[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj")
_MODERNBERT_ATTENTION: Final[tuple[str, ...]] = ("Wqkv", "Wo")

ENCODERS: Final[dict[str, EncoderSpec]] = {
    "BAAI/bge-large-en-v1.5": EncoderSpec(
        revision="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        max_seq_length=512, hidden_size=1024, num_layers=24,
        model_type="bert", lora_target_modules=_BERT_ATTENTION,
        note="The incumbent. 512 tokens is a hard architectural ceiling.",
    ),
    "Alibaba-NLP/gte-modernbert-base": EncoderSpec(
        revision="e7f32e3c00f91d699e8c43b53106206bcc72bb22",
        max_seq_length=8192, hidden_size=768, num_layers=22,
        model_type="modernbert", lora_target_modules=_MODERNBERT_ATTENTION,
        note="Smallest and fastest. Narrower than BGE at 768.",
    ),
    "BAAI/bge-m3": EncoderSpec(
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        max_seq_length=8194, hidden_size=1024, num_layers=24,
        model_type="xlm-roberta", lora_target_modules=_BERT_ATTENTION,
        note="Same shape as BGE-large with 16x the context.",
    ),
    "Qwen/Qwen3-Embedding-0.6B": EncoderSpec(
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        max_seq_length=32768, hidden_size=1024, num_layers=28,
        model_type="qwen3", lora_target_modules=_QWEN_ATTENTION,
        note="Same width as BGE, 1.8x the parameters, 64x the context.",
    ),
    # Small and fast; used by the test suite so the LoRA save/reload
    # regression can run without pulling a multi-gigabyte encoder. Pinned
    # like the rest, because a test that silently changes model is a test
    # that stops testing what it was written for.
    "sentence-transformers/all-MiniLM-L6-v2": EncoderSpec(
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        max_seq_length=512, hidden_size=384, num_layers=6,
        model_type="bert", lora_target_modules=_BERT_ATTENTION,
        note="Test fixture. Too small for a real run.",
    ),
    "Qwen/Qwen3-Embedding-4B": EncoderSpec(
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        max_seq_length=40960, hidden_size=2560, num_layers=36,
        model_type="qwen3", lora_target_modules=_QWEN_ATTENTION,
        note="12x BGE's parameters. 3.75x its activation cost per token-slot.",
    ),
}


def resolve(model_name: str) -> EncoderSpec:
    """Look up an encoder, or refuse with the list of known ones.

    An allowlist rather than a free-text argument, so an unpinned or
    remote-code model cannot reach a training pod by typo.
    """
    spec = ENCODERS.get(model_name)
    if spec is None:
        raise ValueError(
            f"Unknown encoder {model_name!r}. Known: {sorted(ENCODERS)}. "
            f"Add it to tract/encoders.py with a pinned revision and its "
            f"LoRA target modules; a model fetched at 'main' cannot be tied "
            f"to the number it produced."
        )
    return spec
