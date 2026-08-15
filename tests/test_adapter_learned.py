"""Tests for the post-training adapter check in tract/training/loop.py.

PEFT initialises every lora_B to zeros, so an adapter that never receives a
gradient is the identity and the run produces the base model while the record
claims a fine-tune. That is the same silent-success shape as the ST 5.x
auto_model defect, and it is why this is asserted rather than assumed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from tract.training.config import TrainingConfig  # noqa: E402
from tract.training.loop import _assert_adapter_learned  # noqa: E402


class _Model(nn.Module):
    """Minimal stand-in exposing lora_A/lora_B named parameters."""

    def __init__(self, b_values: list[float]) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(len(b_values)))
        self.lora_B = nn.Parameter(torch.tensor(b_values))


def _config(**kwargs: object) -> TrainingConfig:
    return TrainingConfig(name="t", **kwargs)  # type: ignore[arg-type]


def test_a_trained_adapter_passes() -> None:
    _assert_adapter_learned(_Model([0.0, 0.31, 0.0]), _config())


def test_an_untouched_adapter_raises() -> None:
    """lora_B all zeros means no gradient ever arrived."""
    with pytest.raises(RuntimeError, match="still zero after training"):
        _assert_adapter_learned(_Model([0.0, 0.0, 0.0]), _config())


def test_the_error_names_gradient_checkpointing_when_it_is_on() -> None:
    """It is the most likely cause, so the message should say so."""
    with pytest.raises(RuntimeError, match="gradient_checkpointing is on"):
        _assert_adapter_learned(
            _Model([0.0, 0.0]), _config(gradient_checkpointing=True),
        )


def test_the_error_does_not_blame_checkpointing_when_it_is_off() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        _assert_adapter_learned(
            _Model([0.0, 0.0]), _config(gradient_checkpointing=False),
        )
    assert "gradient_checkpointing is on" not in str(excinfo.value)


def test_a_missing_adapter_raises_for_a_lora_run() -> None:
    with pytest.raises(RuntimeError, match="not attached"):
        _assert_adapter_learned(nn.Linear(2, 2), _config())


def test_a_full_finetune_needs_no_adapter() -> None:
    """lora_rank=0 is the A9 full fine-tune arm; there is nothing to check."""
    _assert_adapter_learned(nn.Linear(2, 2), _config(lora_rank=0))


def test_negative_weights_count_as_learned() -> None:
    """A tensor can move away from zero in either direction."""
    _assert_adapter_learned(_Model([-0.4, 0.0]), _config())
