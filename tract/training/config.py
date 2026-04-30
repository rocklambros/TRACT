"""Training configuration and experiment metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tract.config import (
    PHASE1B_BASE_MODEL,
    PHASE1B_BATCH_SIZE,
    PHASE1B_HARD_NEGATIVES,
    PHASE1B_LEARNING_RATE,
    PHASE1B_LORA_ALPHA,
    PHASE1B_LORA_DROPOUT,
    PHASE1B_LORA_RANK,
    PHASE1B_LORA_TARGET_MODULES,
    PHASE1B_MAX_EPOCHS,
    PHASE1B_MAX_GRAD_NORM,
    PHASE1B_MAX_SEQ_LENGTH,
    PHASE1B_SAMPLING_TEMPERATURE,
    PHASE1B_SEED,
    PHASE1B_WARMUP_RATIO,
    PHASE1B_WEIGHT_DECAY,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Full configuration for one training experiment."""

    name: str
    base_model: str = PHASE1B_BASE_MODEL
    training_data: str = "joint-tempscaled"
    checkpoint_path: Path | None = None

    lora_rank: int = PHASE1B_LORA_RANK
    lora_alpha: int = PHASE1B_LORA_ALPHA
    lora_dropout: float = PHASE1B_LORA_DROPOUT
    lora_target_modules: list[str] = field(default_factory=lambda: list(PHASE1B_LORA_TARGET_MODULES))

    sampling_temperature: float = PHASE1B_SAMPLING_TEMPERATURE
    control_text_source: str = "section_name"

    batch_size: int = PHASE1B_BATCH_SIZE
    learning_rate: float = PHASE1B_LEARNING_RATE
    warmup_ratio: float = PHASE1B_WARMUP_RATIO
    weight_decay: float = PHASE1B_WEIGHT_DECAY
    max_grad_norm: float = PHASE1B_MAX_GRAD_NORM
    max_epochs: int = PHASE1B_MAX_EPOCHS
    max_seq_length: int = PHASE1B_MAX_SEQ_LENGTH
    hard_negatives: int = PHASE1B_HARD_NEGATIVES
    seed: int = PHASE1B_SEED

    hub_rep_format: str = "path+name"
    data_hash: str = ""

    def to_dict(self) -> dict:
        """Serialize for WandB/JSON logging."""
        return {
            "name": self.name,
            "base_model": self.base_model,
            "training_data": self.training_data,
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else None,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": self.lora_target_modules,
            "sampling_temperature": self.sampling_temperature,
            "control_text_source": self.control_text_source,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "max_epochs": self.max_epochs,
            "max_seq_length": self.max_seq_length,
            "hard_negatives": self.hard_negatives,
            "seed": self.seed,
            "hub_rep_format": self.hub_rep_format,
            "data_hash": self.data_hash,
        }
