"""Training configuration and experiment metadata."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    # control_text_source is NOT a field. It was a constant default that
    # nothing ever assigned, so every prose run recorded
    # "control_text_source": "section_name" beside "use_prose": true in the
    # same object. to_dict() now derives it from use_prose. Keeping it as a
    # field, even renamed, left `TrainingConfig(control_text_source=...)`
    # looking like it worked while changing nothing.

    batch_size: int = PHASE1B_BATCH_SIZE
    learning_rate: float = PHASE1B_LEARNING_RATE
    warmup_ratio: float = PHASE1B_WARMUP_RATIO
    weight_decay: float = PHASE1B_WEIGHT_DECAY
    max_grad_norm: float = PHASE1B_MAX_GRAD_NORM
    max_epochs: int = PHASE1B_MAX_EPOCHS
    max_seq_length: int = PHASE1B_MAX_SEQ_LENGTH
    hard_negatives: int = PHASE1B_HARD_NEGATIVES
    seed: int = PHASE1B_SEED

    # Recompute activations in the backward pass instead of storing them.
    # This does not change the loss, the batch composition, or the result --
    # it trades roughly 30% throughput for a large drop in activation memory.
    # Needed because the anchors changed: the batch size was tuned when a
    # control was a 22-character title (~8 tokens) and the same batch of 32
    # now carries paragraphs that fill the 512-token window, so peak
    # activation memory grew with the sequence length and the worst-case
    # batch OOMed an 80GB H100 partway through epoch 4.
    #
    # Reducing batch_size would have been the other lever and is the wrong
    # one: MultipleNegativesRankingLoss draws its negatives from within the
    # batch, so a smaller batch is a weaker training signal, and PRD 6.4
    # pre-registers the configuration. Checkpointing changes the arithmetic
    # not at all.
    gradient_checkpointing: bool = True

    hub_rep_format: str = "path+name"
    data_hash: str = ""

    # Text-selection arms. Both are recorded in to_dict(), so a run's anchors
    # can be reconstructed from its checkpoint metadata rather than inferred.
    #
    # use_prose: take each control's full text instead of its section title.
    # The pipeline read section_name unconditionally, training on three-word
    # titles while production is handed paragraphs.
    use_prose: bool = True
    # use_stopword_filter: strip corpus-derived boilerplate from control AND
    # hub text. Off by default and carried as an ablation arm: removing
    # function words moves input off the distribution a contextual encoder was
    # pretrained on, so the trade has to be measured rather than assumed.
    use_stopword_filter: bool = False
    # use_description_only: cut each control at its first remediation heading.
    # The encoder's 512-token budget is an architectural ceiling on BGE-large
    # (BertModel, absolute position embeddings), so the only lever is which
    # tokens it spends. 175 controls carry such a heading and the median puts
    # 47% of its body after it.
    use_description_only: bool = False
    # Temperature for flattening the CRE-branch distribution during batch
    # ordering. 0 disables it, leaving the binary is_ai behaviour untouched;
    # 1.0 is the natural distribution; larger flattens toward uniform.
    #
    # 72.1% of training links point at "Technical application security
    # controls" and 3.3% at the "Cross-cutting concerns" threat branch, and
    # none of CAPEC's 702 adversary-as-subject anchors point at a threat hub.
    # The model learns "attack narrative -> the control that stops it", which
    # is correct for CAPEC and wrong for MITRE ATLAS techniques: measured at
    # -29.4 hit@1 points on that stratum, which is the entire negative ATLAS
    # fold.
    branch_balance_temperature: float = 0.0

    def to_dict(self) -> dict[str, Any]:
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
            "control_text_source": ("full_prose" if self.use_prose else "section_name"),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "max_epochs": self.max_epochs,
            "max_seq_length": self.max_seq_length,
            "hard_negatives": self.hard_negatives,
            "seed": self.seed,
            "branch_balance_temperature": self.branch_balance_temperature,
            "gradient_checkpointing": self.gradient_checkpointing,
            "hub_rep_format": self.hub_rep_format,
            "use_prose": self.use_prose,
            "use_stopword_filter": self.use_stopword_filter,
            "use_description_only": self.use_description_only,
            "data_hash": self.data_hash,
        }
