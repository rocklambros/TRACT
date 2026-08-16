from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tract.api.settings import get_settings


# Module-level constants 
MAX_TEXT_LENGTH = 8192
MAX_BATCH_SIZE = 256
MAX_TOP_K = 10
DEFAULT_TOP_K = 5
DEFAULT_DUPLICATE_THRESHOLD = 0.95
DEFAULT_SIMILAR_THRESHOLD = 0.85


class _ApiModel(BaseModel):
    """Common base — disables pydantic's `model_*` namespace protection
    so we can keep API field names like `model_version` without warnings."""
    model_config = ConfigDict(protected_namespaces=())


# Request models

class AssignRequest(_ApiModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Free-form security control text to assign to a CRE hub.",
    )
    top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        le=MAX_TOP_K,
        description="Number of top-ranked hub candidates to return.",
    )
    include_raw_similarity: bool = Field(
        default=False,
        description="Surface raw cosine similarity. Useful for research; not needed for production use.",
    )
    


class ControlInput(_ApiModel):
    id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        pattern=r"^[A-Za-z0-9._:\-/]+$",  # whitelist — adjust to match your real ID formats
        description="Caller-supplied identifier. Alphanumeric + ._:-/ only, max 256 chars.",
    )
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)

class AssignBatchRequest(_ApiModel):
    controls: list[ControlInput] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K)


class DuplicateRequest(_ApiModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    duplicate_threshold: float = Field(
        default=DEFAULT_DUPLICATE_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Cosine similarity at or above which a match is a near-duplicate.",
    )
    similar_threshold: float = Field(
        default=DEFAULT_SIMILAR_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Cosine similarity at or above which a match is merely similar. Must be <= duplicate_threshold.",
    )

    @model_validator(mode="after")
    def _check_thresholds(self) -> "DuplicateRequest":
        floor = get_settings().duplicates_min_threshold
        if self.similar_threshold < floor:
            raise ValueError(
                f"similar_threshold ({self.similar_threshold}) is below the deployment-configured "
                f"minimum ({floor}). The minimum protects the indexed corpus from enumeration. "
                f"Operators can lower TRACT_API_DUPLICATES_MIN_THRESHOLD if they understand the risk."
            )
        if self.duplicate_threshold < self.similar_threshold:
            raise ValueError(
                f"duplicate_threshold ({self.duplicate_threshold}) must be >= "
                f"similar_threshold ({self.similar_threshold})"
            )
        return self


#  Response models

class HubAssignment(_ApiModel):
    hub_id: str
    hub_name: str
    hierarchy_path: str
    raw_similarity: float | None = Field(
        default=None,
        description="Cosine similarity in [-1, 1] — not calibrated. Only present when the request set include_raw_similarity=true.",
    )
    calibrated_confidence: float = Field(
        ...,
        description="Calibrated probability in [0, 1]. Accuracy varies by framework — see model card.",
    )
    in_conformal_set: bool = Field(
        ...,
        description="True if this hub is in the conformal-prediction set guaranteeing the correct answer with target coverage.",
    )
    rank: int = Field(..., description="1-indexed rank within the top-k results.")


class AssignResponse(_ApiModel):
    decision_class: Literal["advisory_non_authoritative"] = "advisory_non_authoritative"
    model_card_url: str = Field(
        default="https://huggingface.co/rockCO78/tract-cre-assignment",
        description="Link to the documented intended use, accuracy bounds, and known limitations.",
    )
    assignments: list[HubAssignment]
    ood_flag: bool = Field(
        ...,
        description="True if the input is out-of-distribution. Assignments below are unreliable when this is True.",
    )
    model_adapter_hash: str  # full SHA-256, not short prefix — this is the audit-grade identity
    t_deploy: float
    ood_threshold: float
    request_id: str
    ood_score: float = Field(..., description="Max raw similarity across all hubs. Below threshold => OOD.")
    model_version: str


class BatchResultEntry(_ApiModel):
    control_id: str
    assignments: list[HubAssignment]
    ood_flag: bool
    ood_score: float


class BatchAssignResponse(_ApiModel):
    decision_class: Literal["advisory_non_authoritative"] = "advisory_non_authoritative"
    model_card_url: str = Field(
        default="https://huggingface.co/rockCO78/tract-cre-assignment",
        description="Link to the documented intended use, accuracy bounds, and known limitations.",
    )
    results: list[BatchResultEntry]
    model_version: str


class DuplicateMatchOut(_ApiModel):
    control_id: str
    framework_id: str
    similarity: float
    tier: Literal["duplicate", "similar"]


class DuplicateResponse(_ApiModel):
    decision_class: Literal["advisory_non_authoritative"] = "advisory_non_authoritative"
    model_card_url: str = Field(
        default="https://huggingface.co/rockCO78/tract-cre-assignment",
        description="Link to the documented intended use, accuracy bounds, and known limitations.",
    )
    duplicates: list[DuplicateMatchOut]
    similar: list[DuplicateMatchOut]


class HubSummary(_ApiModel):
    hub_id: str
    name: str
    hierarchy_path: str


class HubDetail(_ApiModel):
    hub_id: str
    name: str
    hierarchy_path: str
    parent_id: str | None
    children_ids: list[str]


class HubListResponse(_ApiModel):
    hubs: list[HubSummary]
    total: int
    page: int
    page_size: int


class HealthResponse(_ApiModel):
    status: Literal["ok"]
    model_adapter_hash: str
    t_deploy: float
    ood_threshold: float

class LivenessResponse(_ApiModel):
    status: Literal["alive"]


class VersionResponse(_ApiModel):
    tract_version: str
    model_adapter_hash: str
    deployment_artifact_timestamp: str
    package_version : str
    
