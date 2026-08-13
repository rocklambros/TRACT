"""Record shapes for the reviewer-facing export document.

The review export is a contract, not a scratch dict. `tract review-export`
writes it, a human edits the decision fields, and `tract review-validate` and
`tract review-import` read it back. Declaring the shape here keeps the
producer and both consumers honest about what the file contains, and lets
mypy catch a renamed or dropped key instead of a reviewer discovering it.

Field types mirror the crosswalk schema: assignments.id is INTEGER, every
identifier column is TEXT.
"""
from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class AlternativeHub(TypedDict):
    """A runner-up hub shown to the reviewer next to the assigned one."""

    hub_id: str
    hub_name: str
    confidence: float


class ReviewItem(TypedDict):
    """One control awaiting review.

    Calibration items use the same shape as model predictions on purpose:
    the reviewer must not be able to tell them apart, which is what makes
    them useful as a calibration control. `id` is negative for calibration
    items and the positive assignments.id for real predictions.

    The four reviewer fields start empty and are filled in by hand.
    """

    id: int
    control_id: str
    framework_id: str
    framework_name: str
    section_id: str
    control_title: str
    control_text: str
    assigned_hub_id: str
    assigned_hub_name: str
    assigned_hub_path: str
    confidence: float
    raw_similarity: float
    is_ood: bool
    in_conformal_set: bool
    text_quality: str
    review_priority: str
    provenance: str
    alternative_hubs: list[AlternativeHub]
    decision: str | None
    reviewer_hub_id: str | None
    reviewer_notes: str | None
    status: str


class HubPredictionDict(TypedDict):
    """`HubPrediction.to_dict()` output, which is `asdict()` of the dataclass.

    Spelled out here so consumers index it against a checked contract. The
    confidence key is `calibrated_confidence`; `tract ingest` read a
    non-existent `confidence` key for months and raised KeyError on its own
    happy path, because nothing type-checked this indexing.
    """

    hub_id: str
    hub_name: str
    hierarchy_path: str
    raw_similarity: float
    calibrated_confidence: float
    in_conformal_set: bool
    is_ood: bool


class IngestSummary(TypedDict):
    """Counts block of the ingest review document."""

    total_controls: int
    ood_flagged: int
    duplicate_flagged: int
    similar_flagged: int
    high_confidence: int
    low_confidence: int


class IngestQualitySummary(TypedDict):
    """Aggregate match-quality block of the ingest review document."""

    mean_max_cosine_sim: float
    ood_fraction: float
    below_confidence_floor_count: int
    below_confidence_floor_fraction: float


class IngestControl(TypedDict):
    """One control as written into the ingest review document."""

    control_id: str
    title: str
    description: str
    # Control.full_text is optional in the parsed framework schema.
    full_text: str | None
    predictions: list[HubPredictionDict]
    is_ood: bool
    duplicates: list[dict[str, Any]]
    similar: list[dict[str, Any]]
    review: dict[str, str]


class IngestReviewDocument(TypedDict):
    """Top level of <framework>_review.json written by `tract ingest`.

    calibration_note and quality_summary are attached after construction, so
    they are NotRequired rather than part of the initial literal.
    """

    framework_id: str
    framework_name: str
    version: str
    fetched_date: str
    source_url: str
    generated_at: str
    model_version: str
    context: str
    summary: IngestSummary
    controls: list[IngestControl]
    calibration_note: NotRequired[str]
    quality_summary: NotRequired[IngestQualitySummary]


class ExportMetadata(TypedDict):
    """Summary header for the export, also the return value of the export."""

    generated_at: str
    model_version: str
    total_predictions: int
    calibration_items: int
    framework_breakdown: dict[str, int]
    priority_breakdown: dict[str, int]


class ReviewExportDocument(TypedDict):
    """Top level of review_export.json."""

    metadata: ExportMetadata
    predictions: list[ReviewItem]
