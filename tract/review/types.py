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

from typing import Any, Final, NotRequired, TypedDict


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
    """Summary header written into review_export.json -- the reviewer's copy.

    `calibration_items` is deliberately absent. Written here it is a check
    figure: it tells the reviewer being audited how many of their items are
    controls, which turns a guess about which ones into something they can
    verify (F19). The count is real and the operator needs it, so it lives on
    ExportSummary, which is returned to the caller and never serialised into
    the reviewer's file.
    """

    generated_at: str
    model_version: str
    total_predictions: int
    framework_breakdown: dict[str, int]
    priority_breakdown: dict[str, int]


class ExportSummary(ExportMetadata):
    """What generate_review_export returns to its caller, who is the operator.

    A superset of what is written. Anything added here is visible to the
    operator only; anything added to ExportMetadata reaches the reviewer.
    """

    calibration_items: int


class ReviewExportDocument(TypedDict):
    """Top level of review_export.json."""

    metadata: ExportMetadata
    predictions: list[ReviewItem]


class CoverageMetrics(TypedDict):
    """Review completion counts."""

    total_predictions: int
    reviewed: int
    pending: int
    completion_pct: float


class OverallRates(TypedDict):
    """Accept/reject/reassign counts and percentages across all frameworks."""

    accepted: int
    accepted_rate: float
    rejected: int
    rejected_rate: float
    reassigned: int
    reassigned_rate: float


class FrameworkRates(TypedDict):
    """Per-framework decision counters."""

    framework_name: str
    accepted: int
    rejected: int
    reassigned: int


class CalibrationDisagreement(TypedDict):
    """A calibration item the reviewer did not agree with."""

    id: int | None
    assigned_hub_id: str | None
    status: str
    reviewer_hub_id: str | None


class CalibrationQuality(TypedDict):
    """Reviewer quality measured against the seeded calibration items."""

    total_calibration: int
    reviewed: int
    agreed: int
    quality_score: float | None
    disagreements: list[CalibrationDisagreement]


class AcceptanceRate(TypedDict):
    """Acceptance counts for one confidence band."""

    total: int
    accepted: int
    acceptance_rate: float


class ConfidenceAnalysis(TypedDict):
    """Acceptance broken out by confidence band and OOD flag."""

    high_confidence: AcceptanceRate
    low_confidence: AcceptanceRate
    ood_items: AcceptanceRate


class ReviewMetrics(TypedDict):
    """Top level of review_metrics.json."""

    import_round: int
    coverage: CoverageMetrics
    overall: OverallRates
    per_framework: dict[str, FrameworkRates]
    reviewer_quality: CalibrationQuality
    confidence_analysis: ConfidenceAnalysis


# Used when review_metrics.json is absent. `tract publish-dataset` previously
# passed a bare {} in that case and the card reader papered over it with .get
# defaults; now that the card indexes the document directly, the fallback has
# to be a well-formed empty record rather than an empty dict.
EMPTY_REVIEW_METRICS: Final[ReviewMetrics] = {
    "import_round": 0,
    "coverage": {
        "total_predictions": 0,
        "reviewed": 0,
        "pending": 0,
        "completion_pct": 0.0,
    },
    "overall": {
        "accepted": 0,
        "accepted_rate": 0.0,
        "rejected": 0,
        "rejected_rate": 0.0,
        "reassigned": 0,
        "reassigned_rate": 0.0,
    },
    "per_framework": {},
    "reviewer_quality": {
        "total_calibration": 0,
        "reviewed": 0,
        "agreed": 0,
        "quality_score": None,
        "disagreements": [],
    },
    "confidence_analysis": {
        "high_confidence": {"total": 0, "accepted": 0, "acceptance_rate": 0.0},
        "low_confidence": {"total": 0, "accepted": 0, "acceptance_rate": 0.0},
        "ood_items": {"total": 0, "accepted": 0, "acceptance_rate": 0.0},
    },
}
