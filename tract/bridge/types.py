"""Typed structures for bridge analysis domain objects."""
from __future__ import annotations

from typing import TypedDict


class RawCandidate(TypedDict):
    """Candidate as produced by extract_top_k (before orchestrator enrichment)."""

    ai_hub_id: str
    trad_hub_id: str
    cosine_similarity: float
    rank_for_ai_hub: int


class BridgeCandidate(TypedDict):
    """Fully enriched bridge candidate with all fields set."""

    ai_hub_id: str
    trad_hub_id: str
    cosine_similarity: float
    rank_for_ai_hub: int
    ai_hub_name: str
    trad_hub_name: str
    ai_controls_linked: int
    trad_controls_linked: int
    status: str
    reviewer_notes: str
    description: str


class RawNegative(TypedDict):
    """Negative control as produced by extract_negatives."""

    ai_hub_id: str
    trad_hub_id: str
    cosine_similarity: float
    is_negative: bool


class NegativeControl(TypedDict):
    """Fully enriched negative control with description."""

    ai_hub_id: str
    trad_hub_id: str
    cosine_similarity: float
    is_negative: bool
    ai_hub_name: str
    trad_hub_name: str
    description: str


class SimilarityStats(TypedDict):
    """Summary statistics for the similarity matrix."""

    matrix_shape: list[int]
    mean: float
    std: float
    min: float
    max: float
    percentiles: dict[str, float]
