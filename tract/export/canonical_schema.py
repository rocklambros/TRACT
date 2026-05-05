"""Pydantic v2 models for the canonical export format (spec §2)."""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel


class FilterPolicy(BaseModel):
    confidence_floor: float
    confidence_override: float | None
    excluded_ground_truth: bool = True
    excluded_ood: bool = True
    excluded_null_confidence: bool = True
    review_status_required: str = "accepted"


class CanonicalControl(BaseModel):
    control_id: str
    framework_id: str
    section_id: str
    title: str
    description: str
    hyperlink: str


class CREMapping(BaseModel):
    control_id: str
    hub_id: str
    hub_name: str
    confidence: float
    rank: int
    link_type: str = "TRACT_ML_PREDICTED"
    provenance: str
    model_version: str


class StandardSnapshot(BaseModel):
    schema_version: str = "1.0"
    framework_id: str
    framework_name: str
    export_date: str
    content_hash: str
    tract_version: str
    model_adapter_hash: str
    filter_policy: FilterPolicy
    controls: list[CanonicalControl]
    mappings: list[CREMapping]


class ChangesetEntry(BaseModel):
    operation: Literal[
        "ADD_CONTROL", "UPDATE_CONTROL", "DELETE_CONTROL",
        "ADD_MAPPING", "UPDATE_MAPPING", "DELETE_MAPPING",
    ]
    entity: CanonicalControl | CREMapping | None = None
    before: CanonicalControl | CREMapping | None = None
    key: str | None = None


class ChangesetSummary(BaseModel):
    controls_added: int
    controls_updated: int
    controls_deleted: int
    mappings_added: int
    mappings_updated: int
    mappings_deleted: int


class ImpactAnalysis(BaseModel):
    affected_hubs: list[str]
    affected_frameworks: list[str]
    co_mapped_changes: int
    scope: str


class Changeset(BaseModel):
    schema_version: str = "1.0"
    framework_id: str
    from_version: str | None
    to_version: str
    export_date: str
    operations: list[ChangesetEntry]
    summary: ChangesetSummary
    impact: ImpactAnalysis


def compute_content_hash(snapshot: StandardSnapshot) -> str:
    """Compute SHA-256 of snapshot excluding volatile fields (spec §2.2)."""
    data = snapshot.model_dump(exclude={"content_hash", "export_date"})
    canonical_json = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
