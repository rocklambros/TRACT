"""Tests for canonical export Pydantic models (spec §2.1)."""
from __future__ import annotations

import pytest


def _make_control(
    control_id: str = "fw1:c1",
    framework_id: str = "fw1",
    section_id: str = "c1",
    title: str = "Control 1",
    description: str = "Description 1",
    hyperlink: str = "https://example.com",
) -> dict:
    return {
        "control_id": control_id,
        "framework_id": framework_id,
        "section_id": section_id,
        "title": title,
        "description": description,
        "hyperlink": hyperlink,
    }


def _make_mapping(
    control_id: str = "fw1:c1",
    hub_id: str = "004-517",
    hub_name: str = "Security requirements",
    confidence: float = 0.75,
    rank: int = 1,
    provenance: str = "active_learning_round_2",
    model_version: str = "7e8b8f834db5",
) -> dict:
    return {
        "control_id": control_id,
        "hub_id": hub_id,
        "hub_name": hub_name,
        "confidence": confidence,
        "rank": rank,
        "link_type": "TRACT_ML_PREDICTED",
        "provenance": provenance,
        "model_version": model_version,
    }


class TestCanonicalControl:
    def test_roundtrip(self) -> None:
        from tract.export.canonical_schema import CanonicalControl

        ctrl = CanonicalControl(**_make_control())
        assert ctrl.control_id == "fw1:c1"
        assert ctrl.hyperlink == "https://example.com"
        d = ctrl.model_dump()
        assert CanonicalControl(**d) == ctrl

    def test_rejects_missing_fields(self) -> None:
        from tract.export.canonical_schema import CanonicalControl

        with pytest.raises(Exception):
            CanonicalControl(control_id="x", framework_id="y")  # type: ignore[call-arg]


class TestCREMapping:
    def test_roundtrip(self) -> None:
        from tract.export.canonical_schema import CREMapping

        m = CREMapping(**_make_mapping())
        assert m.rank == 1
        assert m.link_type == "TRACT_ML_PREDICTED"
        assert m.model_version == "7e8b8f834db5"

    def test_default_link_type(self) -> None:
        from tract.export.canonical_schema import CREMapping

        data = _make_mapping()
        del data["link_type"]
        m = CREMapping(**data)
        assert m.link_type == "TRACT_ML_PREDICTED"


class TestFilterPolicy:
    def test_defaults(self) -> None:
        from tract.export.canonical_schema import FilterPolicy

        fp = FilterPolicy(confidence_floor=0.3, confidence_override=None)
        assert fp.excluded_ground_truth is True
        assert fp.excluded_ood is True
        assert fp.review_status_required == "accepted"


class TestStandardSnapshot:
    def test_roundtrip(self) -> None:
        from tract.export.canonical_schema import (
            CanonicalControl,
            CREMapping,
            FilterPolicy,
            StandardSnapshot,
        )

        snap = StandardSnapshot(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**_make_control())],
            mappings=[CREMapping(**_make_mapping())],
        )
        assert snap.schema_version == "1.0"
        assert len(snap.controls) == 1
        assert len(snap.mappings) == 1


class TestContentHash:
    def test_deterministic(self) -> None:
        from tract.export.canonical_schema import compute_content_hash, StandardSnapshot, FilterPolicy, CanonicalControl, CREMapping

        snap = StandardSnapshot(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**_make_control())],
            mappings=[CREMapping(**_make_mapping())],
        )
        h1 = compute_content_hash(snap)
        h2 = compute_content_hash(snap)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_excludes_date_and_hash(self) -> None:
        from tract.export.canonical_schema import compute_content_hash, StandardSnapshot, FilterPolicy, CanonicalControl, CREMapping

        base = dict(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            controls=[CanonicalControl(**_make_control())],
            mappings=[CREMapping(**_make_mapping())],
        )
        snap1 = StandardSnapshot(export_date="2026-05-04T00:00:00Z", **base)
        snap2 = StandardSnapshot(export_date="2026-12-25T00:00:00Z", **base)
        assert compute_content_hash(snap1) == compute_content_hash(snap2)

    def test_different_data_different_hash(self) -> None:
        from tract.export.canonical_schema import compute_content_hash, StandardSnapshot, FilterPolicy, CanonicalControl, CREMapping

        base = dict(
            schema_version="1.0",
            framework_id="fw1",
            framework_name="Framework One",
            export_date="2026-05-04T00:00:00Z",
            content_hash="placeholder",
            tract_version="abc123",
            model_adapter_hash="7e8b8f834db5",
            filter_policy=FilterPolicy(confidence_floor=0.3, confidence_override=None),
            mappings=[CREMapping(**_make_mapping())],
        )
        snap1 = StandardSnapshot(controls=[CanonicalControl(**_make_control())], **base)
        snap2 = StandardSnapshot(controls=[CanonicalControl(**_make_control(title="Different"))], **base)
        assert compute_content_hash(snap1) != compute_content_hash(snap2)
