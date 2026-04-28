"""Tests for tract.descriptions — hub description models and prompt rendering."""
from __future__ import annotations

import pytest
from pydantic import ValidationError


class TestHubDescriptionModel:

    def test_valid_description(self) -> None:
        from tract.descriptions import HubDescription
        desc = HubDescription(
            hub_id="123-456",
            hub_name="Input validation",
            hierarchy_path="Root > Security > Input validation",
            description="Covers input validation controls.",
            model="claude-opus-4-20250514",
            temperature=0.0,
            generated_at="2026-04-28T12:00:00Z",
            review_status="pending",
            reviewed_description=None,
            reviewer_notes=None,
        )
        assert desc.hub_id == "123-456"
        assert desc.review_status == "pending"

    def test_rejects_empty_description(self) -> None:
        from tract.descriptions import HubDescription
        with pytest.raises(ValidationError):
            HubDescription(
                hub_id="123-456",
                hub_name="Test",
                hierarchy_path="Root > Test",
                description="",
                model="test",
                temperature=0.0,
                generated_at="2026-04-28T12:00:00Z",
                review_status="pending",
                reviewed_description=None,
                reviewer_notes=None,
            )

    def test_rejects_invalid_review_status(self) -> None:
        from tract.descriptions import HubDescription
        with pytest.raises(ValidationError):
            HubDescription(
                hub_id="123-456",
                hub_name="Test",
                hierarchy_path="Root > Test",
                description="Some description.",
                model="test",
                temperature=0.0,
                generated_at="2026-04-28T12:00:00Z",
                review_status="invalid_status",
                reviewed_description=None,
                reviewer_notes=None,
            )


class TestHubDescriptionSetModel:

    def test_valid_set(self) -> None:
        from tract.descriptions import HubDescription, HubDescriptionSet
        desc = HubDescription(
            hub_id="123-456",
            hub_name="Test",
            hierarchy_path="Root > Test",
            description="Test description.",
            model="claude-opus-4-20250514",
            temperature=0.0,
            generated_at="2026-04-28T12:00:00Z",
            review_status="pending",
            reviewed_description=None,
            reviewer_notes=None,
        )
        desc_set = HubDescriptionSet(
            descriptions={"123-456": desc},
            generation_model="claude-opus-4-20250514",
            generation_timestamp="2026-04-28T12:00:00Z",
            data_hash="abc123",
            total_generated=1,
            total_pending_review=1,
        )
        assert len(desc_set.descriptions) == 1
