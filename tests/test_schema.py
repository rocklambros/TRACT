"""Tests for tract.schema — Pydantic v2 models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tract.schema import Control, FrameworkOutput, HubLink


# ── Control ────────────────────────────────────────────────────────────────

class TestControl:
    """Tests for the Control model."""

    def test_valid_minimal(self) -> None:
        """Minimal required fields produce a valid Control."""
        c = Control(
            control_id="CTRL-001",
            title="Test Control",
            description="A short description of the control.",
        )
        assert c.control_id == "CTRL-001"
        assert c.title == "Test Control"
        assert c.full_text is None
        assert c.hierarchy_level is None
        assert c.parent_id is None
        assert c.parent_name is None
        assert c.metadata is None

    def test_valid_all_fields(self) -> None:
        """All optional fields populated."""
        c = Control(
            control_id="AICM-AIS-01",
            title="AI System Inventory",
            description="Organizations shall maintain a comprehensive inventory.",
            full_text="Organizations shall maintain a comprehensive inventory of all AI systems deployed...",
            hierarchy_level="control",
            parent_id="AIS",
            parent_name="AI System Lifecycle Security",
            metadata={"domain": "AI System Lifecycle Security"},
        )
        assert c.full_text is not None
        assert c.hierarchy_level == "control"
        assert c.metadata == {"domain": "AI System Lifecycle Security"}

    def test_strips_whitespace(self) -> None:
        """str_strip_whitespace in model_config strips leading/trailing."""
        c = Control(
            control_id="  CTRL-001  ",
            title="  padded title  ",
            description="  padded description  ",
        )
        assert c.control_id == "CTRL-001"
        assert c.title == "padded title"
        assert c.description == "padded description"

    def test_rejects_empty_control_id(self) -> None:
        """control_id must have min_length=1."""
        with pytest.raises(ValidationError, match="control_id"):
            Control(
                control_id="",
                title="Title",
                description="Description text.",
            )

    def test_rejects_empty_description(self) -> None:
        """description must have min_length=1."""
        with pytest.raises(ValidationError, match="description"):
            Control(
                control_id="CTRL-001",
                title="Title",
                description="",
            )

    def test_rejects_whitespace_only_description(self) -> None:
        """Whitespace-only description is stripped to empty -> rejected."""
        with pytest.raises(ValidationError, match="description"):
            Control(
                control_id="CTRL-001",
                title="Title",
                description="   ",
            )

    def test_allows_long_description(self) -> None:
        """Control accepts long descriptions (truncation is done by sanitize pipeline)."""
        c = Control(
            control_id="CTRL-001",
            title="Title",
            description="x" * 3000,
        )
        assert len(c.description) == 3000

    def test_description_at_max_length(self) -> None:
        """Exactly 2000 characters is valid."""
        c = Control(
            control_id="CTRL-001",
            title="Title",
            description="x" * 2000,
        )
        assert len(c.description) == 2000

    def test_serialization_roundtrip(self) -> None:
        """model_dump -> Control(**data) produces identical object."""
        original = Control(
            control_id="CTRL-001",
            title="Test",
            description="A description.",
            hierarchy_level="activity",
            parent_id="DOM-01",
            parent_name="Domain One",
            metadata={"key": "value"},
        )
        data = original.model_dump()
        restored = Control(**data)
        assert restored == original

    def test_json_roundtrip(self) -> None:
        """model_dump_json -> model_validate_json roundtrip."""
        original = Control(
            control_id="CTRL-001",
            title="Test",
            description="A description.",
        )
        json_str = original.model_dump_json()
        restored = Control.model_validate_json(json_str)
        assert restored == original


# ── FrameworkOutput ────────────────────────────────────────────────────────

class TestFrameworkOutput:
    """Tests for the FrameworkOutput model."""

    def _make_control(self) -> Control:
        return Control(
            control_id="C-1",
            title="Test",
            description="Test description.",
        )

    def test_valid(self) -> None:
        fo = FrameworkOutput(
            framework_id="test_fw",
            framework_name="Test Framework",
            version="1.0",
            source_url="https://example.com",
            fetched_date="2026-04-27",
            mapping_unit_level="control",
            controls=[self._make_control()],
        )
        assert fo.framework_id == "test_fw"
        assert len(fo.controls) == 1

    def test_rejects_empty_controls(self) -> None:
        """controls list must have min_length=1."""
        with pytest.raises(ValidationError, match="controls"):
            FrameworkOutput(
                framework_id="test_fw",
                framework_name="Test Framework",
                version="1.0",
                source_url="https://example.com",
                fetched_date="2026-04-27",
                mapping_unit_level="control",
                controls=[],
            )


# ── HubLink ───────────────────────────────────────────────────────────────

class TestHubLink:
    """Tests for the HubLink model."""

    def test_valid(self) -> None:
        hl = HubLink(
            cre_id="123-456",
            cre_name="Data poisoning",
            standard_name="MITRE ATLAS",
            section_id="AML.T0020",
            section_name="Poison Training Data",
            link_type="LinkedTo",
            framework_id="mitre_atlas",
        )
        assert hl.cre_id == "123-456"
        assert hl.link_type == "LinkedTo"

    def test_rejects_empty_cre_id(self) -> None:
        with pytest.raises(ValidationError, match="cre_id"):
            HubLink(
                cre_id="",
                cre_name="Name",
                standard_name="Std",
                section_id="S1",
                section_name="Section",
                link_type="LinkedTo",
                framework_id="fw",
            )
