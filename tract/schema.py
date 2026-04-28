"""TRACT Pydantic v2 models for the standardized control schema.

Defines the data contract between parsers and all downstream phases.
Mirrors PRD Section 4.8.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Control(BaseModel):
    """A single security control / mapping unit from a framework."""

    model_config = ConfigDict(str_strip_whitespace=True)

    control_id: str = Field(..., min_length=1)
    title: str
    description: str = Field(..., min_length=1)
    full_text: str | None = None
    hierarchy_level: str | None = None
    parent_id: str | None = None
    parent_name: str | None = None
    metadata: dict[str, str | list[str]] | None = None


class FrameworkOutput(BaseModel):
    """Top-level output schema for a parsed framework.

    Each parser writes exactly one of these to data/processed/frameworks/.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    framework_id: str = Field(..., min_length=1)
    framework_name: str = Field(..., min_length=1)
    version: str
    source_url: str
    fetched_date: str = Field(..., min_length=1)
    mapping_unit_level: str = Field(..., min_length=1)
    controls: list[Control] = Field(..., min_length=1)


class HubLink(BaseModel):
    """A single standard-section-to-CRE-hub link from OpenCRE.

    Represents one row of training data: a known mapping between
    a framework section and a CRE hub concept.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    cre_id: str = Field(..., min_length=1)
    cre_name: str = Field(..., min_length=1)
    standard_name: str = Field(..., min_length=1)
    section_id: str
    section_name: str
    link_type: str = Field(..., min_length=1)
    framework_id: str
