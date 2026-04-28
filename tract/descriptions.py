"""TRACT hub description models and prompt rendering.

Provides Pydantic models for hub descriptions and the prompt template
used to generate them via Opus.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class HubDescription(BaseModel):
    """A generated description for a single CRE leaf hub."""

    model_config = ConfigDict(str_strip_whitespace=True)

    hub_id: str = Field(..., min_length=1)
    hub_name: str = Field(..., min_length=1)
    hierarchy_path: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    model: str = Field(..., min_length=1)
    temperature: float
    generated_at: str = Field(..., min_length=1)
    review_status: Literal["pending", "accepted", "edited", "rejected"] = "pending"
    reviewed_description: str | None = None
    reviewer_notes: str | None = None


class HubDescriptionSet(BaseModel):
    """Container for all hub descriptions with generation metadata."""

    model_config = ConfigDict(str_strip_whitespace=True)

    descriptions: dict[str, HubDescription]
    generation_model: str = Field(..., min_length=1)
    generation_timestamp: str = Field(..., min_length=1)
    data_hash: str = Field(..., min_length=1)
    total_generated: int = Field(..., ge=0)
    total_pending_review: int = Field(..., ge=0)


DESCRIPTION_SYSTEM_PROMPT: str = (
    "You are a cybersecurity taxonomy expert. Generate a precise 2-3 "
    "sentence description for a CRE (Common Requirements Enumeration) "
    "hub node.\n\n"
    "Write a description that:\n"
    "1. Defines what this hub covers in concrete terms\n"
    "2. Distinguishes it from its sibling hubs\n"
    "3. States the boundary of its scope (what it does NOT cover)\n\n"
    "Be specific and technical. Do not use filler phrases. "
    "Every word must add information."
)


def build_description_prompt(
    hub_name: str,
    hierarchy_path: str,
    sibling_names: list[str],
    linked_section_names: list[str],
) -> str:
    """Build the user message for generating one hub description."""
    siblings_str = ", ".join(sibling_names[:20]) if sibling_names else "(none)"
    linked_str = ", ".join(linked_section_names[:50]) if linked_section_names else "(none)"

    return (
        f"Hub name: {hub_name}\n"
        f"Hierarchy path: {hierarchy_path}\n"
        f"Sibling hubs (same parent): {siblings_str}\n"
        f"Linked standard sections: {linked_str}"
    )
