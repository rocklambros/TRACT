"""LLM-assisted extractor for framework preparation.

Uses Claude API with tool_use structured output to extract controls
from unstructured documents (PDF, HTML, plain text).

Public API:
    LlmExtractor.extract(path, framework_id, output_dir) -> list[Control]
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from tract.config import (
    PREPARE_LLM_CHUNK_TOKEN_LIMIT,
    PREPARE_LLM_MAX_RETRIES,
    PREPARE_LLM_MODEL,
    PREPARE_LLM_RETRY_BACKOFF_FACTOR,
    PREPARE_LLM_RETRY_INITIAL_DELAY_S,
    PREPARE_LLM_TEMPERATURE,
)
from tract.io import atomic_write_json
from tract.schema import Control

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a security framework analyst. Your task is to extract EVERY "
    "security control, requirement, technique, or actionable item from "
    "the provided document. Be exhaustive — do not skip any controls. "
    "Each control must have a unique ID (use the document's numbering if "
    "available, otherwise generate sequential IDs like CTRL-001), a title, "
    "and a description that captures the full requirement text."
)


def _build_tool_schema() -> dict[str, Any]:
    """Build the tool_use schema for structured control extraction."""
    return {
        "name": "extract_controls",
        "description": "Extract security framework controls from the document",
        "input_schema": {
            "type": "object",
            "properties": {
                "controls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "control_id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "full_text": {"type": ["string", "null"]},
                        },
                        "required": ["control_id", "title", "description"],
                    },
                },
            },
            "required": ["controls"],
        },
    }


def _get_anthropic_client() -> Any:
    """Get an Anthropic client, loading the API key from the pass store."""
    try:
        import anthropic
    except ImportError:
        raise RuntimeError(
            "LLM extraction requires additional dependencies. "
            "Install with: pip install tract[llm]"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            result = subprocess.run(
                ["pass", "anthropic/api-key"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            api_key = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    if not api_key:
        raise RuntimeError(
            "Anthropic API key not found. Set ANTHROPIC_API_KEY environment "
            "variable or configure via 'pass anthropic/api-key'."
        )

    return anthropic.Anthropic(api_key=api_key)


def _deduplicate_controls(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate controls by control_id, keeping the longer description."""
    seen: dict[str, dict[str, Any]] = {}
    for ctrl in controls:
        cid = ctrl.get("control_id", "")
        if cid in seen:
            existing_desc = seen[cid].get("description", "")
            new_desc = ctrl.get("description", "")
            if len(new_desc) > len(existing_desc):
                seen[cid] = ctrl
        else:
            seen[cid] = ctrl
    return list(seen.values())


class LlmExtractor:
    """Extract controls from unstructured documents using Claude API."""

    def extract(
        self,
        path: Path,
        *,
        framework_id: str,
        output_dir: Path,
    ) -> list[Control]:
        """Read a document and extract controls via Claude API tool_use."""
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = path.read_text(encoding="utf-8")
        logger.info("Read %d chars from %s for LLM extraction", len(text), path.name)

        client = _get_anthropic_client()
        tool_schema = _build_tool_schema()

        raw_controls = self._call_with_retry(client, text, tool_schema)

        deduped = _deduplicate_controls(raw_controls)

        raw_path = output_dir / f"{framework_id}_llm_raw.json"
        atomic_write_json({"controls": deduped}, raw_path)
        logger.info("Saved raw LLM response to %s", raw_path)

        if not deduped:
            raise ValueError(
                "LLM extraction returned no controls. The document may not "
                "contain a structured framework."
            )

        controls: list[Control] = []
        for item in deduped:
            controls.append(Control(
                control_id=item["control_id"],
                title=item.get("title", ""),
                description=item["description"],
                full_text=item.get("full_text"),
            ))

        logger.info(
            "LLM extracted %d controls (%d before dedup)",
            len(controls), len(raw_controls),
        )
        return controls

    def _call_with_retry(
        self,
        client: Any,
        text: str,
        tool_schema: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Call the Anthropic API with exponential backoff retry."""
        delay = PREPARE_LLM_RETRY_INITIAL_DELAY_S
        last_error: Exception | None = None

        for attempt in range(1, PREPARE_LLM_MAX_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=PREPARE_LLM_MODEL,
                    max_tokens=4096,
                    temperature=PREPARE_LLM_TEMPERATURE,
                    system=_SYSTEM_PROMPT,
                    tools=[tool_schema],
                    tool_choice={"type": "tool", "name": "extract_controls"},
                    messages=[
                        {"role": "user", "content": text},
                    ],
                )

                for block in response.content:
                    if getattr(block, "type", None) == "tool_use":
                        return block.input.get("controls", [])

                logger.warning("API response had no tool_use block on attempt %d", attempt)
                return []

            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM API attempt %d/%d failed: %s",
                    attempt, PREPARE_LLM_MAX_RETRIES, e,
                )
                if attempt < PREPARE_LLM_MAX_RETRIES:
                    time.sleep(delay)
                    delay *= PREPARE_LLM_RETRY_BACKOFF_FACTOR

        raise RuntimeError(
            f"LLM extraction failed after {PREPARE_LLM_MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )
