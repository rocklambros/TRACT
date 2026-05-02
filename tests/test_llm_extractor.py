"""Tests for tract.prepare.llm_extractor — LLM-assisted extraction with mocked API."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tract.prepare.llm_extractor import LlmExtractor, _build_tool_schema
from tract.schema import Control


def _mock_api_response(controls: list[dict]) -> MagicMock:
    """Build a mock Anthropic API response with tool_use content."""
    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "extract_controls"
    tool_use_block.input = {"controls": controls}

    response = MagicMock()
    response.content = [tool_use_block]
    response.stop_reason = "tool_use"
    return response


@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """A plain text file for LLM extraction."""
    content = (
        "AI Security Framework v1.0\n\n"
        "1. Access Control\n"
        "Organizations must enforce strict access control policies "
        "for AI model endpoints and training pipelines.\n\n"
        "2. Data Protection\n"
        "Sensitive training data must be encrypted at rest and in "
        "transit using industry-standard cryptographic methods.\n\n"
        "3. Model Monitoring\n"
        "Continuously monitor AI model behavior for drift, adversarial "
        "inputs, and unexpected output patterns.\n"
    )
    path = tmp_path / "framework.txt"
    path.write_text(content, encoding="utf-8")
    return path


class TestBuildToolSchema:
    def test_schema_has_required_fields(self) -> None:
        schema = _build_tool_schema()
        assert schema["name"] == "extract_controls"
        props = schema["input_schema"]["properties"]["controls"]["items"]["properties"]
        assert "control_id" in props
        assert "title" in props
        assert "description" in props
        assert "full_text" in props


class TestLlmExtractor:
    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_extract_returns_controls(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([
            {"control_id": "AC-01", "title": "Access Control", "description": "Enforce strict access control policies for AI model endpoints"},
            {"control_id": "DP-01", "title": "Data Protection", "description": "Encrypt sensitive training data at rest and in transit"},
            {"control_id": "MM-01", "title": "Model Monitoring", "description": "Monitor AI model behavior for drift and adversarial inputs"},
        ])

        extractor = LlmExtractor()
        controls = extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )

        assert len(controls) == 3
        assert all(isinstance(c, Control) for c in controls)
        assert controls[0].control_id == "AC-01"
        assert controls[0].title == "Access Control"

        mock_client.messages.create.assert_called_once()

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_saves_raw_llm_response(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([
            {"control_id": "AC-01", "title": "Access", "description": "Access control policies for systems"},
        ])

        extractor = LlmExtractor()
        extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )

        raw_path = tmp_path / "test_fw_llm_raw.json"
        assert raw_path.exists()
        raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
        assert "controls" in raw_data

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_zero_controls_raises(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([])

        extractor = LlmExtractor()
        with pytest.raises(ValueError, match="(?i)no controls"):
            extractor.extract(
                sample_text_file,
                framework_id="test_fw",
                output_dir=tmp_path,
            )

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_api_failure_retries(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        mock_client.messages.create.side_effect = [
            Exception("API temporarily unavailable"),
            Exception("Rate limited"),
            _mock_api_response([
                {"control_id": "AC-01", "title": "Access", "description": "Access control policies for system components and users"},
            ]),
        ]

        extractor = LlmExtractor()
        controls = extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )
        assert len(controls) == 1
        assert mock_client.messages.create.call_count == 3

    @patch("tract.prepare.llm_extractor._get_anthropic_client")
    def test_deduplicates_by_control_id(
        self, mock_get_client: MagicMock, sample_text_file: Path, tmp_path: Path,
    ) -> None:
        """When chunking produces duplicates, keep the one with longer description."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _mock_api_response([
            {"control_id": "AC-01", "title": "Access", "description": "Short desc"},
            {"control_id": "AC-01", "title": "Access Control", "description": "Longer description that provides more context about access control"},
        ])

        extractor = LlmExtractor()
        controls = extractor.extract(
            sample_text_file,
            framework_id="test_fw",
            output_dir=tmp_path,
        )
        assert len(controls) == 1
        assert "Longer description" in controls[0].description
