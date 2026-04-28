"""Tests for parsers.fetch_opencre — mocked, no live API calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from parsers.fetch_opencre import PageEnvelope, fetch_all_cres, fetch_page


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_envelope_response(
    data: list[Any],
    page: int = 1,
    total_pages: int = 1,
    status_code: int = 200,
) -> MagicMock:
    """Return a MagicMock that behaves like a successful requests.Response
    returning the standard OpenCRE paginated envelope."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = {"data": data, "page": page, "total_pages": total_pages}
    resp.raise_for_status.return_value = None
    return resp


# ---------------------------------------------------------------------------
# Test 1: fetch_page retries on failure then succeeds
# ---------------------------------------------------------------------------

class TestFetchPageRetryOnFailure:
    """fetch_page should retry after transient failures and return data on success."""

    def test_fetch_page_retry_on_failure(self) -> None:
        """First attempt raises RequestException; second attempt returns valid envelope."""
        cre_items = [{"id": "CRE-001", "name": "Access Control"}]

        session = MagicMock(spec=requests.Session)
        # First call: raises a connection error. Second call: success.
        session.get.side_effect = [
            requests.ConnectionError("connection refused"),
            _make_envelope_response(cre_items, page=1, total_pages=1),
        ]

        with patch("time.sleep"):
            result = fetch_page(1, session)

        assert result["data"] == cre_items
        assert result["total_pages"] == 1
        assert session.get.call_count == 2

    def test_fetch_page_raises_after_all_retries_exhausted(self) -> None:
        """fetch_page must raise RequestException when every attempt fails."""
        session = MagicMock(spec=requests.Session)
        session.get.side_effect = requests.ConnectionError("always down")

        with patch("time.sleep"):
            with pytest.raises(requests.RequestException):
                fetch_page(1, session)

    def test_fetch_page_raises_on_non_dict_response(self) -> None:
        """fetch_page must raise ValueError when the API returns a non-dict body."""
        session = MagicMock(spec=requests.Session)
        resp = MagicMock(spec=requests.Response)
        resp.raise_for_status.return_value = None
        resp.json.return_value = ["not", "an", "envelope"]  # raw list, not envelope
        session.get.return_value = resp

        with patch("time.sleep"):
            with pytest.raises(ValueError, match="Expected JSON object"):
                fetch_page(1, session)

    def test_fetch_page_raises_on_missing_data_key(self) -> None:
        """fetch_page must raise ValueError when 'data' key is absent."""
        session = MagicMock(spec=requests.Session)
        resp = MagicMock(spec=requests.Response)
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"page": 1, "total_pages": 1}  # no 'data'
        session.get.return_value = resp

        with patch("time.sleep"):
            with pytest.raises(ValueError, match="missing 'data' or 'total_pages'"):
                fetch_page(1, session)


# ---------------------------------------------------------------------------
# Test 2: fetch_all_cres writes correct output structure
# ---------------------------------------------------------------------------

class TestFetchAllWritesOutput:
    """fetch_all_cres should produce the correct merged output JSON."""

    def test_fetch_all_writes_output(self, tmp_path: Path) -> None:
        """fetch_all_cres writes merged output with expected keys and counts."""
        page1_cres = [
            {"id": "CRE-001", "name": "Alpha"},
            {"id": "CRE-002", "name": "Beta"},
        ]
        page2_cres = [
            {"id": "CRE-003", "name": "Gamma"},
        ]

        def fake_fetch_page(page: int, session: Any) -> PageEnvelope:
            if page == 1:
                return PageEnvelope(data=page1_cres, page=1, total_pages=2)
            return PageEnvelope(data=page2_cres, page=2, total_pages=2)

        with (
            patch("parsers.fetch_opencre.RAW_OPENCRE_DIR", tmp_path),
            patch("parsers.fetch_opencre.fetch_page", side_effect=fake_fetch_page),
            patch("time.sleep"),
        ):
            fetch_all_cres()

        output_path = tmp_path / "opencre_all_cres.json"
        assert output_path.exists(), "Output file must be created"

        with open(output_path, encoding="utf-8") as fh:
            output = json.load(fh)

        # Structure checks
        assert "fetch_timestamp" in output
        assert "total_pages" in output
        assert "total_cres" in output
        assert "cres" in output

        # Content checks
        assert output["total_cres"] == 3
        assert len(output["cres"]) == 3
        assert output["cres"][0]["id"] == "CRE-001"
        assert output["cres"][2]["id"] == "CRE-003"

    def test_fetch_all_resumes_from_cache(self, tmp_path: Path) -> None:
        """Pages already cached on disk must not trigger additional API calls."""
        pages_dir = tmp_path / "pages"
        pages_dir.mkdir(parents=True)

        # Pre-populate page 1 in cache using the envelope format
        cached_data = [{"id": "CRE-100", "name": "Cached"}]
        cache_file = pages_dir / "page_001.json"
        cache_file.write_text(
            json.dumps({"data": cached_data, "page": 1, "total_pages": 1}),
            encoding="utf-8",
        )

        call_count = 0

        def fake_fetch_page(page: int, session: Any) -> PageEnvelope:
            nonlocal call_count
            call_count += 1
            # Should never be called for page 1 (cached)
            raise AssertionError(f"fetch_page called for page {page} — should be cached")

        with (
            patch("parsers.fetch_opencre.RAW_OPENCRE_DIR", tmp_path),
            patch("parsers.fetch_opencre.fetch_page", side_effect=fake_fetch_page),
            patch("time.sleep"),
        ):
            fetch_all_cres()

        # fetch_page should not have been called at all (only 1 page, fully cached).
        assert call_count == 0, (
            f"Expected 0 API calls (all pages cached), got {call_count}"
        )

        output_path = tmp_path / "opencre_all_cres.json"
        with open(output_path, encoding="utf-8") as fh:
            output = json.load(fh)

        assert output["total_cres"] == 1
        assert output["cres"][0]["id"] == "CRE-100"
