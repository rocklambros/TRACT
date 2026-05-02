"""Tests for tract.sanitize — text sanitization pipeline."""

from __future__ import annotations

import pytest

from tract.sanitize import sanitize_control, sanitize_text, strip_html


class TestStripHtml:
    """Tests for the strip_html function."""

    def test_removes_tags(self) -> None:
        assert strip_html("<p>Hello <b>world</b></p>") == "Hello world"

    def test_unescapes_entities(self) -> None:
        assert strip_html("&amp; &lt; &gt;") == "& < >"

    def test_plain_text_unchanged(self) -> None:
        assert strip_html("no tags here") == "no tags here"


class TestSanitizeText:
    """Tests for the sanitize_text function."""

    def test_strips_null_bytes(self) -> None:
        result = sanitize_text("hello\x00world")
        assert result == "hello world"

    def test_unicode_nfc_normalization(self) -> None:
        # e + combining acute accent -> single char e-acute
        decomposed = "é"  # NFD form
        composed = "é"     # NFC form
        result = sanitize_text(f"caf{decomposed}")
        assert result == f"caf{composed}"

    def test_strips_html_tags(self) -> None:
        result = sanitize_text("<p>Hello <b>world</b></p>")
        assert result == "Hello world"

    def test_fixes_ligature_ff(self) -> None:
        result = sanitize_text("eﬀective")
        assert result == "effective"

    def test_fixes_ligature_fi(self) -> None:
        result = sanitize_text("conﬁgure")
        assert result == "configure"

    def test_fixes_ligature_fl(self) -> None:
        result = sanitize_text("inﬂation")
        assert result == "inflation"

    def test_fixes_ligature_ffi(self) -> None:
        result = sanitize_text("oﬃce")
        assert result == "office"

    def test_fixes_ligature_ffl(self) -> None:
        result = sanitize_text("baﬄe")
        assert result == "baffle"

    def test_fixes_broken_hyphenation(self) -> None:
        result = sanitize_text("secu-\nrity")
        assert result == "security"

    def test_collapses_whitespace(self) -> None:
        result = sanitize_text("hello   \t  \n  world")
        assert result == "hello world"

    def test_strips_leading_trailing(self) -> None:
        result = sanitize_text("  hello  ")
        assert result == "hello"

    def test_truncation_at_word_boundary(self) -> None:
        text = "word " * 600
        result = sanitize_text(text.strip(), max_length=100)
        assert isinstance(result, str)
        assert len(result) <= 100
        assert not result.endswith(" ")

    def test_return_full_when_truncated(self) -> None:
        text = "word " * 600
        short, full = sanitize_text(text.strip(), max_length=100, return_full=True)
        assert len(short) <= 100
        assert full is not None
        assert len(full) == len(text.strip())

    def test_return_full_when_not_truncated(self) -> None:
        text = "short text"
        short, full = sanitize_text(text, return_full=True)
        assert short == "short text"
        assert full is None

    def test_raises_on_empty_result(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sanitize_text("")

    def test_raises_on_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sanitize_text("   \t\n  ")

    def test_raises_on_null_bytes_only(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sanitize_text("\x00\x00\x00")

    def test_raises_on_html_tags_only(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sanitize_text("<br/><hr/>")

    def test_strips_zero_width_space(self) -> None:
        result = sanitize_text("hello​world")
        assert result == "helloworld"

    def test_strips_zero_width_non_joiner(self) -> None:
        result = sanitize_text("hello‌world")
        assert result == "helloworld"

    def test_strips_zero_width_joiner(self) -> None:
        result = sanitize_text("hello‍world")
        assert result == "helloworld"

    def test_strips_bom(self) -> None:
        result = sanitize_text("﻿hello world")
        assert result == "hello world"

    def test_strips_multiple_zero_width(self) -> None:
        result = sanitize_text("a​‌‍﻿b")
        assert result == "ab"

    def test_combined_pipeline(self) -> None:
        """Full pipeline: null bytes + HTML + ligatures + hyphenation + whitespace."""
        raw = "  <p>The eﬀective\x00 secu-\nrity   control.</p>  "
        result = sanitize_text(raw)
        assert result == "The effective security control."


class TestSanitizeControl:
    def test_sanitizes_description(self) -> None:
        ctrl = {
            "control_id": "TC-01",
            "title": "Test",
            "description": "A\x00 description with null bytes",
        }
        result = sanitize_control(ctrl)
        assert "\x00" not in result["description"]
        assert "description" in result["description"]

    def test_sanitizes_title(self) -> None:
        ctrl = {
            "control_id": "TC-01",
            "title": "Title\x00with null",
            "description": "Valid description for the control test case",
        }
        result = sanitize_control(ctrl)
        assert "\x00" not in result["title"]

    def test_sanitizes_full_text(self) -> None:
        ctrl = {
            "control_id": "TC-01",
            "title": "Test",
            "description": "Valid description for the control test case",
            "full_text": "Full\x00text content here for detailed analysis",
        }
        result = sanitize_control(ctrl)
        assert "\x00" not in result["full_text"]

    def test_preserves_non_text_fields(self) -> None:
        ctrl = {
            "control_id": "TC-01",
            "title": "Test",
            "description": "Valid description for the control test case",
            "hierarchy_level": 2,
            "parent_id": "PARENT-01",
            "metadata": {"source": "test"},
        }
        result = sanitize_control(ctrl)
        assert result["hierarchy_level"] == 2
        assert result["parent_id"] == "PARENT-01"
        assert result["metadata"] == {"source": "test"}

    def test_long_description_splits_to_full_text(self) -> None:
        long_desc = "A " * 1500
        ctrl = {
            "control_id": "TC-01",
            "title": "Test",
            "description": long_desc,
        }
        result = sanitize_control(ctrl)
        assert len(result["description"]) <= 2000
        assert result["full_text"] is not None
        assert len(result["full_text"]) > len(result["description"])
