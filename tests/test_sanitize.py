"""Tests for tract.sanitize — text sanitization pipeline."""

from __future__ import annotations

import pytest

from tract.sanitize import sanitize_text, strip_html


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
        assert result == "helloworld"

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

    def test_truncation_at_max_length(self) -> None:
        text = "a" * 3000
        result = sanitize_text(text, max_length=100)
        assert isinstance(result, str)
        assert len(result) == 100

    def test_return_full_when_truncated(self) -> None:
        text = "a" * 3000
        short, full = sanitize_text(text, max_length=100, return_full=True)
        assert len(short) == 100
        assert full is not None
        assert len(full) == 3000

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

    def test_combined_pipeline(self) -> None:
        """Full pipeline: null bytes + HTML + ligatures + hyphenation + whitespace."""
        raw = "  <p>The eﬀective\x00 secu-\nrity   control.</p>  "
        result = sanitize_text(raw)
        assert result == "The effective security control."
