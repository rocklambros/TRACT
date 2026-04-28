"""TRACT text sanitization pipeline.

Every text field passes through this pipeline before storage:
1. Strip null bytes
2. Unicode NFC normalization
3. HTML unescape + strip tags
4. Fix common PDF ligatures
5. Fix broken hyphenation from PDF line-wrapping
6. Collapse whitespace
7. Strip leading/trailing whitespace

Public API:
    sanitize_text(text, *, max_length, return_full) -> str | tuple[str, str | None]
    strip_html(text) -> str
"""

from __future__ import annotations

import html
import re
import unicodedata
from typing import Literal, overload

# ── PDF ligature replacements ──────────────────────────────────────────────
# Order matters: longer ligatures first to avoid partial replacement.
_LIGATURE_MAP: list[tuple[str, str]] = [
    ("ﬄ", "ffl"),
    ("ﬃ", "ffi"),
    ("ﬀ", "ff"),
    ("ﬁ", "fi"),
    ("ﬂ", "fl"),
]

# Matches word-\n continuation (broken hyphenation from PDF extraction)
_HYPHEN_BREAK_RE: re.Pattern[str] = re.compile(r"(\w)-\n(\w)")

# Matches HTML/XML tags
_HTML_TAG_RE: re.Pattern[str] = re.compile(r"</?[a-zA-Z][^>]*>")

# Matches runs of whitespace (spaces, tabs, newlines)
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Unescape HTML entities, then remove HTML/XML tags.

    Unescape first so double-encoded tags (e.g., &lt;script&gt;) are
    decoded and then stripped, preventing literal tags in output.

    Args:
        text: Raw text potentially containing HTML markup.

    Returns:
        Plain text with all tags removed and entities decoded.
    """
    unescaped = html.unescape(text)
    return _HTML_TAG_RE.sub("", unescaped)


def _strip_null_bytes(text: str) -> str:
    """Remove null bytes that can poison downstream processing."""
    return text.replace("\x00", " ")


def _normalize_unicode(text: str) -> str:
    """Normalize to Unicode NFC form for consistent byte representation."""
    return unicodedata.normalize("NFC", text)


def _fix_ligatures(text: str) -> str:
    """Replace common PDF ligature characters with ASCII equivalents."""
    for ligature, replacement in _LIGATURE_MAP:
        text = text.replace(ligature, replacement)
    return text


def _fix_hyphenation(text: str) -> str:
    """Rejoin words broken by PDF line-wrapping (word-\\nword -> wordword)."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def _collapse_whitespace(text: str) -> str:
    """Collapse all whitespace runs to a single space."""
    return _WHITESPACE_RE.sub(" ", text)


@overload
def sanitize_text(
    text: str,
    *,
    max_length: int = ...,
    return_full: Literal[False] = ...,
) -> str: ...


@overload
def sanitize_text(
    text: str,
    *,
    max_length: int = ...,
    return_full: Literal[True],
) -> tuple[str, str | None]: ...


def sanitize_text(
    text: str,
    *,
    max_length: int = 2000,
    return_full: bool = False,
) -> str | tuple[str, str | None]:
    """Run the full sanitization pipeline on a text field.

    Args:
        text: Raw input text.
        max_length: Maximum length for the returned (truncated) text.
            Defaults to 2000 (DESCRIPTION_MAX_LENGTH).
        return_full: If True, return a tuple of (truncated, full_or_None).
            full_or_None is the complete sanitized text when it exceeds
            max_length, otherwise None.

    Returns:
        If return_full is False: the sanitized, truncated string.
        If return_full is True: (truncated_str, full_str_or_None).

    Raises:
        ValueError: If the sanitized text is empty after all processing.
    """
    cleaned = text
    cleaned = _strip_null_bytes(cleaned)
    cleaned = _normalize_unicode(cleaned)
    cleaned = strip_html(cleaned)
    cleaned = _fix_ligatures(cleaned)
    cleaned = _fix_hyphenation(cleaned)
    cleaned = _collapse_whitespace(cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError(
            f"Sanitized text is empty. Original text (first 100 chars): "
            f"{text[:100]!r}"
        )

    if return_full:
        if len(cleaned) > max_length:
            truncated = cleaned[:max_length].rsplit(" ", 1)[0]
            if len(truncated) < max_length // 2:
                truncated = cleaned[:max_length]
            return truncated, cleaned
        return cleaned, None

    if len(cleaned) > max_length:
        truncated = cleaned[:max_length].rsplit(" ", 1)[0]
        if len(truncated) < max_length // 2:
            return cleaned[:max_length]
        return truncated
    return cleaned
