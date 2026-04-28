"""Parser for OWASP GenAI Data Security Risks and Mitigations 2026 (DSGAI).

Source file: pdftotext output of OWASP-GenAI-Data-Security-Risks-and-Mitigations-2026-v1.0.pdf

Each DSGAI section starts on a new PDF page (form-feed character), followed by
the section heading on the same line: e.g. '\x0cDSGAI01 — Sensitive Data Leakage'.
Some titles wrap to the next line when pdftotext breaks a long heading across pages
(e.g. '\x0cDSGAI07 — Data Governance, Lifecycle & Classification for AI' / 'Systems').
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Matches a form-feed followed immediately by a DSGAI ID, capturing the ID
# and the rest of the first title line (may be incomplete when it wraps).
_SECTION_RE = re.compile(
    r"\x0c(DSGAI(?:0[1-9]|1[0-9]|2[01]))\s*[—\-]\s*(.+)",
)

# Lines that look like "genai.owasp.org     Page N" — strip these from body text
_PAGE_FOOTER_RE = re.compile(r"genai\.owasp\.org\s+Page\s+\d+")


def _clean_body(raw: str) -> str:
    """Remove page footer/header artefacts from a section body."""
    lines = raw.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if _PAGE_FOOTER_RE.search(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip()


class OwaspDsgaiParser(BaseParser):
    """Parser for OWASP GenAI Data Security Risks and Mitigations 2026."""

    framework_id = "owasp_dsgai"
    framework_name = "OWASP GenAI Data Security Risks and Mitigations"
    version = "1.0"
    source_url = "https://genai.owasp.org/"
    mapping_unit_level = "risk"
    expected_count = 21

    def parse(self) -> list[Control]:
        """Parse DSGAI sections from the pdftotext output.

        Returns:
            List of 21 Control objects, one per DSGAI section.

        Raises:
            FileNotFoundError: If the manifest or source text file is missing.
            ValueError: If the manifest is malformed or fewer than expected
                controls are found after deduplication.
        """
        manifest_path = self.raw_dir / "MANIFEST.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"MANIFEST.json not found at {manifest_path}")

        with manifest_path.open(encoding="utf-8") as fh:
            manifest: dict = json.load(fh)

        source_file: str = manifest.get("source_file", "")
        if not source_file:
            raise ValueError("MANIFEST.json missing 'source_file' key")

        txt_path: Path = self.raw_dir / source_file
        if not txt_path.exists():
            raise FileNotFoundError(f"Source text file not found: {txt_path}")

        text = txt_path.read_text(encoding="utf-8")
        logger.debug("Read %d characters from %s", len(text), txt_path)

        return self._extract_controls(text)

    def _extract_controls(self, text: str) -> list[Control]:
        """Extract DSGAI controls from the full document text.

        Strategy: find every form-feed+DSGAIXX header, deduplicate by ID
        (first occurrence wins — the table-of-contents uses the same IDs),
        then for each unique ID extract the text between its position and
        the next unique section start.

        Args:
            text: Full content of the pdftotext output file.

        Returns:
            Ordered list of Control objects.

        Raises:
            ValueError: If no section headers are found.
        """
        matches = list(_SECTION_RE.finditer(text))
        if not matches:
            raise ValueError("No DSGAI section headers found in source file")

        logger.debug("Raw regex matches: %d", len(matches))

        # Deduplicate: first occurrence of each ID is the real section.
        seen_ids: set[str] = set()
        unique_matches: list[re.Match] = []
        for m in matches:
            cid = m.group(1)
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_matches.append(m)

        logger.info("Unique DSGAI sections found: %d", len(unique_matches))

        if len(unique_matches) == 0:
            raise ValueError("All DSGAI matches were duplicates — no sections parsed")

        controls: list[Control] = []

        for i, m in enumerate(unique_matches):
            control_id = m.group(1)
            raw_title_fragment = m.group(2).strip()

            # Determine the character offset immediately after this match
            # so we can look ahead for a possible title continuation and body.
            after_header = m.end()

            # Check if the title wraps to the next line.
            # pdftotext wraps a long heading when it hits the page width, putting
            # the remainder on the very next line.  The continuation line is short
            # (< 80 chars) and is NOT one of the standard section sub-headings.
            #
            # Special case: when the first line ends with a hyphen the continuation
            # may start with a lowercase letter (e.g. "LLM-" / "to-SQL/Graph)"),
            # and the hyphen indicates the two fragments should be joined directly
            # (no space).
            _SUBHEADING_STARTS = (
                "How ", "This ", "Attack", "Mitig", "Impact",
                "Risk", "Techni", "For ", "References",
            )

            remainder_from_header = text[after_header:]
            first_newline = remainder_from_header.find("\n")
            if first_newline == -1:
                # Shouldn't happen in a valid file, but handle gracefully.
                continuation_line = ""
                ends_with_hyphen = False
                body_start_offset = after_header
            else:
                next_line_block = remainder_from_header[first_newline + 1:]
                second_newline = next_line_block.find("\n")
                second_line = (
                    next_line_block[:second_newline].strip()
                    if second_newline != -1
                    else next_line_block.strip()
                )

                ends_with_hyphen = raw_title_fragment.endswith("-")

                # Determine whether the second line is a title continuation:
                # - short (< 80 chars)
                # - not a known sub-heading opener
                # - not starting a new DSGAI section reference
                # - if hyphenated end: lowercase-start is OK (e.g. "to-SQL/Graph)")
                # - if no hyphen: must start with uppercase word
                is_upper_start = bool(second_line) and second_line[0].isupper()
                is_lower_ok = ends_with_hyphen and bool(second_line) and second_line[0].islower()

                is_continuation = (
                    bool(second_line)
                    and len(second_line) < 80
                    and (is_upper_start or is_lower_ok)
                    and not any(second_line.startswith(w) for w in _SUBHEADING_STARTS)
                    and not second_line.startswith("DSGAI")
                    # Guard: continuation should not look like a full prose sentence
                    # (those have 80+ chars and would be caught by length already)
                )

                if is_continuation:
                    continuation_line = second_line
                    # Body starts after the continuation line.
                    cont_end = after_header + first_newline + 1 + second_newline + 1
                    body_start_offset = cont_end
                else:
                    continuation_line = ""
                    ends_with_hyphen = False
                    body_start_offset = after_header

            # Build the full title; hyphen-at-end means join without space.
            if continuation_line:
                if ends_with_hyphen:
                    title = f"{raw_title_fragment}{continuation_line}".strip()
                else:
                    title = f"{raw_title_fragment} {continuation_line}".strip()
            else:
                title = raw_title_fragment

            # Extract section body: from body_start_offset to the next section
            if i + 1 < len(unique_matches):
                body_end_offset = unique_matches[i + 1].start()
            else:
                body_end_offset = len(text)

            raw_body = text[body_start_offset:body_end_offset]
            body = _clean_body(raw_body)

            if not body:
                logger.warning(
                    "%s: empty body after cleaning; using title as description",
                    control_id,
                )
                description = title
                full_text: str | None = None
            elif len(body) > 2000:
                description = body[:2000]
                full_text = body
            else:
                description = body
                full_text = None

            controls.append(
                Control(
                    control_id=control_id,
                    title=title,
                    description=description,
                    full_text=full_text,
                    hierarchy_level="risk",
                )
            )
            logger.debug("Parsed %s: %r (body %d chars)", control_id, title, len(body))

        return controls


if __name__ == "__main__":
    parser = OwaspDsgaiParser()
    parser.run()
