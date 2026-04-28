"""Parser for EU AI Act — Regulation (EU) 2024/1689.

Extracts Articles 1-113 and Annexes I-XIII from the EUR-Lex HTML.

HTML structure (EUR-Lex oj-convex-act layout):
  Articles:
    <div class="eli-subdivision" id="art_N">
      <p class="oj-ti-art">Article N</p>
      <div class="eli-title">
        <p class="oj-sti-art">Title text</p>
      </div>
      ... body paragraphs ...
    </div>

  Annexes:
    <div class="eli-container" id="anx_[ROMAN]">
      <p class="oj-doc-ti">ANNEX [ROMAN]</p>
      <p class="oj-doc-ti">Subtitle text</p>
      ... body content ...
    </div>
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from tract.parsers.base import BaseParser
from tract.schema import Control

logger = logging.getLogger(__name__)

# Matches "Article N" where N is a plain integer (no leading text that would
# make this a cross-reference inside prose).
ARTICLE_ID_RE = re.compile(r"^Article\s+(\d+)$", re.ASCII)

# Matches article subdivision IDs like "art_1", "art_42".
ART_DIV_ID_RE = re.compile(r"^art_(\d+)$")

# Matches annex container IDs like "anx_I", "anx_XIII".
ANX_DIV_ID_RE = re.compile(r"^anx_([IVXLCDM]+)$")

# Roman numerals used in annex IDs (I through XIII in this regulation).
ROMAN_NUMERALS: dict[str, int] = {
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4,
    "V": 5,
    "VI": 6,
    "VII": 7,
    "VIII": 8,
    "IX": 9,
    "X": 10,
    "XI": 11,
    "XII": 12,
    "XIII": 13,
}


def _clean_text(tag: Tag) -> str:
    """Return all text from *tag*, collapsing whitespace runs to single spaces.

    Args:
        tag: A BeautifulSoup Tag whose `.get_text()` we want normalised.

    Returns:
        Stripped, whitespace-collapsed plain-text string.
    """
    raw = tag.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", raw).strip()


def _extract_article(div: Tag) -> tuple[int, str, str] | None:
    """Extract (article_number, title, body_text) from an article subdivision.

    Args:
        div: A ``<div class="eli-subdivision" id="art_N">`` element.

    Returns:
        A 3-tuple ``(article_number, title, body_text)`` on success, or
        ``None`` if the element does not look like an article.

    Raises:
        ValueError: If the article number cannot be parsed from the element.
    """
    m = ART_DIV_ID_RE.match(div.get("id", ""))
    if m is None:
        return None

    art_num = int(m.group(1))

    # Article title: <p class="oj-ti-art">Article N</p>
    ti_tag = div.find("p", class_="oj-ti-art")
    if ti_tag is None:
        logger.warning("art_%d: missing oj-ti-art element, skipping", art_num)
        return None

    # Article subtitle: first <p class="oj-sti-art"> in eli-title
    sti_tag = div.find("p", class_="oj-sti-art")
    title_text = _clean_text(sti_tag) if sti_tag else f"Article {art_num}"
    # Strip stray backtick from Article 1 subtitle in source HTML
    title_text = title_text.rstrip("`").strip()

    # Body: all text in the div excluding the header elements
    # Gather text from all direct descendant structural divs and paragraphs
    # that are NOT the eli-title block.
    body_parts: list[str] = []
    for child in div.children:
        if not isinstance(child, Tag):
            continue
        # Skip the article header (oj-ti-art) paragraph
        if child.name == "p" and "oj-ti-art" in (child.get("class") or []):
            continue
        # Skip the eli-title subtitle block
        if child.name == "div" and "eli-title" in (child.get("class") or []):
            continue
        text = _clean_text(child)
        if text:
            body_parts.append(text)

    body = " ".join(body_parts).strip()
    if not body:
        # Fallback: use the title as description when body is empty
        body = title_text

    return art_num, title_text, body


def _extract_annex(div: Tag) -> tuple[str, str, str] | None:
    """Extract (roman_numeral, subtitle, body_text) from an annex container.

    Args:
        div: A ``<div class="eli-container" id="anx_[ROMAN]">`` element.

    Returns:
        A 3-tuple ``(roman_numeral, subtitle, body_text)`` on success, or
        ``None`` if the element does not look like an annex.
    """
    m = ANX_DIV_ID_RE.match(div.get("id", ""))
    if m is None:
        return None

    roman = m.group(1)

    # Find all oj-doc-ti paragraphs inside this container.
    doc_ti_tags = div.find_all("p", class_="oj-doc-ti", recursive=True)

    # First oj-doc-ti: "ANNEX [ROMAN]"
    # Second oj-doc-ti: subtitle
    subtitle = ""
    if len(doc_ti_tags) >= 2:
        subtitle = _clean_text(doc_ti_tags[1])
    elif doc_ti_tags:
        subtitle = _clean_text(doc_ti_tags[0])

    if not subtitle:
        subtitle = f"ANNEX {roman}"

    # Body: all text in the container except the header doc-ti paragraphs
    body_parts: list[str] = []
    header_tags: set[int] = {id(t) for t in doc_ti_tags[:2]}

    for child in div.children:
        if not isinstance(child, Tag):
            continue
        if id(child) in header_tags:
            continue
        # Skip a <p class="oj-doc-ti"> that is a direct header child
        if child.name == "p" and "oj-doc-ti" in (child.get("class") or []):
            continue
        text = _clean_text(child)
        if text:
            body_parts.append(text)

    body = " ".join(body_parts).strip()
    if not body:
        body = subtitle

    return roman, subtitle, body


class EuAiActParser(BaseParser):
    """Parser for EU AI Act (Regulation (EU) 2024/1689).

    Extracts all 113 Articles and 13 Annexes from the EUR-Lex HTML,
    producing one Control per article and one per annex.
    """

    framework_id = "eu_ai_act"
    framework_name = "EU AI Act — Regulation (EU) 2024/1689"
    version = "2024/1689"
    source_url = (
        "https://eur-lex.europa.eu/legal-content/EN/TXT/"
        "?uri=CELEX:32024R1689"
    )
    mapping_unit_level = "article"
    expected_count = 126  # 113 articles + 13 annexes

    def parse(self) -> list[Control]:
        """Parse Articles 1-113 and Annexes I-XIII from the EUR-Lex HTML.

        Returns:
            List of Control objects, one per article then one per annex.

        Raises:
            FileNotFoundError: If the HTML source file is missing.
            ValueError: If fewer than 90 controls are extracted.
        """
        html_path: Path = self.raw_dir / "eu_ai_act_2024_1689.html"
        if not html_path.exists():
            raise FileNotFoundError(
                f"EU AI Act HTML not found at {html_path}"
            )

        logger.info("Loading HTML from %s", html_path)
        html_text = html_path.read_text(encoding="utf-8")

        soup = BeautifulSoup(html_text, "html.parser")
        controls: list[Control] = []
        seen_ids: set[str] = set()

        # --- Articles ---
        article_divs = soup.find_all(
            "div",
            class_="eli-subdivision",
            id=ART_DIV_ID_RE,
        )
        logger.info("Found %d article subdivision elements", len(article_divs))

        for div in article_divs:
            result = _extract_article(div)
            if result is None:
                continue
            art_num, title, body = result
            control_id = f"AIA-Art{art_num}"
            if control_id in seen_ids:
                logger.debug("Duplicate article %s — skipping", control_id)
                continue
            seen_ids.add(control_id)

            controls.append(Control(
                control_id=control_id,
                title=title,
                description=body[:2000],
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="article",
                metadata={"article_number": str(art_num)},
            ))

        logger.info("Extracted %d article controls", len(controls))

        # --- Annexes ---
        annex_divs = soup.find_all(
            "div",
            class_="eli-container",
            id=ANX_DIV_ID_RE,
        )
        logger.info("Found %d annex container elements", len(annex_divs))

        for div in annex_divs:
            result = _extract_annex(div)
            if result is None:
                continue
            roman, subtitle, body = result
            control_id = f"AIA-Annex{roman}"
            if control_id in seen_ids:
                logger.debug("Duplicate annex %s — skipping", control_id)
                continue
            seen_ids.add(control_id)

            ordinal = str(ROMAN_NUMERALS.get(roman, roman))
            controls.append(Control(
                control_id=control_id,
                title=subtitle,
                description=body[:2000],
                full_text=body if len(body) > 2000 else None,
                hierarchy_level="annex",
                metadata={
                    "annex_roman": roman,
                    "annex_number": ordinal,
                },
            ))

        logger.info(
            "Extracted %d total controls (%d articles + %d annexes)",
            len(controls),
            sum(1 for c in controls if c.hierarchy_level == "article"),
            sum(1 for c in controls if c.hierarchy_level == "annex"),
        )

        if len(controls) < 90:
            raise ValueError(
                f"EU AI Act parser produced only {len(controls)} controls; "
                "expected at least 90. Check HTML structure."
            )

        return controls


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = EuAiActParser()
    parser.run()
