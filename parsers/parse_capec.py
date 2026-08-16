"""Parser for MITRE CAPEC — attack patterns with full descriptions.

CAPEC previously reached the corpus through parsers/fetch_opencre.py, which
carries the link graph and each section's name but not the standard's text. That
left 1,799 training links, 41% of the entire training set, anchored on a
three-word attack-pattern title. This parser reads MITRE's own XML so those
links get the real description.

Source: https://capec.mitre.org/data/xml/capec_latest.xml
"""
from __future__ import annotations

import logging
import re
from typing import Final
from xml.etree.ElementTree import Element

# The stdlib XML parser is vulnerable to XXE and entity-expansion. CLAUDE.md
# treats framework data as untrusted input, and this file is downloaded from
# the internet, so it is parsed with defusedxml. Element is imported from the
# stdlib for typing only; defusedxml returns the same node objects.
from defusedxml.ElementTree import parse as parse_xml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CAPEC ships Draft and Deprecated entries alongside stable ones. Deprecated
# patterns are retained in the catalog as tombstones pointing elsewhere, so
# their text describes a redirection rather than an attack.
EXCLUDED_STATUS: Final[frozenset[str]] = frozenset({"Deprecated", "Obsolete"})

_WHITESPACE = re.compile(r"\s+")
# Descriptions are long enough that a cap keeps one pattern from dominating a
# training batch. The remainder is preserved in full_text.
DESCRIPTION_CAP: Final[int] = 2000


def _text_of(element: Element | None) -> str:
    """Flatten an element's mixed content, including nested xhtml markup."""
    if element is None:
        return ""
    return _WHITESPACE.sub(" ", "".join(element.itertext())).strip()


class CapecParser(BaseParser):
    framework_id = "capec"
    framework_name = "CAPEC"
    version = "3.9"
    source_url = "https://capec.mitre.org"
    mapping_unit_level = "attack_pattern"
    # The catalog holds more patterns than OpenCRE links to. Emitting all of the
    # stable ones means every link finds its text regardless of which subset
    # OpenCRE covers, so this is a floor rather than an exact expectation.
    expected_count = 500
    expected_count_is_floor = True
    fetched_date = "2026-08-14"

    def parse(self) -> list[Control]:
        source = self.raw_dir / "capec_latest.xml"
        root = parse_xml(source).getroot()
        if root is None:
            raise ValueError(f"{source} parsed to an empty XML tree.")
        namespace = {"c": root.tag.split("}")[0].strip("{")}
        catalog_version = root.get("Version") or self.version

        controls: list[Control] = []
        skipped_status: dict[str, int] = {}

        for pattern in root.findall(".//c:Attack_Pattern", namespace):
            status = pattern.get("Status") or ""
            if status in EXCLUDED_STATUS:
                skipped_status[status] = skipped_status.get(status, 0) + 1
                continue

            capec_id = pattern.get("ID") or ""
            name = pattern.get("Name") or ""
            if not capec_id or not name:
                continue

            description = _text_of(pattern.find("c:Description", namespace))
            extended = _text_of(pattern.find("c:Extended_Description", namespace))
            body = " ".join(part for part in (description, extended) if part).strip()
            if not body:
                # Control.description is min_length=1, and a pattern with no
                # text is worth no more than its title anyway.
                body = name

            controls.append(Control(
                # Bare numeric id, matching the section_id OpenCRE uses for
                # CAPEC. The title also matches exactly, so the prose joins on
                # either key.
                control_id=capec_id,
                title=name,
                description=body[:DESCRIPTION_CAP],
                full_text=body if len(body) > DESCRIPTION_CAP else None,
                hierarchy_level=pattern.get("Abstraction") or "attack_pattern",
                metadata={
                    "capec_id": f"CAPEC-{capec_id}",
                    "status": status,
                    "catalog_version": catalog_version,
                },
            ))

        if skipped_status:
            logger.info("Skipped by status: %s", skipped_status)
        logger.info("Parsed %d CAPEC attack patterns", len(controls))
        return controls


if __name__ == "__main__":
    CapecParser().run()
