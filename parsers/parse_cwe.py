"""Parser for MITRE CWE — weakness entries with full descriptions.

CWE previously reached the corpus through parsers/fetch_opencre.py, which
carries the link graph and each section's name but not the standard's text. That
left 613 training links, 14% of the entire training set, anchored on a
three-word weakness title. This parser reads MITRE's own XML so those links get
the real description.

Source: https://cwe.mitre.org/data/xml/cwec_latest.xml.zip
"""
from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from typing import ClassVar, Final
from xml.etree.ElementTree import Element

# The stdlib XML parser is vulnerable to XXE and entity-expansion. CLAUDE.md
# treats framework data as untrusted input, and this file is downloaded from
# the internet, so it is parsed with defusedxml. Element is imported from the
# stdlib for typing only; defusedxml returns the same node objects.
from defusedxml.ElementTree import parse as parse_xml

from tract.config import PROSE_MIN_EXTRA_CHARS
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CWE keeps retired entries in the catalog as tombstones whose description is a
# redirection ("DEPRECATED: This entry has been replaced by CWE-123"), so their
# text describes bookkeeping rather than a weakness. Status "Obsolete" is a
# different thing: it marks a category whose grouping is discouraged for
# mapping, while its summary still describes the weakness class. Three of the
# categories OpenCRE links to are Obsolete, so filtering on it would throw away
# links that have perfectly good text.
EXCLUDED_STATUS: Final[frozenset[str]] = frozenset({"Deprecated"})

# The catalog decompresses to roughly 18 MB. A zip that claims far more than
# that is either a different artifact or a decompression bomb, and this is the
# one place the parser commits to writing an attacker-controlled size into
# memory. 64 MB leaves room for several releases of growth.
MAX_UNCOMPRESSED_BYTES: Final[int] = 64 * 1024 * 1024

_WHITESPACE = re.compile(r"\s+")


def _text_of(element: Element | None) -> str:
    """Flatten an element's mixed content, including nested xhtml markup."""
    if element is None:
        return ""
    return _WHITESPACE.sub(" ", "".join(element.itertext())).strip()


def _clears_title(body: str, title: str) -> bool:
    """True when a body carries more than a restatement of its title.

    This is the same test tract.text_selection.ProseIndex applies when it
    decides whether a control's description is worth indexing at all. Mirroring
    it here lets the parser see, at write time, which entries are about to be
    dropped downstream.
    """
    return len(body.strip()) > len(title.strip()) + PROSE_MIN_EXTRA_CHARS


class CweParser(BaseParser):
    framework_id = "cwe"
    framework_name = "CWE"
    version = "4.20"
    source_url = "https://cwe.mitre.org"
    mapping_unit_level = "weakness"
    # The catalog holds far more entries than OpenCRE links to. Emitting all of
    # the live ones means every link finds its text regardless of which subset
    # OpenCRE covers, so this is a floor rather than an exact expectation, and
    # it moves with each quarterly CWE release.
    expected_count = 1300
    expected_count_is_floor: ClassVar[bool] = True
    fetched_date: ClassVar[str] = "2026-08-14"

    def parse(self) -> list[Control]:
        root = self._read_catalog()
        namespace = {"c": root.tag.split("}")[0].strip("{")}
        catalog_version = root.get("Version") or self.version

        controls: list[Control] = []
        skipped_status: dict[str, int] = {}
        below_bar = 0
        title_only = 0

        # Weaknesses first, then categories. Both share the CWE-N id space and
        # both are linked from OpenCRE, but a weakness is the thing the standard
        # actually defines, so it wins any title collision in the prose index.
        entries = [
            (entry, "weakness") for entry in root.findall(".//c:Weakness", namespace)
        ] + [
            (entry, "category") for entry in root.findall(".//c:Category", namespace)
        ]

        for entry, entry_type in entries:
            status = entry.get("Status") or ""
            if status in EXCLUDED_STATUS:
                skipped_status[status] = skipped_status.get(status, 0) + 1
                continue

            cwe_id = entry.get("ID") or ""
            name = entry.get("Name") or ""
            if not cwe_id or not name:
                continue

            body = self._body(entry, name, entry_type, namespace)
            if body is None:
                # Control.description is min_length=1, and an entry with no text
                # is worth no more than its title anyway. Counted rather than
                # passed over, because a run that quietly fell back to titles
                # looks identical downstream to one that found prose.
                body = name
                title_only += 1
            elif not _clears_title(body, name):
                below_bar += 1

            controls.append(Control(
                # Bare numeric id, matching the section_id OpenCRE uses for CWE.
                # The title matches the catalog Name too, so the prose joins on
                # either key. Both matter here: six of the linked entries reach
                # OpenCRE with the bare number as their section_name, so they
                # can only resolve by id, and CWE-598 was renamed after OpenCRE
                # captured it, so it can only resolve by id as well.
                control_id=cwe_id,
                title=name,
                # DESCRIPTION_MAX_LENGTH is applied by BaseParser.run, which
                # truncates on a word boundary and keeps the remainder in
                # full_text. Capping here as well would only add a second cut
                # in the middle of a word.
                description=body,
                hierarchy_level=entry.get("Abstraction") or entry_type,
                metadata={
                    "cwe_id": f"CWE-{cwe_id}",
                    "status": status,
                    "entry_type": entry_type,
                    "catalog_version": catalog_version,
                },
            ))

        if skipped_status:
            logger.info("Skipped by status: %s", skipped_status)
        logger.info(
            "Parsed %d CWE entries (%d have no text but their title, %d more "
            "stay shorter than their own title and so will be read as a "
            "restatement and dropped by the prose index)",
            len(controls), title_only, below_bar,
        )
        return controls

    def _read_catalog(self) -> Element:
        """Open the zipped catalog and return its root element.

        CWE ships one XML file inside a zip. data/raw is immutable, so the
        member is read from the archive rather than unpacked beside it.
        """
        source = self.raw_dir / "cwec_latest.xml.zip"
        # Read through the recording reader so the artifact records which
        # archive bytes produced it, then hand the same bytes to zipfile.
        with zipfile.ZipFile(BytesIO(self.read_source_bytes(source.name))) as archive:
            members = [
                info for info in archive.infolist()
                if info.filename.lower().endswith(".xml")
            ]
            if len(members) != 1:
                raise ValueError(
                    f"{source} holds {len(members)} XML members, expected "
                    f"exactly one: {[m.filename for m in members]}"
                )
            member = members[0]
            if member.file_size > MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"{source} member {member.filename} declares "
                    f"{member.file_size} bytes uncompressed, over the "
                    f"{MAX_UNCOMPRESSED_BYTES} byte ceiling."
                )
            with archive.open(member) as handle:
                parsed = parse_xml(handle).getroot()
                if parsed is None:
                    raise ValueError(
                        f"{source} member {member.filename} parsed to an "
                        f"empty XML tree."
                    )
                root: Element = parsed
        return root

    @staticmethod
    def _body(
        entry: Element,
        name: str,
        entry_type: str,
        namespace: dict[str, str],
    ) -> str | None:
        """Assemble the descriptive prose for one catalog entry.

        Returns None when the entry carries no text at all.

        A category holds a one-line Summary and nothing else. A weakness holds
        Description plus an optional Extended_Description, which is the pair
        that says what the weakness is; that is the whole body for all but a
        handful of entries.

        The exception is an entry whose description is shorter than its own
        title, which the prose index treats as a restatement and drops. CWE-532
        is "Insertion of Sensitive Information into Log File" described as "The
        product writes sensitive information to a log file." That is real prose
        and losing it costs the link its text, so those entries pick up the
        standard's own supporting narrative until they clear the bar. Every
        other entry keeps the canonical description untouched rather than being
        padded.
        """
        if entry_type == "category":
            return _text_of(entry.find("c:Summary", namespace)) or None

        core = " ".join(part for part in (
            _text_of(entry.find("c:Description", namespace)),
            _text_of(entry.find("c:Extended_Description", namespace)),
        ) if part).strip()
        if _clears_title(core, name):
            return core

        supporting = [
            _text_of(element) for element in entry.findall(
                "c:Background_Details/c:Background_Detail", namespace,
            )
        ] + [
            _text_of(element) for element in entry.findall(
                "c:Common_Consequences/c:Consequence/c:Note", namespace,
            )
        ]
        body = " ".join(
            part for part in [core, *supporting] if part
        ).strip()
        return body or None


if __name__ == "__main__":
    CweParser().run()
