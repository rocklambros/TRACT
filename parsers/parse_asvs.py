"""Parser for OWASP ASVS, read from the project's own repository archive.

ASVS reached the corpus through parsers/fetch_opencre.py, which carries the link
graph and each section's name but never the standard's own explanatory text. For
ASVS the section name is the requirement sentence rather than a three-word
title, so the anchor was not as thin as it is for CAPEC. It was still the only
text the pipeline had, and the OpenCRE record repeats that one sentence as both
title and description, which the prose index reads as a title restatement and
drops. All 277 ASVS links, 6% of the training set, therefore fell back to the
link text. This parser reads OWASP's own JSON export so each requirement also
carries the standard's explanatory prose for the section it belongs to.

Version. The archive ships 4.0.3 and 5.0.0. 5.0.0 is newer and complete, and it
is the wrong source here: it renumbered every requirement and dropped 108 of the
277 that OpenCRE links, per 5.0/mappings/mapping_v4.0.3_to_v5.0.0.yml. OpenCRE's
section ids and section names are 4.0.3, so 4.0.3 is what this parser emits and
what the version field records.

Source: https://github.com/OWASP/ASVS (master branch archive)
"""
from __future__ import annotations

import json
import logging
import re
import zipfile
from collections.abc import Mapping
from typing import Final

from tract.config import PROSE_MIN_EXTRA_CHARS
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "asvs.zip"
# Matched rather than hardcoded with the archive's top directory, which carries
# the branch name and changes whenever the tarball is re-fetched.
REQUIREMENTS_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"(?:^|/)4\.0/docs_en/[^/]*4\.0\.3-en\.json$",
)
# 0x10-V1-Architecture.md through 0x22-V14-Config.md. The numeric-then-V prefix
# excludes the front matter and the appendices, which carry no requirements.
CHAPTER_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"(?:^|/)4\.0/en/0x[0-9a-f]+-V\d+[^/]*\.md$",
)
EXPECTED_VERSION: Final[str] = "4.0.3"

# The archive is downloaded from the internet and CLAUDE.md treats framework
# data as untrusted, so every member read is bounded. The largest member this
# parser touches is the 224 KB requirements JSON.
MAX_MEMBER_BYTES: Final[int] = 4 * 1024 * 1024

H2_HEADING: Final[re.Pattern[str]] = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
SECTION_HEADING: Final[re.Pattern[str]] = re.compile(r"^(V\d+\.\d+)\s+\S")
CHAPTER_HEADING: Final[re.Pattern[str]] = re.compile(
    r"^#\s+(V\d+)\s+\S", re.MULTILINE,
)
OBJECTIVE_HEADING: Final[str] = "Control Objective"

# Requirement text ends with a cross-reference to the OWASP Proactive Controls,
# for example "([C3, C4](https://owasp.org/...))". OpenCRE strips it from the
# section name, so the title has to match, and the marker is framework-branded
# noise in the description either way.
CITATION: Final[re.Pattern[str]] = re.compile(r"\s*\(\[C[\dC,\s]+\]\([^)]*\)\)")
# 4.0.3 keeps eight withdrawn requirements as tombstones whose whole text is
# "[DELETED, DUPLICATE OF 4.1.3]". None of them is linked, and two share their
# text, which would collide in the prose index's title map.
DELETED: Final[re.Pattern[str]] = re.compile(r"^\[deleted[,\]]", re.IGNORECASE)

MARKDOWN_LINK: Final[re.Pattern[str]] = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
BARE_URL: Final[re.Pattern[str]] = re.compile(r"https?://\S+")
LIST_MARKER: Final[re.Pattern[str]] = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
TABLE_ROW: Final[re.Pattern[str]] = re.compile(r"^\s*\|.*$", re.MULTILINE)
WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")

DESCRIPTION_CAP: Final[int] = 2000
# The explanatory prose belongs to the section, so every requirement under one
# section repeats it. Bounding it keeps the requirement itself the part of the
# anchor that distinguishes one control from its siblings.
CONTEXT_CAP: Final[int] = 400


def _read_member(archive: zipfile.ZipFile, name: str) -> str:
    """Read one archive member as UTF-8, bounded in size.

    Both the declared size and the read are checked. The declared size lives in
    the archive's own header, so it is attacker-controlled and cannot be the
    only guard against a decompression bomb.
    """
    declared = archive.getinfo(name).file_size
    if declared > MAX_MEMBER_BYTES:
        raise ValueError(
            f"{name}: declares {declared} bytes, over the {MAX_MEMBER_BYTES} "
            f"byte cap for a single ASVS member"
        )
    with archive.open(name) as handle:
        payload = handle.read(MAX_MEMBER_BYTES + 1)
    if len(payload) > MAX_MEMBER_BYTES:
        raise ValueError(f"{name}: expanded past the {MAX_MEMBER_BYTES} byte cap")
    return payload.decode("utf-8")


def _as_map(value: object, where: str) -> Mapping[str, object]:
    """Narrow a decoded JSON value to an object, or say which one was not."""
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _as_list(value: object, where: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected an array, got {type(value).__name__}")
    return value


def _text(record: Mapping[str, object], key: str, where: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: missing or empty {key!r} in {record!r:.300}")
    return value.strip()


def _prose(block: str) -> str:
    """Flatten one markdown block to plain sentences.

    The requirement tables and the reference lists are not prose, and a URL
    reaching the encoder is a run of subword tokens that says nothing about the
    control.
    """
    text = TABLE_ROW.sub(" ", block)
    text = MARKDOWN_LINK.sub(r"\1", text)
    text = BARE_URL.sub(" ", text)
    text = LIST_MARKER.sub(" ", text)
    text = text.replace("*", " ").replace("`", " ")
    return WHITESPACE.sub(" ", text).strip()


def _trim(text: str, cap: int) -> str:
    """Cut to cap characters at the last sentence end, or the last word."""
    if len(text) <= cap:
        return text
    head = text[:cap]
    stop = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if stop > cap // 2:
        return head[: stop + 1]
    return head.rsplit(" ", 1)[0].rstrip()


def _chapter_prose(text: str) -> tuple[str, str, dict[str, str]]:
    """Split one chapter file into its objective and per-section narratives.

    Returns (chapter shortcode, control objective, {section shortcode: prose}).
    """
    chapter = CHAPTER_HEADING.search(text)
    if not chapter:
        raise ValueError("chapter file has no '# V<n> <name>' heading")

    headings = list(H2_HEADING.finditer(text))
    objective = ""
    narratives: dict[str, str] = {}

    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = _prose(text[heading.end():end])
        label = heading.group(1)
        if label == OBJECTIVE_HEADING:
            objective = body
            continue
        section = SECTION_HEADING.match(label)
        if section:
            narratives[section.group(1)] = body

    return chapter.group(1), objective, narratives


class AsvsParser(BaseParser):
    framework_id = "asvs"
    # Must stay the spelling OpenCRE uses as standard_name, which is the key the
    # prose index joins a link to its control on.
    framework_name = "ASVS"
    version = EXPECTED_VERSION
    source_url = "https://github.com/OWASP/ASVS"
    mapping_unit_level = "requirement"
    # 286 requirements in 4.0.3 less the eight withdrawn tombstones.
    expected_count = 278

    def parse(self) -> list[Control]:
        source = self.raw_dir / ARCHIVE_NAME
        with zipfile.ZipFile(source) as archive:
            names = sorted(archive.namelist())
            member = next(
                (name for name in names if REQUIREMENTS_MEMBER.search(name)), None,
            )
            if member is None:
                raise ValueError(
                    f"{source}: no 4.0/docs_en/*4.0.3-en.json member. The archive "
                    f"holds {len(names)} entries; re-fetch it from "
                    f"https://github.com/OWASP/ASVS/archive/refs/heads/master.zip"
                )
            payload = _as_map(
                json.loads(_read_member(archive, member)), member,
            )
            objectives, narratives = self._read_chapter_prose(archive, names)

        found = _text(payload, "Version", member)
        if found != EXPECTED_VERSION:
            raise ValueError(
                f"{member}: expected ASVS {EXPECTED_VERSION}, found {found!r}. "
                f"The OpenCRE links carry 4.0.3 numbering, so a different "
                f"release would emit ids that cannot join."
            )

        controls: list[Control] = []
        skipped_deleted = 0

        for chapter_value in _as_list(payload.get("Requirements"), member):
            chapter = _as_map(chapter_value, f"{member}: chapter")
            chapter_code = _text(chapter, "Shortcode", "chapter")
            chapter_name = _text(chapter, "Name", chapter_code)

            for section_value in _as_list(chapter.get("Items"), chapter_code):
                section = _as_map(section_value, f"{chapter_code}: section")
                section_code = _text(section, "Shortcode", chapter_code)
                section_name = _text(section, "Name", section_code)

                # Most specific prose the standard offers for this requirement.
                # 30 of the 71 sections carry none, and their chapter objective
                # is the next thing up the tree that is still about the topic.
                context = narratives.get(section_code) or ""
                context_source = "section"
                if not context:
                    context = objectives.get(chapter_code, "")
                    context_source = "chapter"
                context = _trim(context, CONTEXT_CAP)

                for item_value in _as_list(section.get("Items"), section_code):
                    item = _as_map(item_value, f"{section_code}: requirement")
                    code = _text(item, "Shortcode", section_code)
                    statement = CITATION.sub("", _text(item, "Description", code))
                    statement = statement.strip()

                    if DELETED.match(statement):
                        skipped_deleted += 1
                        continue
                    if not statement:
                        raise ValueError(f"{code}: requirement text is only a citation")

                    # Checked per emitted control rather than per section, so a
                    # section holding nothing but tombstones cannot fail the run.
                    # Without prose beyond the requirement sentence the prose
                    # index reads the control as a restatement of its own title
                    # and drops it, which looks like coverage and is not.
                    if len(context) <= PROSE_MIN_EXTRA_CHARS:
                        raise ValueError(
                            f"{code}: no explanatory prose in section "
                            f"{section_code} or chapter {chapter_code}"
                        )

                    body = f"{statement} {context}"
                    controls.append(Control(
                        # OpenCRE's section_id for ASVS is "V1.1.1" exactly, and
                        # its section_name is this same statement, so the link
                        # resolves on either key.
                        control_id=code,
                        title=statement,
                        description=body[:DESCRIPTION_CAP],
                        full_text=body if len(body) > DESCRIPTION_CAP else None,
                        hierarchy_level="requirement",
                        parent_id=section_code,
                        parent_name=section_name,
                        metadata={
                            "asvs_version": found,
                            "chapter": f"{chapter_code} {chapter_name}",
                            "section": f"{section_code} {section_name}",
                            "context_source": context_source,
                            "levels": self._levels(item, code),
                            "cwe": self._cwe(item, code),
                        },
                    ))

        logger.info(
            "Parsed %d ASVS %s requirements (%d withdrawn tombstones skipped)",
            len(controls), found, skipped_deleted,
        )
        return controls

    @staticmethod
    def _read_chapter_prose(
        archive: zipfile.ZipFile, names: list[str],
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Collect the control objective and section narrative of every chapter."""
        objectives: dict[str, str] = {}
        narratives: dict[str, str] = {}

        for name in names:
            if not CHAPTER_MEMBER.search(name):
                continue
            code, objective, sections = _chapter_prose(_read_member(archive, name))
            objectives[code] = objective
            narratives.update(sections)

        if not objectives:
            raise ValueError(
                "No 4.0/en/0x*-V*.md chapter files in the archive. Without them "
                "every requirement is its own title and nothing joins as prose."
            )
        logger.info(
            "Chapter prose: %d objectives, %d section narratives",
            len(objectives), sum(1 for text in narratives.values() if text),
        )
        return objectives, narratives

    @staticmethod
    def _levels(item: Mapping[str, object], where: str) -> list[str]:
        """The verification levels a requirement applies to, L1 through L3."""
        levels = []
        for level in ("L1", "L2", "L3"):
            flag = _as_map(item.get(level), f"{where}: {level}")
            if flag.get("Required") is True:
                levels.append(level)
        return levels

    @staticmethod
    def _cwe(item: Mapping[str, object], where: str) -> list[str]:
        return [
            f"CWE-{entry}"
            for entry in _as_list(item.get("CWE"), f"{where}: CWE")
            if isinstance(entry, int)
        ]


if __name__ == "__main__":
    AsvsParser().run()
