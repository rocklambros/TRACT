"""Parser for the OWASP Cheat Sheet Series, one control per cheat sheet.

The Cheat Sheets previously reached the corpus through parsers/fetch_opencre.py,
which carries the link graph and each section's name but not the standard's
text. That left 391 training links, 9% of the training set, anchored on a
four-word document title while production hands the model paragraphs. This
parser reads OWASP's own markdown so those links get the real prose.

The join key is the file name, not the document's H1. OpenCRE's section_id and
section_name are both the file name with underscores turned into spaces
("Docker_Security_Cheat_Sheet.md" -> "Docker Security Cheat Sheet"), and 28 of
the 120 sheets carry an H1 that disagrees with their file name ("XS Leaks Cheat
Sheet" is titled "Cross-site leaks Cheat Sheet"). Titles therefore come from the
file name, and a divergent H1 is recorded as an alternate name instead.

Source: https://github.com/OWASP/CheatSheetSeries (repository archive)
"""
from __future__ import annotations

import logging
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "cheatsheets.zip"
# Published sheets live in cheatsheets/. cheatsheets_draft/ holds eight
# unpublished drafts that OpenCRE does not link and the site does not ship.
SHEET_DIR: Final[str] = "cheatsheets"
# The archive is downloaded from GitHub and CLAUDE.md treats framework data as
# untrusted input, so every member is bounded before it is read into memory.
# The largest real sheet in this snapshot is 73 KB.
MAX_MEMBER_BYTES: Final[int] = 1_000_000
# Descriptions are capped so one 58,000-character sheet cannot dominate a
# training batch. The remainder is preserved in full_text.
DESCRIPTION_CAP: Final[int] = 2000
# A deprecated sheet is reduced to a two-sentence tombstone pointing at its
# successor. That tombstone is long enough to pass the prose test in
# tract.text_selection, so it would look like coverage while telling the model
# nothing about the control. Anything this short is treated as a redirect.
STUB_MAX_CHARS: Final[int] = 400
# Redirects are followed transitively in case OWASP ever chains two of them.
# Bounded, with a visited set, because the source is untrusted.
MAX_REDIRECT_HOPS: Final[int] = 3

# Sheets OWASP renamed without leaving a tombstone file behind, so nothing in
# the tree records the old name. The equivalence is the source's own, declared
# in scripts/Generate_Site_mkDocs.sh:
#     redirect_from: "/cheatsheets/JSON_Web_Token_for_Java_Cheat_Sheet.html"
#       -> cheatsheets/JSON_Web_Token_Cheat_Sheet.html
# The old document no longer exists, so it becomes an alternate name on the
# surviving sheet rather than a control of its own.
RENAMED_SHEETS: Final[dict[str, str]] = {
    "JSON_Web_Token_Cheat_Sheet": "JSON Web Token for Java Cheat Sheet",
}

# ── Markdown reduction ────────────────────────────────────────────────────
# Order matters and is enforced by _clean_markdown. Fenced code goes first so
# that markdown inside a code sample is never interpreted.
_FENCE = re.compile(r"^[ \t]*(?:```|~~~).*?(?:^[ \t]*(?:```|~~~)[ \t]*$|\Z)", re.M | re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF_DEFINITION = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", re.M)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_INLINE_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_AUTOLINK = re.compile(r"<https?://[^>]*>")
_BARE_URL = re.compile(r"https?://\S+")
# Whole table rows are dropped rather than flattened. Cipher suite and header
# tables are dense token noise, and no linked sheet keeps its description in
# one: the two most table-heavy lose 21% and 36% of their bytes, and both still
# open with several paragraphs of prose.
_TABLE_ROW = re.compile(r"^[ \t]*\|.*\|[ \t]*$", re.M)
_HORIZONTAL_RULE = re.compile(r"^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$", re.M)
_HEADING = re.compile(r"^[ \t]*(#{1,6})[ \t]*(.+?)[ \t]*#*[ \t]*$", re.M)
_LIST_MARKER = re.compile(r"^[ \t]*(?:[-*+]|\d+[.)])[ \t]+", re.M)
_STRONG_OR_CODE = re.compile(r"\*+|`+|~~")
# Only underscores that wrap a whole word are emphasis. Leaving the general
# case alone keeps snake_case identifiers intact.
_UNDERSCORE_EMPHASIS = re.compile(r"(?<![\w\\])_([^_\n]+)_(?![\w])")
_BACKSLASH_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|~>])")
_BLANK_LINES = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t]+")

# A markdown link to a sibling sheet, with an optional heading anchor.
_SHEET_LINK = re.compile(r"\]\(\s*(?:\./)?([A-Za-z0-9_.-]+)\.md(?:#([A-Za-z0-9_-]+))?\s*\)")
_ANCHOR_UNSAFE = re.compile(r"[^a-z0-9 -]")
_TRAILING_PUNCTUATION: Final[tuple[str, ...]] = (".", ":", "?", "!")


def _anchor_slug(heading: str) -> str:
    """Reproduce GitHub's heading anchor so a "#fragment" can be resolved."""
    return _ANCHOR_UNSAFE.sub("", heading.strip().lower()).replace(" ", "-")


def _clean_markdown(text: str) -> str:
    """Reduce a cheat sheet to its prose.

    The H1 is dropped. It restates the title for 92 of 120 sheets and
    contradicts it for the rest, so keeping it would put a name in the anchor
    that differs from the one the link carries. Every other heading is kept and
    given a full stop, because a heading names the topic of the paragraph under
    it and the pipeline flattens the document to one line, which would
    otherwise fuse "Key Storage" onto the sentence that follows it.
    """
    text = _FENCE.sub(" ", text)
    text = _COMMENT.sub(" ", text)
    text = _REF_DEFINITION.sub(" ", text)
    text = _IMAGE.sub(" ", text)
    text = _INLINE_LINK.sub(r"\1", text)
    text = _AUTOLINK.sub(" ", text)
    text = _BARE_URL.sub(" ", text)
    text = _TABLE_ROW.sub(" ", text)
    text = _HORIZONTAL_RULE.sub(" ", text)
    text = _LIST_MARKER.sub("", text)

    def _flatten_heading(match: re.Match[str]) -> str:
        if len(match.group(1)) == 1:
            return " "
        body = match.group(2).strip()
        if not body or body.endswith(_TRAILING_PUNCTUATION):
            return body
        return f"{body}."

    text = _HEADING.sub(_flatten_heading, text)
    text = _STRONG_OR_CODE.sub("", text)
    text = _UNDERSCORE_EMPHASIS.sub(r"\1", text)
    text = _BACKSLASH_ESCAPE.sub(r"\1", text)
    text = _SPACES.sub(" ", text)
    return _BLANK_LINES.sub("\n\n", text).strip()


def _first_heading(text: str) -> str:
    """The document's own H1, which is not always its file name."""
    match = _HEADING.search(text)
    if match and len(match.group(1)) == 1:
        return match.group(2).strip()
    return ""


def _section_at_anchor(text: str, anchor: str) -> str:
    """Return the heading subtree a "#fragment" points at, or the whole text.

    A redirect that names a fragment moved one section, not a document. The
    deprecated Java injection sheet points at
    Java_Security_Cheat_Sheet.md#injection-prevention-in-java, and taking the
    whole Java sheet would only happen to be right because that section leads
    the file.
    """
    headings = list(_HEADING.finditer(text))
    for index, heading in enumerate(headings):
        if _anchor_slug(heading.group(2)) != anchor:
            continue
        level = len(heading.group(1))
        end = len(text)
        for following in headings[index + 1:]:
            if len(following.group(1)) <= level:
                end = following.start()
                break
        return text[heading.start():end]
    logger.warning("Anchor #%s not found, using the whole target document", anchor)
    return text


def _resolve_stub(
    stem: str, sources: dict[str, str],
) -> tuple[str, str] | None:
    """Follow a deprecation tombstone to the sheet that absorbed its content.

    Returns (raw_markdown, source_stem), or None when the sheet is not a stub.
    The tombstone's own text says only that the sheet was deprecated, which
    would train the model on a redirect notice.
    """
    seen = {stem}
    current = stem
    for _ in range(MAX_REDIRECT_HOPS):
        raw = sources[current]
        if len(_clean_markdown(raw)) > STUB_MAX_CHARS:
            return (raw, current) if current != stem else None
        link = _SHEET_LINK.search(raw)
        if link is None or link.group(1) not in sources or link.group(1) in seen:
            return (raw, current) if current != stem else None
        current = link.group(1)
        seen.add(current)
        if link.group(2):
            return _section_at_anchor(sources[current], link.group(2)), current
    raise ValueError(
        f"Redirect chain from {stem!r} exceeded {MAX_REDIRECT_HOPS} hops: {seen}"
    )


class OwaspCheatSheetsParser(BaseParser):
    framework_id = "owasp_cheat_sheets"
    # Must match the standard_name OpenCRE uses, since tract.text_selection has
    # no alias for this framework and joins on the name verbatim.
    framework_name = "OWASP Cheat Sheets"
    # The series has no release numbers, it is a rolling repository, so the
    # archived commit is the only version the source states about itself.
    # GitHub records it in the zip comment, and _read_archive refuses to run
    # against any other commit so that a silent refresh of data/raw/ cannot
    # change the corpus while the recorded version keeps saying otherwise.
    version = "07111ee754e832e335377ac64fd0f8f848d9029c"
    source_url = "https://cheatsheetseries.owasp.org"
    mapping_unit_level = "cheat_sheet"
    # The series publishes more sheets than OpenCRE links to (120 against 50),
    # so this tracks the source rather than the link set.
    expected_count = 120
    expected_count_is_floor: ClassVar[bool] = True
    fetched_date: ClassVar[str] = "2026-08-14"

    def parse(self) -> list[Control]:
        archive = self.raw_dir / ARCHIVE_NAME
        if not archive.is_file():
            raise FileNotFoundError(
                f"Missing {archive}. Fetch the repository archive from "
                f"{self.source_url} before running this parser."
            )

        sources, revision = self._read_archive(
            archive, self.read_source_bytes(ARCHIVE_NAME),
        )
        if not sources:
            raise ValueError(f"No cheat sheets found under {SHEET_DIR}/ in {archive}")

        claimed = {stem.replace("_", " ").lower() for stem in sources}
        controls: list[Control] = []
        redirected = 0

        for stem in sorted(sources):
            redirect = _resolve_stub(stem, sources)
            content_stem = stem
            raw = sources[stem]
            if redirect is not None:
                raw, content_stem = redirect
                redirected += 1

            body = _clean_markdown(raw)
            title = stem.replace("_", " ")
            if not body:
                # Control.description is min_length=1 and a sheet with no prose
                # is worth no more than its title, which the link already
                # carries. Skipping keeps the fallback visible in the stats
                # rather than hiding it behind an empty description.
                logger.warning("No prose in %s, skipping it", stem)
                continue

            metadata: dict[str, str | list[str]] = {
                "source_file": f"{SHEET_DIR}/{stem}.md",
                "revision": revision,
            }
            if content_stem != stem:
                metadata["deprecated"] = "true"
                metadata["content_source"] = f"{SHEET_DIR}/{content_stem}.md"
            alternates = self._alternate_names(
                stem, sources[stem], claimed, deprecated=content_stem != stem,
            )
            if alternates:
                metadata["alt_titles"] = alternates

            controls.append(Control(
                # OpenCRE carries this sheet's file-derived name in both
                # section_id and section_name, so id and title agree and the
                # prose joins on either key.
                control_id=title,
                title=title,
                description=body[:DESCRIPTION_CAP],
                full_text=body if len(body) > DESCRIPTION_CAP else None,
                hierarchy_level="cheat_sheet",
                metadata=metadata,
            ))

        logger.info(
            "Parsed %d OWASP cheat sheets at revision %s (%d resolved through a "
            "deprecation redirect)",
            len(controls), revision, redirected,
        )
        return controls

    def _read_archive(
        self, archive: Path, payload: bytes,
    ) -> tuple[dict[str, str], str]:
        """Load every published sheet from the archive without unpacking it.

        data/raw/ is immutable, so members are read in memory. Nothing is
        written to disk, which also means no path-traversal surface.

        *archive* names the file for error messages; *payload* carries the
        bytes the recording reader already hashed into the manifest.
        """
        sources: dict[str, str] = {}
        with zipfile.ZipFile(BytesIO(payload)) as bundle:
            revision = bundle.comment.decode("ascii", errors="replace").strip()
            if not revision:
                # Failing open here would skip the check and then stamp the
                # pinned commit into every control's metadata anyway, which is
                # a provenance claim the archive never made.
                raise ValueError(
                    f"{archive} carries no commit in its zip comment, so the "
                    f"pin to {self.version} cannot be verified. Re-fetch with "
                    f"scripts/fetch_frameworks.py, which preserves it."
                )
            if revision != self.version:
                raise ValueError(
                    f"{archive} holds commit {revision}, but this parser is "
                    f"pinned to {self.version}. Re-read the source for renamed "
                    f"and deprecated sheets, then move the pin."
                )
            for member in bundle.infolist():
                parts = member.filename.split("/")
                if len(parts) != 3 or parts[1] != SHEET_DIR:
                    continue
                if not parts[2].endswith(".md"):
                    continue
                if member.file_size > MAX_MEMBER_BYTES:
                    raise ValueError(
                        f"{member.filename} declares {member.file_size} bytes, "
                        f"above the {MAX_MEMBER_BYTES} byte member limit"
                    )
                sources[parts[2][:-3]] = bundle.read(member).decode(
                    "utf-8", errors="replace",
                )
        return sources, revision or self.version

    @staticmethod
    def _alternate_names(
        stem: str, raw: str, claimed: set[str], deprecated: bool,
    ) -> list[str]:
        """Extra names a sheet answers to, for links made before a rename.

        tract.text_selection indexes alternates but never lets one displace a
        real title, so a collision costs nothing. Names already claimed by a
        file are dropped here anyway, to keep the artifact honest about what it
        asserts. A tombstone's H1 reads "DEPRECATED: ..." and names nothing, so
        deprecated sheets contribute no heading alternate.
        """
        names: list[str] = []
        heading = "" if deprecated else _first_heading(raw)
        if heading and heading.lower() not in claimed:
            names.append(heading)
        renamed = RENAMED_SHEETS.get(stem)
        if renamed and renamed.lower() not in claimed:
            names.append(renamed)
        return sorted(set(names))


if __name__ == "__main__":
    OwaspCheatSheetsParser().run()
