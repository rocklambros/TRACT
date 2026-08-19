"""Parser for OWASP SAMM, at the stream level.

The repository has three granularities and only one of them is what OpenCRE
links against. `model/security_practices/` holds the 15 practices,
`model/streams/` holds the 30 practice-and-stream pairs, and
`model/activities/` holds the 90 (practice, stream, level) triples. Every one
of the 30 curated links carries a `section_id` equal to a stream filename stem.
Activity filenames (`D-SA-1-A`) match no section_id at all. [measured]

The statement is the stream's own `description` plus the three activities'
`shortDescription`, in maturity-level order. Measured over the 30 streams that
lands between 347 and 986 characters. Using `longDescription` instead runs
2,548 to 6,678, which puts every description over DESCRIPTION_MAX_LENGTH, makes
BaseParser._sanitize_control write the overflow into full_text, and hands
ProseIndex -- which prefers full_text unconditionally -- an anchor that
prepare_anchor then cuts at 2,150. Richer text the encoder never reads is not
richer text.

Maturity level comes from the activity filename. The `level` field is a GUID
into SAMM's own model, not an ordinal, and sorting on it produces an arbitrary
order that changes with the release.

`text_origin` is set to synthetic. Every character is the publisher's, and the
paragraph is not: it joins four separate source records in an order this parser
chose, and that composition appears nowhere upstream. The corpus report
separates parser-written anchors from publisher-written ones on this key, so an
unmarked statement would be counted as prose a publisher wrote as one unit.

Three of the 30 curated links spell their stream's name differently from SAMM's
own model. See OPENCRE_TITLE_VARIANTS.
"""
from __future__ import annotations

import hashlib
import logging
import re
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from io import BytesIO
from typing import Any, ClassVar, Final

import yaml

from tract.corpus_report import SYNTHETIC_TEXT_ORIGIN, TEXT_ORIGIN_METADATA_KEY
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "samm_core.zip"
SOURCE_SHA256: Final[str] = (
    "16eb608b70bad3039b14ca4e3f300893d29bbc4205c737ac07fcbdfb4f7493a6"
)

STREAM_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/model/streams/([A-Z]-[A-Z]{2}-[AB])\.yml$"
)
# D-SA-1-A.yml: practice code, maturity level, stream letter. The level is read
# here because the `level` FIELD is a GUID. The stream letter is part of the
# grouping key: D-SA-A and D-SA-B share a practice code and own three
# activities each, so keying on the practice alone merges six into one bucket.
ACTIVITY_MEMBER: Final[re.Pattern[str]] = re.compile(
    r"/model/activities/([A-Z]-[A-Z]{2})-(\d)-([AB])\.yml$"
)

EXPECTED_LEVELS: Final[tuple[int, ...]] = (1, 2, 3)

# Stream stem -> the names OpenCRE's curated links spell for it, where those
# differ from SAMM's own. Declared as alt_titles so the link resolves through
# the title channel the curator wrote.
#
# The plan recorded section_name matching the stream name for 30 of 30. It is
# 27 of 30. [measured 2026-08-19] The other three still resolve, by id, because
# the section_id is the stream's filename stem in all 30 cases. What they cost
# without this table is the channel: three links answer by id, and the id-side
# wrong-anchor detector then reads a name the reached control's title does not
# contain and flags all three. The anchor is provably right -- the id IS the
# stream's own stem -- so the flag would be a fact about OpenCRE's spelling
# rather than about the anchor.
#
#   V-AA-A  "Achitecture validation"  missing r, against "Architecture Validation"
#   V-AA-B  "Achitecture mitigation"  missing r, against "Architecture Mitigation"
#   G-PC-A  "Policy & Standards"      ampersand, against "Policy and Standards"
#
# tests/test_parse_samm.py::TestTitleVariants derives this same set from the
# tracked link file and the tracked artifact, so an entry that stops being
# needed and a mismatch that appears both fail there.
OPENCRE_TITLE_VARIANTS: Final[Mapping[str, tuple[str, ...]]] = {
    "G-PC-A": ("Policy & Standards",),
    "V-AA-A": ("Achitecture validation",),
    "V-AA-B": ("Achitecture mitigation",),
}


class SammParser(BaseParser):
    framework_id: ClassVar[str] = "samm"
    # Matches the curated links' standard_name exactly, so no alias is needed.
    framework_name: ClassVar[str] = "SAMM"
    version: ClassVar[str] = "2.0"
    source_url: ClassVar[str] = "https://owaspsamm.org/model/"
    mapping_unit_level: ClassVar[str] = "stream"
    # 15 practices x 2 streams. [measured]
    expected_count: ClassVar[int] = 30
    fetched_date: ClassVar[str] = "2026-08-15"
    # Every statement is at least 347 characters and none equals its name.
    # [measured]
    min_prose_fraction: ClassVar[float] = 1.0
    # Class-level so a fixture-backed test declares its own digest instead of
    # the real gate being widened to accept two archives.
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256
    # Class-level for the same reason: a fixture archive carries two streams
    # and cannot satisfy a table that names three others.
    title_variants: ClassVar[Mapping[str, tuple[str, ...]]] = (
        OPENCRE_TITLE_VARIANTS
    )

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        streams, activities = self._read_members(payload)
        controls = self.build_controls(streams, activities, self.title_variants)
        logger.info(
            "%s: %d streams, statement length %d..%d characters",
            self.framework_id, len(controls),
            min(len(c.description) for c in controls),
            max(len(c.description) for c in controls),
        )
        return controls

    def _check_digest(self, payload: bytes) -> None:
        """Refuse an archive that is not the pinned one.

        Raises:
            ValueError: If the digest does not match `expected_sha256`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        if actual == self.expected_sha256:
            return
        raise ValueError(
            f"{self.framework_id}: {ARCHIVE_NAME} has sha256 {actual}, not "
            f"the pinned {self.expected_sha256}. The statement-length "
            f"measurements that chose shortDescription over longDescription "
            f"were taken against the pinned bytes, as were expected_count, "
            f"min_prose_fraction and the join floor. Re-measure before moving "
            f"the pin, and move it in scripts/fetch_frameworks.py at the same "
            f"time."
        )

    def _read_members(
        self, payload: bytes,
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, list[tuple[int, dict[str, Any]]]],
    ]:
        """Streams keyed by stem, activities keyed by stem and maturity level.

        Raises:
            ValueError: If the archive carries no stream members.
        """
        streams: dict[str, dict[str, Any]] = {}
        activities: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            for name in sorted(archive.namelist()):
                stream = STREAM_MEMBER.search(name)
                if stream:
                    streams[stream.group(1)] = self._document(archive, name)
                    continue
                activity = ACTIVITY_MEMBER.search(name)
                if activity:
                    key = f"{activity.group(1)}-{activity.group(3)}"
                    activities[key].append(
                        (int(activity.group(2)), self._document(archive, name))
                    )
        if not streams:
            raise ValueError(
                f"{self.framework_id}: no model/streams/*.yml members in "
                f"{ARCHIVE_NAME}. The stream stem is the only identifier "
                f"OpenCRE links against; without it there is no join."
            )
        return streams, dict(activities)

    def _document(self, archive: zipfile.ZipFile, name: str) -> dict[str, Any]:
        """One YAML member, refused unless it is a mapping.

        Raises:
            ValueError: If the member parses to anything but a mapping.
        """
        document = yaml.safe_load(archive.read(name).decode("utf-8"))
        if not isinstance(document, dict):
            raise ValueError(
                f"{self.framework_id}: {name} parsed to a "
                f"{type(document).__name__} where a mapping was expected. "
                f"Every model member is a flat mapping, so a different shape "
                f"means the layout changed and the field reads below would "
                f"return nothing without saying so."
            )
        return document

    @classmethod
    def build_controls(
        cls,
        streams: dict[str, dict[str, Any]],
        activities: dict[str, list[tuple[int, dict[str, Any]]]],
        title_variants: Mapping[str, tuple[str, ...]] | None = None,
    ) -> list[Control]:
        """One Control per stream, statement built from its three activities.

        `title_variants` defaults to the class table. It is a parameter rather
        than a plain `cls.title_variants` read because a fixture archive
        carries two streams and cannot satisfy a table naming three others,
        and an instance attribute is invisible to a classmethod.

        Raises:
            ValueError: If the declared title variants do not match the streams
                given, or a stream has no name, no description, no activities,
                or activities whose levels are not exactly EXPECTED_LEVELS.
        """
        variants = cls.title_variants if title_variants is None else title_variants
        cls._check_title_variants(streams, variants)
        controls: list[Control] = []
        for stem in sorted(streams):
            stream = streams[stem]
            # Sorted on the filename level, explicitly. Sorting the raw pairs
            # would fall through to comparing two dicts on a tie, and a tie is
            # a duplicate level that the check below is meant to report.
            owned = sorted(
                activities.get(stem, []), key=lambda pair: pair[0],
            )
            if not owned:
                raise ValueError(
                    f"samm: stream {stem} has no activities. The statement is "
                    f"built from them, so an empty list would emit a control "
                    f"carrying only the two-sentence stream description."
                )
            levels = tuple(level for level, _ in owned)
            if levels != EXPECTED_LEVELS:
                raise ValueError(
                    f"samm: stream {stem} has maturity levels {levels}, "
                    f"expected {EXPECTED_LEVELS}. A missing level means the "
                    f"statement is short by a third and nothing else would "
                    f"say so."
                )
            controls.append(cls._to_control(stem, stream, owned, variants))
        return controls

    @staticmethod
    def _check_title_variants(
        streams: Mapping[str, Any],
        variants: Mapping[str, tuple[str, ...]],
    ) -> None:
        """Refuse a variant table that has drifted from the model.

        Raises:
            ValueError: If a declared stem is absent, or a declared variant is
                empty or restates the stream's own name.
        """
        for stem in sorted(variants):
            if stem not in streams:
                raise ValueError(
                    f"samm: the title variant table names no stream {stem!r}. "
                    f"Streams read: {sorted(streams)}. A variant for a stream "
                    f"that does not exist reaches no control and still reads "
                    f"as a live alternate."
                )
            name = str(streams[stem].get("name") or "").strip()
            for variant in variants[stem]:
                if not variant.strip():
                    raise ValueError(
                        f"samm: stream {stem} declares an empty title "
                        f"variant. An empty key can never be looked up."
                    )
                if variant.strip().lower() == name.lower():
                    raise ValueError(
                        f"samm: stream {stem} declares the title variant "
                        f"{variant!r}, which is already the stream's own name. "
                        f"ProseIndex indexes real titles first and never lets "
                        f"an alternate displace one, so this entry is dead."
                    )

    @classmethod
    def _to_control(
        cls,
        stem: str,
        stream: Mapping[str, Any],
        owned: list[tuple[int, dict[str, Any]]],
        variants: Mapping[str, tuple[str, ...]],
    ) -> Control:
        """One stream as a Control.

        Raises:
            ValueError: If the stream has no name or no description.
        """
        name = str(stream.get("name") or "").strip()
        if not name:
            raise ValueError(
                f"samm: stream {stem} has no name. The name is what 27 of the "
                f"30 curated links carry as section_name, so a nameless "
                f"stream loses the title channel with nothing to say it did."
            )
        description = str(stream.get("description") or "").strip()
        if not description:
            raise ValueError(
                f"samm: stream {stem} has no description. All 30 carry one, "
                f"110 characters at the shortest, and it is the only part of "
                f"the statement that describes the stream rather than one "
                f"maturity level of it."
            )

        metadata: dict[str, str | list[str]] = {
            "stream_letter": str(stream.get("letter") or ""),
            TEXT_ORIGIN_METADATA_KEY: SYNTHETIC_TEXT_ORIGIN,
        }
        declared = variants.get(stem)
        if declared:
            metadata["alt_titles"] = list(declared)

        return Control(
            control_id=stem,
            title=name,
            description=cls._statement(stem, description, owned),
            hierarchy_level="stream",
            parent_id=str(stream.get("practice") or "").strip() or None,
            metadata=metadata,
        )

    @staticmethod
    def _statement(
        stem: str,
        description: str,
        owned: list[tuple[int, dict[str, Any]]],
    ) -> str:
        """The stream description, then one short description per level.

        Raises:
            ValueError: If an activity carries no shortDescription. All 90 do,
                30 characters at the shortest, so an empty one means the
                schema changed and the statement would be short by a third.
        """
        parts = [description]
        for level, activity in owned:
            short = str(activity.get("shortDescription") or "").strip()
            if not short:
                raise ValueError(
                    f"samm: stream {stem} level {level} has no "
                    f"shortDescription. It is the only field of an activity "
                    f"that fits the encoder budget, and dropping it silently "
                    f"leaves the statement short by a third."
                )
            parts.append(short)
        return "\n\n".join(parts)


def main() -> None:
    SammParser().run()


if __name__ == "__main__":
    main()
