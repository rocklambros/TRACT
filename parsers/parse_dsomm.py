"""Parser for the OWASP DevSecOps Maturity Model.

OpenCRE keys every DSOMM link on the activity's `uuid`, and 214 of 214 curated
links carry a uuid this file defines. What OpenCRE puts in `section_name` is
the SUB-DIMENSION, one level above the activity, so 214 links share 18 names.
Falling back to that name is what the corpus does today and it is why DSOMM
reads 11.89 links per anchor. Joining on the uuid takes it to 182 anchors.

The statement is `description`, `risk` and `measure` concatenated in that
order, and the order is not cosmetic. `description` is present on 53 of the 194
activities and non-empty on 51. `risk` and `measure` are non-empty on all 194.
A parser reading `description` alone would emit 143 empty statements, which
`Control` rejects, and the survivors would fail the prose rule that decides
whether ProseIndex indexes a control at all.

No `text_origin` marker. That marker separates parser-written anchors from
publisher-written ones, and every character here is the publisher's, read from
one record, in the key order the generated model file itself uses. The
frameworks that set it merge text drawn from several separate source
documents into a single control, which is a different claim about provenance.

`level`, `usefulness`, `isImplemented` and `evidence` are deliberately unused:
they are assessment state, not control text.
"""
from __future__ import annotations

import hashlib
import logging
import zipfile
from io import BytesIO
from typing import Any, ClassVar, Final

import yaml

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARCHIVE_NAME: Final[str] = "dsomm_data.zip"
# The archive root carries the pinned commit sha, so the member is located by
# suffix rather than by a path that changes on every re-pin.
MODEL_SUFFIX: Final[str] = "generated/model.yaml"

SOURCE_SHA256: Final[str] = (
    "a6d773129591d59e7c0757651142c39a341400333f40c1555fb2481ae89f2c66"
)

# Statement fields, in the order they are joined. See the module docstring for
# why `description` cannot be the only one.
STATEMENT_FIELDS: Final[tuple[str, ...]] = ("description", "risk", "measure")


class DsommParser(BaseParser):
    framework_id: ClassVar[str] = "dsomm"
    # Matches canonical_framework("DevSecOps Maturity Model (DSOMM)"), which
    # FRAMEWORK_NAME_ALIASES already maps to "DSOMM". No new alias is needed.
    framework_name: ClassVar[str] = "DSOMM"
    version: ClassVar[str] = "4.3.1"
    source_url: ClassVar[str] = (
        "https://github.com/devsecopsmaturitymodel/DevSecOps-MaturityModel-data"
    )
    mapping_unit_level: ClassVar[str] = "activity"
    # 194 leaf activities in the pinned archive. [measured]
    expected_count: ClassVar[int] = 194
    fetched_date: ClassVar[str] = "2026-08-15"
    # 192 of 194 clear the 60-character honest-prose bar and differ from their
    # title. The two that do not are single-sentence measures with long names.
    # [measured] The floor sits just under, close enough that a regression
    # toward name-only extraction still trips it.
    min_prose_fraction: ClassVar[float] = 0.98
    # Class-level so a fixture-backed test declares its own digest instead of
    # the real gate being widened to accept two archives.
    expected_sha256: ClassVar[str | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        payload = self.read_source_bytes(ARCHIVE_NAME)
        self._check_digest(payload)
        model = self._read_model(payload)
        controls = self.activities_to_controls(model)
        logger.info(
            "%s: %d activities across %d dimensions",
            self.framework_id, len(controls),
            len({c.parent_name for c in controls}),
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
            f"the pinned {self.expected_sha256}. expected_count, "
            f"min_prose_fraction and the join floor were all measured against "
            f"the pinned bytes. Re-measure before moving the pin, and move it "
            f"in scripts/fetch_frameworks.py at the same time."
        )

    def _read_model(self, payload: bytes) -> dict[str, Any]:
        """The second YAML document of the generated model file.

        Raises:
            ValueError: If the member is absent or the stream is not two
                documents with a mapping second.
        """
        with zipfile.ZipFile(BytesIO(payload)) as archive:
            names = [n for n in archive.namelist() if n.endswith(MODEL_SUFFIX)]
            if len(names) != 1:
                raise ValueError(
                    f"{self.framework_id}: expected exactly one "
                    f"{MODEL_SUFFIX} in {ARCHIVE_NAME}, found {names}. The "
                    f"generated file is what flattens 26 per-subdimension "
                    f"YAMLs into one document. Without it the join level is "
                    f"guesswork."
                )
            raw = archive.read(names[0]).decode("utf-8")

        documents = list(yaml.safe_load_all(raw))
        if len(documents) != 2 or not isinstance(documents[1], dict):
            raise ValueError(
                f"{self.framework_id}: {MODEL_SUFFIX} is not a meta document "
                f"followed by a model mapping (got {len(documents)} "
                f"document(s)). The layout changed."
            )
        return documents[1]

    @classmethod
    def activities_to_controls(cls, model: dict[str, Any]) -> list[Control]:
        """One Control per leaf activity, in source order.

        Raises:
            ValueError: On an activity with no uuid, or with no statement text
                in any of STATEMENT_FIELDS.
        """
        controls: list[Control] = []
        for dimension, sub_dimensions in model.items():
            for sub_dimension, activities in sub_dimensions.items():
                for name, body in activities.items():
                    controls.append(
                        cls._to_control(dimension, sub_dimension, name, body)
                    )
        return controls

    @classmethod
    def _to_control(
        cls, dimension: str, sub_dimension: str, name: str, body: dict[str, Any],
    ) -> Control:
        uuid = str(body.get("uuid") or "").strip()
        if not uuid:
            raise ValueError(
                f"dsomm: activity {name!r} under {dimension}/{sub_dimension} "
                f"has no uuid. The uuid is what OpenCRE links against, so an "
                f"activity without one cannot be joined and must not be "
                f"emitted as though it could."
            )
        statement = cls._statement(body)
        if not statement:
            raise ValueError(
                f"dsomm: activity {name!r} (uuid {uuid}) has no text in any "
                f"of {STATEMENT_FIELDS}. All 194 activities in the pinned "
                f"archive have risk and measure, so an empty one means the "
                f"schema changed."
            )
        return Control(
            control_id=uuid,
            title=name.strip(),
            description=statement,
            hierarchy_level="activity",
            parent_id=sub_dimension,
            parent_name=dimension,
            metadata={"sub_dimension": sub_dimension, "dimension": dimension},
        )

    @staticmethod
    def _statement(body: dict[str, Any]) -> str:
        parts = [
            str(body.get(field) or "").strip() for field in STATEMENT_FIELDS
        ]
        return "\n\n".join(part for part in parts if part)


def main() -> None:
    DsommParser().run()


if __name__ == "__main__":
    main()
