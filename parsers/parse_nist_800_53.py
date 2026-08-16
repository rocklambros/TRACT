"""Parser for NIST SP 800-53 rev5, read from NIST's own OSCAL catalog.

800-53 previously reached the corpus through parsers/fetch_opencre.py, which
carries the link graph and each section's name but not the standard's text. That
left 300 training links, 7% of the training set, anchored on strings like
"AC-2 Account Management" while production hands the model paragraphs.

The join is on the section name. Every one of the 300 links carries
section_id == section_name == "<label> <title>", so the control title emitted
here is that same concatenation and ProseIndex resolves it on the title key,
which lowercases both sides and therefore survives OpenCRE's inconsistent
casing ("CM-2 BASELINE CONFIGURATION" alongside "CM-6 Configuration Settings").
control_id carries the bare label so a link that cites "AC-2" alone still
resolves on the id key, which is case-sensitive and so must be upper case.

Source: https://github.com/usnistgov/oscal-content (SP 800-53 rev5 catalog)
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any, ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CATALOG_NAME: Final[str] = "nist_800_53_catalog.json"
DESCRIPTION_CAP: Final[int] = 2000

# The parts that carry the control itself, in the order NIST prints them. The
# statement says what the control requires and the guidance discusses it; both
# are description. Everything else in the catalog is SP 800-53A assessment
# procedure ("assessment-objective", "assessment-method"), which restates the
# statement as a checklist and would roughly triple the text with near-duplicate
# wording before the encoder's budget is reached.
BODY_PART_NAMES: Final[tuple[str, ...]] = ("statement", "guidance")

# Withdrawn controls carry no statement at all, only a link recording which
# control absorbed them. They are tombstones, the same case as CAPEC's
# deprecated patterns, and their text describes a redirection rather than a
# control.
WITHDRAWN: Final[str] = "withdrawn"

# "{{ insert: param, ac-02_odp.01 }}". Left in place it reaches the encoder as
# the tokens "insert param ac 02 odp 01", which is noise in every one of the
# 2,945 places the catalog uses it.
INSERT_RE: Final[re.Pattern[str]] = re.compile(
    r"\{\{\s*insert:\s*param,\s*([^}]+?)\s*\}\}"
)
# OSCAL prose carries markdown cross references, "[AU-02](#au-2)" and
# "[PRIVACT](#18e71fec-...)". The anchor is a document-internal UUID or id and
# says nothing; the link text is the control or reference name and does.
MARKDOWN_LINK_RE: Final[re.Pattern[str]] = re.compile(r"\[([^\]]*)\]\([^)]*\)")
# A parameter reference can resolve to a selection whose choices contain further
# references. Three levels covers the catalog; the bound exists so a malformed
# or hostile catalog cannot spin here.
MAX_INSERT_DEPTH: Final[int] = 3
_WHITESPACE = re.compile(r"\s+")
# The catalog writes "{{ insert: param, sc-13_odp.01 }} ; and" often enough to
# matter (131 places), so substitution leaves a space before the punctuation.
_LOOSE_PUNCTUATION = re.compile(r"\s+([;,.])")

JsonDict = dict[str, Any]


def _iter_parts(parts: list[JsonDict] | None) -> Iterator[JsonDict]:
    """Yield a part and every part nested beneath it, in document order."""
    for part in parts or []:
        yield part
        yield from _iter_parts(part.get("parts"))


def _label(node: JsonDict) -> str:
    """The control's printed label, "AC-2".

    Each control carries three: zero-padded "AC-02", the sp800-53a assessment
    form "AC-02", and the classless one NIST prints and OpenCRE cites. Take the
    classless one, which is unique per control across the whole catalog.
    """
    for prop in node.get("props") or []:
        if prop.get("name") == "label" and "class" not in prop:
            return str(prop.get("value") or "").strip()
    return ""


def _is_withdrawn(node: JsonDict) -> bool:
    return any(
        prop.get("name") == "status" and prop.get("value") == WITHDRAWN
        for prop in node.get("props") or []
    )


def _index_params(catalog: JsonDict) -> dict[str, JsonDict]:
    """Map every parameter id and alt-identifier to its parameter.

    Built over the whole catalog, enhancements included, because the index is
    keyed by ids that are globally unique here (verified: no duplicate id, no
    duplicate alt-identifier, no overlap between the two key spaces). A
    duplicate would silently bind one control's parameter text into another
    control's statement, so it raises rather than taking the first writer.
    """
    index: dict[str, JsonDict] = {}

    def register(key: str, param: JsonDict) -> None:
        if not key:
            return
        if key in index and index[key] is not param:
            raise ValueError(
                f"Duplicate OSCAL parameter key {key!r}; resolving inserts "
                f"against it would bind the wrong text into a statement."
            )
        index[key] = param

    def visit(control: JsonDict) -> None:
        for param in control.get("params") or []:
            register(str(param.get("id") or ""), param)
            for prop in param.get("props") or []:
                if prop.get("name") == "alt-identifier":
                    register(str(prop.get("value") or ""), param)
        for nested in control.get("controls") or []:
            visit(nested)

    for group in catalog.get("groups") or []:
        for control in group.get("controls") or []:
            visit(control)
    return index


def _render_param(param: JsonDict, index: dict[str, JsonDict], depth: int) -> str:
    """Render a parameter the way NIST prints it, minus the bracket furniture.

    An assignment prints its label, "organization-defined frequency". A
    selection prints its choices. NIST wraps both in "[Assignment: ...]" and
    "[Selection (one or more): ...]"; the wrapper is typography that repeats
    1,600 times across the catalog, so only the substantive words are kept.
    """
    label = str(param.get("label") or "").strip()
    if label:
        # 158 of 1,467 labels already say "organization-defined". Prefixing
        # unconditionally would print it twice.
        if "organization-defined" in label.lower():
            return label
        return f"organization-defined {label}"

    select = param.get("select") or {}
    choices = [
        _resolve_inserts(str(choice), index, depth + 1)
        for choice in select.get("choice") or []
    ]
    return " or ".join(choice for choice in choices if choice)


def _resolve_inserts(text: str, index: dict[str, JsonDict], depth: int = 0) -> str:
    """Substitute every "{{ insert: param, ... }}" with the parameter's words."""
    if depth > MAX_INSERT_DEPTH:
        return INSERT_RE.sub(" ", text)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1).strip()
        param = index.get(key)
        if param is None:
            # Fail loud: an unresolved reference means the catalog and this
            # parser disagree about the schema, and the alternative is a
            # sentence with a hole in it that no downstream check would catch.
            raise ValueError(f"OSCAL insert references unknown parameter {key!r}")
        return _render_param(param, index, depth)

    return INSERT_RE.sub(replace, text)


def _clean(text: str, index: dict[str, JsonDict]) -> str:
    resolved = MARKDOWN_LINK_RE.sub(r"\1", _resolve_inserts(text, index))
    return _LOOSE_PUNCTUATION.sub(r"\1", _WHITESPACE.sub(" ", resolved)).strip()


def _body(control: JsonDict, index: dict[str, JsonDict]) -> str:
    """The control's statement followed by its guidance, flattened in order.

    A statement is split across nested items that each carry their own prose
    ("a. Define and document ...", "d.1 Authorized users of the system;"), and
    the order is the requirement's order, so the tree is walked depth first
    rather than collected by name. The printed item labels are dropped: they
    survive tokenisation as bare letters and digits that say nothing about
    which hub the control belongs to.

    Statement first so that the 2,000-character cut spends the encoder's budget
    on what the control requires before it reaches the discussion of it.
    """
    fragments: list[str] = []
    for name in BODY_PART_NAMES:
        for part in control.get("parts") or []:
            if part.get("name") != name:
                continue
            for node in _iter_parts([part]):
                prose = _clean(str(node.get("prose") or ""), index)
                if prose:
                    fragments.append(prose)
    return " ".join(fragments).strip()


class Nist80053Parser(BaseParser):
    framework_id = "nist_800_53"
    framework_name = "NIST 800-53"
    version = "5.2.0"
    source_url = "https://csrc.nist.gov/pubs/sp/800/53/r5/upd1/final"
    mapping_unit_level = "control"
    # Active base controls. The catalog also holds 714 active enhancements
    # ("AC-2(1)"), and they are deliberately left out: OpenCRE links exactly two
    # of them, and emitting all 714 would take this framework from 300 of 3,105
    # corpus controls to 1,014 of 3,819. The stop word list is built by document
    # frequency over that corpus, so tripling one framework's share to buy two
    # links would move every other framework's anchors as well.
    expected_count = 300
    fetched_date: ClassVar[str] = "2026-08-14"

    def parse(self) -> list[Control]:
        catalog: JsonDict = json.loads(self.read_source(CATALOG_NAME))["catalog"]

        catalog_version = str(catalog.get("metadata", {}).get("version") or "")
        if catalog_version != self.version:
            logger.warning(
                "Catalog reports version %s, parser declares %s; the emitted "
                "framework version will be wrong until the parser is updated.",
                catalog_version, self.version,
            )

        params = _index_params(catalog)
        controls: list[Control] = []
        skipped_withdrawn = 0
        skipped_empty = 0

        for group in catalog.get("groups") or []:
            family = str(group.get("id") or "").upper()
            family_name = str(group.get("title") or "")

            for control in group.get("controls") or []:
                if _is_withdrawn(control):
                    skipped_withdrawn += 1
                    continue

                label = _label(control)
                title = str(control.get("title") or "").strip()
                if not label or not title:
                    raise ValueError(
                        f"OSCAL control {control.get('id')!r} has no label or "
                        f"no title, so nothing can join to it."
                    )

                body = _body(control, params)
                if not body:
                    # An active control with no statement and no guidance is
                    # worth no more than the title the link already carries.
                    skipped_empty += 1
                    logger.warning("No body text for %s %s", label, title)
                    continue

                controls.append(Control(
                    # Bare label, upper case. ProseIndex matches control_id
                    # case-sensitively, and OpenCRE cites 800-53 in upper case
                    # while OSCAL ids are lower ("ac-2").
                    control_id=label,
                    # The label and title together, which is the string every
                    # one of the 300 links carries as its section name.
                    title=f"{label} {title}",
                    description=body[:DESCRIPTION_CAP],
                    full_text=body if len(body) > DESCRIPTION_CAP else None,
                    hierarchy_level="control",
                    parent_id=family,
                    parent_name=family_name,
                    metadata={
                        "family": family,
                        "control_title": title,
                        "oscal_id": str(control.get("id") or ""),
                        "catalog_version": catalog_version,
                    },
                ))

        logger.info(
            "Parsed %d NIST 800-53 controls (skipped %d withdrawn, %d without text)",
            len(controls), skipped_withdrawn, skipped_empty,
        )
        return controls


if __name__ == "__main__":
    Nist80053Parser().run()
