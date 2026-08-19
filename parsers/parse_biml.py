"""Parser for the Berryville Institute of Machine Learning risk analyses.

Two documents, both required. ara.pdf is the 2020 architectural risk analysis
of machine learning systems, "BIML-78"; BIML-LLM24.pdf is the 2024 analysis of
large language models, "BIML-24(LLM)". Both mark every named risk with an
inline [category:number:label] tag.

A tag that opens a line is not enough to call it a definition. Both documents
wrap their columns, so a mid-sentence cross-reference lands at a line start
nine times across the two files: ara's top-ten summary ends a line with "See"
and the next line opens with "[data:2:transfer], [model:1:improper re-use] and
[model:2:Trojan] below." Reading that as a definition gave data:2, which
carries a curated link, the text of three unrelated summary items, and gave
inference:3 the two characters ".)". So a definition is a line-start tag whose
remainder is empty or opens a sentence, and a remainder opening with
punctuation or a lowercase word is a wrapped cross-reference that belongs to
the risk above it. [measured: 6 in ara, 3 in BIML-LLM24]

A risk body also has to stop somewhere other than at the next definition. The
last risk of every block is followed by the publisher's "Associated controls."
heading or by a page break into unrelated matter, and with no terminator ara's
system:10 absorbed the whole reference list at 39,093 characters. Two
terminators fix that: the "Associated controls." heading, and a page-furniture
line reached at a sentence boundary. The sentence-boundary condition is
load-bearing, because five furniture lines fall mid-sentence where the body
genuinely continues on the next page. [measured: 11 heading cuts, 19 page
cuts, 5 furniture lines correctly skipped]

The two documents reuse the same category:number space for different risks:
ara's raw:3 is Storage and BIML-LLM24's is data feudalism. So a control_id must
carry its document, and it is spelled exactly as OpenCRE's prefixed ids are.

Titles carry the document and the tag, and that is the load-bearing decision
here. ProseIndex.lookup resolves the section NAME before the section id.
OpenCRE gives two different risks the name "Data Confidentiality" and three
link rows the name "Hosting", so seven of the 21 curated rows participate in a
label collision. A bare label as the title would hand all of them one anchor,
which is the same collapse the title-first order was written to fix for NIST AI
100-2. The tag is in the title as well as the document, because a label is not
unique inside one document either: ara names two risks "storage" and
BIML-LLM24 names three "data confidentiality", two of which carry curated
links. [measured: 5 repeated labels in ara over 11 tags, 11 in BIML-LLM24 over
23] A scoped title matches no link name at all, so every row falls through to
the id channel, where the document prefix disambiguates.

Eight curated links carry an unprefixed id. Seven are resolved by exact
tag-label match against one document and only one document, and are declared
in UNPREFIXED_IDS. The eighth, output:2 "Direct Output", matches ara's
[output:1:direct] by name while ara's own output:2 is provenance and
BIML-LLM24's is wrongness. It is resolved by NAME, as an alt_title on ara's
output:1, and the id conflict is written to the repair audit. Aliasing the id
instead would assert OpenCRE made a typo, which the evidence supports no
better.

One of the 21 links duplicates another target. inference:9 appears both as
"BIML-24(LLM): inference:9" and bare, and UNPREFIXED_IDS routes the bare form
to that same control, so 21 links land on 19 anchors rather than 20.
"""
from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from io import BytesIO
from typing import ClassVar, Final

import pdfplumber

from tract.config import DESCRIPTION_MAX_LENGTH
from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

ARA: Final[str] = "BIML-78(2020)"
LLM24: Final[str] = "BIML-24(LLM)"

SOURCE_FILES: Final[dict[str, str]] = {ARA: "ara.pdf", LLM24: "BIML-LLM24.pdf"}
SOURCE_SHA256: Final[dict[str, str]] = {
    ARA: "247d7f06d8c768cc734dc84ab7004c6e4d645e91911af61002fd1743807ef312",
    LLM24: "1a41ba1a9218e6aecdcab46d2cc6cf8a3b99f6cc1c98a3683bf3a6e4964e955f",
}

TAG: Final[re.Pattern[str]] = re.compile(r"\[([a-z]+):(\d+):([^\]]+)\]")

# The publisher's heading between a block of risks and the controls that follow
# it. Both documents spell it the same way and neither uses the phrase anywhere
# else. [measured: 8 lines in ara, 4 in BIML-LLM24]
CONTROLS_HEADING: Final[str] = "Associated controls"

# Running heads and feet. Both forms appear alone on a line, and both also
# appear mid-sentence where a paragraph crosses a page, which is why
# _is_body_end only treats one as a terminator at a sentence boundary.
PAGE_FURNITURE: Final[re.Pattern[str]] = re.compile(
    r"\A(?:\d+\s+Berryville Institute of Machine Learning|BIML(?:\s+\d+)?)\Z"
)

# What a completed sentence ends with, including the curly forms the PDFs use.
SENTENCE_END: Final[tuple[str, ...]] = (".", "?", "!", "”", '"', "’")

# What a body may open with and still be a definition rather than a wrapped
# cross-reference. An uppercase letter or an opening quotation mark starts a
# sentence; a comma, a bracket, a full stop or a lowercase word continues one.
BODY_OPENERS: Final[tuple[str, ...]] = ("“", '"')

# The parser's own bound on a risk body, held strictly below the base class's
# description limit. BaseParser._sanitize_control moves anything longer into
# full_text behind the parser's back, which would hand ProseIndex a different
# anchor from the one measured here. Ruling R14: the parser owns its anchor.
MAX_BODY_CHARS: Final[int] = DESCRIPTION_MAX_LENGTH

# Unprefixed OpenCRE ids, each resolved by exact tag-label match against one
# document and only one. Hand-verified against both PDFs:
#   model:2      ara "trojan";                   BIML-LLM24's model:2 is "improper use"
#   raw:3        ara "storage";                  BIML-LLM24's raw:3 is "data feudalism"
#   input:2      ara "controlled input stream";  BIML-LLM24's input:2 is "prompt injection"
#   inference:4  ara "hosting";                  BIML-LLM24's inference:4 is "stochasticity"
#   alg:11       ara "parameters";               absent from BIML-LLM24
#   inference:9  BIML-LLM24 "hosting";           absent from ara
#   output:4     BIML-LLM24 "data confidentiality"; ara's output:4 is "inscrutability"
UNPREFIXED_IDS: Final[dict[str, tuple[str, str]]] = {
    "model:2": (ARA, "model:2"),
    "raw:3": (ARA, "raw:3"),
    "input:2": (ARA, "input:2"),
    "inference:4": (ARA, "inference:4"),
    "alg:11": (ARA, "alg:11"),
    "inference:9": (LLM24, "inference:9"),
    "output:4": (LLM24, "output:4"),
}

# The one row whose id and name disagree upstream. Resolved by name.
# Keyed by the OpenCRE section_id, so a second conflict on the same name cannot
# silently overwrite this one.
NAME_CONFLICTS: Final[dict[str, tuple[str, str, str, str]]] = {
    "output:2": (
        "Direct Output", ARA, "output:1",
        "OpenCRE's section_id output:2 names ara's provenance risk and "
        "BIML-LLM24's wrongness risk; its section_name matches ara's "
        "[output:1:direct] exactly. Resolved by name, because the name is "
        "the only side of the row that matches anything in either document.",
    ),
}

# Two rows where OpenCRE prefixes the component onto BIML's own descriptor.
# The id is right and the anchor it reaches is right; only the spelling differs,
# so the id-side wrong-anchor detector fires on a fact about OpenCRE's naming
# rather than about the anchor. Declared as alt_titles on the control the id
# already reaches, which is the same remedy parse_samm.py applies to the three
# stream names OpenCRE misspells.
#
#   BIML-78(2020): data:1   "Data Poisoning"             against "Poisoning"
#   BIML-24(LLM): output:4  "Output Data Confidentiality" against "Data Confidentiality"
#
# Neither spelling collides with any other BIML title [measured 2026-08-19], so
# adding them cannot pull a link onto a different control. Titles stay
# document-scoped for exactly that reason and these two are the only exceptions.
# tests/test_parse_biml.py::TestOpenCreTitleVariants derives this table from the
# tracked link file and the parsed artifact, so an entry that stops being needed
# and a divergence that newly appears both fail.
OPENCRE_TITLE_VARIANTS: Final[Mapping[str, tuple[str, ...]]] = {
    "BIML-78(2020): data:1": ("Data Poisoning",),
    "BIML-24(LLM): output:4": ("Output Data Confidentiality",),
}

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def _is_definition(remainder: str) -> bool:
    """Whether a line-start tag defines a risk rather than referring to one.

    `remainder` is everything after the closing bracket. Empty means the tag
    stands alone on its line, which is ara's block style. Otherwise the body
    has to open a sentence: BIML-LLM24 runs its definitions inline and every
    one of them starts with a capital or an opening quotation mark, while every
    wrapped cross-reference continues the sentence above it with punctuation or
    a lowercase word. [measured on both pinned PDFs]
    """
    head = remainder.lstrip()
    if not head:
        return True
    return head[0].isupper() or head[0] in BODY_OPENERS


def _titled(label: str) -> str:
    """The label with each word's first character raised.

    Not str.title(), which lowercases the rest of every word and turns the
    label "API encoding" into "Api Encoding".
    """
    return " ".join(word[:1].upper() + word[1:] for word in label.split())


class BimlParser(BaseParser):
    framework_id: ClassVar[str] = "biml"
    framework_name: ClassVar[str] = "BIML"
    version: ClassVar[str] = "BIML-78 (2020) + BIML-24 LLM (2024)"
    source_url: ClassVar[str] = "https://berryvilleiml.com/results/"
    mapping_unit_level: ClassVar[str] = "risk"
    # 78 definitional tags in ara.pdf and 68 in BIML-LLM24.pdf. [measured]
    expected_count: ClassVar[int] = 146
    fetched_date: ClassVar[str] = "2026-08-15"
    # Exact, not a floor, and per document. A count on the SUM is satisfied by
    # 80 + 66 as readily as by 78 + 68, so one document could lose a dozen
    # risks while the other gained them. Exact rather than a lower bound
    # because expected_sha256 pins the bytes: with the source fixed, the tag
    # census is a fact about these two files and a surplus is a parser defect
    # exactly as an undershoot is. Overridable so a synthetic pair of PDFs can
    # drive parse() in CI. [measured]
    expected_tags: ClassVar[dict[str, int]] = {ARA: 78, LLM24: 68}
    # Every one of the 146 risk bodies clears HONEST_PROSE_MIN_CHARS, the
    # shortest running 84 characters. [measured] The floor is 1.00 because the
    # digest pin fixes the source, so anything below it is the parser losing
    # text rather than the publisher changing it. Reading only the line-start
    # tags without the cross-reference rule above scored 0.9932: ara's
    # inference:3 came out as the two characters ".)".
    min_prose_fraction: ClassVar[float] = 1.00
    expected_sha256: ClassVar[dict[str, str] | None] = SOURCE_SHA256

    def parse(self) -> list[Control]:
        texts: dict[str, str] = {}
        for document, filename in SOURCE_FILES.items():
            payload = self.read_source_bytes(filename)
            self._check_digest(document, filename, payload)
            with pdfplumber.open(BytesIO(payload)) as pdf:
                texts[document] = "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        controls, audit = self.build_controls(texts, require_targets=True)
        self._check_shape(controls)
        self.write_repair_audit(audit)
        for record in audit:
            logger.warning("%s: %s", self.framework_id, record["reason"])
        logger.info(
            "%s: %d risks (%s)", self.framework_id, len(controls),
            ", ".join(
                f"{document} "
                f"{sum(1 for c in controls if c.control_id.startswith(f'{document}: '))}"
                for document in SOURCE_FILES
            ),
        )
        return controls

    def _check_digest(
        self, document: str, filename: str, payload: bytes,
    ) -> None:
        """Refuse a PDF that is not the pinned one.

        Raises:
            ValueError: If the digest does not match the pin for `document`.
        """
        if self.expected_sha256 is None:
            return
        actual = hashlib.sha256(payload).hexdigest()
        expected = self.expected_sha256[document]
        if actual != expected:
            raise ValueError(
                f"{self.framework_id}: {filename} has sha256 {actual}, not "
                f"the pinned {expected}. UNPREFIXED_IDS and NAME_CONFLICTS "
                f"were resolved by comparing tag labels across these exact "
                f"two documents; a different revision can move a label onto a "
                f"different number and silently re-point a link."
            )

    def _check_shape(self, controls: list[Control]) -> None:
        """Refuse a parse where one document lost risks and the other gained.

        Raises:
            ValueError: If a document yields a different count from its census.
        """
        for document, census in sorted(self.expected_tags.items()):
            found = sum(
                1 for c in controls if c.control_id.startswith(f"{document}: ")
            )
            if found != census:
                raise ValueError(
                    f"{self.framework_id}: {document} yielded {found} risk(s) "
                    f"against a census of {census}. expected_count covers the "
                    f"sum, so a shortfall here can be hidden by a surplus in "
                    f"the other document. The source bytes are pinned by "
                    f"sha256, so this is the parser changing rather than the "
                    f"publisher."
                )

    @classmethod
    def risks_from_text(
        cls, text: str, document: str,
    ) -> list[tuple[str, str, str]]:
        """(tag, label, body) per definitional tag, in document order.

        The body keeps its line breaks. sanitize_text rejoins words the PDF
        wrapped across a hyphen, and it can only do that while the newline is
        still there: 121 words in BIML-LLM24 and 7 in ara arrive as "inter-"
        and "face" on separate lines. [measured]

        Raises:
            ValueError: If the text holds no definitional tag at all, which
                means the extraction changed shape rather than the source
                losing every risk.
        """
        found: list[tuple[str, str, list[str]]] = []
        closed: set[int] = set()
        for line in text.split("\n"):
            stripped = line.strip()
            match = TAG.match(stripped)
            if match is not None and _is_definition(stripped[match.end():]):
                found.append((
                    f"{match.group(1)}:{match.group(2)}",
                    _WHITESPACE.sub(" ", match.group(3)).strip(),
                    [stripped[match.end():].lstrip()],
                ))
                continue
            if not found or len(found) - 1 in closed:
                continue
            if cls._is_body_end(stripped, found[-1][2]):
                closed.add(len(found) - 1)
                continue
            found[-1][2].append(stripped)

        if not found:
            raise ValueError(
                f"{cls.framework_id}: no definitional tag in {document}. The "
                f"documents mark every risk with [category:number:label] at a "
                f"line start, so an empty result means extract_text() returned "
                f"a different shape, not that the document lost its risks."
            )

        seen: set[str] = set()
        risks: list[tuple[str, str, str]] = []
        for tag, label, body in found:
            # First definition wins. Later blocks reuse a tag to name the
            # CONTROL for that risk, sometimes under a different descriptor:
            # ara's [data:4:storage] risk is answered by [data:4:disimilarity].
            if tag in seen:
                continue
            seen.add(tag)
            risks.append((tag, label, "\n".join(body).strip()))
        return risks

    @staticmethod
    def _is_body_end(line: str, body: list[str]) -> bool:
        """Whether this line ends the risk body that is being accumulated.

        Two terminators. The publisher's controls heading always closes a block
        of risks. A running head or foot closes one only when the text so far
        ends a sentence, because a paragraph that crosses a page has the same
        furniture sitting in the middle of it.
        """
        if line.startswith(CONTROLS_HEADING):
            return True
        if not PAGE_FURNITURE.match(line):
            return False
        return " ".join(body).strip().endswith(SENTENCE_END)

    @classmethod
    def build_controls(
        cls, texts: dict[str, str], require_targets: bool = False,
    ) -> tuple[list[Control], list[dict[str, object]]]:
        """Document-scoped controls, plus the audit of every repair.

        Raises:
            ValueError: If require_targets is set and a declared alternate id
                or name conflict points at a tag this parse did not produce.
        """
        alt_ids: dict[str, list[str]] = {}
        for unprefixed, (document, tag) in UNPREFIXED_IDS.items():
            alt_ids.setdefault(f"{document}: {tag}", []).append(unprefixed)

        alt_titles: dict[str, list[str]] = {}
        for control_id, variants in OPENCRE_TITLE_VARIANTS.items():
            alt_titles.setdefault(control_id, []).extend(variants)

        audit: list[dict[str, object]] = []
        for section_id, (name, document, tag, reason) in NAME_CONFLICTS.items():
            alt_titles.setdefault(f"{document}: {tag}", []).append(name)
            audit.append({
                "repair": "name_conflict",
                "opencre_section_id": section_id,
                "opencre_section_name": name,
                "resolved_to": f"{document}: {tag}",
                "resolved_by": "section_name",
                "reason": (
                    f"{section_id} resolved by name to {document}: {tag}, not "
                    f"by its own id. {reason}"
                ),
            })

        controls: list[Control] = []
        for document in sorted(texts):
            for tag, label, body in cls.risks_from_text(texts[document], document):
                control_id = f"{document}: {tag}"
                statement = cls._cap(control_id, body, audit)
                metadata: dict[str, str | list[str]] = {
                    "document": document, "tag": tag, "label": label,
                }
                if control_id in alt_ids:
                    metadata["alt_ids"] = alt_ids[control_id]
                if control_id in alt_titles:
                    metadata["alt_titles"] = alt_titles[control_id]
                controls.append(Control(
                    control_id=control_id,
                    # Scoped by document AND tag so no OpenCRE section_name can
                    # match it and no two risks can share it. See the module
                    # docstring: seven of 21 rows share a label across the two
                    # documents, and 34 tags share one inside a document.
                    title=f"{_titled(label)} ({control_id})",
                    description=statement,
                    hierarchy_level="risk",
                    parent_id=f"{document}: {tag.split(':')[0]}",
                    metadata=metadata,
                ))

        cls._check_targets(
            {c.control_id for c in controls}, alt_ids, alt_titles,
            require_targets,
        )
        return controls, audit

    @staticmethod
    def _cap(
        control_id: str, body: str, audit: list[dict[str, object]],
    ) -> str:
        """The body, held under the description limit, cut on a line boundary.

        Ruling R14. A description at or over DESCRIPTION_MAX_LENGTH is rewritten
        by BaseParser._sanitize_control, which moves the whole text into
        full_text and hands ProseIndex an anchor the parser never chose. Cutting
        here keeps that decision in the parser, and the audit record carries the
        text on both sides so a reviewer can see what left rather than only that
        something did.
        """
        if len(body) < MAX_BODY_CHARS:
            return body
        kept: list[str] = []
        length = 0
        for line in body.split("\n"):
            if length + len(line) + 1 >= MAX_BODY_CHARS:
                break
            kept.append(line)
            length += len(line) + 1
        if not kept:
            raise ValueError(
                f"biml: {control_id} has no line short enough to keep under "
                f"{MAX_BODY_CHARS} characters. The extraction returned one "
                f"unbroken run where the PDF wraps at the column, so the cut "
                f"has nowhere to land."
            )
        after = "\n".join(kept)
        audit.append({
            "repair": "body_capped",
            "control_id": control_id,
            "before": body,
            "after": after,
            "before_chars": len(body),
            "after_chars": len(after),
            "reason": (
                f"{control_id} extracted {len(body)} characters against a "
                f"{MAX_BODY_CHARS} limit, cut to {len(after)} on a line "
                f"boundary. The tag is the last one its block defines, so the "
                f"body ran on into matter that belongs to no risk."
            ),
        })
        return after

    @staticmethod
    def _check_targets(
        control_ids: set[str],
        alt_ids: dict[str, list[str]],
        alt_titles: dict[str, list[str]],
        required: bool,
    ) -> None:
        """Refuse declarations pointing at tags this parse did not produce.

        Raises:
            ValueError: If any declared target is absent.
        """
        if not required:
            return
        missing_ids = sorted(set(alt_ids) - control_ids)
        if missing_ids:
            raise ValueError(
                f"biml: UNPREFIXED_IDS declares alt_ids on {missing_ids}, "
                f"which this parse did not produce. Seven curated links reach "
                f"their risk only through that channel; a stale entry leaves "
                f"them resolving to nothing."
            )
        missing_titles = sorted(set(alt_titles) - control_ids)
        if missing_titles:
            # Two tables feed alt_titles, so naming only one sends a reader to
            # the wrong constant. Attribute each missing target to whichever
            # declared it.
            sources = {
                control_id: (
                    "OPENCRE_TITLE_VARIANTS"
                    if control_id in OPENCRE_TITLE_VARIANTS
                    else "NAME_CONFLICTS"
                )
                for control_id in missing_titles
            }
            raise ValueError(
                f"biml: {sources} declare alt_titles on targets this parse did "
                f"not produce. A stale entry puts a title on nothing; a missing "
                f"one puts a wrong-anchor flag on a correct anchor."
            )


def main() -> None:
    BimlParser().run()


if __name__ == "__main__":
    main()
