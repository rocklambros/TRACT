"""Extract Appendix A of the OWASP LLM Top 10 2026 as a crosswalk artifact.

Appendix A is not control text. It is an expert crosswalk from the ten 2026
risks to nine external frameworks, at each target framework's coarse level,
with primary and supporting weights and a written rationale per row. It goes
to `data/processed/owasp_llm_top10_2026_mappings.json`, deliberately NOT to
`data/processed/frameworks/`, because everything in that directory is merged
into the corpus as control text and these rows are not controls.

Named `extract_` rather than `parse_` for the same reason
`parsers/extract_hub_links.py` is. `tests/test_parser_manifest_coverage.py`
scans `parse_*.py` for reads that bypass the source manifest, and its rules
are written for framework parsers reading raw sources. This module does read
its raw source through the recording reader, and it also reads two processed
artifacts to measure the CWE chain, which that scanner would flag as a
manifest bypass it is not.

Row shapes the tables actually use, all three of which have to be handled or
rows go missing:

    | LLM01 Prompt Injection | ● CWE-1427 Improper ... | rationale |
    || ○ CWE-707 Improper Neutralization (Pillar) | rationale |
    ||| rationale continued after a page break |

An empty Risk cell means "same risk as the row above". An empty Element cell
with a non-empty Relevance cell is a rationale resumed below a repeated header,
where the PDF-to-markdown conversion broke the table across pages. Dropped,
that mapping ships a rationale that stops mid-sentence.

The appendix states its own answer twice: once as a coverage matrix of
risk-by-framework marks, and once as the detail tables. Both are extracted and
compared, and a disagreement stops the run. That cross-check is the only cheap
detector of a partially extracted table there is, and it was measured to agree
on all 90 cells of the real document before it was made a gate.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Final

from tract.config import (
    PROCESSED_DIR,
    PROCESSED_FRAMEWORKS_DIR,
    TRAINING_DIR,
)
from tract.io import atomic_write_json, load_json
from tract.parsers.base import SourceReader
from parsers.parse_owasp_llm_top10_2026 import (
    ENTRY_IDS,
    SOURCE_FILE,
    SOURCE_SHA256,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAPPINGS_FILENAME: Final[str] = "owasp_llm_top10_2026_mappings.json"

APPENDIX_START: Final[re.Pattern[str]] = re.compile(r"^##\s+Appendix A\b")
APPENDIX_END: Final[re.Pattern[str]] = re.compile(r"^##\s+Appendix B\b")
SECTION_HEADING: Final[re.Pattern[str]] = re.compile(r"^##\s+(\S.*?)\s*$")
COVERAGE_MATRIX_HEADING: Final[str] = "Coverage matrix"

# "LLM01 Prompt Injection" in a Risk cell, and in a coverage-matrix row label.
RISK_CELL: Final[re.Pattern[str]] = re.compile(r"^(LLM(?:0[1-9]|10))\b")

PRIMARY_MARK: Final[str] = "●"      # filled circle
SUPPORTING_MARK: Final[str] = "○"   # hollow circle
NO_MAPPING_MARK: Final[str] = "-"
WEIGHTS: Final[dict[str, str]] = {
    PRIMARY_MARK: "primary", SUPPORTING_MARK: "supporting",
}
MARKS: Final[dict[str | None, str]] = {
    "primary": PRIMARY_MARK, "supporting": SUPPORTING_MARK,
    None: NO_MAPPING_MARK,
}


@dataclass(frozen=True)
class TargetFramework:
    """One framework the appendix maps the ten risks onto.

    Attributes:
        key: Stable slug for this artifact. Not the framework_id, because two
            targets have no corpus here and would otherwise have no key.
        heading: The section heading in the appendix, matched exactly. An
            exact match is on purpose: a heading that drifted is a section
            this extractor has not read, and skipping it silently would drop
            a whole framework's rows.
        label: The framework name without the version tail.
        version: The version the appendix pins its elements to.
        framework_id: TRACT's id for this framework, or None when the project
            holds no corpus for it.
        element_pattern: Splits an element cell into an id and a name. None
            when the framework numbers nothing, which is how NIST AI 600-1
            names its risk categories.
        element_id_template: Applied to the pattern's match to build the id.
            Exists because the conversion runs some ids together with their
            index, so "MEASURE1" has to become "MEASURE 1".
        element_level: The granularity the appendix maps at, in that
            framework's own words.
        matrix_column: This framework's column in the coverage matrix.
    """

    key: str
    heading: str
    label: str
    version: str
    framework_id: str | None
    element_pattern: re.Pattern[str] | None
    element_level: str
    matrix_column: str
    element_id_template: str = r"\g<id>"


TARGET_FRAMEWORKS: Final[tuple[TargetFramework, ...]] = (
    TargetFramework(
        key="owasp_agentic_top10",
        heading="OWASP Top 10 for Agentic Applications (ASI) -2026 "
                "(announced 2025-12-09)",
        label="OWASP Top 10 for Agentic Applications (ASI)",
        version="2026 (announced 2025-12-09)",
        framework_id="owasp_agentic_top10",
        element_pattern=re.compile(r"^(?P<id>ASI\d{2})\s*-\s*(?P<name>.+)$"),
        element_level="risk",
        matrix_column="ASI",
    ),
    TargetFramework(
        key="owasp_dsgai",
        heading="OWASP GenAI Data Security 2026 (DSGAI) -v1.0 (2026-0317)",
        label="OWASP GenAI Data Security 2026 (DSGAI)",
        # As printed in the section heading. The Framework Sources table in
        # the same appendix writes it 2026-03-17; the heading drops a hyphen.
        version="v1.0 (2026-0317)",
        framework_id="owasp_dsgai",
        element_pattern=re.compile(r"^(?P<id>DSGAI\d{2})\s*-\s*(?P<name>.+)$"),
        element_level="risk category",
        matrix_column="DSGAI",
    ),
    TargetFramework(
        key="mitre_atlas",
        heading="MITRE ATLAS -content v2026.06 (format-version 6.0.0)",
        label="MITRE ATLAS",
        version="content v2026.06 (format-version 6.0.0)",
        framework_id="mitre_atlas",
        element_pattern=re.compile(r"^(?P<id>AML\.TA\d{4})\s+(?P<name>.+)$"),
        # Tactics, while data/processed/frameworks/mitre_atlas.json carries
        # techniques (AML.T####). The two id spaces do not intersect, so these
        # rows resolve to the framework and not to any control in it.
        element_level="tactic",
        matrix_column="ATLAS",
    ),
    TargetFramework(
        key="mitre_attack",
        heading="MITRE ATT&CK -v19.1",
        label="MITRE ATT&CK",
        version="v19.1",
        framework_id=None,
        element_pattern=re.compile(r"^(?P<id>TA\d{4})\s+(?P<name>.+)$"),
        element_level="enterprise tactic",
        matrix_column="ATT&CK",
    ),
    TargetFramework(
        key="cwe",
        heading="MITRE CWE (Common Weakness Enumeration) -4.20",
        label="MITRE CWE (Common Weakness Enumeration)",
        version="4.20",
        framework_id="cwe",
        element_pattern=re.compile(r"^CWE-(?P<id>\d+)\s+(?P<name>.+)$"),
        element_id_template=r"CWE-\g<id>",
        element_level="weakness",
        matrix_column="CWE",
    ),
    TargetFramework(
        key="nist_ai_600_1",
        heading="NIST AI 600-1 (Generative AI Profile) -v1.0 (July 2024)",
        label="NIST AI 600-1 (Generative AI Profile)",
        version="v1.0 (July 2024)",
        framework_id="nist_ai_600_1",
        # This framework numbers nothing. Its elements are the risk category
        # names, which are also the titles in our corpus.
        element_pattern=None,
        element_level="risk category",
        matrix_column="600-1",
    ),
    TargetFramework(
        key="nist_ai_rmf",
        heading="NIST AI RMF (AI 100-1) -v1.0 (2023)",
        label="NIST AI RMF (AI 100-1)",
        version="v1.0 (2023)",
        framework_id="nist_ai_rmf",
        element_pattern=re.compile(
            r"^(?P<fn>GOVERN|MAP|MEASURE|MANAGE)\s*(?P<num>\d+)"
            r"\s*\((?P<name>.+)\)$"
        ),
        element_id_template=r"\g<fn> \g<num>",
        # Categories, while our corpus carries subcategories (GOVERN 1.1).
        element_level="category",
        matrix_column="RMF",
    ),
    TargetFramework(
        key="csa_aicm",
        heading="CSA AI Controls Matrix (AICM) -v1.1 (2026-06-22)",
        label="CSA AI Controls Matrix (AICM)",
        version="v1.1 (2026-06-22)",
        framework_id="csa_aicm",
        # The conversion runs some domain codes into their names, so
        # "STASupply Chain Management" is one token in the source.
        element_pattern=re.compile(r"^(?P<id>[A-Z][A-Z&]{2})\s*(?P<name>.+)$"),
        # Control domains, while our corpus carries controls (AIS-01).
        element_level="control domain",
        matrix_column="AICM",
    ),
    TargetFramework(
        key="owasp_aivss",
        heading="OWASP AIVSS (AI Vulnerability Scoring System) -v0.8",
        label="OWASP AIVSS (AI Vulnerability Scoring System)",
        version="v0.8",
        framework_id=None,
        element_pattern=re.compile(r"^(?P<id>AIVSS-\d+)\s+(?P<name>.+)$"),
        element_level="agentic core security risk",
        matrix_column="AIVSS",
    ),
)

# Measured against the pinned source, exactly, and checked in both directions.
# A ceiling would only catch an extractor that runs away. The failure that
# ships a shorter crosswalk quietly is the opposite one: the conversion moves a
# table, the row pattern stops reaching it, and the artifact is short with
# every gate green. Moving a number here means re-measuring against the source.
EXPECTED_MAPPING_COUNTS: Final[dict[str, int]] = {
    "owasp_agentic_top10": 31,
    "owasp_dsgai": 39,
    "mitre_atlas": 46,
    "mitre_attack": 29,
    "cwe": 48,
    "nist_ai_600_1": 39,
    "nist_ai_rmf": 25,
    "csa_aicm": 44,
    "owasp_aivss": 30,
}

_CWE_CORPUS: Final[Path] = PROCESSED_FRAMEWORKS_DIR / "cwe.json"
_CURATED_LINKS: Final[Path] = TRAINING_DIR / "hub_links_curated.jsonl"


def _link_record(line: str) -> dict[str, Any]:
    """Parse one hub-link JSONL record.

    Raises:
        ValueError: If the line is not a JSON object.
    """
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(
            f"expected a JSON object per line in {_CURATED_LINKS.name}, got "
            f"{type(value).__name__}"
        )
    return value


class Owasp2026AppendixExtractor(SourceReader):
    """Reads Appendix A and writes the crosswalk artifact."""

    framework_id: ClassVar[str] = "owasp_llm_top10_2026"
    source_sha256: ClassVar[str] = SOURCE_SHA256
    targets: ClassVar[tuple[TargetFramework, ...]] = TARGET_FRAMEWORKS
    expected_mapping_counts: ClassVar[dict[str, int]] = EXPECTED_MAPPING_COUNTS

    def __init__(
        self,
        raw_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        super().__init__(raw_dir)
        self.output_dir = output_dir or PROCESSED_DIR

    def extract(self) -> dict[str, Any]:
        """Build the crosswalk record. Raises rather than returning partials."""
        text = self.read_source(SOURCE_FILE)
        self._check_digest()

        appendix = self._appendix_lines(text.splitlines())
        sections = self._sections(appendix)

        mappings: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for target in self.targets:
            rows = self._section_rows(target, sections)
            self._check_count(target, len(rows))
            mappings.extend(rows)
            summaries.append({
                "key": target.key,
                "label": target.label,
                "version": target.version,
                "framework_id": target.framework_id,
                "element_level": target.element_level,
                "mapping_count": len(rows),
                "distinct_elements": len(
                    {row["target_element_name"] for row in rows}
                ),
            })

        cells = self._check_coverage_matrix(sections, mappings)

        record: dict[str, Any] = {
            "source_framework_id": self.framework_id,
            "source_file": SOURCE_FILE,
            "source_sha256": self.recorded_sha256(SOURCE_FILE),
            "mapping_count": len(mappings),
            "coverage_matrix_cells_checked": cells,
            "cwe_chain": self._cwe_chain(mappings),
            "target_frameworks": summaries,
            "mappings": mappings,
        }
        logger.info(
            "%s: %d mappings across %d target frameworks, %d coverage-matrix "
            "cells agreed with the detail tables",
            self.framework_id, len(mappings), len(summaries), cells,
        )
        return record

    def run(self) -> Path:
        """Write the artifact and return its path."""
        path = self.output_dir / MAPPINGS_FILENAME
        atomic_write_json(self.extract(), path)
        logger.info("%s: wrote %s", self.framework_id, path)
        return path

    def _check_digest(self) -> None:
        """Refuse a source that is not the bytes this extractor was built for.

        Raises:
            ValueError: If the read file's sha256 is not the declared pin.
        """
        actual = self.recorded_sha256(SOURCE_FILE)
        if actual == self.source_sha256:
            return
        raise ValueError(
            f"{self.framework_id}: {SOURCE_FILE} has sha256 {actual}, not the "
            f"pinned {self.source_sha256}. Re-measure the appendix against the "
            f"new bytes before moving the pin, because every expected mapping "
            f"count below was measured against the old ones."
        )

    def _appendix_lines(self, lines: list[str]) -> list[str]:
        """The slice between Appendix A and Appendix B.

        Raises:
            ValueError: If either boundary heading is missing.
        """
        start = next(
            (i for i, line in enumerate(lines) if APPENDIX_START.match(line)),
            None,
        )
        end = next(
            (i for i, line in enumerate(lines) if APPENDIX_END.match(line)),
            None,
        )
        if start is None or end is None or end <= start:
            raise ValueError(
                f"{self.framework_id}: could not locate Appendix A "
                f"(line {start}) and Appendix B (line {end}) in that order in "
                f"{SOURCE_FILE}. Without both, the crosswalk would either be "
                f"empty or run on into the reference list."
            )
        return lines[start:end]

    @staticmethod
    def _sections(appendix: list[str]) -> dict[str, list[str]]:
        """Appendix subsections keyed by heading, in source order."""
        sections: dict[str, list[str]] = {}
        current: str | None = None
        for line in appendix:
            match = SECTION_HEADING.match(line)
            if match:
                current = match.group(1)
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(line)
        return sections

    @staticmethod
    def _cells(line: str) -> list[str] | None:
        """Split one markdown table row, or None when the line is not one.

        Strips exactly one leading and one trailing pipe. Stripping every
        leading pipe collapses the "||" continuation shape to two cells and
        loses the empty Risk cell that carries the "same risk as above"
        meaning, which silently dropped 15 of ASI's 31 rows.
        """
        if not line.startswith("|"):
            return None
        body = line[1:]
        if body.endswith("|"):
            body = body[:-1]
        return [cell.strip() for cell in body.split("|")]

    @staticmethod
    def _is_separator(cells: list[str]) -> bool:
        """A markdown ruler row, "|---|---|---|"."""
        return all(set(cell) <= set("-: ") for cell in cells)

    def _section_rows(
        self, target: TargetFramework, sections: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        """Every mapping row in one framework's section.

        Raises:
            ValueError: If the section is missing, empty, or carries a row this
                extractor cannot read.
        """
        if target.heading not in sections:
            raise ValueError(
                f"{self.framework_id}: no appendix section headed "
                f"{target.heading!r}. Either the heading changed or the "
                f"section was removed. Skipping it would drop every mapping "
                f"to {target.label}."
            )

        rows: list[dict[str, Any]] = []
        risk: str | None = None
        for line in sections[target.heading]:
            cells = self._cells(line)
            if cells is None:
                continue
            if len(cells) != 3:
                raise ValueError(
                    f"{self.framework_id}: {target.key} row has "
                    f"{len(cells)} cells, expected 3: {line!r}"
                )
            risk_cell, element_cell, relevance = cells
            if risk_cell == "Risk" and element_cell == "Element":
                continue
            if self._is_separator(cells):
                continue

            if risk_cell:
                match = RISK_CELL.match(risk_cell)
                if match is None:
                    raise ValueError(
                        f"{self.framework_id}: {target.key} row has an "
                        f"unreadable Risk cell {risk_cell!r}. A row whose risk "
                        f"cannot be resolved would be attributed to whichever "
                        f"risk came before it."
                    )
                risk = f"{match.group(1)}:2026"

            if element_cell:
                rows.append(
                    self._mapping(target, risk, element_cell, relevance)
                )
                continue

            # Empty Element with text in Relevance: a rationale resumed below a
            # repeated header where the table broke across a page.
            if relevance:
                if not rows:
                    raise ValueError(
                        f"{self.framework_id}: {target.key} opens with a "
                        f"continuation row that has nothing to continue: "
                        f"{line!r}"
                    )
                rows[-1]["rationale"] = (
                    f"{rows[-1]['rationale']} {relevance}".strip()
                )
        return rows

    def _mapping(
        self,
        target: TargetFramework,
        risk: str | None,
        element_cell: str,
        relevance: str,
    ) -> dict[str, Any]:
        """One mapping record from one Element cell.

        Raises:
            ValueError: If the row has no risk, no weight marker, or an element
                the target's pattern cannot read.
        """
        if risk is None:
            raise ValueError(
                f"{self.framework_id}: {target.key} has a mapping before any "
                f"risk was named: {element_cell!r}"
            )
        weight = WEIGHTS.get(element_cell[0])
        if weight is None:
            raise ValueError(
                f"{self.framework_id}: {target.key} element {element_cell!r} "
                f"carries no primary or supporting marker. The weight is the "
                f"crosswalk's own confidence and must not be guessed."
            )
        element = element_cell[1:].strip()

        element_id: str | None = None
        element_name = element
        if target.element_pattern is not None:
            match = target.element_pattern.match(element)
            if match is None:
                raise ValueError(
                    f"{self.framework_id}: {target.key} element {element!r} "
                    f"does not match its declared id pattern "
                    f"{target.element_pattern.pattern!r}. Emitting it without "
                    f"an id would put an unjoinable row in the crosswalk."
                )
            element_id = match.expand(target.element_id_template)
            element_name = match.group("name").strip()

        return {
            "source_control_id": risk,
            "target_framework": target.key,
            "target_framework_id": target.framework_id,
            "target_element_id": element_id,
            "target_element_name": element_name,
            "weight": weight,
            "rationale": relevance.strip(),
        }

    def _check_count(self, target: TargetFramework, actual: int) -> None:
        """Compare one section's row count against its measured value.

        Raises:
            ValueError: If the target has no measured count, or the count moved.
        """
        expected = self.expected_mapping_counts.get(target.key)
        if expected is None:
            raise ValueError(
                f"{self.framework_id}: target {target.key!r} has no measured "
                f"row count in expected_mapping_counts, so its section runs "
                f"ungated. Measure it against the pinned source and declare "
                f"the number."
            )
        if actual == expected:
            logger.info(
                "%s: %s contributed %d mappings (measured %d)",
                self.framework_id, target.key, actual, expected,
            )
            return
        raise ValueError(
            f"{self.framework_id}: target {target.key!r} yielded {actual} "
            f"mappings against a measured {expected}. Fewer means rows stopped "
            f"being reached and the crosswalk ships short. More means the row "
            f"pattern is picking up something that is not a mapping. Re-measure "
            f"before moving the number."
        )

    def _check_coverage_matrix(
        self, sections: dict[str, list[str]], mappings: list[dict[str, Any]],
    ) -> int:
        """Compare the appendix's own summary against its detail tables.

        Returns the number of cells checked.

        Raises:
            ValueError: If the matrix is missing, its columns do not match the
                declared targets, or any cell disagrees.
        """
        if COVERAGE_MATRIX_HEADING not in sections:
            raise ValueError(
                f"{self.framework_id}: no {COVERAGE_MATRIX_HEADING!r} section. "
                f"That table is the appendix's own restatement of every "
                f"mapping and the only cheap check that no row was dropped."
            )

        columns: list[str] | None = None
        matrix: dict[str, dict[str, str]] = {}
        for line in sections[COVERAGE_MATRIX_HEADING]:
            cells = self._cells(line)
            if cells is None or self._is_separator(cells):
                continue
            if cells[0] == "Risk":
                if columns is not None and cells[1:] != columns:
                    raise ValueError(
                        f"{self.framework_id}: the coverage matrix continues "
                        f"with columns {cells[1:]} after starting with "
                        f"{columns}. The two halves describe different things."
                    )
                columns = cells[1:]
                continue
            match = RISK_CELL.match(cells[0])
            if match is None or columns is None:
                continue
            matrix[f"{match.group(1)}:2026"] = dict(zip(columns, cells[1:]))

        declared = [target.matrix_column for target in self.targets]
        if columns != declared:
            raise ValueError(
                f"{self.framework_id}: the coverage matrix has columns "
                f"{columns}, and the declared targets are {declared}. A column "
                f"with no target is a framework whose rows are not being "
                f"extracted at all."
            )
        if sorted(matrix) != sorted(ENTRY_IDS):
            raise ValueError(
                f"{self.framework_id}: the coverage matrix covers "
                f"{sorted(matrix)}, expected {list(ENTRY_IDS)}."
            )

        strongest: dict[tuple[str, str], str] = {}
        for mapping in mappings:
            key = (mapping["source_control_id"], mapping["target_framework"])
            if strongest.get(key) != "primary":
                strongest[key] = mapping["weight"]

        disagreements: list[str] = []
        for target in self.targets:
            for control_id in ENTRY_IDS:
                expected = MARKS[strongest.get((control_id, target.key))]
                found = matrix[control_id][target.matrix_column]
                if found != expected:
                    disagreements.append(
                        f"{control_id}/{target.matrix_column}: matrix says "
                        f"{found!r}, tables say {expected!r}"
                    )
        if disagreements:
            raise ValueError(
                f"{self.framework_id}: the coverage matrix and the detail "
                f"tables disagree on {len(disagreements)} cell(s): "
                f"{'; '.join(disagreements)}. Either a detail row was dropped "
                f"or the source contradicts itself. Both need reading before "
                f"this artifact is trusted."
            )
        return len(self.targets) * len(ENTRY_IDS)

    def _cwe_chain(self, mappings: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Measure how far the CWE mappings reach into the CRE hub graph.

        The chain is risk -> CWE -> OpenCRE hub, and it is what makes this
        appendix more than documentation: it is the only route by which the
        2026 risks touch the hub space without any 2026 link existing.

        Recorded, not gated. `data/training/hub_links_curated.jsonl` is
        rebuilt by the pending OpenCRE re-fetch, so a declared expectation here
        would be a gate on a moving input.

        Returns None when either input is absent from this checkout.
        """
        pairs = [
            (m["source_control_id"], str(m["target_element_id"]))
            for m in mappings
            if m["target_framework"] == "cwe" and m["target_element_id"]
        ]
        if not pairs:
            logger.info("%s: no CWE mappings, chain not measured",
                        self.framework_id)
            return None
        if not _CWE_CORPUS.is_file() or not _CURATED_LINKS.is_file():
            logger.warning(
                "%s: CWE chain not measured, missing %s or %s",
                self.framework_id, _CWE_CORPUS, _CURATED_LINKS,
            )
            return None

        corpus = load_json(_CWE_CORPUS)
        # Corpus control ids are the bare numbers; the appendix writes CWE-1427.
        known = {str(control["control_id"]) for control in corpus["controls"]}

        hubs_by_cwe: dict[str, set[str]] = {}
        for line in _CURATED_LINKS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            link = _link_record(line)
            if link.get("framework_id") != "cwe":
                continue
            section = str(link.get("section_id", "")).strip()
            hubs_by_cwe.setdefault(section, set()).add(str(link["cre_id"]))

        distinct = {cwe for _, cwe in pairs}
        resolved = {c for c in distinct if c.removeprefix("CWE-") in known}
        linked = {c for c in distinct if hubs_by_cwe.get(c.removeprefix("CWE-"))}
        triples = {
            (risk, cwe, hub)
            for risk, cwe in pairs
            for hub in hubs_by_cwe.get(cwe.removeprefix("CWE-"), set())
        }
        return {
            "cwe_corpus_version": str(corpus.get("version", "")),
            "mappings": len(pairs),
            "distinct_cwes": len(distinct),
            "cwes_in_corpus": len(resolved),
            "cwes_with_opencre_links": len(linked),
            "mappings_that_chain": sum(
                1 for _, cwe in pairs
                if hubs_by_cwe.get(cwe.removeprefix("CWE-"))
            ),
            "risk_cwe_hub_triples": len(triples),
            "distinct_hubs": len({hub for _, _, hub in triples}),
            "risks_reaching_a_hub": len({risk for risk, _, _ in triples}),
        }


def main() -> None:
    Owasp2026AppendixExtractor().run()


if __name__ == "__main__":
    main()
