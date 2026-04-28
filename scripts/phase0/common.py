"""Phase 0 shared utilities — data loading, hierarchy, evaluation corpus.

Provides core infrastructure used by all four Phase 0 experiments:
- CRE hierarchy tree construction
- Hub link extraction and normalization
- Evaluation corpus builder with full-text / title-only tracks

LOFO cross-validation, scoring metrics, and bootstrap CIs are added
in a subsequent task.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ───────────────────────────────────────────────────────────────────

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
OPENCRE_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "opencre" / "opencre_all_cres.json"
HUB_LINKS_PATH: Final[Path] = PROJECT_ROOT / "data" / "training" / "hub_links.jsonl"
PROCESSED_FRAMEWORKS_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "frameworks"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results" / "phase0"

# ── AI Framework Names (as they appear in OpenCRE) ─────────────────────────

AI_FRAMEWORK_NAMES: Final[frozenset[str]] = frozenset({
    "MITRE ATLAS",
    "OWASP AI Exchange",
    "NIST AI 100-2",
    "OWASP Top10 for LLM",
    "OWASP Top10 for ML",
})

AI_FRAMEWORK_ID_MAP: Final[dict[str, str]] = {
    "MITRE ATLAS": "mitre_atlas",
    "OWASP AI Exchange": "owasp_ai_exchange",
    "NIST AI 100-2": "nist_ai_100_2",
    "OWASP Top10 for LLM": "owasp_llm_top10",
    "OWASP Top10 for ML": "owasp_ml_top10",
}

PARSED_FRAMEWORK_IDS: Final[frozenset[str]] = frozenset({
    "mitre_atlas",
    "owasp_ai_exchange",
    "owasp_llm_top10",
})

# ── Bootstrap Settings ──────────────────────────────────────────────────────

BOOTSTRAP_N_RESAMPLES: Final[int] = 10_000
BOOTSTRAP_CI_LEVEL: Final[float] = 0.95
BOOTSTRAP_SEED: Final[int] = 42

# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class CREHierarchy:
    """In-memory CRE hub tree with parent/child/depth lookups."""

    hubs: dict[str, str] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    parent: dict[str, str] = field(default_factory=dict)
    children: dict[str, list[str]] = field(default_factory=dict)
    depth: dict[str, int] = field(default_factory=dict)
    branch: dict[str, str] = field(default_factory=dict)

    def hierarchy_path(self, hub_id: str) -> str:
        """Build 'Root > Parent > ... > Hub' path string."""
        if hub_id not in self.hubs:
            raise ValueError(f"Unknown hub ID: {hub_id}")
        parts: list[str] = []
        current: str | None = hub_id
        while current:
            parts.append(self.hubs[current])
            current = self.parent.get(current)
        return " > ".join(reversed(parts))

    def leaf_hub_ids(self) -> list[str]:
        """Return hub IDs that have no children."""
        return [hid for hid in self.hubs if hid not in self.children or not self.children[hid]]

    def branch_hub_ids(self, root_id: str) -> list[str]:
        """Return all hub IDs under a given root (including the root)."""
        if root_id not in self.hubs:
            raise ValueError(f"Unknown root ID: {root_id}")
        result: list[str] = []
        queue: deque[str] = deque([root_id])
        while queue:
            current = queue.popleft()
            result.append(current)
            queue.extend(self.children.get(current, []))
        return result


@dataclass
class HubStandardLink:
    """One link between a CRE hub and a standard section."""

    cre_id: str
    cre_name: str
    standard_name: str
    section_id: str
    section_name: str


@dataclass
class EvalItem:
    """One evaluation data point: a control mapped to a ground-truth hub."""

    control_text: str
    ground_truth_hub_id: str
    ground_truth_hub_name: str
    framework_name: str
    section_id: str
    track: str  # "full-text" or "all"


# ── Hierarchy Builder ───────────────────────────────────────────────────────

TRAINING_LINK_TYPES: Final[frozenset[str]] = frozenset({
    "linked to", "automatically linked to",
})


def build_hierarchy(cres: list[dict]) -> CREHierarchy:
    """Build CRE hierarchy tree from raw OpenCRE CRE list."""
    tree = CREHierarchy()

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        tree.hubs[cre["id"]] = cre["name"]

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        cre_id = cre["id"]
        for link in cre.get("links", []):
            if link.get("ltype") == "Contains" and link["document"].get("doctype") == "CRE":
                child_id = link["document"]["id"]
                if child_id in tree.hubs:
                    tree.children.setdefault(cre_id, []).append(child_id)
                    tree.parent[child_id] = cre_id

    contained_ids = set(tree.parent.keys())
    tree.roots = sorted(hid for hid in tree.hubs if hid not in contained_ids)

    queue: deque[str] = deque()
    for root_id in tree.roots:
        tree.depth[root_id] = 0
        tree.branch[root_id] = root_id
        queue.append(root_id)

    while queue:
        current = queue.popleft()
        for child_id in tree.children.get(current, []):
            if child_id not in tree.depth:
                tree.depth[child_id] = tree.depth[current] + 1
                tree.branch[child_id] = tree.branch[current]
                queue.append(child_id)

    logger.info(
        "Built hierarchy: %d hubs, %d roots, %d leaves",
        len(tree.hubs), len(tree.roots), len(tree.leaf_hub_ids()),
    )
    return tree


# ── Link Extraction ─────────────────────────────────────────────────────────


def extract_hub_standard_links(cres: list[dict]) -> list[HubStandardLink]:
    """Extract all standard-to-hub links from CRE data."""
    links: list[HubStandardLink] = []

    for cre in cres:
        if cre.get("doctype") != "CRE":
            continue
        cre_id = cre["id"]
        cre_name = cre["name"]

        for link in cre.get("links", []):
            doc = link.get("document", {})
            if doc.get("doctype") != "Standard":
                continue

            ltype = link.get("ltype", "")
            if ltype.lower() not in TRAINING_LINK_TYPES:
                continue

            standard_name = doc.get("name", "")
            if not standard_name:
                continue

            section_id = doc.get("sectionID") or doc.get("section", "")
            section_name = doc.get("section") or doc.get("sectionID", "")

            links.append(HubStandardLink(
                cre_id=cre_id,
                cre_name=cre_name,
                standard_name=standard_name,
                section_id=str(section_id) if section_id else "",
                section_name=str(section_name) if section_name else "",
            ))

    logger.info("Extracted %d standard-to-hub links", len(links))
    return links


# ── Evaluation Corpus ───────────────────────────────────────────────────────


def normalize_owasp_aie_id(s: str) -> str:
    """Normalize OWASP AI Exchange IDs for matching."""
    return s.lower().replace("_", "").replace(" ", "").replace("-", "").replace("/", "")


def load_parsed_controls() -> dict[tuple[str, str], str]:
    """Load parsed control descriptions from processed frameworks.

    Returns dict mapping (standard_name, control_id) -> full description text.
    Uses normalized IDs for OWASP AI Exchange matching.
    """
    parsed: dict[tuple[str, str], str] = {}

    framework_to_standard: dict[str, str] = {
        "mitre_atlas": "MITRE ATLAS",
        "owasp_ai_exchange": "OWASP AI Exchange",
        "owasp_llm_top10": "OWASP Top10 for LLM",
    }

    for fw_id in PARSED_FRAMEWORK_IDS:
        path = PROCESSED_FRAMEWORKS_DIR / f"{fw_id}.json"
        if not path.exists():
            logger.warning("Parsed framework not found: %s", path)
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        standard_name = framework_to_standard[fw_id]
        for ctrl in data.get("controls", []):
            cid = ctrl["control_id"]
            desc = ctrl.get("full_text") or ctrl["description"]
            parsed[(standard_name, cid)] = desc

    logger.info("Loaded %d parsed control descriptions", len(parsed))
    return parsed


def build_evaluation_corpus(
    links: list[HubStandardLink],
    ai_framework_names: set[str] | frozenset[str],
    parsed_controls: dict[tuple[str, str], str],
) -> list[EvalItem]:
    """Build evaluation corpus from AI framework hub links.

    For each AI framework link, uses parsed full-text description if available,
    otherwise falls back to section_name (title-only).
    """
    corpus: list[EvalItem] = []

    for link in links:
        if link.standard_name not in ai_framework_names:
            continue

        control_text: str | None = None
        track = "all"

        key_direct = (link.standard_name, link.section_id)
        if key_direct in parsed_controls:
            control_text = parsed_controls[key_direct]
            track = "full-text"
        elif link.standard_name == "OWASP AI Exchange":
            norm_link = normalize_owasp_aie_id(link.section_id or link.section_name)
            for (std, cid), desc in parsed_controls.items():
                if std == "OWASP AI Exchange" and normalize_owasp_aie_id(cid) == norm_link:
                    control_text = desc
                    track = "full-text"
                    break

        if control_text is None:
            control_text = link.section_name or link.section_id
            track = "all"

        corpus.append(EvalItem(
            control_text=control_text,
            ground_truth_hub_id=link.cre_id,
            ground_truth_hub_name=link.cre_name,
            framework_name=link.standard_name,
            section_id=link.section_id,
            track=track,
        ))

    logger.info(
        "Built evaluation corpus: %d items (%d full-text, %d title-only)",
        len(corpus),
        sum(1 for e in corpus if e.track == "full-text"),
        sum(1 for e in corpus if e.track == "all"),
    )
    return corpus


# ── I/O Helpers ─────────────────────────────────────────────────────────────


def load_opencre_cres(path: Path | None = None) -> list[dict]:
    """Load CRE list from OpenCRE JSON dump."""
    p = path or OPENCRE_PATH
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data["cres"]
