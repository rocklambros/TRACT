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
import math
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np

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


# ── Hub Text Builder ────────────────────────────────────────────────────────


def build_hub_texts(
    tree: CREHierarchy,
    links: list[HubStandardLink],
    held_out_framework: str | None = None,
    template: str = "default",
    descriptions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build text representation for each hub.

    Templates:
      - "default": "{hub_name}: {linked standard names}"
      - "path": "{hierarchy_path} | {hub_name}: {linked standard names}"
      - "description": "{hub_name}: {description}. Linked: {linked standard names}"

    The held_out_framework parameter enables LOFO firewall — links from that
    framework are excluded from hub text construction.
    """
    hub_standards: dict[str, list[str]] = defaultdict(list)
    for link in links:
        if held_out_framework and link.standard_name == held_out_framework:
            continue
        hub_standards[link.cre_id].append(link.section_name or link.section_id)

    texts: dict[str, str] = {}
    for hub_id, hub_name in tree.hubs.items():
        standard_names = hub_standards.get(hub_id, [])
        linked_text = ", ".join(sorted(set(standard_names))) if standard_names else ""

        if template == "path":
            path = tree.hierarchy_path(hub_id)
            texts[hub_id] = f"{path} | {hub_name}: {linked_text}" if linked_text else f"{path} | {hub_name}"
        elif template == "description" and descriptions and hub_id in descriptions:
            desc = descriptions[hub_id]
            texts[hub_id] = f"{hub_name}: {desc}. Linked: {linked_text}" if linked_text else f"{hub_name}: {desc}"
        else:
            texts[hub_id] = f"{hub_name}: {linked_text}" if linked_text else hub_name

    return texts


# ── Scoring Metrics ─────────────────────────────────────────────────────────


def _reciprocal_rank(predicted: list[str], truth: str) -> float:
    """Reciprocal rank of truth in predicted list. 0 if not found."""
    for i, p in enumerate(predicted):
        if p == truth:
            return 1.0 / (i + 1)
    return 0.0


def _ndcg_at_k(predicted: list[str], truth: str, k: int = 10) -> float:
    """NDCG@k for single-relevant-item retrieval."""
    dcg = 0.0
    for i, p in enumerate(predicted[:k]):
        if p == truth:
            dcg = 1.0 / math.log2(i + 2)
            break
    idcg = 1.0 / math.log2(2)
    return dcg / idcg


def score_predictions(
    predictions: list[list[str]],
    ground_truth: list[str],
) -> dict[str, float]:
    """Score ranked predictions against ground truth hub IDs.

    Args:
        predictions: List of ranked hub ID lists (one per eval item).
        ground_truth: List of correct hub IDs (one per eval item).

    Returns:
        Dict with hit_at_1, hit_at_5, mrr, ndcg_at_10.
    """
    n = len(predictions)
    if n == 0:
        raise ValueError("Empty predictions list")
    if n != len(ground_truth):
        raise ValueError(f"Prediction count {n} != ground truth count {len(ground_truth)}")

    hit1 = sum(1 for pred, gt in zip(predictions, ground_truth) if pred and pred[0] == gt) / n
    hit5 = sum(1 for pred, gt in zip(predictions, ground_truth) if gt in pred[:5]) / n
    mrr = sum(_reciprocal_rank(pred, gt) for pred, gt in zip(predictions, ground_truth)) / n
    ndcg = sum(_ndcg_at_k(pred, gt) for pred, gt in zip(predictions, ground_truth)) / n

    return {
        "hit_at_1": hit1,
        "hit_at_5": hit5,
        "mrr": mrr,
        "ndcg_at_10": ndcg,
    }


# ── Bootstrap Confidence Intervals ──────────────────────────────────────────


def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for the mean of values."""
    rng = np.random.default_rng(seed)
    n = len(values)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[indices].mean(axis=1)

    alpha = (1 - ci_level) / 2
    ci_low = float(np.percentile(boot_means, 100 * alpha))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha)))

    return {
        "mean": float(values.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def bootstrap_paired_delta(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci_level: float = BOOTSTRAP_CI_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    """Compute bootstrap CI for the difference (B - A) on paired samples."""
    if len(values_a) != len(values_b):
        raise ValueError("Arrays must have same length for paired bootstrap")

    rng = np.random.default_rng(seed)
    n = len(values_a)
    indices = rng.integers(0, n, size=(n_resamples, n))

    boot_a = values_a[indices].mean(axis=1)
    boot_b = values_b[indices].mean(axis=1)
    boot_deltas = boot_b - boot_a

    alpha = (1 - ci_level) / 2
    return {
        "delta_mean": float((values_b - values_a).mean()),
        "ci_low": float(np.percentile(boot_deltas, 100 * alpha)),
        "ci_high": float(np.percentile(boot_deltas, 100 * (1 - alpha))),
    }


# ── LOFO Cross-Validation ──────────────────────────────────────────────────


@dataclass
class LOFOFold:
    """One fold of LOFO cross-validation."""

    held_out_framework: str
    eval_items: list[EvalItem]
    hub_texts: dict[str, str]
    hub_ids: list[str]


def build_lofo_folds(
    tree: CREHierarchy,
    links: list[HubStandardLink],
    corpus: list[EvalItem],
    ai_framework_names: set[str] | frozenset[str],
    template: str = "default",
    descriptions: dict[str, str] | None = None,
) -> list[LOFOFold]:
    """Build LOFO folds — one per AI framework.

    For each fold:
    1. Hold out one AI framework
    2. Rebuild hub texts excluding that framework's links
    3. Collect eval items for the held-out framework
    """
    folds: list[LOFOFold] = []
    all_hub_ids = sorted(tree.hubs.keys())

    for framework_name in sorted(ai_framework_names):
        eval_items = [e for e in corpus if e.framework_name == framework_name]
        if not eval_items:
            logger.warning("No eval items for %s, skipping fold", framework_name)
            continue

        hub_texts = build_hub_texts(
            tree, links,
            held_out_framework=framework_name,
            template=template,
            descriptions=descriptions,
        )

        folds.append(LOFOFold(
            held_out_framework=framework_name,
            eval_items=eval_items,
            hub_texts=hub_texts,
            hub_ids=all_hub_ids,
        ))

        logger.info(
            "LOFO fold %s: %d eval items, %d hubs",
            framework_name, len(eval_items), len(hub_texts),
        )

    return folds


def aggregate_lofo_metrics(
    fold_results: list[dict],
    folds: list[LOFOFold],
    track_filter: str | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate LOFO fold predictions into metrics with bootstrap CIs.

    Args:
        fold_results: List of dicts, one per fold. Each maps
            eval item index (int) -> ranked hub ID predictions (list[str]).
        folds: The LOFO folds (for ground truth and track info).
        track_filter: If "full-text", only include full-text items.
            None means include all items (the all-198 track).

    Returns:
        Dict with hit_at_1, hit_at_5, mrr, ndcg_at_10, each containing
        mean, ci_low, ci_high from bootstrap.
    """
    all_predictions: list[list[str]] = []
    all_ground_truth: list[str] = []

    for fold, results in zip(folds, fold_results):
        for i, item in enumerate(fold.eval_items):
            if track_filter == "full-text" and item.track != "full-text":
                continue
            all_predictions.append(results.get(i, []))
            all_ground_truth.append(item.ground_truth_hub_id)

    if not all_predictions:
        return {
            m: {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            for m in ["hit_at_1", "hit_at_5", "mrr", "ndcg_at_10"]
        }

    hit1_arr = np.array([
        1.0 if pred and pred[0] == gt else 0.0
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])
    hit5_arr = np.array([
        1.0 if gt in pred[:5] else 0.0
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])
    mrr_arr = np.array([
        _reciprocal_rank(pred, gt)
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])
    ndcg_arr = np.array([
        _ndcg_at_k(pred, gt)
        for pred, gt in zip(all_predictions, all_ground_truth)
    ])

    return {
        "hit_at_1": bootstrap_ci(hit1_arr),
        "hit_at_5": bootstrap_ci(hit5_arr),
        "mrr": bootstrap_ci(mrr_arr),
        "ndcg_at_10": bootstrap_ci(ndcg_arr),
    }


# ── Results I/O ─────────────────────────────────────────────────────────────


def save_results(results: dict, filename: str) -> Path:
    """Save results dict to results/phase0/ as formatted JSON (atomic write)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename

    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(results, fh, sort_keys=True, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.info("Saved results to %s", path)
    return path
