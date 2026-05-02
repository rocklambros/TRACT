"""LLM-generated bridge descriptions for candidate hub pairs."""
from __future__ import annotations

import logging

import anthropic

from tract.config import BRIDGE_LLM_MODEL, BRIDGE_LLM_TEMPERATURE
from tract.hierarchy import CREHierarchy
from tract.sanitize import sanitize_text

logger = logging.getLogger(__name__)


def count_controls_for_hub(
    cre_id: str,
    hub_links: dict[str, list[dict]],
) -> int:
    """Count total controls linked to a hub across all frameworks."""
    return sum(
        1 for links in hub_links.values()
        for link in links
        if link["cre_id"] == cre_id
    )


def _build_prompt(
    candidate: dict,
    hierarchy: CREHierarchy,
    hub_links: dict[str, list[dict]],
) -> str:
    ai_id = candidate["ai_hub_id"]
    trad_id = candidate["trad_hub_id"]
    ai_name = candidate.get("ai_hub_name", ai_id)
    trad_name = candidate.get("trad_hub_name", trad_id)

    ai_path = hierarchy.hubs[ai_id].hierarchy_path if ai_id in hierarchy.hubs else ai_name
    trad_path = hierarchy.hubs[trad_id].hierarchy_path if trad_id in hierarchy.hubs else trad_name

    n_ai = count_controls_for_hub(ai_id, hub_links)
    n_trad = count_controls_for_hub(trad_id, hub_links)

    return (
        "You are a security standards expert. Describe the conceptual bridge "
        "between these two CRE hubs in 2-3 sentences.\n\n"
        f'AI Security Hub: "{ai_name}" (path: {ai_path})\n'
        f"- Linked from {n_ai} controls in AI security frameworks\n\n"
        f'Traditional Security Hub: "{trad_name}" (path: {trad_path})\n'
        f"- Linked from {n_trad} controls in traditional security frameworks\n\n"
        "Explain why these hubs address related security concerns, "
        "despite originating from different domains. Be specific about "
        "the shared concepts."
    )


def generate_bridge_descriptions(
    candidates: list[dict],
    hierarchy: CREHierarchy,
    hub_links: dict[str, list[dict]],
) -> None:
    """Add 'description' field to each candidate. Modifies in-place.

    Skips candidates that already have a non-empty description (idempotent).
    On API failure, sets description to '' (does not crash).
    """
    client = anthropic.Anthropic()
    for candidate in candidates:
        if candidate.get("description"):
            continue
        prompt = _build_prompt(candidate, hierarchy, hub_links)
        try:
            response = client.messages.create(
                model=BRIDGE_LLM_MODEL,
                temperature=BRIDGE_LLM_TEMPERATURE,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            candidate["description"] = sanitize_text(raw_text, max_length=500)
        except Exception:
            logger.warning(
                "LLM description failed for %s -> %s",
                candidate["ai_hub_id"], candidate["trad_hub_id"],
            )
            candidate["description"] = ""


def generate_negative_descriptions(
    negatives: list[dict],
    hierarchy: CREHierarchy,
    hub_links: dict[str, list[dict]],
) -> list[dict]:
    """Add descriptions to negative control candidates.

    Args:
        negatives: List from extract_negatives() — each has ai_hub_id,
            trad_hub_id, cosine_similarity, is_negative.

    Returns:
        The same list with hub names and descriptions added.
    """
    for neg in negatives:
        ai_id = neg["ai_hub_id"]
        trad_id = neg["trad_hub_id"]
        neg["ai_hub_name"] = hierarchy.hubs[ai_id].name if ai_id in hierarchy.hubs else ai_id
        neg["trad_hub_name"] = hierarchy.hubs[trad_id].name if trad_id in hierarchy.hubs else trad_id

    generate_bridge_descriptions(negatives, hierarchy, hub_links)
    return negatives
