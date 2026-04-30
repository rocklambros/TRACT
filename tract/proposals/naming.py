"""LLM-generated hub names for proposed clusters.

Opt-in via --name-with-llm flag. Without it, clusters get descriptive
placeholder names based on nearest hub.
"""
from __future__ import annotations

import logging

from tract.config import PHASE1D_PROPOSAL_NAMING_MODEL
from tract.hierarchy import CREHierarchy
from tract.proposals.guardrails import GuardrailResult

logger = logging.getLogger(__name__)


def _get_anthropic_client():
    import anthropic
    return anthropic.Anthropic()


def generate_hub_names(
    results: list[GuardrailResult],
    hierarchy: CREHierarchy,
    control_metadata: dict[str, dict],
    model: str = PHASE1D_PROPOSAL_NAMING_MODEL,
) -> dict[int, str]:
    """Call Claude API to generate hub names for passing clusters."""
    client = _get_anthropic_client()
    names: dict[int, str] = {}

    passing = [r for r in results if r.passed]
    for result in passing:
        cluster = result.cluster
        member_texts = []
        for cid in cluster.control_ids[:10]:
            meta = control_metadata.get(cid, {})
            title = meta.get("title", cid)
            desc = meta.get("description", "")
            member_texts.append(f"- {title}: {desc[:200]}")

        nearest_name = ""
        if cluster.nearest_hub_id and cluster.nearest_hub_id in hierarchy.hubs:
            nearest_name = hierarchy.hubs[cluster.nearest_hub_id].name

        parent_name = ""
        if result.parent_hub_id and result.parent_hub_id in hierarchy.hubs:
            parent_name = hierarchy.hubs[result.parent_hub_id].name

        prompt = (
            f"These security controls were clustered together as potentially needing "
            f"a new category in the CRE (Common Requirements Enumeration) taxonomy.\n\n"
            f"Member controls:\n{''.join(member_texts)}\n\n"
            f"Nearest existing hub: {nearest_name}\n"
            f"Suggested parent hub: {parent_name}\n\n"
            f"Generate a concise hub name (2-5 words) that captures the shared concept. "
            f"Reply with ONLY the hub name, nothing else."
        )

        response = client.messages.create(
            model=model,
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        name = response.content[0].text.strip()
        names[cluster.cluster_id] = name
        logger.info("Named cluster %d: %s", cluster.cluster_id, name)

    return names


def generate_placeholder_names(
    results: list[GuardrailResult],
    hub_names: dict[str, str],
) -> dict[int, str]:
    """Generate placeholder names based on nearest hub. No LLM call."""
    names: dict[int, str] = {}
    for result in results:
        if not result.passed:
            continue
        nearest = hub_names.get(result.cluster.nearest_hub_id, "Unknown")
        names[result.cluster.cluster_id] = f"Cluster {result.cluster.cluster_id} (near: {nearest})"
    return names
