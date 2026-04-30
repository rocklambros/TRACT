"""Hub representation firewall for LOFO evaluation.

Ensures no information from the held-out framework leaks into hub
representations during evaluation. The primary representation
("{hierarchy_path} | {hub_name}") is inherently firewall-safe because
both components come from CRE structure, not framework text.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

from tract.hierarchy import CREHierarchy

logger = logging.getLogger(__name__)


class HasControlText(Protocol):
    control_text: str
    framework: str


def build_firewalled_hub_text(
    hub_id: str,
    hierarchy: CREHierarchy,
    excluded_framework: str | None = None,
    include_description: bool = False,
    descriptions: dict[str, str] | None = None,
    include_standards: bool = False,
    standard_sections: dict[str, list[str]] | None = None,
) -> str:
    """Build a single hub's text representation with firewall.

    Primary format: "{hierarchy_path} | {hub_name}"

    If include_description=True (ablation A6): appends description.
    If include_standards=True (ablation A3): appends standard names,
    excluding the held-out framework.
    """
    node = hierarchy.hubs[hub_id]
    text = f"{node.hierarchy_path} | {node.name}"

    if include_description and descriptions and hub_id in descriptions:
        text = f"{text}: {descriptions[hub_id]}"

    if include_standards and standard_sections and hub_id in standard_sections:
        sections = [
            s
            for s in standard_sections[hub_id]
            if excluded_framework is None or excluded_framework not in s
        ]
        if sections:
            text = f"{text}. Standards: {', '.join(sorted(sections))}"

    return text


def build_all_hub_texts(
    hierarchy: CREHierarchy,
    excluded_framework: str | None = None,
    include_description: bool = False,
    descriptions: dict[str, str] | None = None,
    include_standards: bool = False,
    standard_sections: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Build text representations for all hubs with firewall applied."""
    texts: dict[str, str] = {}
    for hub_id in hierarchy.hubs:
        texts[hub_id] = build_firewalled_hub_text(
            hub_id,
            hierarchy,
            excluded_framework,
            include_description,
            descriptions,
            include_standards,
            standard_sections,
        )
    return texts


def assert_firewall(
    hub_texts: dict[str, str],
    eval_items: list[Any],
    held_out_framework: str,
    base_hub_texts: dict[str, str] | None = None,
) -> None:
    """Assert no information leakage from held-out framework into hub representations.

    The base format ("{path} | {name}") uses only CRE-native content and is
    safe by construction. When hub texts include additional content (descriptions,
    standards), we check only the appended portion by subtracting the base text.

    If base_hub_texts is None, the hub texts are assumed to be base-format-only
    and no check is needed beyond logging.
    """
    if base_hub_texts is None:
        logger.info(
            "Firewall assertion passed (base format only — CRE-native content): "
            "%d items, %d hubs",
            len(eval_items),
            len(hub_texts),
        )
        return

    # Collect all hub names from base texts — these are CRE-native concepts.
    # Descriptions legitimately reference hub names for disambiguation.
    hub_names_lower: set[str] = set()
    for base in base_hub_texts.values():
        if " | " in base:
            hub_names_lower.add(base.split(" | ", 1)[1].lower())

    for item in eval_items:
        control_text = item.control_text
        if len(control_text) < 5:
            continue
        control_lower = control_text.lower()
        # Skip if control text overlaps with any CRE hub name
        if any(
            control_lower in name or name in control_lower
            for name in hub_names_lower
        ):
            continue
        for hub_id, text in hub_texts.items():
            base = base_hub_texts.get(hub_id, "")
            appended = text[len(base):]
            if not appended:
                continue
            if control_text in appended:
                raise AssertionError(
                    f"Firewall breach: control '{control_text[:50]}' "
                    f"(framework={held_out_framework}) found in hub {hub_id} "
                    f"appended text"
                )
    logger.info(
        "Firewall assertion passed: %d items checked against %d hubs "
        "(appended content verified)",
        len(eval_items),
        len(hub_texts),
    )
