"""Hub representation firewall for LOFO evaluation.

Ensures no information from the held-out framework leaks into hub
representations during evaluation. The primary representation
("{hierarchy_path} | {hub_name}") is inherently firewall-safe because
both components come from CRE structure, not framework text.
"""
from __future__ import annotations

import logging
from collections.abc import Collection
from typing import Any, Protocol

from tract.hierarchy import CREHierarchy

logger = logging.getLogger(__name__)


class HasControlText(Protocol):
    control_text: str
    framework: str


def excluded_framework_set(
    excluded: str | Collection[str] | None,
) -> frozenset[str]:
    """Re-exported from tract.training.data so this module has no import cycle.

    See the definition there for why equality was replaced by set membership.
    """
    from tract.training.data import excluded_framework_set as _impl

    return _impl(excluded)


def assert_exclusion_fired(
    pairs: list[Any],
    tiered_links: list[Any],
    excluded: Collection[str],
) -> None:
    """Assert the firewall actually removed the frameworks it named.

    Two failure modes, both silent, both producing a fold record shaped exactly
    like an honest one.

    **A name that matches nothing.** A typo, a roster that drifted from the
    corpus spelling ("OWASP LLM Top 10" for "OWASP Top10 for LLM"), or a set
    passed where a string was compared. Nothing is removed, everything trains,
    hit@1 lands around 0.9, and no downstream instrument objects.

    **A framework that survived the filter anyway.** Cheaper to check directly
    than to reason about: no surviving training pair may name an excluded
    framework.

    Raises rather than logs. A firewall that did not fire has invalidated the
    run, and continuing produces a number whose only defect is being far too
    good.
    """
    excluded_set = excluded_framework_set(excluded)
    if not excluded_set:
        return

    survived = sorted({p.framework for p in pairs} & excluded_set)
    if survived:
        raise AssertionError(
            f"Firewall breach: training pairs still carry excluded "
            f"framework(s) {survived}. The exclusion set was "
            f"{sorted(excluded_set)}."
        )

    n_matching = sum(
        1 for t in tiered_links
        if t.link.get("standard_name", "") in excluded_set
    )
    if n_matching == 0:
        raise AssertionError(
            f"Firewall exclusion matched no links at all. Holding out "
            f"{sorted(excluded_set)} removed 0 of {len(tiered_links)} links, "
            "so the run trained on everything it claimed to firewall. The "
            "usual cause is a framework name that does not match the corpus "
            "spelling of `standard_name`."
        )
    logger.info(
        "Firewall exclusion verified: %d links removed for %s, %d pairs remain",
        n_matching, sorted(excluded_set), len(pairs),
    )


def build_firewalled_hub_text(
    hub_id: str,
    hierarchy: CREHierarchy,
    excluded_framework: str | Collection[str] | None = None,
    include_description: bool = False,
    descriptions: dict[str, str] | None = None,
    include_standards: bool = False,
    standard_sections: dict[str, list[str]] | None = None,
    stopwords: frozenset[str] | None = None,
) -> str:
    """Build a single hub's text representation with firewall.

    Primary format: "{hierarchy_path} | {hub_name}"

    If include_description=True (ablation A6): appends description.
    If include_standards=True (ablation A3): appends standard names,
    excluding the held-out framework.

    stopwords, when given, must be the same set applied to control text.
    assert_firewall compares the two by exact substring, so filtering one side
    only would make a genuine leak unmatchable and the check would pass on a
    breach.
    """
    node = hierarchy.hubs[hub_id]
    text = f"{node.hierarchy_path} | {node.name}"

    if include_description and descriptions and hub_id in descriptions:
        text = f"{text}: {descriptions[hub_id]}"

    if include_standards and standard_sections and hub_id in standard_sections:
        excluded = excluded_framework_set(excluded_framework)
        sections = [
            s
            for s in standard_sections[hub_id]
            if not any(name in s for name in excluded)
        ]
        if sections:
            text = f"{text}. Standards: {', '.join(sorted(sections))}"

    if stopwords:
        from tract.stopwords import filter_stopwords

        text = filter_stopwords(text, stopwords)

    return text


def build_all_hub_texts(
    hierarchy: CREHierarchy,
    excluded_framework: str | Collection[str] | None = None,
    include_description: bool = False,
    descriptions: dict[str, str] | None = None,
    include_standards: bool = False,
    standard_sections: dict[str, list[str]] | None = None,
    stopwords: frozenset[str] | None = None,
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
            stopwords,
        )
    return texts


def assert_firewall(
    hub_texts: dict[str, str],
    eval_items: list[Any],
    held_out_framework: str | Collection[str],
    base_hub_texts: dict[str, str] | None = None,
    hub_names: set[str] | None = None,
) -> None:
    """Assert no information leakage from held-out framework into hub representations.

    The base format ("{path} | {name}") uses only CRE-native content and is
    safe by construction. When hub texts include additional content (descriptions,
    standards), we check only the appended portion by subtracting the base text.

    If base_hub_texts is None the hub texts are base-format only, so the whole
    text is scanned rather than an appended slice. This function previously
    returned here after logging a pass, which meant the default configuration
    (hub_rep_format="path+name") asserted nothing at all: the firewall CLAUDE.md
    calls non-negotiable could not fail on any code path. Scanning the full text
    costs one substring test per (item, hub) pair and makes the base case a real
    assertion.
    """
    scan_appended_only = base_hub_texts is not None

    # CRE-native concept names, used to exempt controls that ARE the vocabulary.
    #
    # Recovering these by splitting hub text on " | " only works while the text
    # still contains that separator. Stop word filtering rebuilds text from
    # alphabetic tokens and drops all punctuation, so the separator vanished
    # from 522 of 522 hub texts, this set came out empty, the exemption never
    # fired, and the firewall raised a breach on a control that merely shares
    # its name with a CRE hub. A control named "AI model performance
    # validation" is not a leak; the check was reporting the absence of a
    # separator as evidence of contamination.
    #
    # Callers pass the names from the hierarchy instead, filtered with the same
    # stop words as everything else so both sides of the comparison match.
    if hub_names is not None:
        hub_names_lower = {name.strip().lower() for name in hub_names if name.strip()}
    else:
        hub_names_lower = set()
        for base in (base_hub_texts or hub_texts).values():
            if " | " in base:
                hub_names_lower.add(base.split(" | ", 1)[1].lower())
        if not hub_names_lower and hub_texts:
            raise ValueError(
                "Could not recover any hub names from the hub texts, so the "
                "CRE-vocabulary exemption would be empty and every control "
                "sharing a hub's name would raise a false breach. Pass "
                "hub_names explicitly when the text has been transformed."
            )

    for item in eval_items:
        control_text = item.control_text
        if len(control_text) < 5:
            continue
        control_lower = control_text.lower()
        # A control that is itself CRE vocabulary is not a leak: "Adversarial
        # training" measured against a hub named "Adversarial training" matches
        # because CRE named the concept, not because the framework text reached
        # the hub. Exempt controls that are a substring of some hub name.
        #
        # The converse test, `name in control_lower`, was also part of this
        # condition and was far too broad. Real hub names include short generic
        # nouns -- "Data", "Logging" -- so any control sentence that happened to
        # mention one was skipped against EVERY hub, including a hub carrying a
        # verbatim copy of that control. It exempted a whole sentence on the
        # strength of a single shared word, which is most of what the firewall
        # was supposed to catch.
        if any(control_lower in name for name in hub_names_lower):
            continue
        for hub_id, text in hub_texts.items():
            if scan_appended_only:
                assert base_hub_texts is not None
                base = base_hub_texts.get(hub_id, "")
                haystack = text[len(base):]
                where = "appended text"
            else:
                haystack = text
                where = "hub text"
            if not haystack:
                continue
            if control_text in haystack:
                raise AssertionError(
                    f"Firewall breach: control '{control_text[:50]}' "
                    f"(held out={sorted(excluded_framework_set(held_out_framework))}) "
                    f"found in hub {hub_id} "
                    f"{where}"
                )
    logger.info(
        "Firewall assertion passed: %d items checked against %d hubs "
        "(appended content verified)",
        len(eval_items),
        len(hub_texts),
    )
