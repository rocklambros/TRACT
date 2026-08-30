"""Whether an eval item's anchor already contains its own answer.

An item whose anchor lexically covers its ground-truth hub name is not testing
semantic mapping -- a string match scores a hit and looks like comprehension.
Campaign 1's headline was withdrawn partly for that reason, and
`results/phase1b/CAMPAIGN3.md` makes the non-echo stratum a BINDING side
condition on the gate.

A binding condition needs a partition that does not move. Two ways the previous
one moved:

`lexical_overlap_diagnostic` computes the split PER ARM against that arm's own
anchors, which is the right question for a per-run record ("how much of THIS
arm's score is lexical") and the wrong one for a gate ("which items are echo").
Titles give 32 echo items and prose gives 38, with a symmetric difference of 28.

Worse, the prose partition is computed from the TRUNCATED anchor, so it moves
with `max_seq_length`. Measured on the Campaign 2 test corpus: 38 echo at the
2,150-char budget, 41 at 4,300, 44 at 8,601 and above. Six items become echo
purely because a longer budget restored the tail that names their hub. Under the
truncated partition the non-echo stratum is 98 items; under an untruncated one
it is 93 -- and the five items that move carry trained hit == zero-shot hit, so
re-partitioning alone lifts the side-condition metric with no change in model
behaviour whatsoever. A ruler that moves when the experiment moves cannot bind
the experiment.

So the frozen partition is computed from the MAXIMUM text an item could ever
present -- its title and its full untruncated prose, unioned -- and is therefore
a property of the item and the hub tree alone. No budget, arm, or filter setting
can change it. It is deliberately the most generous definition of echo
available: an item counts as echo if ANY text the project could put in front of
the model would contain its hub name, which makes the non-echo stratum the
conservative one to make a claim about.
"""
from __future__ import annotations

import logging
from typing import Any, Final

from tract.hierarchy import CREHierarchy

logger = logging.getLogger(__name__)

# Words carried by almost every control statement, so their presence says
# nothing about whether the anchor names its hub. Kept byte-identical to
# orchestrate.lexical_overlap_diagnostic's list: the two answer different
# questions and must not also differ in their tokenisation.
FUNCTION_WORDS: Final[frozenset[str]] = frozenset({
    "the", "a", "an", "of", "for", "to", "in", "and", "or", "with", "by",
    "on", "at", "is", "are", "be", "as", "that", "this", "from", "its",
    "it", "their", "which", "when", "where", "all", "any", "not", "no",
    "if", "then", "than", "into", "via", "using", "use", "used",
})

# Big enough that no control in the corpus reaches it (the longest is ~50k
# chars), so prepare_anchor's cut never fires and the partition is computed
# against the whole document.
UNTRUNCATED: Final[int] = 10_000_000


def content_words(text: str) -> set[str]:
    """Lowercased content tokens, function words removed."""
    from tract.stopwords import tokenize

    return {token.lower() for token in tokenize(text)} - FUNCTION_WORDS


def is_echo(anchor: str, hub_name: str) -> bool:
    """True when every content word of the hub name appears in the anchor.

    Subset rather than overlap: a hub named "Anomalous AI input handling" is
    only echoed when the anchor carries all of anomalous/ai/input/handling. A
    hub with no content words at all is never echo, because the test would
    otherwise be vacuously true for every item.
    """
    hub_words = content_words(hub_name)
    return bool(hub_words) and hub_words <= content_words(anchor)


def frozen_echo_keys(
    corpus: list[Any],
    hierarchy: CREHierarchy,
    prose_index: Any | None,
    stopwords: frozenset[str] | None,
) -> set[tuple[str, str]]:
    """(framework_name, section_id) for every item that is echo under ANY text.

    Union of two partitions: the item's title, and its full prose with no
    length budget. `corpus` must be the title-keyed corpus that
    `build_evaluation_corpus` returns, BEFORE prose is applied -- item identity
    is fixed from titles, and this function needs both forms.

    Keyed on (framework, section) rather than section alone: section ids are not
    unique across frameworks in this corpus, and collapsing them undercounts.
    """
    from tract.text_selection import apply_prose_to_corpus

    names = {hub_id: node.name for hub_id, node in hierarchy.hubs.items()}

    def keys_for(items: list[Any]) -> set[tuple[str, str]]:
        return {
            (item.framework_name, item.section_id)
            for item in items
            if is_echo(item.control_text, names.get(item.ground_truth_hub_id, ""))
        }

    title_echo = keys_for(corpus)
    prose_echo: set[tuple[str, str]] = set()
    if prose_index is not None:
        prose_echo = keys_for(
            apply_prose_to_corpus(
                corpus, prose_index, stopwords, max_chars=UNTRUNCATED,
            )
        )

    frozen = title_echo | prose_echo
    logger.info(
        "Frozen echo partition: %d of %d items (title %d, untruncated prose "
        "%d). Budget-independent by construction.",
        len(frozen), len(corpus), len(title_echo), len(prose_echo),
    )
    return frozen
