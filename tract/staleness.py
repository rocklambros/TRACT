"""Which recorded results were produced from inputs that have since moved.

Every fold records the digests of the files it read: the curated links, the
merged corpus, the stopword list and the framework-identity tokens. That makes
staleness DETECTABLE. It does not make it detected, and this project has
already published one figure that did not survive audit.

So this module answers one question a reader should be able to ask cheaply:
"is the number I am about to quote still describing the inputs that produced
it?" It reports rather than fails, because staleness after a corpus rebuild is
expected and correct. Quoting a stale number is the error, not having one.

The rule that follows from it: a result whose inputs have moved may be kept,
may be compared against its own recorded inputs, and may not be quoted as a
current measurement without re-running it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tract.config import PROJECT_ROOT
from tract.framework_identity import FRAMEWORK_IDENTITY_PATH
from tract.io import repo_relative
from tract.stopwords import STOPWORDS_PATH
from tract.text_selection import merged_corpus_path
from tract.training.data_quality import BRIDGE_PATH, CURATED_PATH

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"


def tracked_inputs() -> dict[str, Path]:
    """The file each digest in a fold's ``inputs`` block describes.

    Keyed by the field name in fold_result.json["inputs"], so a field that
    stops being written shows up as unrecorded rather than as fresh.

    Every path is imported from the module that WROTE the digest rather than
    spelled a second time here, because the one path that was spelled twice is
    the one that cost a campaign its result. fold_input_digests hashes
    merged_corpus_path(), which prefers the licensed overlay whenever it is
    staged -- and it is staged on every real run, because `provision` refuses
    on a corpus mismatch and the ISO 27001 fold would otherwise have no
    controls. This module held the literal data/processed/all_controls.json
    instead. The overlay is that file plus the restricted frameworks, so the
    writer's digest and the reader's could not match on any run that was
    configured correctly: replayed with the overlay on disk, every fold of A1
    (prose+sw) and A3 (prose+sw+qwen) came back stale on all_controls_sha256
    alone, and `aggregate` refuses stale folds. A flawless five-fold campaign
    would have produced no number anyone was allowed to quote.

    A function rather than a literal mapping because one of these four paths is
    a property of the checkout rather than of the repository: staging or
    clearing the overlay moves it, and a value resolved once at import cannot
    follow it.
    """
    return {
        "curated_links_sha256": CURATED_PATH,
        "all_controls_sha256": merged_corpus_path(),
        "stopwords_sha256": STOPWORDS_PATH,
        # Written by fold_input_digests since the framework-identity arm landed
        # and unread here until now. A digest nobody checks cannot go stale, so
        # a token set rebuilt between one fold and the next was invisible to the
        # one instrument whose whole job is to notice that.
        "framework_identity_sha256": FRAMEWORK_IDENTITY_PATH,
        # Phase 2C. Absent on a run that used no bridge corpus, in which case
        # the fold records None and _artifact_sha256 returns None here too, so
        # the two agree. Present, it is what distinguishes two runs that agree
        # on every other digest and disagree on the metric.
        "bridge_links_sha256": BRIDGE_PATH,
    }


# The import-time snapshot, for callers that want the field NAMES rather than a
# live path: describe() lists them and is_checkable counts them, and both are
# fixed by the writer's schema rather than by which corpus this checkout holds.
# check_result calls tracked_inputs() instead, so the paths it hashes are
# resolved at the moment it runs.
TRACKED_INPUTS: Final[dict[str, Path]] = tracked_inputs()


@dataclass(frozen=True)
class StaleInput:
    """One recorded digest that no longer matches the file it names."""

    field: str
    path: str
    recorded: str
    current: str


@dataclass(frozen=True)
class ResultStatus:
    """One fold result, and whether its inputs still hold."""

    result_path: str
    stale: tuple[StaleInput, ...]
    unrecorded: tuple[str, ...]

    @property
    def is_stale(self) -> bool:
        return bool(self.stale)

    @property
    def is_checkable(self) -> bool:
        """False when a result records none of the tracked digests.

        An unrecorded input is worse than a stale one. A stale digest says the
        number is old; a missing digest says nothing at all, and cannot be
        distinguished from a number that is current.
        """
        return len(self.unrecorded) < len(TRACKED_INPUTS)


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_result(result_path: Path) -> ResultStatus:
    """Compare one fold_result.json's recorded input digests against the files."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    inputs = payload.get("inputs") or {}
    stale: list[StaleInput] = []
    unrecorded: list[str] = []
    # Resolved here rather than read from the snapshot above, so that a check
    # run after the overlay was staged hashes the corpus the fold actually
    # read. See tracked_inputs().
    for field, path in tracked_inputs().items():
        recorded = inputs.get(field)
        if not recorded:
            unrecorded.append(field)
            continue
        current = _digest(path)
        if current is not None and current != recorded:
            stale.append(
                StaleInput(
                    field=field,
                    path=repo_relative(path),
                    recorded=str(recorded),
                    current=current,
                )
            )
    return ResultStatus(
        result_path=repo_relative(result_path),
        stale=tuple(stale),
        unrecorded=tuple(unrecorded),
    )


def scan(results_dir: Path | None = None) -> list[ResultStatus]:
    """Every fold result under results/, sorted, with its input status."""
    root = results_dir or RESULTS_DIR
    if not root.exists():
        return []
    return [
        check_result(path)
        for path in sorted(root.glob("**/fold_result.json"))
    ]


def describe(statuses: list[ResultStatus]) -> str:
    """A report a reader can act on, naming the file and what moved."""
    if not statuses:
        return "no fold results found"
    stale = [s for s in statuses if s.is_stale]
    uncheckable = [s for s in statuses if not s.is_checkable]
    lines = [
        f"{len(statuses)} fold results, {len(stale)} stale, "
        f"{len(uncheckable)} recording no input digest at all"
    ]
    for status in stale:
        lines.append(f"  {status.result_path}")
        for item in status.stale:
            lines.append(
                f"      {item.field}: {item.path} moved "
                f"{item.recorded[:12]} -> {item.current[:12]}"
            )
    for status in uncheckable:
        lines.append(
            f"  {status.result_path}: records none of "
            f"{sorted(TRACKED_INPUTS)}, so it cannot be checked"
        )
    if stale:
        lines.append(
            "A stale result may be kept and may be compared against its own "
            "recorded inputs. It may not be quoted as a current measurement "
            "without re-running it."
        )
    return "\n".join(lines)


def main() -> None:
    print(describe(scan()))


if __name__ == "__main__":
    main()
