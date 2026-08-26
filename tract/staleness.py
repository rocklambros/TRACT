"""Which recorded results were produced from inputs that have since moved.

Every fold records the digests of the three files it read: the curated links,
the merged corpus, and the stopword list. That makes staleness DETECTABLE. It
does not make it detected, and this project has already published one figure
that did not survive audit.

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

from tract.config import PROCESSED_DIR, PROJECT_ROOT, TRAINING_DIR
from tract.io import repo_relative

RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

# The three inputs a fold records, mapped to the file each digest describes.
# Keyed by the field name in fold_result.json["inputs"], so a field that stops
# being written shows up as unrecorded rather than as fresh.
TRACKED_INPUTS: Final[dict[str, Path]] = {
    "curated_links_sha256": TRAINING_DIR / "hub_links_curated.jsonl",
    "all_controls_sha256": PROCESSED_DIR / "all_controls.json",
    "stopwords_sha256": PROCESSED_DIR / "stopwords.json",
}


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
        """False when a result records none of the three digests.

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
    for field, path in TRACKED_INPUTS.items():
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
