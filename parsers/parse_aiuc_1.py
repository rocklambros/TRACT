"""Parser for AIUC-1 Standard — Tier 1 structured JSON.

Mapping unit is activity (leaf nodes under controls).

The Q1 2026 update retired two controls and left their records in place as
withdrawal notices. E007 reads "RETIRED - Merged with E004 at Q1 2026 update."
and E014 reads "RETIRED - Merged into E017 at Q1 2026 update.", and each keeps
one activity whose whole statement is "RETIRED - merged into E004." or
"RETIRED - merged into E017.". [measured 2026-08-19] Shipped as controls, those
two rows hand a consumer a redirect where a security requirement belongs, and
that is what the published crosswalk carries today.

They are dropped. Ruling R17 settled the shape for WSTG, where eight archive
members were withdrawal notices: a notice that names a successor does not ship
as a control, and the retired id is declared as an `alt_ids` entry on the
successor so curated links still reach real prose. The first half applies here
unchanged. The second half does not, for two reasons that were checked rather
than assumed.

The successor is named at the wrong level. E007 and E014 name E004 and E017,
which are controls, while this parser's unit is the activity. E004 has two
activities and E017 has three, and the source nowhere says which one absorbed
E007.1 or E014.1. An `alt_ids` entry would have to pick one, which asserts an
equivalence the publisher did not state.

Nothing would read it. AIUC-1 carries none of the 4,405 curated OpenCRE links,
so no link targets E007.1, E014.1, E007 or E014, and the alias channel exists
to let a curated link reach prose. [measured 2026-08-19]

Keeping them with a damaged marker was the third option and is worse than
dropping. The marker takes a control out of the prose ratio, not out of the
artifact, so both redirect notices would still publish as crosswalk rows.

The drop writes a repair audit record per activity carrying the notice text,
the retired control's own statement and the successor's statement, so the two
ids leave the corpus in the open rather than silently.
"""
from __future__ import annotations

import json
import logging
import re

from typing import Any, ClassVar, Final

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SOURCE_NAME: Final[str] = "aiuc-1-standard.json"

# The publisher's withdrawal marker, at the head of the statement it replaces.
# Anchored so a control that merely discusses retiring a model is not read as
# a withdrawal notice.
RETIRED: Final[re.Pattern[str]] = re.compile(r"^\s*RETIRED\b", re.IGNORECASE)

# The successor a withdrawal notice names. Both spellings the source uses,
# "Merged with E004" and "Merged into E017", against the one control id shape
# the standard issues: a domain letter and three digits.
SUCCESSOR: Final[re.Pattern[str]] = re.compile(
    r"\bmerged\s+(?:with|into)\s+([A-Z]\d{3})\b", re.IGNORECASE,
)


class Aiuc1Parser(BaseParser):
    framework_id = "aiuc_1"
    framework_name = "AIUC-1 Standard"
    version = "1.0"
    source_url = "https://www.aiuc-1.com"
    mapping_unit_level = "activity"
    # 132 activities in the source, less the two that are withdrawal notices
    # for the controls the Q1 2026 update retired. [measured 2026-08-19] See
    # the module docstring for why those two do not ship.
    expected_count = 130
    fetched_date: ClassVar[str] = "2026-04-28"
    # 110 of the 130 shipped activities carry a statement longer than
    # HONEST_PROSE_MIN_CHARS, giving 0.8462, and the floor fires at 109/130
    # (0.8385). [measured 2026-08-19] The previous 0.83 was 110/132, with the
    # two tombstones counted against the parser as if they were terse controls.
    #
    # The 20 short ones are short in the source, not lost by the parser. Each
    # activity record holds exactly id, description, category and
    # evidence_types, parse() copies description verbatim, and no field is
    # discarded. Median source description is 76 characters and the minimum is
    # 44. [measured 2026-08-19]
    #
    # The floor sits well below 1.0 because the source is terse. It still stops
    # a collapse onto the parent control title, which is what all 130
    # activities would inherit if the description read broke.
    min_prose_fraction: ClassVar[float] = 0.84

    def parse(self) -> list[Control]:
        data = json.loads(self.read_source(SOURCE_NAME))
        by_id = {
            ctrl["id"]: ctrl
            for domain in data["domains"]
            for ctrl in domain["controls"]
        }
        controls: list[Control] = []
        audit: list[dict[str, object]] = []

        for domain in data["domains"]:
            domain_name = domain["name"]
            for ctrl in domain["controls"]:
                if RETIRED.match(ctrl["description"]):
                    self._record_withdrawal(ctrl, by_id, audit)
                    continue
                controls.extend(
                    self._activities(ctrl, domain_name),
                )

        self.write_repair_audit(audit)
        logger.info(
            "%s: %d activities, %d dropped as withdrawal notices",
            self.framework_id, len(controls), len(audit),
        )
        return controls

    @staticmethod
    def _activities(
        ctrl: dict[str, Any], domain_name: str,
    ) -> list[Control]:
        """One Control per activity of a live control.

        Raises:
            ValueError: If a live control carries a withdrawal notice as an
                activity. The source has no such record, and shipping one
                would put a redirect back into the corpus through a path the
                control-level check does not cover.
        """
        built: list[Control] = []
        for activity in ctrl.get("activities", []):
            if RETIRED.match(activity["description"]):
                raise ValueError(
                    f"aiuc_1: activity {activity['id']} is a withdrawal "
                    f"notice ({activity['description']!r}) but its control "
                    f"{ctrl['id']} is live ({ctrl['description']!r}). This "
                    f"parser drops withdrawn activities with their control, "
                    f"so a notice on a live control has no successor to name "
                    f"and would ship as a redirect."
                )
            built.append(Control(
                control_id=activity["id"],
                title=ctrl["title"],
                description=activity["description"],
                hierarchy_level="activity",
                parent_id=ctrl["id"],
                parent_name=ctrl["title"],
                metadata={
                    "category": activity.get("category", ""),
                    "domain": domain_name,
                    "evidence_types": activity.get("evidence_types", []),
                },
            ))
        return built

    @classmethod
    def _record_withdrawal(
        cls,
        ctrl: dict[str, Any],
        by_id: dict[str, dict[str, Any]],
        audit: list[dict[str, object]],
    ) -> None:
        """Write what each dropped activity said and what replaced it.

        Raises:
            ValueError: If the notice names no successor, names one the
                standard does not issue, names one that is itself retired, or
                carries an activity that is not itself a notice.
        """
        successor_id = cls._successor(ctrl, by_id)
        successor = by_id[successor_id]
        for activity in ctrl.get("activities", []):
            if not RETIRED.match(activity["description"]):
                raise ValueError(
                    f"aiuc_1: control {ctrl['id']} is retired "
                    f"({ctrl['description']!r}) but its activity "
                    f"{activity['id']} carries a statement rather than a "
                    f"notice ({activity['description']!r}). Dropping the "
                    f"control would discard that statement. Decide whether "
                    f"the activity moved to {successor_id} or survived, and "
                    f"say so in the parser."
                )
            audit.append({
                "control_id": activity["id"],
                "repair": "retired_activity_dropped",
                # The statement the corpus would otherwise have anchored on.
                "before": activity["description"],
                # Nothing replaces it under this id. The successor's own
                # activities already carry the requirement.
                "after": "",
                "parent_id": ctrl["id"],
                "parent_statement": ctrl["description"],
                "successor_id": successor_id,
                "successor_statement": successor["description"],
                "reason": (
                    "the publisher retired this control and left the record "
                    "in place as a redirect, so its activity statement is a "
                    "withdrawal notice rather than a requirement. The "
                    "successor is named at control level and this parser's "
                    "unit is the activity, so there is no activity the source "
                    "names to alias onto"
                ),
            })

    @staticmethod
    def _successor(
        ctrl: dict[str, Any], by_id: dict[str, dict[str, Any]],
    ) -> str:
        """The live control a withdrawal notice redirects to.

        Raises:
            ValueError: If no successor is named, the named id is unknown, or
                the named id is itself retired.
        """
        named = SUCCESSOR.search(ctrl["description"])
        if named is None:
            raise ValueError(
                f"aiuc_1: control {ctrl['id']} is retired "
                f"({ctrl['description']!r}) and names no successor. Dropping "
                f"it would remove its activities with nothing recorded in "
                f"their place. Read the source and decide whether the "
                f"requirement moved or was withdrawn outright."
            )
        successor_id = named.group(1).upper()
        if successor_id not in by_id:
            raise ValueError(
                f"aiuc_1: control {ctrl['id']} redirects to {successor_id}, "
                f"which the standard does not issue. The redirect would drop "
                f"{ctrl['id']} out of the corpus with nothing to reach in its "
                f"place."
            )
        if RETIRED.match(by_id[successor_id]["description"]):
            raise ValueError(
                f"aiuc_1: control {ctrl['id']} redirects to {successor_id}, "
                f"which is itself retired "
                f"({by_id[successor_id]['description']!r}). This parser "
                f"resolves one hop, so a chain has to be read and declared "
                f"rather than followed."
            )
        return successor_id


def main() -> None:
    Aiuc1Parser().run()


if __name__ == "__main__":
    main()
