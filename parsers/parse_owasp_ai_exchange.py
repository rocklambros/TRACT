"""Parser for OWASP AI Exchange — Tier 3 Hugo markdown extraction."""
from __future__ import annotations

import logging
import re

from tract.parsers.base import BaseParser
from tract.schema import Control

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONTROL_HEADER_RE = re.compile(r"^####\s+#(?P<id>[A-Z][A-Z0-9 /]+)\s*$", re.MULTILINE)

SOURCE_FILES_WITH_CONTROLS = [
    "src_1_general_controls.md",
    "src_2_threats_through_use.md",
    "src_3_development_time_threats.md",
    "src_4_runtime_application_security_threats.md",
]

CATEGORY_MAP: dict[str, str] = {
    "src_1_general_controls.md": "General Controls",
    "src_2_threats_through_use.md": "Threats Through Use",
    "src_3_development_time_threats.md": "Development Time Threats",
    "src_4_runtime_application_security_threats.md": "Runtime Application Security Threats",
}


class OwaspAiExchangeParser(BaseParser):
    framework_id = "owasp_ai_exchange"
    framework_name = "OWASP AI Exchange"
    version = "2024"
    source_url = "https://owaspai.org"
    mapping_unit_level = "control"
    expected_count = 54

    def parse(self) -> list[Control]:
        controls: list[Control] = []
        seen_ids: set[str] = set()

        for filename in SOURCE_FILES_WITH_CONTROLS:
            filepath = self.raw_dir / filename
            if not filepath.exists():
                logger.warning("Missing file: %s", filepath)
                continue

            text = filepath.read_text(encoding="utf-8")
            category = CATEGORY_MAP.get(filename, filename)
            matches = list(CONTROL_HEADER_RE.finditer(text))

            for i, m in enumerate(matches):
                raw_id = m.group("id").strip()
                control_id = raw_id.replace(" ", "_")
                if control_id in seen_ids:
                    continue
                seen_ids.add(control_id)

                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
                body = text[start:end].strip()

                next_section = re.search(r"^#{1,3}\s+\d", body, re.MULTILINE)
                if next_section:
                    body = body[:next_section.start()].strip()

                title = raw_id.replace("_", " ").title()

                is_threat = any(
                    kw in filename
                    for kw in ["threats_through_use", "development_time", "runtime_application"]
                )
                level = "threat" if is_threat and not body.startswith("Control") else "control"

                controls.append(Control(
                    control_id=control_id,
                    title=title,
                    description=body[:2000] if body else title,
                    full_text=body if len(body) > 2000 else None,
                    hierarchy_level=level,
                    metadata={"category": category, "source_file": filename},
                ))

        return controls


if __name__ == "__main__":
    parser = OwaspAiExchangeParser()
    parser.run()
