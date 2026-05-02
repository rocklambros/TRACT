"""TRACT CRE hierarchy tree model.

Provides HubNode and CREHierarchy Pydantic models for the CRE taxonomy.
This is the coordinate system — every downstream component depends on it.
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tract.io import atomic_write_json, load_json

logger = logging.getLogger(__name__)


class HubNode(BaseModel):
    """A single hub in the CRE hierarchy tree."""

    model_config = ConfigDict(frozen=True)

    hub_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    parent_id: str | None = None
    children_ids: list[str] = Field(default_factory=list)
    depth: int = Field(..., ge=0)
    branch_root_id: str = Field(..., min_length=1)
    hierarchy_path: str = Field(..., min_length=1)
    is_leaf: bool
    sibling_hub_ids: list[str] = Field(default_factory=list)
    related_hub_ids: list[str] = Field(default_factory=list)


class CREHierarchy(BaseModel):
    """The full CRE hub tree — the coordinate system for hub assignment."""

    model_config = ConfigDict(frozen=True)

    hubs: dict[str, HubNode]
    roots: list[str]
    label_space: list[str]
    fetch_timestamp: str = Field(..., min_length=1)
    data_hash: str = Field(..., min_length=1)
    version: str = "1.1"

    @classmethod
    def from_opencre(
        cls,
        cres: list[dict[str, Any]],
        fetch_timestamp: str,
        data_hash: str,
    ) -> CREHierarchy:
        """Build a CREHierarchy from raw OpenCRE CRE records."""
        hub_names: dict[str, str] = {}
        for cre in cres:
            if cre.get("doctype") != "CRE":
                continue
            cre_id = cre["id"]
            cre_name = cre["name"]
            if not cre_id or not cre_name:
                raise ValueError(f"CRE record missing id or name: {cre!r:.200}")
            hub_names[cre_id] = cre_name

        children_map: dict[str, list[str]] = {}
        parent_map: dict[str, str] = {}
        for cre in cres:
            if cre.get("doctype") != "CRE":
                continue
            cre_id = cre["id"]
            for link in cre.get("links", []):
                if link.get("ltype") != "Contains":
                    continue
                doc = link.get("document", {})
                if doc.get("doctype") != "CRE":
                    continue
                child_id = doc.get("id", "")
                if child_id in hub_names:
                    children_map.setdefault(cre_id, []).append(child_id)
                    parent_map[child_id] = cre_id

        roots = sorted(hid for hid in hub_names if hid not in parent_map)

        depth_map: dict[str, int] = {}
        branch_map: dict[str, str] = {}
        path_map: dict[str, str] = {}
        queue: deque[str] = deque()

        for root_id in roots:
            depth_map[root_id] = 0
            branch_map[root_id] = root_id
            path_map[root_id] = hub_names[root_id]
            queue.append(root_id)

        while queue:
            current = queue.popleft()
            for child_id in children_map.get(current, []):
                if child_id not in depth_map:
                    depth_map[child_id] = depth_map[current] + 1
                    branch_map[child_id] = branch_map[current]
                    path_map[child_id] = f"{path_map[current]} > {hub_names[child_id]}"
                    queue.append(child_id)

        # Detect orphans (hubs with no depth assigned — not reachable from any root)
        unreachable = set(hub_names.keys()) - set(depth_map.keys())
        if unreachable:
            logger.warning(
                "Found %d unreachable hubs (treating as orphan roots): %s",
                len(unreachable),
                sorted(unreachable),
            )
            for orphan_id in sorted(unreachable):
                depth_map[orphan_id] = 0
                branch_map[orphan_id] = orphan_id
                path_map[orphan_id] = hub_names[orphan_id]
                if orphan_id not in roots:
                    roots.append(orphan_id)
            roots = sorted(roots)

        # Compute sibling_hub_ids per hub
        sibling_map: dict[str, list[str]] = {}
        for hub_id in hub_names:
            pid = parent_map.get(hub_id)
            if pid is not None:
                sibling_map[hub_id] = sorted(
                    cid for cid in children_map.get(pid, []) if cid != hub_id
                )
            else:
                sibling_map[hub_id] = []

        # Log root hubs that are also leaves (no children)
        orphan_ids = [
            hid for hid in roots
            if hid not in children_map or not children_map[hid]
        ]
        if orphan_ids:
            logger.warning(
                "Orphan hubs (root + leaf): %s",
                [(oid, hub_names[oid]) for oid in orphan_ids],
            )

        # Build HubNode objects
        hub_nodes: dict[str, HubNode] = {}
        for hub_id, name in hub_names.items():
            kids = children_map.get(hub_id, [])
            hub_nodes[hub_id] = HubNode(
                hub_id=hub_id,
                name=name,
                parent_id=parent_map.get(hub_id),
                children_ids=sorted(kids),
                depth=depth_map[hub_id],
                branch_root_id=branch_map[hub_id],
                hierarchy_path=path_map[hub_id],
                is_leaf=len(kids) == 0,
                sibling_hub_ids=sibling_map[hub_id],
            )

        label_space = sorted(
            hid for hid, node in hub_nodes.items() if node.is_leaf
        )

        logger.info(
            "Built hierarchy: %d hubs, %d roots, %d leaves",
            len(hub_nodes), len(roots), len(label_space),
        )

        hierarchy = cls(
            hubs=hub_nodes,
            roots=roots,
            label_space=label_space,
            fetch_timestamp=fetch_timestamp,
            data_hash=data_hash,
            version="1.0",
        )
        hierarchy.validate_integrity()
        return hierarchy

    def validate_integrity(self) -> None:
        """Validate the hierarchy tree. Raises ValueError on failure."""
        # 1. No dangling references
        for hub_id, node in self.hubs.items():
            if node.parent_id is not None and node.parent_id not in self.hubs:
                raise ValueError(
                    f"Hub {hub_id} has dangling parent_id: {node.parent_id}"
                )
            for child_id in node.children_ids:
                if child_id not in self.hubs:
                    raise ValueError(
                        f"Hub {hub_id} has dangling child_id: {child_id}"
                    )

        # 2. Depth consistency
        for hub_id, node in self.hubs.items():
            if node.parent_id is not None:
                parent = self.hubs[node.parent_id]
                if node.depth != parent.depth + 1:
                    raise ValueError(
                        f"Hub {hub_id} depth={node.depth} but parent "
                        f"{node.parent_id} depth={parent.depth}"
                    )

        # 3. All leaves reachable from a root
        for leaf_id in self.label_space:
            current: str | None = leaf_id
            visited: set[str] = set()
            while current is not None:
                if current in visited:
                    raise ValueError(f"Cycle detected at hub {current}")
                visited.add(current)
                current = self.hubs[current].parent_id
            top = leaf_id
            for v in visited:
                if self.hubs[v].parent_id is None:
                    top = v
                    break
            if top not in self.roots:
                raise ValueError(
                    f"Leaf {leaf_id} not reachable from any root"
                )

        # 4. Label space determinism
        if self.label_space != sorted(self.label_space):
            raise ValueError("label_space is not sorted")

        # 5. Related hub IDs: exist and are bidirectional
        for hub_id, node in self.hubs.items():
            for related_id in node.related_hub_ids:
                if related_id not in self.hubs:
                    raise ValueError(
                        f"Hub {hub_id} has dangling related_hub_id: {related_id}"
                    )
                if hub_id not in self.hubs[related_id].related_hub_ids:
                    raise ValueError(
                        f"Hub {hub_id} has related_hub_id {related_id} but "
                        f"{related_id} does not list {hub_id}"
                    )

        # 6. Expected counts (warnings only)
        if len(self.hubs) != 522:
            logger.warning(
                "Expected 522 hubs, got %d", len(self.hubs)
            )
        if len(self.label_space) != 400:
            logger.warning(
                "Expected 400 leaves, got %d", len(self.label_space)
            )
        if len(self.roots) != 5:
            logger.warning(
                "Expected 5 roots, got %d", len(self.roots)
            )

    # ── Query methods ──────────────────────────────────────────────────

    def leaf_hub_ids(self) -> list[str]:
        """Return the label space (sorted leaf hub IDs)."""
        return list(self.label_space)

    def get_parent(self, hub_id: str) -> HubNode | None:
        """Return the parent node, or None if root."""
        node = self.hubs[hub_id]
        if node.parent_id is None:
            return None
        return self.hubs[node.parent_id]

    def get_children(self, hub_id: str) -> list[HubNode]:
        """Return child nodes of a hub."""
        return [self.hubs[cid] for cid in self.hubs[hub_id].children_ids]

    def get_siblings(self, hub_id: str) -> list[HubNode]:
        """Return sibling nodes (same parent, excluding self)."""
        return [self.hubs[sid] for sid in self.hubs[hub_id].sibling_hub_ids]

    def get_branch_hub_ids(self, root_id: str) -> list[str]:
        """Return all hub IDs under a root (including the root)."""
        if root_id not in self.hubs:
            raise ValueError(f"Unknown hub ID: {root_id}")
        result: list[str] = []
        q: deque[str] = deque([root_id])
        while q:
            current = q.popleft()
            result.append(current)
            q.extend(self.hubs[current].children_ids)
        return result

    def get_hierarchy_path(self, hub_id: str) -> str:
        """Return the cached hierarchy path string."""
        return self.hubs[hub_id].hierarchy_path

    def hub_by_name(self, name: str) -> HubNode | None:
        """Case-insensitive hub lookup by name. Returns first match or None."""
        lower = name.lower()
        for node in self.hubs.values():
            if node.name.lower() == lower:
                return node
        return None

    # ── Serialization ──────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Atomically save hierarchy to JSON."""
        atomic_write_json(self.model_dump(), path)
        logger.info("Saved hierarchy to %s", path)

    @classmethod
    def load(cls, path: Path) -> CREHierarchy:
        """Load hierarchy from JSON and validate."""
        data = load_json(path)
        hierarchy = cls.model_validate(data)
        hierarchy.validate_integrity()
        return hierarchy
