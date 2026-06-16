from __future__ import annotations

from pathlib import Path


SENSITIVE_NAMES = {".env"}


def resolve_workspace_path(workspace: Path, requested: str, *, allow_sensitive: bool = False) -> Path:
    root = workspace.resolve()
    path = (root / requested).resolve()
    if path == root:
        return path
    if root not in path.parents:
        raise ValueError(f"path is outside workspace: {requested}")
    if not allow_sensitive and any(part in SENSITIVE_NAMES for part in path.parts):
        raise ValueError(f"path is sensitive and requires explicit approval: {requested}")
    return path
