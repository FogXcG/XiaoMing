from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.context.truncation import truncate_middle
from xiaoming.llm.types import ToolSpec
from xiaoming.permissions.types import PermissionBehavior
from xiaoming.policy.paths import resolve_workspace_path
from xiaoming.tools.base import ToolResult


IGNORED_DIRS = {".git", "node_modules", ".venv", "dist", "build", "__pycache__"}


class ListFilesTool:
    name = "list_files"
    description = "List files in the workspace, ignoring common generated directories."
    supports_parallel_tool_calls = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": ["string", "null"]},
            "pattern": {"type": ["string", "null"]},
        },
        "required": ["path", "pattern"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Path, permission_engine=None):
        self.workspace = workspace
        self.permission_engine = permission_engine

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            root = self.workspace.resolve()
            path_text = args.get("path") or "."
            if self.permission_engine is not None:
                decision = self.permission_engine.decide_file("List", path_text)
                if decision.behavior == PermissionBehavior.DENY:
                    return ToolResult(self.name, "error", error=f"list rejected: {decision.reason}")
                if decision.behavior == PermissionBehavior.ASK:
                    return ToolResult(self.name, "error", error=f"list requires approval: {decision.reason}")
            base = resolve_workspace_path(root, path_text, allow_sensitive=self.permission_engine is not None)
            pattern = args.get("pattern") or "*"
            files: list[str] = []
            for path in sorted(base.rglob(pattern)):
                if not path.is_file():
                    continue
                rel = path.relative_to(root)
                if any(part in IGNORED_DIRS for part in rel.parts):
                    continue
                files.append(str(rel))
            return ToolResult(self.name, "success", output=truncate_middle("\n".join(files), 12000))
        except Exception as exc:
            return ToolResult(self.name, "error", error=str(exc))
