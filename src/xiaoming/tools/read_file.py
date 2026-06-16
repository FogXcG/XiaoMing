from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.context.truncation import truncate_middle
from xiaoming.llm.types import ToolSpec
from xiaoming.permissions.types import PermissionBehavior
from xiaoming.policy.paths import resolve_workspace_path
from xiaoming.tools.base import ToolResult


class ReadFileTool:
    name = "read_file"
    description = "Read a UTF-8 text file slice from the workspace."
    supports_parallel_tool_calls = True
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": ["integer", "null"]},
            "limit": {"type": ["integer", "null"]},
        },
        "required": ["path", "start_line", "limit"],
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
            path_text = args["path"]
            if self.permission_engine is not None:
                decision = self.permission_engine.decide_file("Read", path_text)
                if decision.behavior == PermissionBehavior.DENY:
                    return ToolResult(self.name, "error", error=f"read rejected: {decision.reason}")
                if decision.behavior == PermissionBehavior.ASK:
                    return ToolResult(self.name, "error", error=f"read requires approval: {decision.reason}")
            path = resolve_workspace_path(self.workspace, path_text, allow_sensitive=self.permission_engine is not None)
            data = path.read_bytes()
            if b"\x00" in data:
                return ToolResult(self.name, "error", error="binary file rejected")
            lines = data.decode("utf-8").splitlines()
            start = int(args.get("start_line") or 1)
            limit = int(args.get("limit") or 200)
            selected = lines[start - 1 : start - 1 + limit]
            numbered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))
            return ToolResult(self.name, "success", output=truncate_middle(numbered, 20000))
        except Exception as exc:
            return ToolResult(self.name, "error", error=str(exc))
