from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.context.truncation import truncate_middle
from xiaoming.llm.types import ToolSpec
from xiaoming.permissions.types import PermissionBehavior
from xiaoming.policy.paths import resolve_workspace_path
from xiaoming.subprocess_utils import run_noninteractive
from xiaoming.tools.base import ToolResult


class SearchCodeTool:
    name = "search_code"
    description = "Search code in the workspace using ripgrep."
    supports_parallel_tool_calls = True
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "path": {"type": ["string", "null"]},
        },
        "required": ["query", "path"],
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
            path_text = args.get("path") or "."
            if self.permission_engine is not None:
                decision = self.permission_engine.decide_file("Search", path_text)
                if decision.behavior == PermissionBehavior.DENY:
                    return ToolResult(self.name, "error", error=f"search rejected: {decision.reason}")
                if decision.behavior == PermissionBehavior.ASK:
                    return ToolResult(self.name, "error", error=f"search requires approval: {decision.reason}")
            base = resolve_workspace_path(self.workspace, path_text, allow_sensitive=self.permission_engine is not None)
            completed = run_noninteractive(
                ["rg", "--line-number", "--no-heading", args["query"], str(base)],
                cwd=self.workspace,
                text=True,
                capture_output=True,
                timeout=20,
                check=False,
            )
            if completed.returncode not in (0, 1):
                return ToolResult(self.name, "error", error=completed.stderr.strip())
            return ToolResult(self.name, "success", output=truncate_middle(completed.stdout.strip(), 20000))
        except Exception as exc:
            return ToolResult(self.name, "error", error=str(exc))
