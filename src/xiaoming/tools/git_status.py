from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Any

from xiaoming.llm.types import ToolSpec
from xiaoming.tools.base import ToolResult


class GitStatusTool:
    name = "git_status"
    description = "Return git status --short for the workspace."
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Path):
        self.workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=self.workspace,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            return ToolResult(self.name, "error", error=completed.stderr.strip())
        return ToolResult(self.name, "success", output=completed.stdout.strip())
