from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.context.truncation import truncate_middle
from xiaoming.llm.types import ToolSpec
from xiaoming.permissions.engine import PermissionEngine
from xiaoming.permissions.types import PermissionBehavior
from xiaoming.policy.approvals import ApprovalCallback
from xiaoming.policy.shell_policy import ShellDecision, decide_shell
from xiaoming.subprocess_utils import run_noninteractive
from xiaoming.tools.base import ToolResult


class ShellTool:
    name = "shell"
    description = "Run a verification shell command in the workspace after policy checks."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Path, approval_mode: str, approve: ApprovalCallback, permission_engine: PermissionEngine | None = None):
        self.workspace = workspace
        self.approval_mode = approval_mode
        self.approve = approve
        self.permission_engine = permission_engine

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        command = args["command"]
        decision = self._decide(command)
        if decision == ShellDecision.REJECT:
            return ToolResult(self.name, "error", error=f"command rejected by policy: {command}")
        if decision == ShellDecision.APPROVE and not self.approve(summarize_shell_for_approval(command)):
            return ToolResult(self.name, "denied", error=f"command denied by user: {command}")
        completed = run_noninteractive(
            command,
            cwd=self.workspace,
            text=True,
            capture_output=True,
            shell=True,
            timeout=120,
            check=False,
        )
        output = completed.stdout
        if completed.stderr:
            output = output + ("\n" if output else "") + completed.stderr
        status = "success" if completed.returncode == 0 else "error"
        if status == "success":
            return ToolResult(self.name, status, output=truncate_middle(output.strip(), 20000))
        return ToolResult(self.name, status, error=truncate_middle(output.strip(), 20000))

    def _decide(self, command: str) -> ShellDecision:
        if self.permission_engine is None:
            return decide_shell(command, self.approval_mode)
        decision = self.permission_engine.decide_shell(command)
        if decision.behavior == PermissionBehavior.ALLOW:
            return ShellDecision.ALLOW
        if decision.behavior == PermissionBehavior.DENY:
            return ShellDecision.REJECT
        return ShellDecision.APPROVE


def summarize_shell_for_approval(command: str) -> str:
    return f"Tool: shell\nCommand:\n{command}"
