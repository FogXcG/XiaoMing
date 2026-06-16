from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.llm.types import ToolSpec
from xiaoming.async_runtime.leases import WriteLeaseCallback
from xiaoming.permissions.engine import PermissionEngine
from xiaoming.permissions.types import PermissionBehavior, PermissionDecision
from xiaoming.policy.approvals import ApprovalCallback
from xiaoming.policy.paths import resolve_workspace_path
from xiaoming.tools.base import ToolResult


class EditFileTool:
    name = "edit_file"
    description = "Replace one unique UTF-8 text snippet in an existing workspace file."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path,
        approval_mode: str,
        approve: ApprovalCallback,
        permission_engine: PermissionEngine | None = None,
        checkpoint_store=None,
        checkpoint_id: str | None = None,
        lease_callback: WriteLeaseCallback | None = None,
    ):
        self.workspace = workspace
        self.approval_mode = approval_mode
        self.approve = approve
        self.permission_engine = permission_engine
        self.checkpoint_store = checkpoint_store
        self.checkpoint_id = checkpoint_id
        self.lease_callback = lease_callback

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        path_text = args["path"]
        old_text = args["old_text"]
        new_text = args["new_text"]
        decision = self._decide(path_text)
        if decision.behavior == PermissionBehavior.DENY:
            return ToolResult(self.name, "error", error=f"edit rejected by policy: {decision.reason}")
        path = resolve_workspace_path(self.workspace, path_text, allow_sensitive=True)
        original = path.read_text()
        count = original.count(old_text)
        if count == 0:
            return ToolResult(self.name, "error", error="old_text was not found; read the file with line numbers before retrying")
        if count > 1:
            return ToolResult(
                self.name,
                "error",
                error=f"old_text matched {count} times; use a longer unique old_text or read_file with line numbers before retrying",
            )
        if decision.behavior == PermissionBehavior.ASK and not self.approve(summarize_edit_for_approval(path_text, old_text, new_text)):
            return ToolResult(self.name, "denied", error="edit denied by user")
        if self.lease_callback is not None and not self.lease_callback(self.name, [path_text]):
            return ToolResult(self.name, "error", error=f"write lease denied for {path_text}")
        if self.checkpoint_store is not None:
            checkpoint_id = self.checkpoint_id() if callable(self.checkpoint_id) else self.checkpoint_id
            self.checkpoint_store.snapshot_paths(checkpoint_id, [path_text])
        path.write_text(original.replace(old_text, new_text, 1))
        return ToolResult(self.name, "success", output=f"Edited file: {path_text}")

    def _decide(self, path_text: str) -> PermissionDecision:
        if self.permission_engine is not None:
            return self.permission_engine.decide_file("Edit", path_text)
        if self.approval_mode == "suggest":
            return PermissionDecision(PermissionBehavior.ASK, reason="suggest mode")
        return PermissionDecision(PermissionBehavior.ALLOW, reason=f"{self.approval_mode} mode")


def summarize_edit_for_approval(path_text: str, old_text: str, new_text: str, preview_chars: int = 4000) -> str:
    old_preview = old_text[:preview_chars]
    new_preview = new_text[:preview_chars]
    if len(old_text) > preview_chars:
        old_preview += "\n... truncated ..."
    if len(new_text) > preview_chars:
        new_preview += "\n... truncated ..."
    return f"Tool: edit_file\nFile: {path_text}\nOld text:\n{old_preview}\nNew text:\n{new_preview}"
