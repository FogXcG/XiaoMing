from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.llm.types import ToolSpec
from xiaoming.async_runtime.leases import WriteLeaseCallback
from xiaoming.permissions.engine import PermissionEngine
from xiaoming.permissions.types import PermissionBehavior
from xiaoming.policy.approvals import ApprovalCallback
from xiaoming.policy.paths import resolve_workspace_path
from xiaoming.tools.base import ToolResult
from xiaoming.tools.write_file import DEFAULT_MAX_CONTENT_CHARS, summarize_write_for_approval


class AppendFileTool:
    name = "append_file"
    description = "Append UTF-8 text to an existing workspace file. Use this for later chunks of large new files."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path,
        approval_mode: str,
        approve: ApprovalCallback,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
        permission_engine: PermissionEngine | None = None,
        checkpoint_store=None,
        checkpoint_id: str | None = None,
        lease_callback: WriteLeaseCallback | None = None,
    ):
        self.workspace = workspace
        self.approval_mode = approval_mode
        self.approve = approve
        self.max_content_chars = max_content_chars
        self.permission_engine = permission_engine
        self.checkpoint_store = checkpoint_store
        self.checkpoint_id = checkpoint_id
        self.lease_callback = lease_callback

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        path_text = args["path"]
        content = args["content"]
        if len(content) > self.max_content_chars:
            return ToolResult(self.name, "error", error=f"content is too large ({len(content)} chars); split it into smaller append_file calls")
        decision = self._decide(path_text)
        if decision.behavior == PermissionBehavior.DENY:
            return ToolResult(self.name, "error", error=f"append rejected by policy: {decision.reason}")
        path = resolve_workspace_path(self.workspace, path_text, allow_sensitive=True)
        if not path.exists():
            return ToolResult(self.name, "error", error="file does not exist; use write_file to create the first chunk")
        if decision.behavior == PermissionBehavior.ASK and not self.approve(summarize_write_for_approval(path_text, content).replace("Tool: write_file", "Tool: append_file", 1)):
            return ToolResult(self.name, "denied", error="append denied by user")
        if self.lease_callback is not None and not self.lease_callback(self.name, [path_text]):
            return ToolResult(self.name, "error", error=f"write lease denied for {path_text}")
        if self.checkpoint_store is not None:
            checkpoint_id = self.checkpoint_id() if callable(self.checkpoint_id) else self.checkpoint_id
            self.checkpoint_store.snapshot_paths(checkpoint_id, [path_text])
        with path.open("a") as handle:
            handle.write(content)
        return ToolResult(
            self.name,
            "success",
            output=f"Appended file: {path_text}\nBytes: {len(content.encode())}\nLines: {len(content.splitlines())}",
        )

    def _decide(self, path_text: str):
        if self.permission_engine is not None:
            return self.permission_engine.decide_file("Write", path_text)
        from xiaoming.permissions.types import PermissionDecision

        if self.approval_mode == "suggest":
            return PermissionDecision(PermissionBehavior.ASK, reason="suggest mode")
        return PermissionDecision(PermissionBehavior.ALLOW, reason=f"{self.approval_mode} mode")
