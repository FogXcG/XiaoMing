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


DEFAULT_MAX_CONTENT_CHARS = 20_000


class WriteFileTool:
    name = "write_file"
    description = "Create a new UTF-8 text file in the workspace. Does not overwrite existing files."
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
            return ToolResult(
                self.name,
                "error",
                error=f"content is too large ({len(content)} chars); split it with write_file for the first chunk and append_file for later chunks",
            )
        decision = self._decide(path_text)
        if decision.behavior == PermissionBehavior.DENY:
            return ToolResult(self.name, "error", error=f"write rejected by policy: {decision.reason}")
        path = resolve_workspace_path(self.workspace, path_text, allow_sensitive=True)
        if path.exists():
            return ToolResult(self.name, "error", error="file already exists; use edit_file or apply_patch to modify existing files")
        if decision.behavior == PermissionBehavior.ASK and not self.approve(summarize_write_for_approval(path_text, content)):
            return ToolResult(self.name, "denied", error="write denied by user")
        if self.lease_callback is not None and not self.lease_callback(self.name, [path_text]):
            return ToolResult(self.name, "error", error=f"write lease denied for {path_text}")
        _snapshot(self.checkpoint_store, self.checkpoint_id, [path_text])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return ToolResult(
            self.name,
            "success",
            output=f"Wrote file: {path_text}\nBytes: {len(content.encode())}\nLines: {len(content.splitlines())}",
        )

    def _decide(self, path_text: str):
        if self.permission_engine is not None:
            return self.permission_engine.decide_file("Write", path_text)
        if self.approval_mode == "suggest":
            from xiaoming.permissions.types import PermissionDecision

            return PermissionDecision(PermissionBehavior.ASK, reason="suggest mode")
        from xiaoming.permissions.types import PermissionDecision

        return PermissionDecision(PermissionBehavior.ALLOW, reason=f"{self.approval_mode} mode")


def _snapshot(checkpoint_store, checkpoint_id, paths: list[str]) -> None:
    if checkpoint_store is not None:
        checkpoint_store.snapshot_paths(checkpoint_id() if callable(checkpoint_id) else checkpoint_id, paths)


def summarize_write_for_approval(path_text: str, content: str, preview_lines: int = 80) -> str:
    lines = content.splitlines()
    preview = "\n".join(lines[:preview_lines])
    if len(lines) > preview_lines:
        preview += "\n... truncated ..."
    return (
        f"Tool: write_file\n"
        f"File: {path_text}\n"
        f"Bytes: {len(content.encode())}\n"
        f"Lines: {len(lines)}\n"
        f"Content preview:\n{preview}"
    )
