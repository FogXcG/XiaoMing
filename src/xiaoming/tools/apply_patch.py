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


class ApplyPatchTool:
    name = "apply_patch"
    description = "Apply a limited Codex-style patch to workspace files."
    input_schema = {
        "type": "object",
        "properties": {
            "patch": {"type": "string"},
        },
        "required": ["patch"],
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
        return ToolSpec(self.name, self.description, self.input_schema, input_mode="freeform", freeform_arg="patch")

    def run(self, args: dict[str, Any]) -> ToolResult:
        patch = args["patch"]
        decision = self._decide(patch)
        if decision.behavior == PermissionBehavior.DENY:
            return ToolResult(self.name, "error", error=f"patch rejected by policy: {decision.reason}")
        if decision.behavior == PermissionBehavior.ASK and not self.approve(summarize_patch_for_approval(patch)):
            return ToolResult(self.name, "denied", error="patch denied by user")
        try:
            changed_paths = _extract_patch_files(patch)
            if self.lease_callback is not None and not self.lease_callback(self.name, changed_paths):
                return ToolResult(self.name, "error", error=f"write lease denied for {', '.join(changed_paths)}")
            if self.checkpoint_store is not None:
                checkpoint_id = self.checkpoint_id() if callable(self.checkpoint_id) else self.checkpoint_id
                self.checkpoint_store.snapshot_paths(checkpoint_id, changed_paths)
            changed = _apply_limited_patch(self.workspace, patch)
            return ToolResult(self.name, "success", output="Updated files:\n" + "\n".join(changed))
        except Exception as exc:
            return ToolResult(self.name, "error", error=str(exc))

    def _decide(self, patch: str) -> PermissionDecision:
        if self.permission_engine is None:
            if self.approval_mode == "suggest":
                return PermissionDecision(PermissionBehavior.ASK, reason="suggest mode")
            return PermissionDecision(PermissionBehavior.ALLOW, reason=f"{self.approval_mode} mode")
        decisions = [self.permission_engine.decide_file("Edit", path) for path in _extract_patch_files(patch)]
        if not decisions:
            return PermissionDecision(PermissionBehavior.ASK, reason="unknown patch target")
        severity = {PermissionBehavior.ALLOW: 0, PermissionBehavior.ASK: 1, PermissionBehavior.DENY: 2}
        return max(decisions, key=lambda item: severity[item.behavior])


def _apply_limited_patch(workspace: Path, patch: str) -> list[str]:
    lines = patch.splitlines()
    if not lines:
        raise ValueError("empty patch")
    if lines[0].startswith("--- "):
        return _apply_unified_patch(workspace, lines)
    if lines[0] == "*** Begin Patch" and lines[-1] == "*** End Patch" and len(lines) > 2 and lines[1].startswith("--- "):
        return _apply_unified_patch(workspace, lines[1:-1])
    if lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ValueError("patch must start with *** Begin Patch and end with *** End Patch")
    index = 1
    changed: list[str] = []
    while index < len(lines) - 1:
        header = lines[index]
        if header.startswith("*** Update File: "):
            path_text = header.removeprefix("*** Update File: ").strip()
            path = resolve_workspace_path(workspace, path_text, allow_sensitive=True)
            index += 1
            old_lines: list[str] = []
            new_lines: list[str] = []
            if index >= len(lines) or lines[index] != "@@":
                raise ValueError("only simple @@ hunks are supported")
            index += 1
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                line = lines[index]
                if line.startswith("-"):
                    old_lines.append(line[1:])
                elif line.startswith("+"):
                    new_lines.append(line[1:])
                elif line.startswith(" "):
                    old_lines.append(line[1:])
                    new_lines.append(line[1:])
                else:
                    raise ValueError(f"unsupported hunk line: {line}")
                index += 1
            original = path.read_text()
            old = "\n".join(old_lines) + "\n"
            new = "\n".join(new_lines) + "\n"
            if old not in original:
                raise ValueError(f"old block not found in {path_text}")
            path.write_text(original.replace(old, new, 1))
            changed.append(path_text)
        elif header.startswith("*** Add File: "):
            path_text = header.removeprefix("*** Add File: ").strip()
            path = resolve_workspace_path(workspace, path_text, allow_sensitive=True)
            index += 1
            new_lines: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                line = lines[index]
                if not line.startswith("+"):
                    raise ValueError("add file lines must start with +")
                new_lines.append(line[1:])
                index += 1
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(new_lines) + "\n")
            changed.append(path_text)
        else:
            raise ValueError(f"unsupported patch header: {header}")
    return changed


def summarize_patch_for_approval(patch: str, preview_lines: int = 80) -> str:
    files = _extract_patch_files(patch)
    preview = "\n".join(patch.splitlines()[:preview_lines])
    if len(patch.splitlines()) > preview_lines:
        preview += "\n... truncated ..."
    return f"Tool: apply_patch\nFiles: {', '.join(files) if files else '(unknown)'}\nPatch preview:\n{preview}"


def _extract_patch_files(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("*** Add File: "):
            files.append(line.removeprefix("*** Add File: ").strip())
        elif line.startswith("*** Update File: "):
            files.append(line.removeprefix("*** Update File: ").strip())
        elif line.startswith("+++ "):
            path = _clean_unified_path(line.removeprefix("+++ ").strip())
            if path != "/dev/null":
                files.append(path)
    return list(dict.fromkeys(files))


def _apply_unified_patch(workspace: Path, lines: list[str]) -> list[str]:
    index = 0
    changed: list[str] = []
    while index < len(lines):
        if not lines[index].startswith("--- "):
            raise ValueError(f"unsupported unified patch header: {lines[index]}")
        old_path = lines[index].removeprefix("--- ").strip()
        index += 1
        if index >= len(lines) or not lines[index].startswith("+++ "):
            raise ValueError("unified patch missing +++ header")
        new_path = _clean_unified_path(lines[index].removeprefix("+++ ").strip())
        index += 1
        old_lines: list[str] = []
        new_lines: list[str] = []
        while index < len(lines) and not lines[index].startswith("--- "):
            line = lines[index]
            if line.startswith("@@"):
                index += 1
                continue
            if line.startswith("+"):
                new_lines.append(line[1:])
            elif line.startswith("-"):
                old_lines.append(line[1:])
            elif line.startswith(" "):
                old_lines.append(line[1:])
                new_lines.append(line[1:])
            elif line == "\\ No newline at end of file":
                pass
            elif line:
                raise ValueError(f"unsupported unified hunk line: {line}")
            index += 1
        path = resolve_workspace_path(workspace, new_path, allow_sensitive=True)
        new_text = "\n".join(new_lines) + ("\n" if new_lines else "")
        if old_path == "/dev/null":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_text)
        else:
            original = path.read_text()
            old_text = "\n".join(old_lines) + ("\n" if old_lines else "")
            if old_text and old_text not in original:
                raise ValueError(f"old block not found in {new_path}")
            path.write_text(original.replace(old_text, new_text, 1) if old_text else new_text)
        changed.append(new_path)
    return changed


def _clean_unified_path(path_text: str) -> str:
    if path_text.startswith("b/") or path_text.startswith("a/"):
        return path_text[2:]
    return path_text
