from __future__ import annotations

from enum import Enum
from pathlib import Path

from xiaoming.permissions.engine import PermissionEngine
from xiaoming.permissions.types import PermissionBehavior, PermissionMode


class ShellDecision(str, Enum):
    ALLOW = "allow"
    APPROVE = "approve"
    REJECT = "reject"


EXACT_ALLOWLIST = {
    "git status",
    "git diff",
    "pytest",
    "python -m pytest",
    "npm test",
    "npm run test",
    "npm run lint",
}

CONTROL_TOKENS = (";", "&&", "||", "|", ">", "<", "$(", "`")
REJECT_PREFIXES = ("rm -rf", "git reset --hard", "git checkout --", "sudo ")


def decide_shell(command: str, approval_mode: str) -> ShellDecision:
    if approval_mode == "full_auto" and " ".join(command.strip().split()) in EXACT_ALLOWLIST:
        return ShellDecision.ALLOW
    mode = approval_mode_to_permission_mode(approval_mode)
    decision = PermissionEngine(Path.cwd(), mode=mode).decide_shell(command)
    if decision.behavior == PermissionBehavior.ALLOW:
        return ShellDecision.ALLOW
    if decision.behavior == PermissionBehavior.DENY:
        return ShellDecision.REJECT
    return ShellDecision.APPROVE


def approval_mode_to_permission_mode(approval_mode: str) -> PermissionMode:
    if approval_mode == "auto_edit":
        return PermissionMode.ACCEPT_EDITS
    if approval_mode == "full_auto":
        return PermissionMode.AUTO
    if approval_mode in {"default", "plan", "accept_edits", "auto", "bypass"}:
        return PermissionMode(approval_mode)
    return PermissionMode.DEFAULT
    return ShellDecision.APPROVE
