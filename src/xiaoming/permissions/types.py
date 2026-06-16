from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PermissionBehavior(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


class PermissionMode(str, Enum):
    DEFAULT = "default"
    PLAN = "plan"
    ACCEPT_EDITS = "accept_edits"
    AUTO = "auto"
    BYPASS = "bypass"


@dataclass(frozen=True)
class PermissionRule:
    behavior: PermissionBehavior
    tool: str
    pattern: str
    source: str = "session"


@dataclass(frozen=True)
class PermissionDecision:
    behavior: PermissionBehavior
    reason: str = ""
    matched_rule: PermissionRule | None = None

    @property
    def allowed(self) -> bool:
        return self.behavior == PermissionBehavior.ALLOW
