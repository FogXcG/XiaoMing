from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal


HookEvent = Literal["SessionStart", "UserPromptSubmit", "PreToolUse", "PostToolUse", "PermissionRequest", "PreCompact", "PostCompact", "Stop"]
HookDecision = Literal["allow", "deny", "ask", "none"]


@dataclass(frozen=True)
class HookResult:
    continue_: bool = True
    decision: HookDecision = "none"
    reason: str = ""
    updated_input: str | None = None
    additional_context: str = ""
    suppress_output: bool = False

    @classmethod
    def from_value(cls, value: "HookResult | dict[str, Any] | None") -> "HookResult":
        if value is None:
            return cls()
        if isinstance(value, HookResult):
            return value
        return cls(
            continue_=bool(value.get("continue", value.get("continue_", True))),
            decision=value.get("decision") if value.get("decision") in {"allow", "deny", "ask", "none"} else "none",
            reason=str(value.get("reason") or value.get("stop_reason") or ""),
            updated_input=str(value["updated_input"]) if value.get("updated_input") is not None else None,
            additional_context=str(value.get("additional_context") or value.get("system_message") or ""),
            suppress_output=bool(value.get("suppress_output", False)),
        )


HookCallback = Callable[[dict[str, Any]], HookResult | dict[str, Any] | None]


class HookManager:
    def __init__(self, hooks: dict[HookEvent, list[HookCallback]] | None = None):
        self._hooks = hooks or {}

    def has_hooks(self) -> bool:
        return any(self._hooks.values())

    def with_prepended(self, event: HookEvent, callback: HookCallback) -> "HookManager":
        hooks = {name: list(callbacks) for name, callbacks in self._hooks.items()}
        hooks.setdefault(event, []).insert(0, callback)
        return HookManager(hooks)

    def run(self, event: HookEvent, payload: dict[str, Any]) -> HookResult:
        merged = HookResult()
        for callback in self._hooks.get(event, []):
            result = HookResult.from_value(callback(payload))
            merged = _merge(merged, result)
            if not merged.continue_ or merged.decision in {"deny", "ask"}:
                break
        return merged


def _merge(left: HookResult, right: HookResult) -> HookResult:
    return HookResult(
        continue_=left.continue_ and right.continue_,
        decision=right.decision if right.decision != "none" else left.decision,
        reason=right.reason or left.reason,
        updated_input=right.updated_input if right.updated_input is not None else left.updated_input,
        additional_context="\n".join(part for part in [left.additional_context, right.additional_context] if part),
        suppress_output=left.suppress_output or right.suppress_output,
    )
