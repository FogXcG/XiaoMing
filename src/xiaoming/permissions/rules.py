from __future__ import annotations

from fnmatch import fnmatchcase

from xiaoming.permissions.types import PermissionBehavior, PermissionDecision, PermissionRule


_SEVERITY = {
    PermissionBehavior.ALLOW: 0,
    PermissionBehavior.ASK: 1,
    PermissionBehavior.DENY: 2,
}


def match_rules(rules: list[PermissionRule], tool: str, value: str) -> PermissionDecision | None:
    matches = [rule for rule in rules if rule.tool == tool and _matches(rule.pattern, value)]
    if not matches:
        return None
    rule = max(matches, key=lambda item: _SEVERITY[item.behavior])
    return PermissionDecision(rule.behavior, reason=f"matched {rule.source} rule {rule.tool}({rule.pattern})", matched_rule=rule)


def _matches(pattern: str, value: str) -> bool:
    normalized_pattern = " ".join(pattern.strip().split())
    normalized_value = " ".join(value.strip().split())
    if normalized_pattern == normalized_value:
        return True
    return fnmatchcase(normalized_value, normalized_pattern)
