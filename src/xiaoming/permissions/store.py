from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from xiaoming.permissions.types import PermissionBehavior, PermissionRule


def permissions_path(workspace: Path) -> Path:
    return workspace / ".xiaoming" / "permissions.json"


def load_project_rules(workspace: Path) -> list[PermissionRule]:
    path = permissions_path(workspace)
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    rules = data.get("rules", [])
    loaded: list[PermissionRule] = []
    for item in rules:
        loaded.append(
            PermissionRule(
                behavior=PermissionBehavior(item["behavior"]),
                tool=item["tool"],
                pattern=item["pattern"],
                source=item.get("source") or "project",
            )
        )
    return loaded


def save_project_rules(workspace: Path, rules: list[PermissionRule]) -> None:
    path = permissions_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"rules": [_rule_to_json(rule) for rule in rules]}
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def add_project_rule(workspace: Path, rule: PermissionRule) -> list[PermissionRule]:
    rules = [existing for existing in load_project_rules(workspace) if not _same_rule(existing, rule)]
    rules.append(rule)
    save_project_rules(workspace, rules)
    return rules


def _same_rule(left: PermissionRule, right: PermissionRule) -> bool:
    return left.tool == right.tool and left.pattern == right.pattern and left.behavior == right.behavior


def _rule_to_json(rule: PermissionRule) -> dict[str, str]:
    data = asdict(rule)
    data["behavior"] = rule.behavior.value
    return data
