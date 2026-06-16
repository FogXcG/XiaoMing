from pathlib import Path

from xiaoming.permissions.store import add_project_rule, load_project_rules
from xiaoming.permissions.types import PermissionBehavior, PermissionRule


def test_add_project_rule_persists_permissions(tmp_path: Path):
    add_project_rule(tmp_path, PermissionRule(PermissionBehavior.ALLOW, "Bash", "pytest *", "project"))

    rules = load_project_rules(tmp_path)

    assert rules == [PermissionRule(PermissionBehavior.ALLOW, "Bash", "pytest *", "project")]
    assert (tmp_path / ".xiaoming" / "permissions.json").exists()
