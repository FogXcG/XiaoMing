from pathlib import Path

from xiaoming.policy.shell_policy import ShellDecision, decide_shell
from xiaoming.tools.shell import ShellTool, summarize_shell_for_approval


def test_exact_whitelist_command_is_allowed():
    assert decide_shell("git status", approval_mode="full_auto") == ShellDecision.ALLOW


def test_control_syntax_is_rejected():
    assert decide_shell("pytest; rm -rf .", approval_mode="full_auto") == ShellDecision.REJECT


def test_destructive_command_is_rejected():
    assert decide_shell("rm -rf .", approval_mode="suggest") == ShellDecision.REJECT


def test_auto_mode_allows_targeted_pytest_command():
    assert decide_shell("pytest tests/test_config.py", approval_mode="full_auto") == ShellDecision.ALLOW


def test_shell_tool_returns_denied_when_approval_callback_declines(tmp_path: Path):
    approvals = []
    tool = ShellTool(tmp_path, approval_mode="suggest", approve=lambda command: approvals.append(command) or False)

    result = tool.run({"command": "npm install"})

    assert result.status == "denied"
    assert "denied" in result.error
    assert approvals[0] == "Tool: shell\nCommand:\nnpm install"


def test_summarize_shell_for_approval_shows_command():
    assert summarize_shell_for_approval("ls -la") == "Tool: shell\nCommand:\nls -la"
