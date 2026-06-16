from pathlib import Path

from xiaoming.context.formatting import format_tool_result
from xiaoming.context.truncation import truncate_middle
from xiaoming.policy.paths import resolve_workspace_path
from xiaoming.tools.base import ToolResult


def test_resolve_workspace_path_allows_child(tmp_path: Path):
    child = tmp_path / "src" / "app.py"
    child.parent.mkdir()
    child.write_text("print('ok')\n")

    assert resolve_workspace_path(tmp_path, "src/app.py") == child.resolve()


def test_resolve_workspace_path_rejects_escape(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"

    try:
        resolve_workspace_path(tmp_path, f"../{outside.name}")
    except ValueError as exc:
        assert "outside workspace" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_truncate_middle_keeps_short_text():
    assert truncate_middle("abc", limit=10) == "abc"


def test_truncate_middle_keeps_start_and_end():
    text = "0123456789abcdefghijklmnopqrstuvwxyz"

    assert truncate_middle(text, limit=27) == "0123\n... truncated ...\nwxyz"


def test_format_tool_result_includes_status_and_output():
    result = ToolResult(tool="read_file", status="success", output="hello")

    assert format_tool_result(result) == "Tool: read_file\nStatus: success\nOutput:\nhello"
