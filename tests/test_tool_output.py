from pathlib import Path

from xiaoming.context.formatting import format_tool_result
from xiaoming.tools.base import ToolResult


def test_format_tool_result_spills_large_output_to_workspace_file(tmp_path: Path):
    result = ToolResult("shell", "success", output="a" * 80)

    text = format_tool_result(result, workspace=tmp_path, max_inline_chars=20, max_saved_chars=50)

    assert "Output too large" in text
    assert ".xiaoming/tool-outputs/" in text
    saved_files = list((tmp_path / ".xiaoming" / "tool-outputs").iterdir())
    assert len(saved_files) == 1
    assert saved_files[0].read_text() == "a" * 50
    assert "saved output truncated to 50 chars" in text


def test_tool_result_to_text_accepts_workspace_for_output_spill(tmp_path: Path):
    result = ToolResult("dummy", "success", output="b" * 40)

    text = result.to_text(workspace=tmp_path, max_inline_chars=10, max_saved_chars=40)

    assert "Tool: dummy" in text
    assert "Output too large" in text
