from __future__ import annotations

from pathlib import Path

from xiaoming.context.tool_output import maybe_spill_output
from xiaoming.tools.base import ToolResult


def format_tool_result(
    result: ToolResult,
    workspace: Path | None = None,
    max_inline_chars: int = 30000,
    max_saved_chars: int = 5_000_000,
) -> str:
    if result.status == "success":
        output = maybe_spill_output(result.tool, result.output, workspace, max_inline_chars, max_saved_chars)
        return f"Tool: {result.tool}\nStatus: success\nOutput:\n{output}"
    return f"Tool: {result.tool}\nStatus: {result.status}\nError:\n{result.error}"
