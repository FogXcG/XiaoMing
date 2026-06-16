from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from xiaoming.llm.types import ToolSpec


@dataclass(frozen=True)
class ToolResult:
    tool: str
    status: str
    output: str = ""
    error: str = ""

    def to_text(
        self,
        workspace: Path | None = None,
        max_inline_chars: int = 30000,
        max_saved_chars: int = 5_000_000,
    ) -> str:
        from xiaoming.context.formatting import format_tool_result

        return format_tool_result(self, workspace=workspace, max_inline_chars=max_inline_chars, max_saved_chars=max_saved_chars)


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    supports_parallel_tool_calls: bool

    @property
    def spec(self) -> ToolSpec:
        ...

    def run(self, args: dict[str, Any]) -> ToolResult:
        ...
