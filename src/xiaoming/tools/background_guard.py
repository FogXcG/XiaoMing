from __future__ import annotations

from typing import Any, Callable

from xiaoming.llm.types import ToolSpec
from xiaoming.tools.base import Tool, ToolResult


class BackgroundGuardedTool:
    def __init__(self, tool: Tool, coordinator_getter: Callable[[], Any]):
        self.tool = tool
        self.coordinator_getter = coordinator_getter
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.input_schema

    @property
    def spec(self) -> ToolSpec:
        return self.tool.spec

    def run(self, args: dict[str, Any]) -> ToolResult:
        coordinator = self.coordinator_getter()
        if coordinator is not None and coordinator.has_active_tasks():
            return ToolResult(
                self.name,
                "error",
                error="background task is still active; use background_tasks_status and do not inspect files to decide whether it is complete",
            )
        return self.tool.run(args)
