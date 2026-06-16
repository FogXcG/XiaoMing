from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.llm.types import ToolSpec
from xiaoming.logging import XiaomingLogger
from xiaoming.tools.base import Tool, ToolResult


class ToolRegistry:
    def __init__(self, tools: list[Tool], logger: XiaomingLogger | None = None):
        self._tools = {tool.name: tool for tool in tools}
        self.logger = logger
        self.workspace: Path | None = next((getattr(tool, "workspace", None) for tool in tools if getattr(tool, "workspace", None) is not None), None)

    def specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, "error", error=f"unknown tool: {name}")
        try:
            return tool.run(args)
        except KeyError as exc:
            if self.logger is not None:
                self.logger.error("tool_missing_argument", tool=name, exc=exc)
            return ToolResult(name, "error", error=f"missing required argument: {exc}")
        except Exception as exc:
            if self.logger is not None:
                self.logger.error("tool_crashed", tool=name, exc=exc)
            return ToolResult(name, "error", error=f"tool crashed: {type(exc).__name__}: {exc}")

    def supports_parallel_tool_calls(self, name: str) -> bool:
        tool = self._tools.get(name)
        return bool(getattr(tool, "supports_parallel_tool_calls", False))

    def format_result(self, result: ToolResult) -> str:
        return result.to_text(workspace=self.workspace)

    def describe_call(self, name: str, args: dict[str, Any]) -> str:
        path = args.get("path")
        if name == "write_file" and path:
            return f"create file {path}"
        if name == "append_file" and path:
            return f"append to {path}"
        if name == "edit_file" and path:
            return f"edit {path}"
        if name == "read_file" and path:
            return f"read {path}"
        if name == "list_files":
            return f"list files under {path or '.'}"
        if name == "search_code":
            query = args.get("query")
            return f"search for {query}" if query else "search code"
        if name == "web_search":
            query = args.get("query")
            return f"search the web for {query}" if query else "search the web"
        if name == "web_fetch":
            url = args.get("url")
            return f"fetch {url}" if url else "fetch web page"
        if name == "apply_patch":
            return "apply a patch"
        if name == "shell":
            command = args.get("cmd") or args.get("command")
            return f"run command: {command}" if command else "run shell command"
        if name == "git_status":
            return "check git status"
        if name == "load_skill":
            skill_name = args.get("name")
            return f"load skill {skill_name}" if skill_name else "load skill instructions"
        if name == "install_skill":
            url = args.get("url")
            return f"install skill from {url}" if url else "install skill"
        if name == "schedule_background_task":
            request = args.get("message") or args.get("request")
            return f"schedule background task: {request}" if request else "schedule background task"
        if name == "background_tasks_status":
            return "check background task status"
        if name == "follow_background_task":
            task_id = args.get("task_id")
            return f"follow background task {task_id}" if task_id else "follow background task"
        if name == "cancel_background_task":
            task_id = args.get("task_id")
            return f"cancel background task {task_id}" if task_id else "cancel background task"
        if name == "reply_mailbox_message":
            message_id = args.get("message_id")
            return f"reply mailbox message {message_id}" if message_id else "reply mailbox message"
        if name == "talk":
            purpose = args.get("purpose")
            return f"ask coordinator for {purpose}" if purpose else "ask coordinator"
        return ""
