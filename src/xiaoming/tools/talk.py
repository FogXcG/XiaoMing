from __future__ import annotations

from typing import Callable

from xiaoming.llm.types import ToolSpec
from xiaoming.tools.base import ToolResult


TalkCallback = Callable[[str, str, str, list[str]], str]


class TalkTool:
    name = "talk"
    description = (
        "Ask the coordinator a blocking question needed to continue the background task. "
        "Use this for requirement clarification or decisions that need an answer before continuing. "
        "Do not use this for progress updates, permission approval, final status, or ordinary notifications."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "purpose": {"type": "string", "enum": ["clarify", "decision"]},
            "message": {"type": "string"},
            "context": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["purpose", "message"],
        "additionalProperties": False,
    }

    def __init__(self, callback: TalkCallback):
        self.callback = callback

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict) -> ToolResult:
        purpose = str(args.get("purpose") or "")
        if purpose not in {"clarify", "decision"}:
            return ToolResult(self.name, "error", error=f"invalid talk purpose: {purpose}")
        message = str(args.get("message") or "").strip()
        if not message:
            return ToolResult(self.name, "error", error="talk message is required")
        context = str(args.get("context") or "")
        options = [str(option) for option in args.get("options") or []]
        answer = self.callback(purpose, message, context, options)
        return ToolResult(self.name, "success", output=answer)
