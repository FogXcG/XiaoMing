from __future__ import annotations

from dataclasses import dataclass, field
import json
from json import JSONDecodeError
from typing import Any

from xiaoming.agent_errors import FatalTurnError, RecoverableToolError
from xiaoming.llm.types import LLMResponse, TokenUsage, ToolCall


@dataclass(frozen=True)
class StreamTextDelta:
    text: str


@dataclass(frozen=True)
class StreamToolCallDelta:
    index: int
    call_id: str | None = None
    name: str | None = None
    arguments_delta: str = ""


@dataclass(frozen=True)
class StreamDone:
    finish_reason: str | None = None


@dataclass(frozen=True)
class StreamUsage:
    usage: TokenUsage


@dataclass(frozen=True)
class StreamError:
    message: str


ModelStreamEvent = StreamTextDelta | StreamToolCallDelta | StreamDone | StreamUsage | StreamError


@dataclass
class ToolBuffer:
    call_id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class StreamAccumulator:
    source: str
    text_parts: list[str] = field(default_factory=list)
    tool_buffers: dict[int, ToolBuffer] = field(default_factory=dict)
    usage: TokenUsage | None = None

    def add_text(self, text: str) -> None:
        self.text_parts.append(text)

    def add_tool_delta(self, delta: StreamToolCallDelta) -> None:
        buffer = self.tool_buffers.setdefault(delta.index, ToolBuffer())
        if delta.call_id:
            buffer.call_id = delta.call_id
        if delta.name:
            buffer.name = delta.name
        if delta.arguments_delta:
            buffer.arguments += delta.arguments_delta

    def set_usage(self, usage: TokenUsage) -> None:
        self.usage = usage

    @property
    def text(self) -> str:
        return "".join(self.text_parts)

    def to_response(self, finish_reason: str | None = None) -> LLMResponse:
        tool_calls: list[ToolCall] = []
        recoverable_errors: list[RecoverableToolError] = []
        fatal_error: FatalTurnError | None = None
        output_tool_calls: list[dict[str, Any]] = []
        for index in sorted(self.tool_buffers):
            buffer = self.tool_buffers[index]
            if not buffer.call_id or not buffer.name:
                fatal_error = FatalTurnError(
                    source=self.source,
                    message="model returned an incomplete streamed tool call without a call id or function name",
                    hint="Retry the request; if it repeats, switch to non-streaming mode or split the task.",
                )
                continue
            output_tool_calls.append(
                {
                    "id": buffer.call_id,
                    "type": "function",
                    "function": {"name": buffer.name, "arguments": buffer.arguments or "{}"},
                }
            )
            try:
                args = json.loads(buffer.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must decode to an object")
            except JSONDecodeError as exc:
                fatal_error = FatalTurnError(
                    source=self.source,
                    message=f"streamed tool arguments were incomplete or invalid for {buffer.name}: {exc}",
                    hint="Retry with streaming disabled using /stream off, or ask the model to split large writes into write_file plus append_file chunks.",
                )
                continue
            except ValueError as exc:
                recoverable_errors.append(
                    RecoverableToolError(
                        call_id=buffer.call_id,
                        tool_name=buffer.name,
                        message=f"failed to parse streamed tool arguments: {exc}",
                        retry_hint="Retry with a smaller valid JSON tool call, or split the change into smaller chunks.",
                    )
                )
                continue
            tool_calls.append(ToolCall(id=buffer.call_id, name=buffer.name, args=args))
        output_item: dict[str, Any] = {"role": "assistant", "content": self.text or None}
        if output_tool_calls:
            output_item["tool_calls"] = output_tool_calls
        return LLMResponse(
            message=self.text or None,
            tool_calls=tool_calls,
            output_items=[output_item],
            raw=None,
            finish_reason=finish_reason,
            usage=self.usage,
            recoverable_errors=recoverable_errors,
            fatal_error=fatal_error,
        )
