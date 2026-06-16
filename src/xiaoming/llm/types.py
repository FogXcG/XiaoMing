from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xiaoming.agent_errors import FatalTurnError, RecoverableToolError


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    input_mode: str = "json"
    freeform_arg: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]
    output_type: str = "function_call_output"


@dataclass(frozen=True)
class LLMRequest:
    instructions: str
    input_items: list[dict[str, Any]]
    tools: list[ToolSpec]
    model: str
    temperature: float
    max_output_tokens: int


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    reasoning_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_tokens",
            "cache_hit_tokens",
            "cache_miss_tokens",
            "reasoning_tokens",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.raw:
            data["raw"] = self.raw
        return data


@dataclass(frozen=True)
class LLMResponse:
    message: str | None
    tool_calls: list[ToolCall]
    output_items: list[dict[str, Any]]
    raw: Any
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    recoverable_errors: list[RecoverableToolError] = field(default_factory=list)
    fatal_error: FatalTurnError | None = None
