from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ErrorKind = Literal[
    "context_window_exceeded",
    "usage_limit_exceeded",
    "rate_limited",
    "server_overloaded",
    "http_connection_failed",
    "stream_disconnected",
    "stream_timeout",
    "bad_request",
    "unauthorized",
    "tool_call_parse_error",
    "tool_execution_error",
    "turn_aborted",
    "worker_failed",
    "unknown",
]


@dataclass(frozen=True)
class AgentErrorInfo:
    kind: ErrorKind
    message: str
    retryable: bool
    affects_turn_status: bool = True
    details: dict[str, Any] = field(default_factory=dict)


class ProviderCallError(RuntimeError):
    def __init__(self, error_info: AgentErrorInfo):
        super().__init__(error_info.message)
        self.error_info = error_info


@dataclass(frozen=True)
class RecoverableToolError:
    call_id: str
    tool_name: str
    message: str
    retry_hint: str = ""


@dataclass(frozen=True)
class FatalTurnError:
    source: str
    message: str
    hint: str = ""


def format_recoverable_tool_error(error: RecoverableToolError) -> str:
    parts = [
        f"Tool: {error.tool_name}",
        "Status: error",
        "Error:",
        error.message,
    ]
    if error.retry_hint:
        parts.extend(["Recovery hint:", error.retry_hint])
    return "\n".join(parts)


def format_fatal_turn_error(error: FatalTurnError) -> str:
    text = f"{error.source}: {error.message}"
    if error.hint:
        text += f"\nHint: {error.hint}"
    return text
