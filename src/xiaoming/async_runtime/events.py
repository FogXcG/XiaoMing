from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4


QuestionKind = Literal["approval_request", "clarification_request", "decision_request"]
WorkerEventKind = Literal[
    "started",
    "heartbeat",
    "progress",
    "assistant_delta",
    "tool_started",
    "tool_finished",
    "approval_request",
    "clarification_request",
    "decision_request",
    "reported",
    "completed",
    "peer_reply",
    "failed",
    "cancelled",
]


@dataclass(frozen=True)
class UserMessage:
    content: str
    message_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class CoordinatorNotice:
    message: str
    task_id: str | None = None


@dataclass(frozen=True)
class Question:
    task_id: str
    kind: QuestionKind
    prompt: str
    request_id: str = field(default_factory=lambda: str(uuid4()))
    purpose: str = ""
    context: str = ""
    options: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WorkerEvent:
    task_id: str
    kind: WorkerEventKind
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "WorkerEvent":
        return cls(
            task_id=str(payload["task_id"]),
            kind=payload["kind"],
            message=str(payload.get("message") or ""),
            data=dict(payload.get("data") or {}),
        )

    def to_json(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "kind": self.kind, "message": self.message, "data": self.data}
