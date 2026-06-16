from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from xiaoming.time_meta import now_iso, time_metadata


PromptRole = Literal["developer", "user", "assistant", "tool"]


@dataclass
class PromptItem:
    role: PromptRole
    content: str
    kind: str
    id: str = field(default_factory=lambda: str(uuid4()))
    turn_id: str | None = None
    durable: bool = True
    consumed: bool = False
    created_at: str = field(default_factory=now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def user(cls, turn_id: str, content: str) -> "PromptItem":
        return cls(role="user", content=content, kind="user_message", turn_id=turn_id, durable=True)

    def to_input_item(self) -> dict[str, Any]:
        if self.role == "developer":
            return {
                "role": "developer",
                "content": self.content,
                "xiaoming": self.to_event_payload(),
            }
        return {"role": self.role, "content": self.content, "xiaoming": self.to_event_payload()}

    def to_event_payload(self) -> dict[str, Any]:
        time_meta = time_metadata(self.created_at)
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
            "durable": self.durable,
            "consumed": self.consumed,
            **time_meta,
            "metadata": self.metadata,
        }

    @classmethod
    def from_event_payload(cls, payload: dict[str, Any]) -> "PromptItem":
        return cls(
            id=str(payload.get("id") or uuid4()),
            turn_id=payload.get("turn_id"),
            role=payload.get("role") if payload.get("role") in {"developer", "user", "assistant", "tool"} else "user",
            kind=str(payload.get("kind") or "message"),
            content=str(payload.get("content") or ""),
            durable=bool(payload.get("durable", True)),
            consumed=bool(payload.get("consumed", False)),
            created_at=str(payload.get("created_at") or now_iso()),
            metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        )


@dataclass
class TurnContext:
    cwd: str
    current_date: str
    provider: str
    model: str
    stream: bool
    permission_mode: str
    approval_policy: str | None = None
    project_instructions_hash: str | None = None
    project_instructions_text: str | None = None
    skills_summary_hash: str | None = None
    skills_summary_text: str | None = None
    plugins_summary_hash: str | None = None
    plugins_summary_text: str | None = None
    session_id: str | None = None
    resumed: bool = False
    checkpoint_id: str | None = None
    last_error_id: str | None = None
    interrupted_turn_id: str | None = None
    pending_worker_questions_text: str | None = None
    background_tasks_text: str | None = None

    def durable_snapshot(self) -> dict[str, Any]:
        return {
            "cwd": self.cwd,
            "provider": self.provider,
            "model": self.model,
            "stream": self.stream,
            "permission_mode": self.permission_mode,
            "approval_policy": self.approval_policy,
            "project_instructions_hash": self.project_instructions_hash,
            "skills_summary_hash": self.skills_summary_hash,
            "plugins_summary_hash": self.plugins_summary_hash,
            "session_id": self.session_id,
            "resumed": self.resumed,
        }

    def to_event_payload(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_event_payload(cls, payload: dict[str, Any]) -> "TurnContext":
        return cls(
            cwd=str(payload.get("cwd") or ""),
            current_date=str(payload.get("current_date") or ""),
            provider=str(payload.get("provider") or ""),
            model=str(payload.get("model") or ""),
            stream=bool(payload.get("stream", True)),
            permission_mode=str(payload.get("permission_mode") or "default"),
            approval_policy=payload.get("approval_policy"),
            project_instructions_hash=payload.get("project_instructions_hash"),
            project_instructions_text=payload.get("project_instructions_text"),
            skills_summary_hash=payload.get("skills_summary_hash"),
            skills_summary_text=payload.get("skills_summary_text"),
            plugins_summary_hash=payload.get("plugins_summary_hash"),
            plugins_summary_text=payload.get("plugins_summary_text"),
            session_id=payload.get("session_id"),
            resumed=bool(payload.get("resumed", False)),
            checkpoint_id=payload.get("checkpoint_id"),
            last_error_id=payload.get("last_error_id"),
            interrupted_turn_id=payload.get("interrupted_turn_id"),
            pending_worker_questions_text=payload.get("pending_worker_questions_text"),
            background_tasks_text=payload.get("background_tasks_text"),
        )


@dataclass
class PendingTurn:
    turn_id: str
    user_input: str
    staged_items: list[PromptItem]
    turn_context: TurnContext
    status: Literal["staged", "sent", "committed", "interrupted", "failed"] = "staged"
    created_at: str = field(default_factory=now_iso)

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "user_input": self.user_input,
            "staged_items": [item.to_event_payload() for item in self.staged_items],
            "turn_context": self.turn_context.to_event_payload(),
            "status": self.status,
            "created_at": self.created_at,
        }
