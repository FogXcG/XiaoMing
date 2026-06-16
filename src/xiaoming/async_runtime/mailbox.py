from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


MailboxKind = Literal["approval_request", "clarification_request", "decision_request", "progress", "result", "review"]
MailboxStatus = Literal["pending", "answered", "cancelled", "dismissed"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MailboxMessage:
    task_id: str
    from_role: str
    to_role: str
    kind: MailboxKind
    content: str
    requires_reply: bool
    worker_id: str | None = None
    status: MailboxStatus = "pending"
    message_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    reply: str | None = None
    presented_count: int = 0
    last_presented_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.message_id

    def answer(self, reply: str) -> None:
        if self.status != "pending":
            return
        self.status = "answered"
        self.reply = reply
        self.updated_at = _now()

    def cancel(self) -> None:
        if self.status != "pending":
            return
        self.status = "cancelled"
        self.updated_at = _now()

    def dismiss(self) -> None:
        if self.status != "pending":
            return
        self.status = "dismissed"
        self.updated_at = _now()

    def mark_presented(self, at: str | None = None) -> None:
        self.presented_count += 1
        self.last_presented_at = at or _now()
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "task_id": self.task_id,
            "worker_id": self.worker_id,
            "from_role": self.from_role,
            "to_role": self.to_role,
            "kind": self.kind,
            "content": self.content,
            "requires_reply": self.requires_reply,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reply": self.reply,
            "presented_count": self.presented_count,
            "last_presented_at": self.last_presented_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MailboxMessage":
        return cls(
            task_id=str(data.get("task_id") or ""),
            worker_id=str(data.get("worker_id")) if data.get("worker_id") is not None else None,
            from_role=str(data.get("from_role") or ""),
            to_role=str(data.get("to_role") or ""),
            kind=_mailbox_kind(data.get("kind")),
            content=str(data.get("content") or ""),
            requires_reply=bool(data.get("requires_reply")),
            status=_mailbox_status(data.get("status")),
            message_id=str(data.get("message_id") or uuid4()),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
            reply=str(data.get("reply")) if data.get("reply") is not None else None,
            presented_count=int(data.get("presented_count") or 0),
            last_presented_at=str(data.get("last_presented_at")) if data.get("last_presented_at") is not None else None,
            metadata=dict(data.get("metadata") or {}) if isinstance(data.get("metadata"), dict) else {},
        )


@dataclass(frozen=True)
class MailboxTaskUpdate:
    message_id: str
    task_id: str
    task_title: str
    kind: MailboxKind
    content: str
    status: MailboxStatus
    created_at: str


@dataclass(frozen=True)
class MailboxNoticeCandidate:
    message_id: str
    task_id: str
    task_title: str
    kind: MailboxKind
    text: str
    requires_reply: bool
    created_at: str
    presented_count: int


class MailboxStore:
    def __init__(self) -> None:
        self._messages: dict[str, MailboxMessage] = {}
        self._order: list[str] = []

    def create_message(
        self,
        *,
        task_id: str,
        from_role: str,
        to_role: str,
        kind: MailboxKind,
        content: str,
        requires_reply: bool,
        worker_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MailboxMessage:
        message = MailboxMessage(
            task_id=task_id,
            from_role=from_role,
            to_role=to_role,
            kind=kind,
            content=content,
            requires_reply=requires_reply,
            worker_id=worker_id,
            metadata=dict(metadata or {}),
        )
        self._messages[message.message_id] = message
        self._order.append(message.message_id)
        return message

    def get(self, message_id: str) -> MailboxMessage | None:
        return self._messages.get(message_id)

    def list(self) -> list[MailboxMessage]:
        return [self._messages[message_id] for message_id in self._order if message_id in self._messages]

    def snapshot(self) -> list[dict[str, Any]]:
        return [message.to_dict() for message in self.list()]

    @classmethod
    def from_snapshot(cls, snapshot: object) -> "MailboxStore":
        store = cls()
        if not isinstance(snapshot, list):
            return store
        for item in snapshot:
            if not isinstance(item, dict):
                continue
            try:
                message = MailboxMessage.from_dict(item)
            except Exception:
                continue
            if not message.task_id or not message.message_id:
                continue
            store._messages[message.message_id] = message
            store._order.append(message.message_id)
        return store

    def pending_for_main(self) -> list[MailboxMessage]:
        return [
            message
            for message in self.list()
            if message.status == "pending" and message.to_role == "main" and message.requires_reply
        ]

    def pending_reply_messages(self) -> list[MailboxMessage]:
        return self.pending_for_main()

    def pending_reply_lines(self, registry: Any | None = None) -> list[str]:
        return [_format_pending_reply_line(message, registry) for message in self.pending_reply_messages()]

    def pending_reply_digest(self, registry: Any | None = None) -> str | None:
        lines = self.pending_reply_lines(registry)
        if not lines:
            return None
        return "\n".join(lines)

    def latest_task_updates(self, registry: Any | None = None, limit: int | None = None) -> list[MailboxTaskUpdate]:
        task_order: list[str] = []
        latest_by_task: dict[str, MailboxTaskUpdate] = {}
        for message in self.list():
            if message.to_role != "main" or message.requires_reply or message.status != "pending":
                continue
            if message.kind not in {"progress", "result", "review"}:
                continue
            if message.task_id not in latest_by_task:
                task_order.append(message.task_id)
            latest_by_task[message.task_id] = MailboxTaskUpdate(
                message_id=message.message_id,
                task_id=message.task_id,
                task_title=_task_title(message, registry),
                kind=message.kind,
                content=message.content,
                status=message.status,
                created_at=message.created_at,
            )
        updates = [latest_by_task[task_id] for task_id in task_order if task_id in latest_by_task]
        if limit is not None:
            return updates[: max(limit, 0)]
        return updates

    def notice_candidates(
        self,
        registry: Any | None = None,
        *,
        include_presented: bool = False,
        limit: int | None = None,
    ) -> list[MailboxNoticeCandidate]:
        candidates: list[MailboxNoticeCandidate] = []
        for message in self.pending_reply_messages():
            if message.presented_count > 0 and not include_presented:
                continue
            task_title = _task_title(message, registry)
            candidates.append(
                MailboxNoticeCandidate(
                    message_id=message.message_id,
                    task_id=message.task_id,
                    task_title=task_title,
                    kind=message.kind,
                    text=f"{task_title} 需要你确认：{message.content}",
                    requires_reply=message.requires_reply,
                    created_at=message.created_at,
                    presented_count=message.presented_count,
                )
            )
        if limit is not None:
            return candidates[: max(limit, 0)]
        return candidates

    def messages_for_task(self, task_id: str) -> list[MailboxMessage]:
        return [message for message in self.list() if message.task_id == task_id]

    def reply(self, message_id: str, reply: str) -> MailboxMessage | None:
        message = self.get(message_id)
        if message is None:
            return None
        message.answer(reply)
        return message

    def mark_presented(self, message_id: str, at: str | None = None) -> MailboxMessage | None:
        message = self.get(message_id)
        if message is None:
            return None
        message.mark_presented(at=at)
        return message

    def cancel_task_messages(self, task_id: str) -> list[MailboxMessage]:
        cancelled: list[MailboxMessage] = []
        for message in self.list():
            if message.task_id != task_id:
                continue
            if message.status != "pending":
                continue
            message.cancel()
            cancelled.append(message)
        return cancelled


def _mailbox_kind(value: object) -> MailboxKind:
    text = str(value or "")
    if text in {"approval_request", "clarification_request", "decision_request", "progress", "result", "review"}:
        return text  # type: ignore[return-value]
    return "progress"


def _mailbox_status(value: object) -> MailboxStatus:
    text = str(value or "")
    if text in {"pending", "answered", "cancelled", "dismissed"}:
        return text  # type: ignore[return-value]
    return "pending"


def _format_pending_reply_line(message: MailboxMessage, registry: Any | None = None) -> str:
    request_id = str(message.metadata.get("request_id") or message.message_id)
    detail = f"- message_id: {message.message_id}; question_id: {request_id}; task: {_task_title(message, registry)}; kind: {message.kind}; prompt: {message.content}"
    purpose = _metadata_string(message.metadata.get("purpose"))
    context = _metadata_string(message.metadata.get("context"))
    options = _metadata_string_list(message.metadata.get("options"))
    if purpose:
        detail += f"; purpose: {purpose}"
    if context:
        detail += f"; context: {context}"
    if options:
        detail += "; options: " + " | ".join(options)
    return detail


def _task_title(message: MailboxMessage, registry: Any | None = None) -> str:
    title = _metadata_string(message.metadata.get("task_title"))
    if title:
        return title
    task = _registry_get(registry, message.task_id)
    task_title = _metadata_string(getattr(task, "title", ""))
    if task_title:
        return task_title
    return "后台任务"


def _registry_get(registry: Any | None, task_id: str) -> Any | None:
    if registry is None or not hasattr(registry, "get"):
        return None
    try:
        return registry.get(task_id)
    except Exception:
        return None


def _metadata_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _metadata_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [text for item in value if (text := str(item).strip())]
