from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


FragmentVisibility = Literal["working", "archived"]
DiaryScope = Literal["day", "week", "month", "year"]
DiaryStatus = Literal["draft", "active", "archived", "rejected"]
DreamStatus = Literal["running", "accepted", "rejected", "failed"]


@dataclass(frozen=True)
class MemoryFragment:
    id: str
    source_event_id: str
    role_or_type: str
    created_at: str
    timezone: str
    token_estimate: int
    content: str
    visibility: FragmentVisibility = "working"
    covered_by_diary_ids: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_event_id": self.source_event_id,
            "role_or_type": self.role_or_type,
            "created_at": self.created_at,
            "timezone": self.timezone,
            "token_estimate": self.token_estimate,
            "content": self.content,
            "visibility": self.visibility,
            "covered_by_diary_ids": list(self.covered_by_diary_ids),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryFragment":
        return cls(
            id=str(payload.get("id") or ""),
            source_event_id=str(payload.get("source_event_id") or ""),
            role_or_type=str(payload.get("role_or_type") or ""),
            created_at=str(payload.get("created_at") or ""),
            timezone=str(payload.get("timezone") or ""),
            token_estimate=int(payload.get("token_estimate") or 0),
            content=str(payload.get("content") or ""),
            visibility="archived" if payload.get("visibility") == "archived" else "working",
            covered_by_diary_ids=[str(item) for item in payload.get("covered_by_diary_ids") or []],
        )


@dataclass(frozen=True)
class MemoryDiary:
    id: str
    scope: DiaryScope
    start_time: str
    end_time: str
    timezone: str
    status: DiaryStatus
    source_fragment_ids: list[str]
    body: str
    created_at: str
    supersedes_diary_ids: list[str] = field(default_factory=list)
    accepted_at: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "scope": self.scope,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "timezone": self.timezone,
            "status": self.status,
            "source_fragment_ids": list(self.source_fragment_ids),
            "supersedes_diary_ids": list(self.supersedes_diary_ids),
            "body": self.body,
            "created_at": self.created_at,
            "accepted_at": self.accepted_at,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "MemoryDiary":
        scope = str(payload.get("scope") or "day")
        status = str(payload.get("status") or "draft")
        return cls(
            id=str(payload.get("id") or ""),
            scope=scope if scope in {"day", "week", "month", "year"} else "day",
            start_time=str(payload.get("start_time") or ""),
            end_time=str(payload.get("end_time") or ""),
            timezone=str(payload.get("timezone") or ""),
            status=status if status in {"draft", "active", "archived", "rejected"} else "draft",
            source_fragment_ids=[str(item) for item in payload.get("source_fragment_ids") or []],
            supersedes_diary_ids=[str(item) for item in payload.get("supersedes_diary_ids") or []],
            body=str(payload.get("body") or ""),
            created_at=str(payload.get("created_at") or ""),
            accepted_at=str(payload.get("accepted_at") or "") or None,
        )


@dataclass(frozen=True)
class DreamRun:
    id: str
    started_at: str
    status: DreamStatus
    ended_at: str | None = None
    snapshot_id: str | None = None
    draft_diary_ids: list[str] = field(default_factory=list)
    reason: str = ""
    tokens_before: int | None = None
    tokens_after: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "status": self.status,
            "snapshot_id": self.snapshot_id,
            "draft_diary_ids": list(self.draft_diary_ids),
            "reason": self.reason,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DreamRun":
        status = str(payload.get("status") or "running")
        return cls(
            id=str(payload.get("id") or ""),
            started_at=str(payload.get("started_at") or ""),
            ended_at=str(payload.get("ended_at") or "") or None,
            status=status if status in {"running", "accepted", "rejected", "failed"} else "failed",
            snapshot_id=str(payload.get("snapshot_id") or "") or None,
            draft_diary_ids=[str(item) for item in payload.get("draft_diary_ids") or []],
            reason=str(payload.get("reason") or ""),
            tokens_before=int(payload["tokens_before"]) if payload.get("tokens_before") is not None else None,
            tokens_after=int(payload["tokens_after"]) if payload.get("tokens_after") is not None else None,
        )
