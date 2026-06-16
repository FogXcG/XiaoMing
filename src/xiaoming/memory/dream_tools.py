from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from xiaoming.llm.types import ToolSpec
from xiaoming.memory.models import MemoryDiary, MemoryFragment
from xiaoming.memory.view import build_memory_view
from xiaoming.time_meta import now_iso, runtime_timezone_name
from xiaoming.tools.base import ToolResult
from xiaoming.tools.registry import ToolRegistry


@dataclass
class DreamToolState:
    packets: dict[str, list[MemoryFragment]]
    draft_diaries: list[MemoryDiary] = field(default_factory=list)
    accepted: bool = False
    rejected: bool = False
    decision_reason: str = ""


def dream_tool_registry(state: DreamToolState) -> ToolRegistry:
    return ToolRegistry(
        [
            ListMemoryPacketsTool(state),
            ReadMemoryPacketTool(state),
            WriteDiaryDraftTool(state),
            ReviseDiaryDraftTool(state),
            BuildCandidateMemoryViewTool(state),
            AcceptDreamTool(state),
            RejectDreamTool(state),
        ]
    )


def _spec(tool) -> ToolSpec:
    return ToolSpec(tool.name, tool.description, tool.input_schema)


class ListMemoryPacketsTool:
    name = "list_memory_packets"
    description = "List memory source packets available in dream mode."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, state: DreamToolState):
        self.state = state

    @property
    def spec(self) -> ToolSpec:
        return _spec(self)

    def run(self, args: dict[str, Any]) -> ToolResult:
        lines = []
        for packet_id, fragments in self.state.packets.items():
            tokens = sum(fragment.token_estimate for fragment in fragments)
            lines.append(f"{packet_id}: fragments={len(fragments)} tokens={tokens}")
        return ToolResult(self.name, "success", output="\n".join(lines))


class ReadMemoryPacketTool:
    name = "read_memory_packet"
    description = "Read one memory source packet by id."
    input_schema = {
        "type": "object",
        "properties": {"packet_id": {"type": "string"}},
        "required": ["packet_id"],
        "additionalProperties": False,
    }

    def __init__(self, state: DreamToolState):
        self.state = state

    @property
    def spec(self) -> ToolSpec:
        return _spec(self)

    def run(self, args: dict[str, Any]) -> ToolResult:
        packet_id = str(args.get("packet_id") or "")
        fragments = self.state.packets.get(packet_id)
        if fragments is None:
            return ToolResult(self.name, "error", error=f"unknown packet: {packet_id}")
        output = "\n\n".join(
            f'<fragment id="{fragment.id}" time="{fragment.created_at}" kind="{fragment.role_or_type}">\n{fragment.content}\n</fragment>'
            for fragment in fragments
        )
        return ToolResult(self.name, "success", output=output)


class WriteDiaryDraftTool:
    name = "write_diary_draft"
    description = "Write a first-person draft diary covering source memory fragments."
    input_schema = {
        "type": "object",
        "properties": {
            "scope": {"type": "string"},
            "start_time": {"type": "string"},
            "end_time": {"type": "string"},
            "body": {"type": "string"},
            "source_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["scope", "start_time", "end_time", "body", "source_ids"],
        "additionalProperties": False,
    }

    def __init__(self, state: DreamToolState):
        self.state = state

    @property
    def spec(self) -> ToolSpec:
        return _spec(self)

    def run(self, args: dict[str, Any]) -> ToolResult:
        scope = str(args.get("scope") or "day")
        if scope not in {"day", "week", "month", "year"}:
            return ToolResult(self.name, "error", error=f"invalid diary scope: {scope}")
        diary = MemoryDiary(
            id=f"diary-{uuid4()}",
            scope=scope,
            start_time=str(args.get("start_time") or ""),
            end_time=str(args.get("end_time") or ""),
            timezone=runtime_timezone_name(),
            status="draft",
            source_fragment_ids=[str(item) for item in args.get("source_ids") or []],
            body=str(args.get("body") or ""),
            created_at=now_iso(),
        )
        self.state.draft_diaries.append(diary)
        return ToolResult(self.name, "success", output=f"draft diary created: {diary.id}")


class ReviseDiaryDraftTool:
    name = "revise_diary_draft"
    description = "Revise an existing draft diary."
    input_schema = {
        "type": "object",
        "properties": {
            "diary_id": {"type": "string"},
            "body": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["diary_id", "body", "reason"],
        "additionalProperties": False,
    }

    def __init__(self, state: DreamToolState):
        self.state = state

    @property
    def spec(self) -> ToolSpec:
        return _spec(self)

    def run(self, args: dict[str, Any]) -> ToolResult:
        diary_id = str(args.get("diary_id") or "")
        body = str(args.get("body") or "")
        for index, diary in enumerate(self.state.draft_diaries):
            if diary.id != diary_id:
                continue
            self.state.draft_diaries[index] = MemoryDiary(
                id=diary.id,
                scope=diary.scope,
                start_time=diary.start_time,
                end_time=diary.end_time,
                timezone=diary.timezone,
                status=diary.status,
                source_fragment_ids=list(diary.source_fragment_ids),
                supersedes_diary_ids=list(diary.supersedes_diary_ids),
                body=body,
                created_at=diary.created_at,
                accepted_at=diary.accepted_at,
            )
            return ToolResult(self.name, "success", output=f"draft diary revised: {diary_id}")
        return ToolResult(self.name, "error", error=f"unknown draft diary: {diary_id}")


class BuildCandidateMemoryViewTool:
    name = "build_candidate_memory_view"
    description = "Render the candidate memory view from current draft diaries."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, state: DreamToolState):
        self.state = state

    @property
    def spec(self) -> ToolSpec:
        return _spec(self)

    def run(self, args: dict[str, Any]) -> ToolResult:
        active_like = [
            MemoryDiary(
                id=diary.id,
                scope=diary.scope,
                start_time=diary.start_time,
                end_time=diary.end_time,
                timezone=diary.timezone,
                status="active",
                source_fragment_ids=list(diary.source_fragment_ids),
                supersedes_diary_ids=list(diary.supersedes_diary_ids),
                body=diary.body,
                created_at=diary.created_at,
                accepted_at=diary.accepted_at,
            )
            for diary in self.state.draft_diaries
        ]
        items = build_memory_view(active_like, recent_items=[])
        output = "\n\n".join(str(item.get("content") or "") for item in items)
        return ToolResult(self.name, "success", output=output)


class AcceptDreamTool:
    name = "accept_dream"
    description = "Accept the dream after checking the candidate memory view."
    input_schema = {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
        "additionalProperties": False,
    }

    def __init__(self, state: DreamToolState):
        self.state = state

    @property
    def spec(self) -> ToolSpec:
        return _spec(self)

    def run(self, args: dict[str, Any]) -> ToolResult:
        self.state.accepted = True
        self.state.rejected = False
        self.state.decision_reason = str(args.get("reason") or "")
        return ToolResult(self.name, "success", output="dream accepted")


class RejectDreamTool:
    name = "reject_dream"
    description = "Reject the dream and leave working memory unchanged."
    input_schema = {
        "type": "object",
        "properties": {"reason": {"type": "string"}},
        "required": ["reason"],
        "additionalProperties": False,
    }

    def __init__(self, state: DreamToolState):
        self.state = state

    @property
    def spec(self) -> ToolSpec:
        return _spec(self)

    def run(self, args: dict[str, Any]) -> ToolResult:
        self.state.accepted = False
        self.state.rejected = True
        self.state.decision_reason = str(args.get("reason") or "")
        return ToolResult(self.name, "success", output="dream rejected")
