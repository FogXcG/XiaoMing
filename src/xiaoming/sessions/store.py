from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from xiaoming.context.truncation import truncate_middle
from xiaoming.memory.models import DreamRun, MemoryDiary
from xiaoming.prompting.items import PromptItem, TurnContext
from xiaoming.session import BootstrapContext, LoadedSkill, Session
from xiaoming.time_meta import ensure_time_metadata, now_iso


SESSION_VERSION = 1
RESUMABLE_EVENT_TYPES = {"assistant_message", "assistant_output", "tool_result", "tool_output", "error", "turn_aborted"}
TERMINAL_TURN_EVENT_TYPES = {"assistant_message", "tool_output", "error", "turn_aborted", "turn_failed", "turn_completed"}


@dataclass(frozen=True)
class SessionRecord:
    id: str
    title: str
    created_at: str
    updated_at: str
    workspace: str
    provider: str
    model: str
    path: Path
    turns: int = 0


class SessionStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / ".xiaoming" / "sessions"
        self.index_path = self.root / "index.json"

    def create(self, title: str, provider: str, model: str) -> SessionRecord:
        self.root.mkdir(parents=True, exist_ok=True)
        now = _now()
        session_id = str(uuid4())
        path = self.root / f"{session_id}.jsonl"
        record = SessionRecord(
            id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            workspace=str(self.workspace),
            provider=provider,
            model=model,
            path=path,
            turns=0,
        )
        self._write_event(path, "session_meta", session_id, {"title": title, "workspace": str(self.workspace), "provider": provider, "model": model})
        self._upsert(record)
        return record

    def append(self, session_id: str | None, event_type: str, payload: dict[str, Any]) -> None:
        if not session_id:
            return
        record = self.get(session_id)
        if record is None:
            return
        safe_payload = payload if event_type in {"bootstrap_context", "loaded_skill", "context_compaction_completed", "memory_diary", "dream_run"} else _truncate_payload(payload)
        self._write_event(record.path, event_type, session_id, safe_payload)
        turns = record.turns + 1 if event_type == "user_message" else record.turns
        title = record.title
        if event_type == "user_message" and title == "New session":
            title = _title_from_user_message(str(payload.get("content") or ""))
        self._upsert(
            SessionRecord(
                id=record.id,
                title=title,
                created_at=record.created_at,
                updated_at=_now(),
                workspace=record.workspace,
                provider=record.provider,
                model=record.model,
                path=record.path,
                turns=turns,
            )
        )

    def get(self, session_id: str) -> SessionRecord | None:
        for record in self.list():
            if record.id == session_id:
                return record
        return None

    def latest_for_workspace(self) -> SessionRecord | None:
        records = [
            record
            for record in self.list()
            if Path(record.workspace).resolve() == self.workspace and self.is_resumable(record.id)
        ]
        if not records:
            return None
        return records[0]

    def is_resumable(self, session_id: str) -> bool:
        return any(event.get("type") in RESUMABLE_EVENT_TYPES for event in self.read_events(session_id))

    def list(self) -> list[SessionRecord]:
        data = self._read_index()
        records: list[SessionRecord] = []
        for item in data.get("sessions", []):
            try:
                records.append(
                    SessionRecord(
                        id=str(item["id"]),
                        title=str(item.get("title") or "Untitled"),
                        created_at=str(item["created_at"]),
                        updated_at=str(item["updated_at"]),
                        workspace=str(item["workspace"]),
                        provider=str(item.get("provider") or ""),
                        model=str(item.get("model") or ""),
                        path=self.workspace / item["path"] if not Path(str(item["path"])).is_absolute() else Path(str(item["path"])),
                        turns=int(item.get("turns") or 0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return records

    def read_events(self, session_id: str) -> list[dict[str, Any]]:
        record = self.get(session_id)
        if record is None or not record.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in record.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def _write_event(self, path: Path, event_type: str, session_id: str, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "version": SESSION_VERSION,
            "event_id": str(uuid4()),
            "session_id": session_id,
            "created_at": _now(),
            "type": event_type,
            "payload": payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    def _read_index(self) -> dict[str, Any]:
        if not self.index_path.exists():
            return {"sessions": []}
        try:
            data = json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return {"sessions": []}
        if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
            return {"sessions": []}
        return data

    def _upsert(self, record: SessionRecord) -> None:
        data = self._read_index()
        rel_path = str(record.path.relative_to(self.workspace)) if _is_relative_to(record.path, self.workspace) else str(record.path)
        item = {
            "id": record.id,
            "title": record.title,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "workspace": record.workspace,
            "provider": record.provider,
            "model": record.model,
            "path": rel_path,
            "turns": record.turns,
        }
        sessions = [entry for entry in data["sessions"] if entry.get("id") != record.id]
        sessions.insert(0, item)
        data["sessions"] = sessions
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def rehydrate_session(events: list[dict[str, Any]], session_id: str | None = None) -> Session:
    session = Session(session_id=session_id, resumed=True)
    unfinished_turn_id: str | None = None
    for event in events:
        payload = event.get("payload") or {}
        event_type = event.get("type")
        if event_type == "turn_started":
            unfinished_turn_id = str(payload.get("turn_id") or "") or None
        elif event_type in TERMINAL_TURN_EVENT_TYPES:
            unfinished_turn_id = None
        if event_type == "base_instructions":
            session.base_instructions = str(payload.get("text") or "")
            session.base_instructions_recorded = True
        elif event_type == "prompt_item":
            item = payload.get("item")
            if isinstance(item, dict):
                session.input_items.append(PromptItem.from_event_payload(item).to_input_item())
        elif event_type == "turn_context":
            data = payload.get("data")
            if isinstance(data, dict):
                session.reference_turn_context = TurnContext.from_event_payload(data)
        elif event_type == "context_compaction_completed":
            replacement_items = payload.get("replacement_items")
            if isinstance(replacement_items, list):
                session.input_items = [item for item in replacement_items if isinstance(item, dict)]
                session.reference_turn_context = None
                session.compaction_count += 1
                session.last_compacted_at = str(payload.get("created_at") or "") or None
        elif event_type == "user_message":
            session.input_items.append(ensure_time_metadata({"role": "user", "content": payload.get("content") or "", "xiaoming": {"kind": "user_message"}}, created_at=str(event.get("created_at") or "")))
        elif event_type == "assistant_message":
            session.input_items.append(ensure_time_metadata({"role": "assistant", "content": payload.get("content") or "", "xiaoming": {"kind": "assistant_message"}}, created_at=str(event.get("created_at") or "")))
        elif event_type == "assistant_output":
            for item in payload.get("items") or []:
                if isinstance(item, dict):
                    session.input_items.append(ensure_time_metadata(item, created_at=str(event.get("created_at") or "")))
        elif event_type == "tool_output":
            item = payload.get("item")
            if isinstance(item, dict):
                session.input_items.append(ensure_time_metadata(item, created_at=str(event.get("created_at") or "")))
        elif event_type == "loaded_skill":
            if isinstance(payload, dict):
                skill = LoadedSkill.from_payload(payload)
                session.remember_loaded_skill(skill)
        elif event_type == "bootstrap_context":
            if isinstance(payload, dict):
                context = BootstrapContext.from_payload(payload)
                session.remember_bootstrap_context(context)
        elif event_type == "memory_diary":
            if isinstance(payload, dict):
                diary = MemoryDiary.from_payload(payload)
                if diary.id:
                    session.memory_diaries[diary.id] = diary
        elif event_type == "dream_run":
            if isinstance(payload, dict):
                run = DreamRun.from_payload(payload)
                if run.id:
                    session.dream_runs[run.id] = run
        elif event_type == "turn_aborted":
            session.input_items.append(_turn_aborted_item(str(payload.get("message") or _default_turn_aborted_message())))
    if unfinished_turn_id is not None:
        session.input_items.append(_turn_aborted_item(_default_turn_aborted_message()))
    session.input_items = _repair_dangling_tool_calls(session.input_items)
    return session


def _repair_dangling_tool_calls(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    repaired: list[dict[str, Any]] = []
    index = 0
    while index < len(input_items):
        item = input_items[index]
        repaired.append(item)
        tool_call_ids = _assistant_tool_call_ids(item)
        if not tool_call_ids:
            index += 1
            continue
        next_index = index + 1
        output_ids: set[str] = set()
        while next_index < len(input_items) and input_items[next_index].get("type") == "function_call_output":
            output = input_items[next_index]
            output_ids.add(str(output.get("call_id") or ""))
            repaired.append(output)
            next_index += 1
        for call_id in sorted(tool_call_ids - output_ids):
            repaired.append(_interrupted_tool_output(call_id))
        index = next_index
    return repaired


def _assistant_tool_call_ids(item: dict[str, Any]) -> set[str]:
    if item.get("role") != "assistant":
        return set()
    tool_calls = item.get("tool_calls")
    if not isinstance(tool_calls, list):
        return set()
    return {str(call.get("id") or "") for call in tool_calls if isinstance(call, dict) and call.get("id")}


def _interrupted_tool_output(call_id: str) -> dict[str, Any]:
    return ensure_time_metadata({
        "type": "function_call_output",
        "call_id": call_id,
        "output": "Tool: unknown\nStatus: interrupted\nError:\nTool execution was interrupted before completion.",
    })


def _turn_aborted_item(message: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": f"<turn_aborted>\n{message}\n</turn_aborted>",
        "xiaoming": {"kind": "turn_aborted", "durable": True},
    }


def _default_turn_aborted_message() -> str:
    return "The previous turn was interrupted before completion. Tools or commands may have partially executed; inspect state before continuing."


def _now() -> str:
    return now_iso()


def _title_from_user_message(content: str) -> str:
    title = " ".join(content.split())
    return title[:40] or "Untitled"


def _truncate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if len(encoded) <= 40000:
        return payload
    return {"truncated": True, "summary": truncate_middle(encoded, 40000)}


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
