from __future__ import annotations

import re
from typing import Any, Callable

from xiaoming.async_runtime.tasks import TaskSpec
from xiaoming.llm.types import ToolSpec
from xiaoming.tools.base import ToolResult


class ScheduleBackgroundTaskTool:
    name = "schedule_background_task"
    description = (
        "Spawn a background worker for a concrete task. "
        "Use this only after you have first told the user what will be handled in the background. "
        "Do not use this for simple questions or answers that can be handled directly in chat. "
        "Pass a plain-language message describing what the worker should do. "
        "Do not pass internal routing, context, skill, artifact, path, or verification details as separate fields. "
        "Include any user-stated constraints directly in message. "
        "Use task_name only as a short human-readable label when useful."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "task_name": {"type": "string"},
        },
        "required": ["message"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        coordinator_getter: Callable[[], Any],
        turn_context_getter: Callable[[], str] | None = None,
        promoted_task_getter: Callable[[], str] | None = None,
    ):
        self.coordinator_getter = coordinator_getter
        self.turn_context_getter = turn_context_getter
        self.promoted_task_getter = promoted_task_getter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        message = str(args.get("message") or "").strip()
        task_name = str(args.get("task_name") or "").strip()
        if not message:
            return ToolResult(self.name, "error", error="message is required")
        promoted_task_id = self.promoted_task_getter() if self.promoted_task_getter is not None else ""
        if promoted_task_id:
            return ToolResult(
                self.name,
                "denied",
                error=(
                    "This run is already running as a background task "
                    f"({promoted_task_id[:8]}). Do not schedule another worker for the same work; "
                    "continue the current task directly."
                ),
            )
        task_spec = TaskSpec.from_request(message, title=task_name or None)
        turn_context = self.turn_context_getter() if self.turn_context_getter is not None else ""
        requested_executor = _requested_executor_from_turn_context(turn_context)
        if requested_executor:
            task_spec.notes = _append_internal_note(task_spec.notes, f"requested_executor={requested_executor}")
        coordinator = self.coordinator_getter()
        if coordinator is None:
            return ToolResult(self.name, "error", error="background coordinator is not running")
        return coordinator.schedule_background_task(task_spec)


def _requested_executor_from_turn_context(text: str) -> str:
    normalized = text.lower()
    if re.search(r"(?:用|使用|让|请|调用|交给)\s*codex(?![a-z0-9_])", normalized):
        return "codex"
    if re.search(r"(?<![a-z0-9_])codex\s*(?:帮|来|处理|开发|写|执行|做)", normalized):
        return "codex"
    return ""


def _append_internal_note(notes: str, note: str) -> str:
    stripped = notes.strip()
    if not stripped:
        return note
    if note in stripped.splitlines():
        return stripped
    return f"{stripped}\n{note}"


class BackgroundTasksStatusTool:
    name = "background_tasks_status"
    description = (
        "Return one snapshot of current background task status. "
        "Use this when the user asks about progress or whether a background task finished. "
        "Do not call this repeatedly in the same turn to wait for completion; answer from the snapshot. "
        "If the user explicitly asks you to wait or follow a task, use follow_background_task instead."
    )
    input_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, coordinator_getter: Callable[[], Any]):
        self.coordinator_getter = coordinator_getter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        coordinator = self.coordinator_getter()
        if coordinator is None:
            return ToolResult(self.name, "error", error="background coordinator is not running")
        return ToolResult(self.name, "success", output=coordinator.current_tasks_text())


class FollowBackgroundTaskTool:
    name = "follow_background_task"
    description = (
        "Briefly wait for a specific background task to change status. "
        "Use only when the user explicitly asks to wait, follow, or watch a task. "
        "This tool returns after a state change or a bounded runtime timeout; it does not poll indefinitely."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    }

    def __init__(self, coordinator_getter: Callable[[], Any]):
        self.coordinator_getter = coordinator_getter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        coordinator = self.coordinator_getter()
        if coordinator is None:
            return ToolResult(self.name, "error", error="background coordinator is not running")
        task_id = str(args.get("task_id") or "").strip()
        if not task_id:
            return ToolResult(self.name, "error", error="task_id is required")
        return coordinator.follow_task(task_id)


class CancelBackgroundTaskTool:
    name = "cancel_background_task"
    description = (
        "Cancel one background task by exact task_id. "
        "Use background_tasks_status first if the user did not provide or imply an exact task_id. "
        "For multiple cancellations, call this tool once per task_id."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
        },
        "required": ["task_id"],
        "additionalProperties": False,
    }

    def __init__(self, coordinator_getter: Callable[[], Any]):
        self.coordinator_getter = coordinator_getter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        coordinator = self.coordinator_getter()
        if coordinator is None:
            return ToolResult(self.name, "error", error="background coordinator is not running")
        return coordinator.cancel_task(str(args.get("task_id") or ""))


class TalkToPeerTool:
    name = "talk_to_peer"
    description = (
        "Send a natural language message to an existing peer such as a background worker, and return its reply. "
        "Use this for questions, discussion, follow-up, or natural continuation with an existing peer. "
        "This is a transparent talk channel: it does not create a new background task and does not trigger task verification."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "peer_id": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["peer_id", "message"],
        "additionalProperties": False,
    }

    def __init__(self, coordinator_getter: Callable[[], Any]):
        self.coordinator_getter = coordinator_getter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        coordinator = self.coordinator_getter()
        if coordinator is None:
            return ToolResult(self.name, "error", error="background coordinator is not running")
        return coordinator.talk_to_peer(str(args.get("peer_id") or ""), str(args.get("message") or ""))


class ReplyMailboxMessageTool:
    name = "reply_mailbox_message"
    description = (
        "Reply to a pending mailbox message from a background worker. "
        "Use this only when the user clearly answered a pending worker request shown in the async context. "
        "If the user is ambiguous, ask a normal chat clarification and do not call this tool."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "message_id": {"type": "string"},
            "normalized_answer": {"type": "string"},
            "decision": {"type": "string", "enum": ["approved", "denied", "none"]},
            "message_to_user": {"type": "string"},
            "authorization_note": {
                "type": "string",
                "description": "Optional updated authorization note for this worker when the user delegates future similar decisions.",
            },
        },
        "required": ["message_id", "normalized_answer", "decision", "message_to_user", "authorization_note"],
        "additionalProperties": False,
    }

    def __init__(self, coordinator_getter: Callable[[], Any]):
        self.coordinator_getter = coordinator_getter

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        coordinator = self.coordinator_getter()
        if coordinator is None:
            return ToolResult(self.name, "error", error="background coordinator is not running")
        return coordinator.reply_mailbox_message(
            message_id=str(args.get("message_id") or ""),
            normalized_answer=str(args.get("normalized_answer") or ""),
            decision=str(args.get("decision") or "none"),
            message_to_user=str(args.get("message_to_user") or ""),
            authorization_note=str(args.get("authorization_note") or ""),
        )
