from __future__ import annotations

from dataclasses import dataclass, field
import json
import queue
import threading
from typing import Literal, Protocol

from xiaoming.llm.provider import LLMProvider
from xiaoming.llm.types import LLMRequest
from xiaoming.async_runtime.tasks import TaskRegistry


DecisionAction = Literal["start_new_task", "attach_to_task", "queue_task", "ask_user", "reject_duplicate", "cancel_and_restart", "chat_reply"]


@dataclass(frozen=True)
class SchedulerDecision:
    action: DecisionAction
    user_intent: str
    task_title: str
    visible_message: str
    reason: str = ""
    target_task_id: str | None = None
    affected_files: set[str] = field(default_factory=set)
    affected_modules: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    conflict_task_ids: set[str] = field(default_factory=set)
    duplicate_task_ids: set[str] = field(default_factory=set)


class Scheduler(Protocol):
    def schedule(self, user_message: str, registry: TaskRegistry) -> SchedulerDecision:
        ...


class SchedulerError(RuntimeError):
    pass


class LLMScheduler:
    def __init__(self, provider: LLMProvider, model: str, timeout_seconds: float = 8, max_attempts: int = 3):
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def schedule(self, user_message: str, registry: TaskRegistry) -> SchedulerDecision:
        prompt = {
            "user_message": user_message,
            "tasks": registry.active_snapshot(),
            "allowed_actions": [
                "start_new_task",
                "attach_to_task",
                "queue_task",
                "ask_user",
                "reject_duplicate",
                "cancel_and_restart",
                "chat_reply",
            ],
        }
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                response_text = self._complete_once(prompt)
                return _decision_from_json(response_text, registry, user_message)
            except BaseException as exc:
                errors.append(f"attempt {attempt}: {exc}")
        raise SchedulerError("Xiaoming scheduler failed after 3 attempts: " + "; ".join(errors))

    def _complete_once(self, prompt: dict) -> str:
        result_queue: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                response = self.provider.complete(
                    LLMRequest(
                        model=self.model,
                        instructions=_SCHEDULER_INSTRUCTIONS,
                        input_items=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False, sort_keys=True)}],
                        tools=[],
                        temperature=0,
                        max_output_tokens=2000,
                    )
                )
                result_queue.put((response.message or "").strip())
            except BaseException as exc:
                result_queue.put(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            result = result_queue.get(timeout=self.timeout_seconds)
        except queue.Empty:
            raise TimeoutError(f"scheduler model timed out after {self.timeout_seconds:g} seconds")
        if isinstance(result, BaseException):
            raise result
        if not result:
            raise ValueError("empty scheduler response")
        return result


_SCHEDULER_INSTRUCTIONS = """In this runtime, act as the background task scheduler.
Return only one JSON object. Do not use markdown.
Choose one action from the allowed actions.
Prefer parallel execution for independent tasks. Do not choose queue_task merely because another task is active.
Be conservative only for duplicate or overlapping tasks.
If a new request clearly may modify the same files, modules, or domains as a running task, choose queue_task and include the conflicting active task ids in conflict_task_ids.
For queue_task, reason must explain the exact conflict, including which file/module/domain overlaps with which active task. If you cannot name a concrete conflict, choose start_new_task.
If it is a small supplement to a running task, choose attach_to_task.
Required JSON keys:
action, user_intent, task_title, visible_message, reason,
target_task_id, affected_files, affected_modules, domains,
conflict_task_ids, duplicate_task_ids.
Array fields must be arrays of strings. Unknown nullable fields must be null.
"""


def _decision_from_json(text: str, registry: TaskRegistry, user_message: str) -> SchedulerDecision:
    parsed = json.loads(_extract_json_object(text))
    action = parsed.get("action")
    if action not in {
        "start_new_task",
        "attach_to_task",
        "queue_task",
        "ask_user",
        "reject_duplicate",
        "cancel_and_restart",
        "chat_reply",
    }:
        raise ValueError(f"invalid scheduler action: {action}")
    files = _string_set(parsed.get("affected_files"))
    modules = _string_set(parsed.get("affected_modules"))
    domains = _string_set(parsed.get("domains"))
    conflict_ids = _string_set(parsed.get("conflict_task_ids"))
    duplicate_ids = _string_set(parsed.get("duplicate_task_ids"))
    visible_message = parsed.get("visible_message")
    if not isinstance(visible_message, str) or not visible_message.strip():
        raise ValueError("scheduler response missing visible_message")
    decision = SchedulerDecision(
        action=action,
        user_intent=str(parsed.get("user_intent") or user_message),
        task_title=str(parsed.get("task_title") or _title(user_message)),
        visible_message=visible_message.strip(),
        reason=str(parsed.get("reason") or ""),
        target_task_id=parsed.get("target_task_id") or None,
        affected_files=files,
        affected_modules=modules,
        domains=domains,
        conflict_task_ids=conflict_ids,
        duplicate_task_ids=duplicate_ids,
    )
    _validate_scheduler_decision(decision, registry)
    return decision


def _validate_scheduler_decision(decision: SchedulerDecision, registry: TaskRegistry) -> None:
    if decision.action == "queue_task":
        valid_conflict_ids = {
            task.task_id
            for task in registry.active()
            if task.task_id in decision.conflict_task_ids and task.status in {"planning", "running", "needs_user", "reported_completed"}
        }
        if not valid_conflict_ids:
            raise ValueError("queue_task requires conflict_task_ids containing at least one active conflicting task id")
        if not decision.reason.strip():
            raise ValueError("queue_task requires a reason explaining the exact conflict")


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("scheduler response did not contain a JSON object")
    return stripped[start : end + 1]


def _string_set(value) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {str(item) for item in value if str(item)}


def _title(message: str) -> str:
    cleaned = " ".join(message.split())
    return cleaned[:24] or "后台任务"
