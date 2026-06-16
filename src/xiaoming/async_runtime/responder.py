from __future__ import annotations

import json
import queue
import threading
from typing import Protocol

from xiaoming.async_runtime.events import Question, WorkerEvent
from xiaoming.async_runtime.tasks import TaskRecord, TaskRegistry
from xiaoming.llm.provider import LLMProvider
from xiaoming.llm.types import LLMRequest


class CoordinatorResponder(Protocol):
    def user_reply(self, user_message: str, registry: TaskRegistry, mode: str, question: Question | None = None) -> str:
        ...

    def worker_notice(self, event: WorkerEvent, task: TaskRecord, registry: TaskRegistry) -> str:
        ...

    def command_reply(self, command: str, payload: dict, registry: TaskRegistry) -> str:
        ...


class ResponderError(RuntimeError):
    pass


class LLMMessagingResponder:
    def __init__(self, provider: LLMProvider, model: str, timeout_seconds: float = 8, max_attempts: int = 3):
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def user_reply(self, user_message: str, registry: TaskRegistry, mode: str, question: Question | None = None) -> str:
        payload = {
            "kind": "user_reply",
            "mode": mode,
            "user_message": user_message,
            "pending_question": question.prompt if question is not None else None,
            "tasks": registry.snapshot(),
        }
        return self._complete(payload)

    def worker_notice(self, event: WorkerEvent, task: TaskRecord, registry: TaskRegistry) -> str:
        payload = {
            "kind": "worker_notice",
            "event": event.to_json(),
            "task": task.snapshot(),
            "tasks": registry.snapshot(),
        }
        return self._complete(payload)

    def command_reply(self, command: str, payload: dict, registry: TaskRegistry) -> str:
        message = {"kind": "command_reply", "command": command, "payload": payload, "tasks": registry.snapshot()}
        return self._complete(message)

    def _complete(self, payload: dict) -> str:
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                text = self._complete_once(payload)
            except BaseException as exc:
                errors.append(f"attempt {attempt}: {exc}")
                continue
            if text:
                return text
            errors.append(f"attempt {attempt}: empty model response")
        raise ResponderError("Xiaoming responder failed after 3 attempts: " + "; ".join(errors))

    def _complete_once(self, payload: dict) -> str:
        result_queue: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                response = self.provider.complete(
                    LLMRequest(
                        model=self.model,
                        instructions=_RESPONDER_INSTRUCTIONS,
                        input_items=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
                        tools=[],
                        temperature=0.2,
                        max_output_tokens=500,
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
            raise TimeoutError(f"responder model timed out after {self.timeout_seconds:g} seconds")
        if isinstance(result, BaseException):
            raise result
        return result


_RESPONDER_INSTRUCTIONS = """In this runtime, act as the user-facing coordinator for background coding workers.
Write one concise Chinese message to the user.
Do not expose worker ids, queues, protocols, locks, JSON, or internal implementation details.
For user_reply, acknowledge the user's message in context and say what the coordinator will do next.
For worker_notice, summarize only user-relevant state changes, questions, completion, failure, or cancellation.
For command_reply, answer the user's command using the payload and current task context.
Keep it natural and specific to the given context. Do not use markdown unless the message contains multiple questions.
"""
