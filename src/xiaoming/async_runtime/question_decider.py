from __future__ import annotations

from dataclasses import dataclass
import json
import queue
import threading
from typing import Literal, Protocol

from xiaoming.async_runtime.events import Question
from xiaoming.async_runtime.tasks import TaskRecord
from xiaoming.llm.provider import LLMProvider
from xiaoming.llm.types import LLMRequest


Decision = Literal["approved", "denied", "ask_user"]


@dataclass(frozen=True)
class WorkerQuestionDecision:
    decision: Decision
    answer: str
    reason: str = ""


class WorkerQuestionDecider(Protocol):
    def decide(self, task: TaskRecord, question: Question) -> WorkerQuestionDecision:
        ...


class LLMWorkerQuestionDecider:
    def __init__(self, provider: LLMProvider, model: str, timeout_seconds: float = 8, max_attempts: int = 3):
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts

    def decide(self, task: TaskRecord, question: Question) -> WorkerQuestionDecision:
        if not task.authorization_note.strip():
            return WorkerQuestionDecision("ask_user", "", "no delegation note")
        payload = {
            "task": {
                "title": task.title,
                "goal": task.current_goal,
                "authorization_note": task.authorization_note,
            },
            "question": {
                "kind": question.kind,
                "prompt": question.prompt,
                "purpose": question.purpose,
                "context": question.context,
                "options": question.options,
            },
        }
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                return _decision_from_json(self._complete_once(payload))
            except BaseException as exc:
                errors.append(f"attempt {attempt}: {exc}")
        return WorkerQuestionDecision("ask_user", "", "; ".join(errors))

    def _complete_once(self, payload: dict) -> str:
        result_queue: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                response = self.provider.complete(
                    LLMRequest(
                        model=self.model,
                        instructions=_DECIDER_INSTRUCTIONS,
                        input_items=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}],
                        tools=[],
                        temperature=0,
                        max_output_tokens=800,
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
            raise TimeoutError(f"worker question decider timed out after {self.timeout_seconds:g} seconds")
        if isinstance(result, BaseException):
            raise result
        if not result:
            raise ValueError("empty worker question decision")
        return result


def _decision_from_json(text: str) -> WorkerQuestionDecision:
    data = json.loads(_extract_json_object(text))
    decision = data.get("decision")
    if decision not in {"approved", "denied", "ask_user"}:
        raise ValueError(f"invalid worker question decision: {decision}")
    return WorkerQuestionDecision(decision=decision, answer=str(data.get("answer") or ""), reason=str(data.get("reason") or ""))


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("worker question decision did not contain a JSON object")
    return stripped[start : end + 1]


_DECIDER_INSTRUCTIONS = """You decide whether the coordinator can answer a background worker question on behalf of the human user.
Return only one JSON object. Do not use markdown.
Use the task goal, the worker question, and the task authorization_note.
If authorization_note clearly covers the worker's current request, choose approved or denied and provide a concise answer for the worker.
If authorization_note is missing, unclear, or does not cover this exact request, choose ask_user.
If the request involves destructive, irreversible, credential-related, account-related, payment-related, or workspace-external behavior and the authorization_note does not explicitly allow that exact behavior, choose ask_user.
Required JSON keys: decision, answer, reason.
decision must be one of: approved, denied, ask_user.
For ask_user, answer must be an empty string.
"""
