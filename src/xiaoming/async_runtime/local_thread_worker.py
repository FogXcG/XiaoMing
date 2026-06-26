from __future__ import annotations

import queue
import threading
from typing import Callable
from uuid import uuid4

from xiaoming.async_runtime.events import WorkerEvent


class ForegroundWorkerHandle:
    def __init__(self, task_id: str, on_event: Callable[[WorkerEvent], None], cancel_event: threading.Event):
        self.task_id = task_id
        self.pid: int | None = None
        self._on_event = on_event
        self._cancel_event = cancel_event
        self._inbox: queue.Queue[dict] = queue.Queue()
        self._pending_answers: dict[str, queue.Queue[dict]] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        return None

    def send(self, kind: str, **payload) -> None:
        if kind == "answer_question":
            self._deliver_answer(payload)
            return
        self._inbox.put({"kind": kind, **payload})

    def terminate(self, timeout_seconds: float = 5) -> None:
        self._cancel_event.set()
        self.send("cancel")

    def request_approval(self, action: str) -> bool:
        request_id = str(uuid4())
        answer_queue: queue.Queue[dict] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending_answers[request_id] = answer_queue
        self._on_event(
            WorkerEvent(
                task_id=self.task_id,
                kind="approval_request",
                message=action,
                data={"request_id": request_id, "purpose": "tool_approval"},
            )
        )
        try:
            payload = answer_queue.get()
        finally:
            with self._lock:
                self._pending_answers.pop(request_id, None)
        return str(payload.get("decision") or "") == "approved"

    def emit_completed(self, message: str) -> None:
        self._on_event(WorkerEvent(self.task_id, "completed", message))

    def emit_failed(self, message: str) -> None:
        self._on_event(WorkerEvent(self.task_id, "failed", message))

    def drain_inbox(self) -> list[dict]:
        items: list[dict] = []
        while True:
            try:
                items.append(self._inbox.get_nowait())
            except queue.Empty:
                return items

    def _deliver_answer(self, payload: dict) -> None:
        request_id = str(payload.get("request_id") or "")
        if not request_id:
            return
        with self._lock:
            answer_queue = self._pending_answers.get(request_id)
        if answer_queue is None:
            return
        try:
            answer_queue.put_nowait(dict(payload))
        except queue.Full:
            pass
