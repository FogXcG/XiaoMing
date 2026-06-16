from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import subprocess
import sys
import threading
from typing import Any, Callable

from xiaoming.async_runtime.events import WorkerEvent
from xiaoming.async_runtime.protocol import ProtocolError, decode_message, write_message
from xiaoming.async_runtime.context_packets import WorkerContextPacket
from xiaoming.async_runtime.tasks import TaskSpec
from xiaoming.logging import redact_secrets


@dataclass
class WorkerConfig:
    task_id: str
    task: str
    workspace: Path
    provider: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    model_timeout_seconds: float | None = None
    stream: bool | None = None
    task_spec: TaskSpec | None = None
    agent_type: str = "worker"
    context_policy: str = "forked"
    skills_to_preload: list[str] | None = None
    context_packet: WorkerContextPacket | None = None
    forked_instructions: str = ""
    forked_input_items: list[dict[str, Any]] | None = None
    forked_loaded_skills: list[dict[str, str]] | None = None


class WorkerProcess:
    def __init__(self, config: WorkerConfig, on_event: Callable[[WorkerEvent], None] | None = None):
        self.config = config
        self.on_event = on_event
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._events: queue.Queue[WorkerEvent] = queue.Queue()
        self._terminal_event_seen = False

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    def start(self) -> None:
        command = [sys.executable, "-m", "xiaoming.worker_main"]
        self.process = subprocess.Popen(
            command,
            cwd=str(self.config.workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        write_message(
            self.process.stdin,
            "start_task",
            task_id=self.config.task_id,
            task=self.config.task,
            workspace=str(self.config.workspace),
            provider=self.config.provider,
            model=self.config.model,
            approval_mode=self.config.approval_mode,
            permission_mode=self.config.permission_mode,
            max_turns=self.config.max_turns,
            model_timeout_seconds=self.config.model_timeout_seconds,
            stream=self.config.stream,
            task_spec=self.config.task_spec.to_dict() if self.config.task_spec is not None else None,
            agent_type=self.config.agent_type,
            context_policy=self.config.context_policy,
            skills_to_preload=list(self.config.skills_to_preload or []),
            context_packet=self.config.context_packet.to_dict() if self.config.context_packet is not None else None,
            forked_instructions=self.config.forked_instructions,
            forked_input_items=list(self.config.forked_input_items or []),
            forked_loaded_skills=list(self.config.forked_loaded_skills or []),
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self._stderr_reader.start()

    def send(self, kind: str, **payload) -> None:
        if self.process is None or self.process.stdin is None:
            return
        write_message(self.process.stdin, kind, **payload)

    def cancel(self) -> None:
        self.send("cancel")

    def terminate(self, timeout_seconds: float = 5) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            return
        self.cancel()
        try:
            self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout_seconds)

    def poll_event(self, timeout: float = 0) -> WorkerEvent | None:
        try:
            return self._events.get(timeout=timeout)
        except queue.Empty:
            return None

    def _read_stdout(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            if self._terminal_event_seen:
                continue
            try:
                payload = decode_message(line)
                event = WorkerEvent.from_json(payload)
            except (ProtocolError, KeyError, TypeError, ValueError) as exc:
                event = WorkerEvent(self.config.task_id, "failed", f"invalid worker event: {exc}", {"error_kind": "invalid_worker_event"})
                self._terminal_event_seen = True
                self.terminate(timeout_seconds=1)
            if event.kind in {"failed", "cancelled"}:
                self._terminal_event_seen = True
            self._events.put(event)
            if self.on_event is not None:
                self.on_event(event)

    def _read_stderr(self) -> None:
        assert self.process is not None
        stderr = getattr(self.process, "stderr", None)
        if stderr is None:
            return
        path = self.config.workspace / ".xiaoming" / "logs" / "workers" / f"{self.config.task_id}.stderr.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for line in stderr:
                handle.write(str(redact_secrets(line)))
                handle.flush()
