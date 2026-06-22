from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import select
import subprocess
import threading
from typing import Callable

from xiaoming.async_runtime.events import WorkerEvent
from xiaoming.async_runtime.worker_process import WorkerConfig
from xiaoming.logging import redact_secrets


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExternalSessionRecord:
    peer_id: str
    provider: str
    title: str
    workspace: str
    session_id: str = ""
    status: str = "active"
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @classmethod
    def from_dict(cls, data: dict) -> "ExternalSessionRecord":
        return cls(
            peer_id=str(data.get("peer_id") or ""),
            provider=str(data.get("provider") or ""),
            title=str(data.get("title") or ""),
            workspace=str(data.get("workspace") or ""),
            session_id=str(data.get("session_id") or ""),
            status=str(data.get("status") or "active"),
            created_at=str(data.get("created_at") or _now()),
            updated_at=str(data.get("updated_at") or _now()),
        )

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "provider": self.provider,
            "title": self.title,
            "workspace": self.workspace,
            "session_id": self.session_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ExternalSessionStore:
    def __init__(self, workspace: Path):
        self.path = workspace / ".xiaoming" / "external_sessions.json"

    def load(self) -> list[ExternalSessionRecord]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return []
        if not isinstance(data, list):
            return []
        records: list[ExternalSessionRecord] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                record = ExternalSessionRecord.from_dict(item)
            except Exception:
                continue
            if record.peer_id and record.provider:
                records.append(record)
        return records

    def save(self, records: list[ExternalSessionRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([record.to_dict() for record in records], ensure_ascii=False, indent=2, sort_keys=True) + "\n")


class CodexRemoteControlSession:
    def __init__(self, workspace: Path, session_id: str = "", timeout_seconds: float = 900):
        self.workspace = workspace
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds
        self._process: subprocess.Popen[str] | None = None
        self._next_request_id = 1
        self._lock = threading.Lock()

    def send(self, message: str, on_progress: Callable[[str], None] | None = None) -> tuple[str, str]:
        with self._lock:
            self._ensure_started()
            thread_id = self._ensure_thread()
            answer = self._run_turn(thread_id, message, on_progress=on_progress)
            return answer, self.session_id

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            ["codex", "app-server", "--stdio"],
            cwd=str(self.workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        response = self._request(
            "initialize",
            {
                "clientInfo": {"name": "xiaoming", "title": "Xiaoming", "version": "0"},
                "capabilities": {"experimentalApi": True, "requestAttestation": False},
            },
        )
        if "result" not in response:
            raise RuntimeError(_jsonrpc_error_text(response))

    def _ensure_thread(self) -> str:
        if self.session_id:
            response = self._request(
                "thread/resume",
                {
                    "threadId": self.session_id,
                    "cwd": str(self.workspace),
                    "runtimeWorkspaceRoots": [str(self.workspace)],
                    "approvalPolicy": "never",
                    "sandbox": "danger-full-access",
                    "excludeTurns": True,
                },
            )
        else:
            response = self._request(
                "thread/start",
                {
                    "cwd": str(self.workspace),
                    "runtimeWorkspaceRoots": [str(self.workspace)],
                    "approvalPolicy": "never",
                    "sandbox": "danger-full-access",
                },
            )
        if "result" not in response:
            raise RuntimeError(_jsonrpc_error_text(response))
        thread = response.get("result", {}).get("thread", {})
        thread_id = str(thread.get("id") or self.session_id)
        if not thread_id:
            raise RuntimeError("Codex app-server did not return a thread id")
        self.session_id = thread_id
        return thread_id

    def _run_turn(self, thread_id: str, message: str, on_progress: Callable[[str], None] | None = None) -> str:
        response = self._request(
            "turn/start",
            {
                "threadId": thread_id,
                "input": [{"type": "text", "text": message, "text_elements": []}],
                "cwd": str(self.workspace),
                "runtimeWorkspaceRoots": [str(self.workspace)],
                "approvalPolicy": "never",
                "sandboxPolicy": {"type": "dangerFullAccess"},
            },
        )
        if "result" not in response:
            raise RuntimeError(_jsonrpc_error_text(response))
        turn = response.get("result", {}).get("turn", {})
        turn_id = str(turn.get("id") or "")
        if not turn_id:
            raise RuntimeError("Codex app-server did not return a turn id")
        return self._read_turn_answer(thread_id, turn_id, on_progress=on_progress)

    def _request(self, method: str, params: dict | None = None) -> dict:
        request_id = self._next_request_id
        self._next_request_id += 1
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method}
        if params is not None:
            payload["params"] = params
        self._write_json(payload)
        deadline = _deadline(self.timeout_seconds)
        while True:
            item = self._read_json(deadline)
            if item.get("id") == request_id:
                return item

    def _read_turn_answer(self, thread_id: str, turn_id: str, on_progress: Callable[[str], None] | None = None) -> str:
        deltas: list[str] = []
        completed_messages: list[str] = []
        deadline = _deadline(self.timeout_seconds)
        while True:
            try:
                item = self._read_json(deadline)
            except TimeoutError:
                if on_progress is not None:
                    on_progress("Codex is still working; waiting for the next event.")
                deadline = _deadline(self.timeout_seconds)
                continue
            method = item.get("method")
            params = item.get("params") if isinstance(item.get("params"), dict) else {}
            if method == "error":
                raise RuntimeError(str(params.get("error") or params))
            if params.get("threadId") != thread_id:
                continue
            if method == "item/agentMessage/delta" and params.get("turnId") == turn_id:
                deltas.append(str(params.get("delta") or ""))
                continue
            if method == "item/completed" and params.get("turnId") == turn_id:
                item_payload = params.get("item") if isinstance(params.get("item"), dict) else {}
                if item_payload.get("type") == "agentMessage" and item_payload.get("text"):
                    completed_messages.append(str(item_payload["text"]))
                continue
            if method == "thread/status/changed":
                status = params.get("status") if isinstance(params.get("status"), dict) else {}
                if status.get("type") == "idle" and (deltas or completed_messages):
                    return ("".join(deltas).strip() or "\n".join(completed_messages[-1:]).strip())
            if method == "turn/completed":
                turn = params.get("turn") if isinstance(params.get("turn"), dict) else {}
                if turn.get("id") != turn_id:
                    continue
                if turn.get("status") == "failed":
                    raise RuntimeError(str(turn.get("error") or "Codex turn failed"))
                answer = "".join(deltas).strip() or "\n".join(completed_messages[-1:]).strip()
                return answer

    def _write_json(self, payload: dict) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Codex app-server is not running")
        process.stdin.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        process.stdin.flush()

    def _read_json(self, deadline: float) -> dict:
        process = self._process
        if process is None or process.stdout is None:
            raise RuntimeError("Codex app-server is not running")
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"Codex app-server exited with code {process.returncode}")
            if datetime.now(timezone.utc).timestamp() > deadline:
                raise TimeoutError("Codex app-server response timed out")
            readable, _, _ = select.select([process.stdout], [], [], 0.25)
            if not readable:
                continue
            line = process.stdout.readline()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload


def _deadline(timeout_seconds: float) -> float:
    return datetime.now(timezone.utc).timestamp() + max(timeout_seconds, 1)


def _jsonrpc_error_text(response: dict) -> str:
    error = response.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error)
        code = error.get("code")
        return f"{message} (code={code})" if code is not None else message
    return str(error or response or "unknown Codex app-server error")


class CodexWorkerProcess:
    def __init__(self, config: WorkerConfig, on_event: Callable[[WorkerEvent], None] | None = None):
        self.config = config
        self.on_event = on_event
        self.session = CodexRemoteControlSession(config.workspace, timeout_seconds=float(config.model_timeout_seconds or 900))
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def pid(self) -> int | None:
        return None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._queue.put(("task", self.config.task))

    def send(self, kind: str, **payload) -> None:
        if kind == "cancel":
            self._stop.set()
            self.session.close()
            self._emit("cancelled", "Task cancelled.")
            return
        if kind in {"talk", "append_user_message", "review_feedback"}:
            message = str(payload.get("message") or payload.get("feedback") or "")
            if message.strip():
                self._queue.put(("talk" if kind == "talk" else "task", message))
            return
        if kind == "review_accepted":
            self._stop.set()
            self.session.close()

    def cancel(self) -> None:
        self.send("cancel")

    def terminate(self, timeout_seconds: float = 5) -> None:
        self.cancel()

    def _run(self) -> None:
        self._emit("started", "codex session started")
        while not self._stop.is_set():
            try:
                kind, message = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                answer, session_id = self.session.send(message, on_progress=lambda progress: self._emit("progress", progress, self._external_session_data()))
            except Exception as exc:
                self._write_error_log(exc)
                self._emit("failed", f"external Codex worker failed: {exc}")
                return
            data = {"external_provider": "codex", "external_session_id": session_id}
            if kind == "talk":
                self._emit("peer_reply", answer, data)
            else:
                self._emit("completed", answer, data)

    def _emit(self, kind: str, message: str, data: dict | None = None) -> None:
        if self.on_event is not None:
            self.on_event(WorkerEvent(self.config.task_id, kind, message, data or {}))

    def _external_session_data(self) -> dict:
        return {"external_provider": "codex", "external_session_id": self.session.session_id}

    def _write_error_log(self, exc: Exception) -> None:
        path = self.config.workspace / ".xiaoming" / "logs" / "workers" / f"{self.config.task_id}.codex.err.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(redact_secrets(str(exc))) + "\n")
