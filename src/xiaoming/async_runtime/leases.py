from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from threading import RLock
from typing import Callable


WriteLeaseCallback = Callable[[str, list[str]], bool]


@dataclass(frozen=True)
class LeaseResult:
    granted: bool
    reason: str = ""


class WriteLeaseServer:
    def __init__(self) -> None:
        self._lock = RLock()
        self._holders: dict[str, str] = {}

    def acquire(self, task_id: str, paths: list[str]) -> LeaseResult:
        normalized = [_normalize(path) for path in paths]
        with self._lock:
            for path in normalized:
                holder = self._holders.get(path)
                if holder is not None and holder != task_id:
                    return LeaseResult(False, f"{path} is already leased by another task")
            for path in normalized:
                self._holders[path] = task_id
        return LeaseResult(True)

    def release(self, task_id: str, paths: list[str] | None = None) -> None:
        with self._lock:
            if paths is None:
                for path, holder in list(self._holders.items()):
                    if holder == task_id:
                        del self._holders[path]
                return
            for path in [_normalize(path) for path in paths]:
                if self._holders.get(path) == task_id:
                    del self._holders[path]

    def callback_for(self, task_id: str) -> WriteLeaseCallback:
        def acquire(tool_name: str, paths: list[str]) -> bool:
            result = self.acquire(task_id, paths)
            return result.granted

        return acquire


class FileWriteLeaseClient:
    def __init__(self, root: Path, task_id: str) -> None:
        self.root = root
        self.task_id = task_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._held: set[Path] = set()

    def acquire(self, tool_name: str, paths: list[str]) -> bool:
        acquired: list[Path] = []
        for path in paths:
            lease_path = self._lease_path(path)
            payload = {"task_id": self.task_id, "path": _normalize(path), "tool": tool_name}
            try:
                fd = os.open(lease_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._held_by_current_task(lease_path):
                    acquired.append(lease_path)
                    continue
                self._release_paths(acquired)
                return False
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            self._held.add(lease_path)
            acquired.append(lease_path)
        return True

    def release_all(self) -> None:
        self._release_paths(list(self._held))

    def _lease_path(self, path: str) -> Path:
        digest = hashlib.sha256(_normalize(path).encode("utf-8")).hexdigest()
        return self.root / f"{digest}.json"

    def _held_by_current_task(self, lease_path: Path) -> bool:
        try:
            payload = json.loads(lease_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("task_id") == self.task_id

    def _release_paths(self, paths: list[Path]) -> None:
        for path in paths:
            try:
                if self._held_by_current_task(path):
                    path.unlink()
            except FileNotFoundError:
                pass
            self._held.discard(path)


def _normalize(path: str) -> str:
    return path.replace("\\", "/").strip()
