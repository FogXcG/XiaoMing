from __future__ import annotations

from typing import Protocol


class WorkerHandle(Protocol):
    task_id: str
    pid: int | None

    def start(self) -> None:
        ...

    def send(self, kind: str, **payload) -> None:
        ...

    def terminate(self, timeout_seconds: float = 5) -> None:
        ...
