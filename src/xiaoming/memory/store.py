from __future__ import annotations

from xiaoming.memory.models import DreamRun, MemoryDiary
from xiaoming.session import Session


class MemoryStore:
    def __init__(self, session: Session, recorder: object | None = None):
        self.session = session
        self.recorder = recorder

    def save_diary(self, diary: MemoryDiary) -> None:
        self.session.memory_diaries[diary.id] = diary
        self._record("memory_diary", diary.to_payload())

    def save_dream_run(self, run: DreamRun) -> None:
        self.session.dream_runs[run.id] = run
        self._record("dream_run", run.to_payload())

    def active_diaries(self) -> list[MemoryDiary]:
        return sorted(
            [diary for diary in self.session.memory_diaries.values() if diary.status == "active"],
            key=lambda diary: (diary.start_time, diary.scope, diary.id),
        )

    def _record(self, event_type: str, payload: dict) -> None:
        if self.recorder is not None and hasattr(self.recorder, "append"):
            self.recorder.append(self.session.session_id, event_type, payload)
