from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from xiaoming.context.truncation import truncate_middle
from xiaoming.logging import redact_secrets


class WorkerSessionRecorder:
    def __init__(self, workspace: Path, task_id: str):
        self.path = workspace / ".xiaoming" / "worker_sessions" / f"{task_id}.jsonl"

    def append(self, session_id: str | None, event_type: str, payload: dict[str, Any]) -> None:
        if not session_id:
            return
        safe_payload = redact_secrets(payload)
        if event_type not in {"bootstrap_context", "loaded_skill", "context_compaction_completed"}:
            safe_payload = _truncate_payload(safe_payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "version": 1,
            "event_id": str(uuid4()),
            "session_id": session_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "payload": safe_payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()


def _truncate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(payload, ensure_ascii=False, default=str)
    if len(encoded) <= 40000:
        return payload
    return {"truncated": True, "summary": truncate_middle(encoded, 40000)}
