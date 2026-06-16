from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import traceback
from typing import Any
from uuid import uuid4


SECRET_KEYWORDS = ("api_key", "authorization", "token", "password", "secret")
SAFE_TOKEN_FIELD_NAMES = {
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_tokens",
    "cache_hit_tokens",
    "cache_miss_tokens",
    "reasoning_tokens",
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
}
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE),
]


@dataclass
class XiaomingLogger:
    path: Path
    session_id: str

    @classmethod
    def create(cls, workspace: Path) -> "XiaomingLogger":
        path = workspace / ".xiaoming" / "logs" / "xiaoming.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path=path, session_id=uuid4().hex)

    @classmethod
    def create_worker(cls, workspace: Path, task_id: str) -> "XiaomingLogger":
        path = workspace / ".xiaoming" / "logs" / "workers" / f"{task_id}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(path=path, session_id=task_id)

    def info(self, event: str, **fields: Any) -> None:
        self._write("info", event, fields)

    def error(self, event: str, exc: BaseException | None = None, **fields: Any) -> None:
        if exc is not None:
            fields.setdefault("exception", f"{type(exc).__name__}: {exc}")
            fields.setdefault("traceback", "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        self._write("error", event, fields)

    def _write(self, level: str, event: str, fields: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "event": event,
            "session_id": self.session_id,
            **fields,
        }
        line = json.dumps(redact_secrets(record), ensure_ascii=False, sort_keys=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if _is_secret_key(key) else redact_secrets(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted
    return value


def summarize_text(value: str, max_chars: int = 500) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 15] + "...[truncated]"


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in SAFE_TOKEN_FIELD_NAMES:
        return False
    return any(keyword in lowered for keyword in SECRET_KEYWORDS)
