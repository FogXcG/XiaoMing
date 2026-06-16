from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def now_iso() -> str:
    return datetime.now(_runtime_zoneinfo()).isoformat()


def time_metadata(created_at: str | None = None) -> dict[str, str]:
    dt = _parse_datetime(created_at) if created_at else datetime.now(_runtime_zoneinfo())
    local = dt.astimezone(_runtime_zoneinfo())
    return {
        "created_at": local.isoformat(),
        "date": local.date().isoformat(),
        "time": local.strftime("%H:%M"),
        "timezone": runtime_timezone_name(),
    }


def ensure_time_metadata(item: dict[str, Any], created_at: str | None = None) -> dict[str, Any]:
    meta = item.get("xiaoming")
    if not isinstance(meta, dict):
        meta = {}
    if "created_at" in meta and "date" in meta and "time" in meta and "timezone" in meta:
        return item
    updated = dict(item)
    updated_meta = dict(meta)
    updated_meta.update(time_metadata(str(meta.get("created_at") or created_at or "")))
    updated["xiaoming"] = updated_meta
    return updated


def runtime_timezone_name() -> str:
    for value in (os.environ.get("XIAOMING_TIMEZONE"), os.environ.get("TZ"), _system_timezone_name()):
        name = (value or "").strip()
        if not name:
            continue
        try:
            ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            continue
        return name
    tzname = datetime.now().astimezone().tzname()
    return tzname or "local"


def _runtime_zoneinfo():
    name = runtime_timezone_name()
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return datetime.now().astimezone().tzinfo


def _system_timezone_name() -> str | None:
    path = Path("/etc/timezone")
    if not path.exists():
        return None
    try:
        return path.read_text().strip().splitlines()[0]
    except Exception:
        return None


def _parse_datetime(value: str | None) -> datetime:
    text = (value or "").strip()
    if not text:
        return datetime.now(_runtime_zoneinfo())
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.now(_runtime_zoneinfo())
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_runtime_zoneinfo())
    return parsed
