from __future__ import annotations

from typing import Any

from xiaoming.memory.models import MemoryDiary


SCOPE_ORDER = {"year": 0, "month": 1, "week": 2, "day": 3}
PROTECTED_KINDS = {"worker_protocol", "hook_context", "runtime_context"}


def build_memory_view(diaries: list[MemoryDiary], recent_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = [_diary_item(diary) for diary in _ordered_active_diaries(diaries)]
    items.extend(recent_items)
    return items


def protected_recent_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    protected: list[dict[str, Any]] = []
    for item in items:
        meta = item.get("xiaoming") if isinstance(item.get("xiaoming"), dict) else {}
        if meta.get("kind") in PROTECTED_KINDS:
            protected.append(item)
    return protected


def _ordered_active_diaries(diaries: list[MemoryDiary]) -> list[MemoryDiary]:
    return sorted(
        [diary for diary in diaries if diary.status == "active"],
        key=lambda diary: (SCOPE_ORDER[diary.scope], diary.start_time, diary.id),
    )


def _diary_item(diary: MemoryDiary) -> dict[str, Any]:
    return {
        "role": "user",
        "content": (
            f'<memory_diary scope="{diary.scope}" start="{diary.start_time}" '
            f'end="{diary.end_time}" timezone="{diary.timezone}" id="{diary.id}">\n'
            f"{diary.body}\n"
            "</memory_diary>"
        ),
        "xiaoming": {
            "kind": "memory_diary",
            "diary_id": diary.id,
            "scope": diary.scope,
            "durable": True,
        },
    }
