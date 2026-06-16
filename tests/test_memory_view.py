from xiaoming.memory.models import MemoryDiary
from xiaoming.memory.view import build_memory_view


def diary(diary_id, scope, start):
    return MemoryDiary(
        id=diary_id,
        scope=scope,
        start_time=start,
        end_time=start,
        timezone="Asia/Shanghai",
        status="active",
        source_fragment_ids=[],
        body=f"I am {diary_id}.",
        created_at=start,
    )


def test_memory_view_orders_diaries_by_scope_then_time():
    items = build_memory_view(
        diaries=[
            diary("day-1", "day", "2026-06-04T00:00:00+08:00"),
            diary("year-1", "year", "2025-01-01T00:00:00+08:00"),
            diary("month-1", "month", "2026-01-01T00:00:00+08:00"),
            diary("week-1", "week", "2026-05-25T00:00:00+08:00"),
        ],
        recent_items=[{"role": "user", "content": "recent"}],
    )

    contents = [item["content"] for item in items]
    assert 'scope="year"' in contents[0]
    assert 'scope="month"' in contents[1]
    assert 'scope="week"' in contents[2]
    assert 'scope="day"' in contents[3]
    assert contents[4] == "recent"


def test_memory_view_ignores_non_active_diaries():
    inactive = diary("draft-1", "day", "2026-06-04T00:00:00+08:00")
    inactive = MemoryDiary(**{**inactive.to_payload(), "status": "draft"})

    items = build_memory_view(diaries=[inactive], recent_items=[])

    assert items == []


def test_memory_view_keeps_protected_recent_items_after_diaries():
    items = build_memory_view(
        diaries=[diary("day-1", "day", "2026-06-04T00:00:00+08:00")],
        recent_items=[
            {"role": "developer", "content": "pending worker question", "xiaoming": {"kind": "worker_protocol"}},
            {"role": "user", "content": "latest user input", "xiaoming": {"kind": "user_message"}},
        ],
    )

    assert "<memory_diary" in items[0]["content"]
    assert items[1]["content"] == "pending worker question"
    assert items[2]["content"] == "latest user input"
