from xiaoming.memory.models import DreamRun, MemoryDiary, MemoryFragment
from xiaoming.session import Session


def test_memory_fragment_defaults_to_working_visibility():
    fragment = MemoryFragment(
        id="fragment-1",
        source_event_id="event-1",
        role_or_type="user",
        created_at="2026-06-05T10:42:00+08:00",
        timezone="Asia/Shanghai",
        token_estimate=12,
        content="hello",
    )

    assert fragment.visibility == "working"
    assert fragment.covered_by_diary_ids == []


def test_diary_round_trips_payload():
    diary = MemoryDiary(
        id="diary-day-1",
        scope="day",
        start_time="2026-06-04T00:00:00+08:00",
        end_time="2026-06-05T00:00:00+08:00",
        timezone="Asia/Shanghai",
        status="draft",
        source_fragment_ids=["fragment-1"],
        body="I remembered the day.",
        created_at="2026-06-05T01:00:00+08:00",
    )

    restored = MemoryDiary.from_payload(diary.to_payload())

    assert restored == diary


def test_session_clear_resets_memory_state():
    session = Session()
    session.memory_diaries["diary-day-1"] = MemoryDiary(
        id="diary-day-1",
        scope="day",
        start_time="2026-06-04T00:00:00+08:00",
        end_time="2026-06-05T00:00:00+08:00",
        timezone="Asia/Shanghai",
        status="active",
        source_fragment_ids=[],
        body="I remembered the day.",
        created_at="2026-06-05T01:00:00+08:00",
    )
    session.dream_runs["dream-1"] = DreamRun(
        id="dream-1",
        started_at="2026-06-05T01:00:00+08:00",
        status="accepted",
    )

    session.clear()

    assert session.memory_diaries == {}
    assert session.dream_runs == {}
