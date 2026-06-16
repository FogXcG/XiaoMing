from xiaoming.memory.models import DreamRun, MemoryDiary
from xiaoming.memory.store import MemoryStore
from xiaoming.session import Session


class Recorder:
    def __init__(self):
        self.events = []

    def append(self, session_id, event_type, payload):
        self.events.append((session_id, event_type, payload))


def test_memory_store_records_diary_and_run():
    session = Session(session_id="session-1")
    recorder = Recorder()
    store = MemoryStore(session, recorder=recorder)
    diary = MemoryDiary(
        id="diary-day-1",
        scope="day",
        start_time="2026-06-04T00:00:00+08:00",
        end_time="2026-06-05T00:00:00+08:00",
        timezone="Asia/Shanghai",
        status="draft",
        source_fragment_ids=["fragment-1"],
        body="I wrote a diary.",
        created_at="2026-06-05T01:00:00+08:00",
    )
    dream = DreamRun(id="dream-1", started_at="2026-06-05T01:00:00+08:00", status="running")

    store.save_diary(diary)
    store.save_dream_run(dream)

    assert session.memory_diaries["diary-day-1"] == diary
    assert session.dream_runs["dream-1"] == dream
    assert [event[1] for event in recorder.events] == ["memory_diary", "dream_run"]
