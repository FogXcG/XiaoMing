from __future__ import annotations

import json
from pathlib import Path

from xiaoming.sessions.store import SessionStore, rehydrate_session


def test_session_store_appends_events_and_updates_index(tmp_path: Path):
    store = SessionStore(tmp_path)

    record = store.create(title="Build README", provider="deepseek", model="deepseek-v4-flash")
    store.append(record.id, "user_message", {"content": "Build README"})
    store.append(record.id, "assistant_message", {"content": "Done."})

    events = store.read_events(record.id)
    index = json.loads((tmp_path / ".xiaoming" / "sessions" / "index.json").read_text())

    assert [event["type"] for event in events] == ["session_meta", "user_message", "assistant_message"]
    assert index["sessions"][0]["id"] == record.id
    assert index["sessions"][0]["title"] == "Build README"
    assert index["sessions"][0]["turns"] == 1


def test_session_store_continues_latest_workspace_session(tmp_path: Path):
    store = SessionStore(tmp_path)
    old = store.create(title="Old", provider="deepseek", model="deepseek-v4-flash")
    store.append(old.id, "user_message", {"content": "old"})
    store.append(old.id, "assistant_message", {"content": "done"})
    new = store.create(title="New", provider="deepseek", model="deepseek-v4-flash")

    latest = store.latest_for_workspace()

    assert latest is not None
    assert latest.id == old.id
    assert latest.id != new.id


def test_session_store_ignores_user_only_sessions_when_resuming_latest(tmp_path: Path):
    store = SessionStore(tmp_path)
    completed = store.create(title="Completed", provider="deepseek", model="deepseek-v4-flash")
    store.append(completed.id, "user_message", {"content": "real task"})
    store.append(completed.id, "assistant_message", {"content": "done"})
    interrupted = store.create(title="Interrupted", provider="deepseek", model="deepseek-v4-flash")
    store.append(interrupted.id, "user_message", {"content": "session"})

    latest = store.latest_for_workspace()

    assert latest is not None
    assert latest.id == completed.id


def test_session_store_skips_bad_json_lines_when_reading(tmp_path: Path):
    store = SessionStore(tmp_path)
    record = store.create(title="Recover", provider="deepseek", model="deepseek-v4-flash")
    record.path.write_text(record.path.read_text() + "{bad json\n")
    store.append(record.id, "user_message", {"content": "hello"})

    events = store.read_events(record.id)

    assert [event["type"] for event in events] == ["session_meta", "user_message"]


def test_rehydrate_session_repairs_dangling_tool_call_output():
    session = rehydrate_session(
        [
            {"type": "user_message", "payload": {"content": "install"}},
            {
                "type": "assistant_output",
                "payload": {
                    "items": [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call_1", "type": "function"}],
                        }
                    ]
                },
            },
        ],
        session_id="session-1",
    )

    assert session.input_items[-1]["type"] == "function_call_output"
    assert session.input_items[-1]["call_id"] == "call_1"
    assert session.input_items[-1]["output"] == "Tool: unknown\nStatus: interrupted\nError:\nTool execution was interrupted before completion."
    assert session.input_items[-1]["xiaoming"]["time"]


def test_rehydrate_session_uses_event_time_for_message_metadata():
    session = rehydrate_session(
        [
            {
                "type": "user_message",
                "created_at": "2026-06-05T10:42:00+08:00",
                "payload": {"content": "hello"},
            },
            {
                "type": "assistant_message",
                "created_at": "2026-06-05T10:43:00+08:00",
                "payload": {"content": "hi"},
            },
        ],
        session_id="session-1",
    )

    assert session.input_items[0]["xiaoming"]["created_at"] == "2026-06-05T10:42:00+08:00"
    assert session.input_items[0]["xiaoming"]["time"] == "10:42"
    assert session.input_items[1]["xiaoming"]["created_at"] == "2026-06-05T10:43:00+08:00"
    assert session.input_items[1]["xiaoming"]["time"] == "10:43"


def test_rehydrate_session_restores_memory_diaries_and_dream_runs():
    from xiaoming.memory.models import DreamRun, MemoryDiary

    diary = MemoryDiary(
        id="diary-day-1",
        scope="day",
        start_time="2026-06-04T00:00:00+08:00",
        end_time="2026-06-05T00:00:00+08:00",
        timezone="Asia/Shanghai",
        status="active",
        source_fragment_ids=["fragment-1"],
        body="I wrote a diary.",
        created_at="2026-06-05T01:00:00+08:00",
        accepted_at="2026-06-05T01:10:00+08:00",
    )
    dream = DreamRun(
        id="dream-1",
        started_at="2026-06-05T01:00:00+08:00",
        ended_at="2026-06-05T01:10:00+08:00",
        status="accepted",
    )

    session = rehydrate_session(
        [
            {"type": "memory_diary", "payload": diary.to_payload()},
            {"type": "dream_run", "payload": dream.to_payload()},
        ],
        session_id="session-1",
    )

    assert session.memory_diaries["diary-day-1"] == diary
    assert session.dream_runs["dream-1"] == dream


def test_rehydrate_session_restores_loaded_skills():
    session = rehydrate_session(
        [
            {
                "type": "loaded_skill",
                "payload": {
                    "name": "brainstorming",
                    "description": "Explore requirements.",
                    "content": "Ask first.",
                    "path": ".agents/skills/brainstorming/SKILL.md",
                },
            }
        ],
        session_id="session-1",
    )

    assert session.loaded_skills["brainstorming"].content == "Ask first."
    assert session.loaded_skills["brainstorming"].path == ".agents/skills/brainstorming/SKILL.md"


def test_rehydrate_session_restores_bootstrap_contexts():
    session = rehydrate_session(
        [
            {
                "type": "bootstrap_context",
                "payload": {
                    "plugin_name": "superpowers",
                    "source": "superpowers:using-superpowers",
                    "content": "You have superpowers.",
                    "path": ".agents/skills/superpowers/skills/using-superpowers/SKILL.md",
                },
            }
        ],
        session_id="session-1",
    )

    assert session.bootstrap_contexts["superpowers:using-superpowers"].content == "You have superpowers."


def test_rehydrate_session_restores_turn_aborted_marker():
    session = rehydrate_session(
        [
            {"type": "user_message", "payload": {"content": "write file"}},
            {
                "type": "turn_aborted",
                "payload": {
                    "turn_id": "turn-1",
                    "reason": "user_interrupted",
                    "message": "The previous turn was interrupted by the user.",
                },
            },
        ],
        session_id="session-1",
    )

    assert session.input_items[-1]["role"] == "user"
    assert "<turn_aborted>" in session.input_items[-1]["content"]
    assert "interrupted by the user" in session.input_items[-1]["content"]


def test_rehydrate_session_replaces_history_on_context_compaction():
    session = rehydrate_session(
        [
            {"type": "user_message", "payload": {"content": "old"}},
            {"type": "assistant_message", "payload": {"content": "old answer"}},
            {
                "type": "context_compaction_completed",
                "payload": {
                    "created_at": "2026-05-11T00:00:00+00:00",
                    "replacement_items": [
                        {"role": "user", "content": "<conversation_summary>\nsummary\n</conversation_summary>", "xiaoming": {"kind": "context_summary"}}
                    ],
                },
            },
            {"type": "user_message", "payload": {"content": "new"}},
        ],
        session_id="session-1",
    )

    assert [item["content"] for item in session.input_items if item.get("role") == "user"] == [
        "<conversation_summary>\nsummary\n</conversation_summary>",
        "new",
    ]
    assert session.compaction_count == 1
    assert session.last_compacted_at == "2026-05-11T00:00:00+00:00"


def test_rehydrate_session_marks_unfinished_turn_as_aborted():
    session = rehydrate_session(
        [
            {"type": "turn_started", "payload": {"turn_id": "turn-1"}},
            {"type": "user_message", "payload": {"content": "write file"}},
        ],
        session_id="session-1",
    )

    assert session.input_items[-1]["role"] == "user"
    assert "<turn_aborted>" in session.input_items[-1]["content"]
