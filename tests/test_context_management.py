from __future__ import annotations

from xiaoming.context.manager import ContextManager, build_summary_item, estimate_tokens, recent_user_messages


def test_context_manager_repairs_dangling_tool_calls_for_prompt():
    items = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function"}]},
        {"role": "user", "content": "next"},
    ]

    prompt_items = ContextManager(items).for_prompt()

    assert prompt_items[1]["type"] == "function_call_output"
    assert prompt_items[1]["call_id"] == "call_1"
    assert "interrupted" in prompt_items[1]["output"]


def test_context_manager_drops_orphan_tool_outputs_for_prompt():
    items = [
        {"type": "function_call_output", "call_id": "missing", "output": "orphan"},
        {"role": "user", "content": "hello"},
    ]

    assert ContextManager(items).for_prompt() == [{"role": "user", "content": "hello"}]


def test_context_manager_renders_short_times_and_date_reminders():
    items = [
        {
            "role": "user",
            "content": "first",
            "xiaoming": {
                "kind": "user_message",
                "created_at": "2026-06-05T10:42:00+08:00",
                "date": "2026-06-05",
                "time": "10:42",
                "timezone": "Asia/Shanghai",
            },
        },
        {
            "role": "assistant",
            "content": "ok",
            "xiaoming": {
                "kind": "assistant_message",
                "created_at": "2026-06-05T10:43:00+08:00",
                "date": "2026-06-05",
                "time": "10:43",
                "timezone": "Asia/Shanghai",
            },
        },
        {
            "role": "user",
            "content": "next day",
            "xiaoming": {
                "kind": "user_message",
                "created_at": "2026-06-06T00:03:00+08:00",
                "date": "2026-06-06",
                "time": "00:03",
                "timezone": "Asia/Shanghai",
            },
        },
    ]

    prompt_items = ContextManager(items).for_prompt()

    assert prompt_items[0]["content"] == "[date=2026-06-05 tz=Asia/Shanghai]\n[@10:42] first"
    assert prompt_items[1]["content"] == "[@10:43] ok"
    assert prompt_items[2]["content"] == "[date=2026-06-06 tz=Asia/Shanghai]\n[@00:03] next day"
    assert items[0]["content"] == "first"


def test_recent_user_messages_keeps_newest_messages_within_budget():
    items = [
        {"role": "user", "content": "old " * 100},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "new"},
    ]

    recent = recent_user_messages(items, token_budget=estimate_tokens([{"role": "user", "content": "new"}]))

    assert recent == [{"role": "user", "content": "new"}]


def test_recent_user_messages_excludes_dynamic_worker_protocol_context():
    items = [
        {"role": "developer", "content": "protocol", "xiaoming": {"kind": "worker_protocol"}},
        {"role": "user", "content": "real user goal"},
    ]

    assert recent_user_messages(items, token_budget=1000) == [{"role": "user", "content": "real user goal"}]


def test_build_summary_item_marks_compacted_context():
    item = build_summary_item("important state")

    assert item["role"] == "user"
    assert "<conversation_summary>" in item["content"]
    assert item["xiaoming"]["kind"] == "context_summary"
