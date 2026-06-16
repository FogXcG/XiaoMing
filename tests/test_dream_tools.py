from xiaoming.memory.dream_tools import DreamToolState, dream_tool_registry
from xiaoming.memory.models import MemoryFragment


def fragment(fragment_id):
    return MemoryFragment(
        id=fragment_id,
        source_event_id=fragment_id,
        role_or_type="user_message",
        created_at="2026-06-05T10:00:00+08:00",
        timezone="Asia/Shanghai",
        token_estimate=10,
        content=f"content {fragment_id}",
    )


def test_dream_tool_registry_exposes_only_dream_tools():
    state = DreamToolState(packets={"packet-1": [fragment("fragment-1")]})
    registry = dream_tool_registry(state)

    assert [spec.name for spec in registry.specs()] == [
        "list_memory_packets",
        "read_memory_packet",
        "write_diary_draft",
        "revise_diary_draft",
        "build_candidate_memory_view",
        "accept_dream",
        "reject_dream",
    ]


def test_write_diary_draft_records_draft_in_state():
    state = DreamToolState(packets={"packet-1": [fragment("fragment-1")]})
    registry = dream_tool_registry(state)

    result = registry.run(
        "write_diary_draft",
        {
            "scope": "day",
            "start_time": "2026-06-05T00:00:00+08:00",
            "end_time": "2026-06-06T00:00:00+08:00",
            "body": "I wrote a diary.",
            "source_ids": ["fragment-1"],
        },
    )

    assert result.status == "success"
    assert len(state.draft_diaries) == 1
    assert state.draft_diaries[0].body == "I wrote a diary."
