from xiaoming.async_runtime.context_packets import (
    ContextPacketBuilder,
    ResourceClaim,
    WorkerContextPacket,
    WorkerContextPolicy,
)
from xiaoming.async_runtime.tasks import TaskRegistry, TaskSpec
from xiaoming.session import Session


def test_context_packet_hash_is_stable_for_same_content():
    claim = ResourceClaim(resource="gomoku.html", kind="file", confidence="explicit")
    packet = WorkerContextPacket(
        task_id="task-1",
        session_id="session-1",
        agent_type="general-worker",
        context_policy="briefed",
        workspace="/repo",
        selected_skills=["brainstorming"],
        active_tasks_summary="none",
        source_item_ids=["item-1"],
        resource_claims=[claim],
        handoff_summary="Build a playable gomoku page.",
        decisions_already_made=["Use a standalone page."],
        relevant_messages=[],
    )

    restored = WorkerContextPacket.from_dict(packet.to_dict())

    assert restored.content_hash == packet.content_hash
    assert restored.resource_claims[0].resource == "gomoku.html"


def test_policy_modes_are_validated():
    policy = WorkerContextPolicy(mode="filtered_context_packet")

    assert policy.mode == "filtered_context_packet"


def test_isolated_packet_does_not_copy_history(tmp_path):
    session = Session(session_id="session-1")
    session.input_items.append({"role": "user", "content": "old context", "xiaoming": {"id": "old-1"}})
    builder = ContextPacketBuilder(tmp_path)

    packet = builder.build(
        session=session,
        task_id="task-1",
        agent_type="skill-installer-worker",
        context_policy="isolated",
        task_spec=TaskSpec(title="Install skill", goal="Install skill"),
        registry=TaskRegistry(),
        selected_skills=["skill-installer"],
    )

    assert packet.relevant_messages == []
    assert packet.selected_skills == ["skill-installer"]


def test_briefed_packet_records_source_item_ids(tmp_path):
    session = Session(session_id="session-1")
    session.input_items.append({"role": "user", "content": "Use standalone page", "xiaoming": {"id": "msg-1"}})
    builder = ContextPacketBuilder(tmp_path)

    packet = builder.build(
        session=session,
        task_id="task-1",
        agent_type="general-worker",
        context_policy="briefed",
        task_spec=TaskSpec(title="Build gomoku", goal="Build gomoku"),
        registry=TaskRegistry(),
        selected_skills=["brainstorming"],
    )

    assert "msg-1" in packet.source_item_ids
    assert packet.handoff_summary
