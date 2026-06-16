from pathlib import Path

from xiaoming.llm.types import ToolSpec
from xiaoming.prompting.base import build_base_instructions
from xiaoming.prompting.runtime import PromptRuntime, RuntimeState
from xiaoming.session import BootstrapContext, LoadedSkill, Session
from xiaoming.prompting.items import PromptItem
from xiaoming.time_meta import runtime_timezone_name


def test_base_instructions_has_no_dynamic_fields():
    instructions = build_base_instructions("rules")

    assert "rules" in instructions
    assert "cwd" not in instructions.lower()
    assert "last_error" not in instructions


def test_prompt_item_records_local_time_metadata():
    item = PromptItem.user("turn-1", "hello")
    item.created_at = "2026-06-05T10:42:30+08:00"

    meta = item.to_input_item()["xiaoming"]

    assert meta["created_at"] == "2026-06-05T10:42:30+08:00"
    assert meta["date"] == "2026-06-05"
    assert meta["time"] == "10:42"
    assert meta["timezone"] == "Asia/Shanghai"


def test_runtime_timezone_ignores_invalid_absolute_tz_path(monkeypatch):
    monkeypatch.setenv("XIAOMING_TIMEZONE", "/UTC")
    monkeypatch.setenv("TZ", "UTC")

    assert runtime_timezone_name() == "UTC"


def test_first_turn_stages_context_before_user_without_committing():
    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")

    compiled = runtime.prepare_turn(
        session,
        "hello",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [],
    )

    assert session.input_items == []
    assert [item.get("xiaoming", {}).get("kind") for item in compiled.input_items] == [
        "developer_context",
        "environment_context",
        "ephemeral_context",
        "user_message",
    ]
    assert compiled.input_items[-1]["content"].endswith("hello")
    assert "[@ " not in compiled.input_items[-1]["content"]
    assert "[@" in compiled.input_items[-1]["content"]


def test_commit_staged_turn_appends_once_and_sets_baseline():
    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    state = RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True)
    runtime.prepare_turn(session, "hello", state, [])

    first = runtime.commit_turn(session)
    second = runtime.commit_turn(session)

    assert len(first) == 4
    assert second == []
    assert session.reference_turn_context is not None
    assert session.input_items[-1]["content"] == "hello"


def test_second_turn_only_injects_ephemeral_and_user_when_context_unchanged():
    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    state = RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True)
    runtime.prepare_turn(session, "first", state, [])
    runtime.commit_turn(session)

    compiled = runtime.prepare_turn(session, "second", state, [])
    staged_kinds = [item.get("xiaoming", {}).get("kind") for item in compiled.input_items[len(session.input_items) :]]

    assert staged_kinds == ["ephemeral_context", "user_message"]


def test_tool_schema_not_rendered_into_context():
    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    tool = ToolSpec(name="dangerous_tool", description="tool schema text", input_schema={"type": "object"})

    compiled = runtime.prepare_turn(
        session,
        "hello",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [tool],
    )
    rendered = "\n".join(str(item.get("content") or "") for item in compiled.input_items)

    assert compiled.tools == [tool]
    assert "tool schema text" not in rendered


def test_prepare_turn_replaces_stale_base_instructions():
    session = Session(session_id="s1", base_instructions=build_base_instructions("old"), base_instructions_recorded=True)
    runtime = PromptRuntime("new")

    compiled = runtime.prepare_turn(
        session,
        "hello",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [],
    )

    assert "new" in compiled.instructions
    assert "old" not in compiled.instructions
    assert session.base_instructions_recorded is False


def test_prepare_turn_prepends_loaded_skill_context_without_committing_it():
    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    session.remember_loaded_skill(
        LoadedSkill.create(
            name="brainstorming",
            description="Explore requirements.",
            content="Ask questions before implementation.",
            path=".agents/skills/brainstorming/SKILL.md",
        )
    )

    compiled = runtime.prepare_turn(
        session,
        "build a page",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [],
    )

    assert compiled.input_items[0]["xiaoming"]["kind"] == "loaded_skill"
    assert "<name>brainstorming</name>" in compiled.input_items[0]["content"]
    assert "Ask questions before implementation." in compiled.input_items[0]["content"]
    assert session.input_items == []


def test_prepare_turn_uses_memory_view_when_active_diaries_exist():
    from xiaoming.memory.models import MemoryDiary

    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    session.input_items.append({"role": "user", "content": "raw old"})
    session.memory_diaries["diary-day-1"] = MemoryDiary(
        id="diary-day-1",
        scope="day",
        start_time="2026-06-04T00:00:00+08:00",
        end_time="2026-06-05T00:00:00+08:00",
        timezone="Asia/Shanghai",
        status="active",
        source_fragment_ids=[],
        body="I wrote a diary.",
        created_at="2026-06-05T01:00:00+08:00",
    )

    compiled = runtime.prepare_turn(
        session,
        "hello",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [],
    )
    rendered = "\n".join(str(item.get("content") or "") for item in compiled.input_items)

    assert "<memory_diary" in rendered
    assert "I wrote a diary." in rendered
    assert "raw old" not in rendered


def test_prepare_turn_keeps_protected_worker_context_with_diaries():
    from xiaoming.memory.models import MemoryDiary

    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    session.input_items.append({"role": "developer", "content": "pending worker question", "xiaoming": {"kind": "worker_protocol"}})
    session.memory_diaries["diary-day-1"] = MemoryDiary(
        id="diary-day-1",
        scope="day",
        start_time="2026-06-04T00:00:00+08:00",
        end_time="2026-06-05T00:00:00+08:00",
        timezone="Asia/Shanghai",
        status="active",
        source_fragment_ids=[],
        body="I wrote a diary.",
        created_at="2026-06-05T01:00:00+08:00",
    )

    compiled = runtime.prepare_turn(
        session,
        "hello",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [],
    )
    rendered = "\n".join(str(item.get("content") or "") for item in compiled.input_items)

    assert "I wrote a diary." in rendered
    assert "pending worker question" in rendered


def test_prepare_turn_prepends_bootstrap_context_before_loaded_skills():
    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    session.remember_bootstrap_context(
        BootstrapContext.create(
            plugin_name="superpowers",
            source="superpowers:using-superpowers",
            content="You have superpowers.",
            path=".agents/skills/superpowers/skills/using-superpowers/SKILL.md",
        )
    )
    session.remember_loaded_skill(
        LoadedSkill.create(
            name="brainstorming",
            description="Explore requirements.",
            content="Ask questions before implementation.",
        )
    )

    compiled = runtime.prepare_turn(
        session,
        "build a page",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [],
    )

    assert compiled.input_items[0]["xiaoming"]["kind"] == "bootstrap_context"
    assert compiled.input_items[1]["xiaoming"]["kind"] == "loaded_skill"
    assert '<bootstrap_context source="superpowers:using-superpowers"' in compiled.input_items[0]["content"]


def test_prepare_turn_places_dynamic_worker_bootstrap_after_loaded_skills():
    runtime = PromptRuntime("rules")
    session = Session(session_id="s1")
    session.remember_bootstrap_context(
        BootstrapContext.create(
            plugin_name="runtime",
            source="runtime:worker-protocol",
            content="stable worker protocol",
        )
    )
    session.remember_bootstrap_context(
        BootstrapContext.create(
            plugin_name="runtime",
            source="runtime:task-contract",
            content="dynamic task contract",
        )
    )
    session.remember_loaded_skill(
        LoadedSkill.create(
            name="brainstorming",
            description="Explore requirements.",
            content="Ask questions before implementation.",
        )
    )

    compiled = runtime.prepare_turn(
        session,
        "start",
        RuntimeState(cwd=Path("/tmp/work"), provider="deepseek", model="deepseek-v4-flash", stream=True),
        [],
    )

    contents = [str(item.get("content") or "") for item in compiled.input_items]
    protocol_index = next(index for index, content in enumerate(contents) if "stable worker protocol" in content)
    skill_index = next(index for index, content in enumerate(contents) if "<name>brainstorming</name>" in content)
    contract_index = next(index for index, content in enumerate(contents) if "dynamic task contract" in content)
    assert protocol_index < skill_index < contract_index
