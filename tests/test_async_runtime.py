from xiaoming.async_runtime.tasks import TaskRecord, TaskRegistry, TaskResultReport, TaskSpec
from xiaoming.async_runtime.verifier import LLMTaskVerifier, _verification_from_json
from xiaoming.llm.types import LLMResponse, ToolCall
from xiaoming.tools.background_task import ScheduleBackgroundTaskTool
from xiaoming.tools.base import ToolResult


def test_task_registry_tracks_status_and_duplicates():
    registry = TaskRegistry()
    task = registry.add(TaskRecord(title="README 任务", original_request="写 README", current_goal="写 README", domains={"docs"}))
    task.transition("running", "started")

    assert registry.get(task.task_id) is task
    assert registry.find_duplicate("README 任务", "写 README") is task
    assert registry.active() == [task]


def test_task_registry_detects_file_conflicts():
    registry = TaskRegistry()
    registry.add(TaskRecord(title="CLI", original_request="改 cli", current_goal="改 cli", status="running", affected_files={"src/xiaoming/cli.py"}))

    conflicts = registry.conflicts_for({"src/xiaoming/cli.py"}, set(), set())

    assert [task.title for task in conflicts] == ["CLI"]


def test_task_spec_accepts_python_style_list_strings_from_models():
    spec = TaskSpec.from_dict(
        {
            "title": "开发五子棋网页",
            "goal": "创建 gomoku.html",
            "success_criteria": "['生成文件']",
            "expected_artifacts": "['gomoku.html']",
            "allowed_write_paths": "['gomoku.html']",
            "verification_commands": "['ls -la gomoku.html']",
            "notes": "",
        }
    )

    assert spec.expected_artifacts == ["gomoku.html"]
    assert spec.allowed_write_paths == ["gomoku.html"]
    assert spec.verification_commands == ["ls -la gomoku.html"]


def test_schedule_background_task_schema_only_exposes_message_and_task_name():
    spec = ScheduleBackgroundTaskTool(lambda: None).spec

    assert set(spec.input_schema["properties"]) == {"message", "task_name"}
    assert spec.input_schema["required"] == ["message"]
    assert "worker type" not in spec.description.lower()
    assert "context_policy" not in spec.description
    assert "skills_to_preload" not in spec.description


def test_schedule_background_task_accepts_plain_message():
    class FakeCoordinator:
        def __init__(self):
            self.task_spec = None

        def schedule_background_task(self, task_spec):
            self.task_spec = task_spec
            return ToolResult("schedule_background_task", "success", output="ok")

    coordinator = FakeCoordinator()
    tool = ScheduleBackgroundTaskTool(lambda: coordinator)

    result = tool.run(
        {
            "message": "Install the superpowers skill from https://github.com/obra/superpowers/tree/main/skills/brainstorming",
            "task_name": "Install brainstorming skill",
        }
    )

    assert result.status == "success"
    assert coordinator.task_spec.title == "Install brainstorming skill"
    assert coordinator.task_spec.goal == "Install the superpowers skill from https://github.com/obra/superpowers/tree/main/skills/brainstorming"
    assert coordinator.task_spec.agent_type == ""
    assert coordinator.task_spec.context_policy == ""
    assert coordinator.task_spec.skills_to_preload == []


def test_schedule_background_task_preserves_codex_request_from_turn_context():
    class FakeCoordinator:
        def __init__(self):
            self.task_spec = None

        def schedule_background_task(self, task_spec):
            self.task_spec = task_spec
            return ToolResult("schedule_background_task", "success", output="ok")

    coordinator = FakeCoordinator()
    tool = ScheduleBackgroundTaskTool(
        lambda: coordinator,
        turn_context_getter=lambda: "帮我用codex开发一个简单的象棋",
    )

    result = tool.run({"message": "开发一个简单的中国象棋网页", "task_name": "开发象棋网页"})

    assert result.status == "success"
    assert coordinator.task_spec.title == "开发象棋网页"
    assert "requested_executor=codex" in coordinator.task_spec.notes


def test_schedule_background_task_does_not_route_plain_codex_mentions():
    class FakeCoordinator:
        def __init__(self):
            self.task_spec = None

        def schedule_background_task(self, task_spec):
            self.task_spec = task_spec
            return ToolResult("schedule_background_task", "success", output="ok")

    coordinator = FakeCoordinator()
    tool = ScheduleBackgroundTaskTool(
        lambda: coordinator,
        turn_context_getter=lambda: "帮我写一个类似 codex 的介绍页面",
    )

    result = tool.run({"message": "写一个介绍页面", "task_name": "介绍页面"})

    assert result.status == "success"
    assert "requested_executor=codex" not in coordinator.task_spec.notes


def test_llm_verifier_response_parses_json_object():
    result = _verification_from_json('```json\n{"accepted": true, "reasons": ["ok"]}\n```')

    assert result.accepted is True
    assert result.reasons == ["ok"]


def test_llm_verifier_can_call_read_only_tools(tmp_path):
    (tmp_path / "gomoku.html").write_text("<!doctype html>\n")
    calls = []

    class FakeProvider:
        def complete(self, request):
            calls.append(request)
            if len(calls) == 1:
                return LLMResponse(
                    message=None,
                    tool_calls=[ToolCall(id="call-1", name="list_files", args={"path": ".", "pattern": "gomoku.html"})],
                    output_items=[{"role": "assistant", "tool_calls": [{"id": "call-1"}]}],
                    raw=None,
                )
            assert any(item.get("type") == "function_call_output" and "gomoku.html" in item.get("output", "") for item in request.input_items)
            return LLMResponse(message='{"accepted": true, "reasons": ["gomoku.html exists"]}', tool_calls=[], output_items=[{"role": "assistant", "content": '{"accepted": true, "reasons": ["gomoku.html exists"]}'}], raw=None)

    verifier = LLMTaskVerifier(tmp_path, FakeProvider(), "fake")

    result = verifier.verify(
        TaskSpec(title="开发五子棋", goal="生成 gomoku.html", expected_artifacts=["gomoku.html"]),
        report=TaskResultReport.from_dict(
            {"status": "completed", "summary": "done", "changed_files": [], "created_files": ["gomoku.html"], "artifacts": ["gomoku.html"], "verification": [], "blockers": [], "evidence": []}
        ),
    )

    assert result.accepted is True
    assert len(calls) == 2
