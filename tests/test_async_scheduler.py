import pytest
import json

from xiaoming.async_runtime.scheduler import LLMScheduler, SchedulerError
from xiaoming.async_runtime.tasks import TaskRecord, TaskRegistry
from xiaoming.llm.types import LLMResponse


class FakeProvider:
    def __init__(self, message):
        self.message = message
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(message=self.message, tool_calls=[], output_items=[], raw=None)


def test_llm_scheduler_uses_structured_model_decision():
    provider = FakeProvider(
        """
        {"action":"start_new_task","user_intent":"写 README","task_title":"README 任务","visible_message":"我已启动 README 任务。","reason":"new","target_task_id":null,"affected_files":["README.md"],"affected_modules":[],"domains":["docs"],"conflict_task_ids":[],"duplicate_task_ids":[]}
        """
    )

    decision = LLMScheduler(provider, "test-model").schedule("写 README", TaskRegistry())

    assert decision.action == "start_new_task"
    assert decision.affected_files == {"README.md"}
    assert provider.requests[0].model == "test-model"


def test_llm_scheduler_falls_back_when_model_returns_invalid_json():
    provider = FakeProvider("not json")

    with pytest.raises(SchedulerError) as exc:
        LLMScheduler(provider, "test-model").schedule("写 README", TaskRegistry())

    assert "failed after 3 attempts" in str(exc.value)
    assert len(provider.requests) == 3


def test_llm_scheduler_preserves_model_start_decision_without_action_guard():
    provider = FakeProvider(
        """
        {"action":"start_new_task","user_intent":"调整输入提示","task_title":"输入提示任务","visible_message":"开始。","reason":"model missed conflict","target_task_id":null,"affected_files":["src/xiaoming/cli.py"],"affected_modules":[],"domains":[],"conflict_task_ids":[],"duplicate_task_ids":[]}
        """
    )
    registry = TaskRegistry()
    registry.add(TaskRecord(title="改 CLI", original_request="改 CLI", current_goal="改 CLI", status="running", affected_files={"src/xiaoming/cli.py"}))

    decision = LLMScheduler(provider, "test-model").schedule("调整输入提示", registry)

    assert decision.action == "start_new_task"
    assert decision.conflict_task_ids == set()
    assert decision.visible_message == "开始。"


def test_llm_scheduler_rejects_queue_without_valid_conflict_ids():
    provider = FakeProvider(
        """
        {"action":"queue_task","user_intent":"开发五子棋","task_title":"五子棋任务","visible_message":"已排队。","reason":"another task is active","target_task_id":null,"affected_files":["gomoku.html"],"affected_modules":[],"domains":[],"conflict_task_ids":[],"duplicate_task_ids":[]}
        """
    )
    registry = TaskRegistry()
    registry.add(TaskRecord(title="安装 skill", original_request="安装 skill", current_goal="安装 skill", status="needs_user", affected_files={".agents/skills/superpowers/**"}, domains={"github.com"}))

    with pytest.raises(SchedulerError) as exc:
        LLMScheduler(provider, "test-model").schedule("开发五子棋", registry)

    assert "queue_task requires conflict_task_ids" in str(exc.value)
    assert len(provider.requests) == 3


def test_llm_scheduler_keeps_queue_with_valid_conflict_id():
    registry = TaskRegistry()
    running = registry.add(TaskRecord(title="改 gomoku", original_request="改 gomoku", current_goal="改 gomoku", status="running", affected_files={"gomoku.html"}))
    provider = FakeProvider(
        f"""
        {{"action":"queue_task","user_intent":"继续改 gomoku","task_title":"继续改 gomoku","visible_message":"已排队。","reason":"same file","target_task_id":null,"affected_files":["gomoku.html"],"affected_modules":[],"domains":[],"conflict_task_ids":["{running.task_id}"],"duplicate_task_ids":[]}}
        """
    )

    decision = LLMScheduler(provider, "test-model").schedule("继续改 gomoku", registry)

    assert decision.action == "queue_task"
    assert decision.conflict_task_ids == {running.task_id}


def test_llm_scheduler_only_sends_active_tasks_to_model():
    registry = TaskRegistry()
    registry.add(TaskRecord(title="已取消安装", original_request="安装 skill", current_goal="安装 skill", status="cancelled"))
    running = registry.add(TaskRecord(title="运行任务", original_request="查 openhuman", current_goal="查 openhuman", status="running"))
    provider = FakeProvider(
        """
        {"action":"start_new_task","user_intent":"写 README","task_title":"README 任务","visible_message":"开始。","reason":"new","target_task_id":null,"affected_files":[],"affected_modules":[],"domains":[],"conflict_task_ids":[],"duplicate_task_ids":[]}
        """
    )

    LLMScheduler(provider, "test-model").schedule("写 README", registry)

    prompt = json.loads(provider.requests[0].input_items[0]["content"])
    assert [task["task_id"] for task in prompt["tasks"]] == [running.task_id]


def test_llm_scheduler_rejects_queue_without_exact_conflict_reason():
    registry = TaskRegistry()
    running = registry.add(TaskRecord(title="改 gomoku", original_request="改 gomoku", current_goal="改 gomoku", status="running", affected_files={"gomoku.html"}))
    provider = FakeProvider(
        f"""
        {{"action":"queue_task","user_intent":"继续改 gomoku","task_title":"继续改 gomoku","visible_message":"已排队。","reason":"","target_task_id":null,"affected_files":["gomoku.html"],"affected_modules":[],"domains":[],"conflict_task_ids":["{running.task_id}"],"duplicate_task_ids":[]}}
        """
    )

    with pytest.raises(SchedulerError) as exc:
        LLMScheduler(provider, "test-model").schedule("继续改 gomoku", registry)

    assert "queue_task requires a reason" in str(exc.value)
    assert len(provider.requests) == 3


def test_llm_scheduler_rejects_missing_visible_message():
    provider = FakeProvider(
        """
        {"action":"start_new_task","user_intent":"写 README","task_title":"README 任务","reason":"new","target_task_id":null,"affected_files":["README.md"],"affected_modules":[],"domains":["docs"],"conflict_task_ids":[],"duplicate_task_ids":[]}
        """
    )

    with pytest.raises(SchedulerError) as exc:
        LLMScheduler(provider, "test-model").schedule("写 README", TaskRegistry())

    assert "missing visible_message" in str(exc.value)
