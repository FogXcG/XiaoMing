from xiaoming.async_runtime.events import WorkerEvent
import time

import pytest

from xiaoming.async_runtime.responder import LLMMessagingResponder, ResponderError
from xiaoming.async_runtime.tasks import TaskRecord, TaskRegistry
from xiaoming.llm.types import LLMResponse


class FakeProvider:
    def __init__(self, message):
        self.message = message
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return LLMResponse(message=self.message, tool_calls=[], output_items=[], raw=None)


class FailingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise RuntimeError("model down")


class EmptyProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        return LLMResponse(message="", tool_calls=[], output_items=[], raw=None)


def test_llm_responder_generates_user_reply_from_model():
    provider = FakeProvider("我会先判断这是不是 README 相关任务。")
    responder = LLMMessagingResponder(provider, "test-model")

    reply = responder.user_reply("顺便更新 README", TaskRegistry(), mode="received")

    assert reply == "我会先判断这是不是 README 相关任务。"
    assert provider.requests[0].model == "test-model"


def test_llm_responder_generates_worker_notice_from_model():
    provider = FakeProvider("README 已经更新完成。")
    responder = LLMMessagingResponder(provider, "test-model")
    registry = TaskRegistry()
    task = registry.add(TaskRecord(title="README 任务", original_request="更新 README", current_goal="更新 README"))

    notice = responder.worker_notice(WorkerEvent(task.task_id, "completed", "done"), task, registry)

    assert notice == "README 已经更新完成。"


def test_llm_responder_generates_command_reply_from_model():
    provider = FakeProvider("我已经取消当前任务。")
    responder = LLMMessagingResponder(provider, "test-model")

    reply = responder.command_reply("cancel_current", {"cancelled": True}, TaskRegistry())

    assert reply == "我已经取消当前任务。"


def test_llm_responder_retries_and_raises_after_failures():
    provider = FailingProvider()
    responder = LLMMessagingResponder(provider, "test-model", max_attempts=3)

    with pytest.raises(ResponderError) as exc:
        responder.user_reply("hi", TaskRegistry(), mode="received")

    assert provider.calls == 3
    assert "failed after 3 attempts" in str(exc.value)


def test_llm_responder_retries_empty_responses():
    provider = EmptyProvider()
    responder = LLMMessagingResponder(provider, "test-model", max_attempts=3)

    with pytest.raises(ResponderError) as exc:
        responder.user_reply("hi", TaskRegistry(), mode="received")

    assert provider.calls == 3
    assert "empty model response" in str(exc.value)
