import copy
import json
from io import BytesIO
import threading
from typing import Any
import time
import zipfile

from xiaoming.agent_loop import AgentLoop
from xiaoming.agent_errors import AgentErrorInfo, FatalTurnError, ProviderCallError, RecoverableToolError
from xiaoming.hooks import HookManager
from xiaoming.logging import XiaomingLogger
from xiaoming.llm.streaming import StreamDone, StreamTextDelta, StreamToolCallDelta, StreamUsage
from xiaoming.llm.types import LLMRequest, LLMResponse, TokenUsage, ToolCall
from xiaoming.progress import ProgressEvent
from xiaoming.session import Session
from xiaoming.skills import Skill, SkillLibrary
from xiaoming.tools.base import ToolResult
from xiaoming.tools.load_skill import LoadSkillTool
from xiaoming.tools.install_skill import InstallSkillTool
from xiaoming.tools.registry import ToolRegistry


class RecordingSessionEvents:
    def __init__(self):
        self.events = []

    def append(self, session_id, event_type, payload):
        self.events.append((session_id, event_type, payload))


class FakeProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                message=None,
                tool_calls=[ToolCall(id="call_1", name="dummy", args={})],
                output_items=[{"type": "function_call", "call_id": "call_1", "name": "dummy", "arguments": "{}"}],
                raw=None,
            )
        return LLMResponse(message="Done.", tool_calls=[], output_items=[], raw=None)


class MultiToolProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                message=None,
                tool_calls=[
                    ToolCall(id="call_1", name="slow_read", args={"value": "first"}),
                    ToolCall(id="call_2", name="slow_read", args={"value": "second"}),
                ],
                output_items=[
                    {"type": "function_call", "call_id": "call_1", "name": "slow_read", "arguments": '{"value":"first"}'},
                    {"type": "function_call", "call_id": "call_2", "name": "slow_read", "arguments": '{"value":"second"}'},
                ],
                raw=None,
            )
        return LLMResponse(message="Done.", tool_calls=[], output_items=[], raw=None)


class RecoverableErrorProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                message=None,
                tool_calls=[],
                output_items=[{"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function"}]}],
                raw=None,
                recoverable_errors=[
                    RecoverableToolError(
                        call_id="call_1",
                        tool_name="apply_patch",
                        message="failed to parse tool arguments",
                        retry_hint="Retry with a smaller patch.",
                    )
                ],
            )
        return LLMResponse(message="Recovered.", tool_calls=[], output_items=[{"role": "assistant", "content": "Recovered."}], raw=None)


class FatalErrorProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            message=None,
            tool_calls=[],
            output_items=[],
            raw=None,
            fatal_error=FatalTurnError(source="deepseek", message="model output was truncated", hint="Try again with smaller chunks."),
        )


class SlowProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        time.sleep(0.03)
        return LLMResponse(message="Done.", tool_calls=[], output_items=[], raw=None)


class HangingProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        time.sleep(0.06)
        return LLMResponse(message="late", tool_calls=[], output_items=[], raw=None)


class TransientProvider:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) <= self.fail_times:
            raise ProviderCallError(AgentErrorInfo(kind="server_overloaded", message="server overloaded", retryable=True))
        return LLMResponse(message="Recovered.", tool_calls=[], output_items=[{"role": "assistant", "content": "Recovered."}], raw=None)


class FatalProviderCallProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise ProviderCallError(AgentErrorInfo(kind="bad_request", message="invalid schema", retryable=False))


class CapturingTextProvider:
    def __init__(self, message: str = "Done."):
        self.message = message
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(message=self.message, tool_calls=[], output_items=[{"role": "assistant", "content": self.message}], raw=None)


class CacheHitProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return LLMResponse(
            message="Worker done.",
            tool_calls=[],
            output_items=[{"role": "assistant", "content": "Worker done."}],
            raw=None,
            usage=TokenUsage(input_tokens=12_000, output_tokens=24, total_tokens=12_024, cached_tokens=10_800),
        )


class CompactingProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(message="summary of previous work", tool_calls=[], output_items=[], raw=None)
        return LLMResponse(message="Done.", tool_calls=[], output_items=[], raw=None, usage=TokenUsage(input_tokens=9, output_tokens=2, total_tokens=11, cached_tokens=3))


class StreamingTextProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming path should not call complete")

    def stream(self, request: LLMRequest):
        self.requests.append(request)
        yield StreamTextDelta("Hel")
        yield StreamTextDelta("lo")
        yield StreamDone("stop")


class StreamingUsageProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming path should not call complete")

    def stream(self, request: LLMRequest):
        yield StreamTextDelta("Hi")
        yield StreamDone("stop")
        yield StreamUsage(TokenUsage(input_tokens=10, output_tokens=2, total_tokens=12, cached_tokens=4))


class StreamingToolProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming path should not call complete")

    def stream(self, request: LLMRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield StreamTextDelta("I will use a tool.")
            yield StreamToolCallDelta(index=0, call_id="call_1", name="dummy", arguments_delta="{")
            yield StreamToolCallDelta(index=0, arguments_delta="}")
            yield StreamDone("tool_calls")
            return
        yield StreamTextDelta("Done.")
        yield StreamDone("stop")


class StreamingBadJsonProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming path should not call complete")

    def stream(self, request: LLMRequest):
        self.requests.append(request)
        if len(self.requests) == 1:
            yield StreamToolCallDelta(index=0, call_id="call_1", name="dummy", arguments_delta="{bad")
            yield StreamDone("tool_calls")
            return
        yield StreamTextDelta("Recovered.")
        yield StreamDone("stop")


class StreamingBadJsonFallbackProvider:
    def __init__(self):
        self.stream_requests: list[LLMRequest] = []
        self.complete_requests: list[LLMRequest] = []

    def stream(self, request: LLMRequest):
        self.stream_requests.append(request)
        yield StreamToolCallDelta(index=0, call_id="call_1", name="dummy", arguments_delta='{"content": "unterminated')
        yield StreamDone("tool_calls")

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.complete_requests.append(request)
        if len(self.complete_requests) == 1:
            return LLMResponse(
                message=None,
                tool_calls=[ToolCall(id="call_1", name="dummy", args={})],
                output_items=[{"type": "function_call", "call_id": "call_1", "name": "dummy", "arguments": "{}"}],
                raw=None,
            )
        return LLMResponse(message="Done.", tool_calls=[], output_items=[{"role": "assistant", "content": "Done."}], raw=None)


class StreamingIncompleteToolProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming path should not call complete")

    def stream(self, request: LLMRequest):
        yield StreamToolCallDelta(index=0, arguments_delta="{}")
        yield StreamDone("tool_calls")


class HangingStreamProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("streaming path should not call complete")

    def stream(self, request: LLMRequest):
        time.sleep(0.06)
        yield StreamTextDelta("late")


class ToolPrefaceProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                message="I will create the file with write_file.",
                tool_calls=[ToolCall(id="call_1", name="dummy", args={})],
                output_items=[
                    {
                        "role": "assistant",
                        "content": "I will create the file with write_file.",
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    }
                ],
                raw=None,
            )
        return LLMResponse(message="Done.", tool_calls=[], output_items=[], raw=None)


class LoadSkillProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if len(self.requests) == 1:
            return LLMResponse(
                message="I will load the relevant skill first.",
                tool_calls=[ToolCall(id="call_1", name="load_skill", args={"name": "frontend"})],
                output_items=[
                    {
                        "role": "assistant",
                        "content": "I will load the relevant skill first.",
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    }
                ],
                raw=None,
            )
        return LLMResponse(message="Done.", tool_calls=[], output_items=[], raw=None)


class DummyTool:
    name = "dummy"
    description = "Dummy."
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    @property
    def spec(self):
        from xiaoming.llm.types import ToolSpec

        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(self.name, "success", output="ok")


class SlowParallelTool:
    name = "slow_read"
    description = "Slow read-only test tool."
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    supports_parallel_tool_calls = True

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    @property
    def spec(self):
        from xiaoming.llm.types import ToolSpec

        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self.lock:
            self.active -= 1
        return ToolResult(self.name, "success", output=str(args["value"]))


class RepeatingStatusProvider:
    def __init__(self):
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        call_id = f"status-{len(self.requests)}"
        return LLMResponse(
            message=None,
            tool_calls=[ToolCall(id=call_id, name="background_tasks_status", args={})],
            output_items=[{"type": "function_call", "call_id": call_id, "name": "background_tasks_status", "arguments": "{}"}],
            raw=None,
        )


class BackgroundStatusTool:
    name = "background_tasks_status"
    description = "Show current background task status."
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    @property
    def spec(self):
        from xiaoming.llm.types import ToolSpec

        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(self.name, "success", output="查看项目文件结构  running  worker started")


class InterruptingTool:
    name = "interrupting_tool"
    description = "Interrupts."
    input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    @property
    def spec(self):
        from xiaoming.llm.types import ToolSpec

        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        raise KeyboardInterrupt()


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def test_agent_loop_appends_tool_output_and_finishes():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    answer = loop.run("do work")

    assert answer == "Done."
    second_input = provider.requests[1].input_items
    assert second_input[-1]["type"] == "function_call_output"
    assert second_input[-1]["call_id"] == "call_1"
    assert second_input[-1]["output"] == "Tool: dummy\nStatus: success\nOutput:\nok"
    assert second_input[-1]["xiaoming"]["time"]


def test_agent_loop_runs_parallel_safe_tools_concurrently_and_preserves_output_order():
    provider = MultiToolProvider()
    tool = SlowParallelTool()
    session = Session()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([tool]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    answer = loop.run("use tools", session=session)

    assert answer == "Done."
    assert tool.max_active == 2
    tool_outputs = [
        (item["call_id"], item["output"])
        for item in session.input_items
        if item.get("type") == "function_call_output" and item.get("call_id") in {"call_1", "call_2"}
    ]
    assert tool_outputs == [
        ("call_1", "Tool: slow_read\nStatus: success\nOutput:\nfirst"),
        ("call_2", "Tool: slow_read\nStatus: success\nOutput:\nsecond"),
    ]


def test_worker_forked_prefix_is_byte_identical_in_model_request_and_records_cache_usage(tmp_path):
    forked_prefix = [
        {"role": "developer", "content": "<bootstrap_context>stable context</bootstrap_context>", "xiaoming": {"kind": "bootstrap_context"}},
        {"role": "user", "content": "[@09:00] 先安装 superpowers", "xiaoming": {"id": "msg-1", "date": "2026-06-09", "time": "09:00", "timezone": "Asia/Shanghai"}},
        {"role": "assistant", "content": "[@09:00] 我会放到后台。", "xiaoming": {"id": "msg-2", "date": "2026-06-09", "time": "09:00", "timezone": "Asia/Shanghai"}},
        {"role": "user", "content": "[@09:05] 再开发五子棋", "xiaoming": {"id": "msg-3", "date": "2026-06-09", "time": "09:05", "timezone": "Asia/Shanghai"}},
        {"role": "assistant", "content": "[@09:05] 继续安排后台 worker。", "xiaoming": {"id": "msg-4", "date": "2026-06-09", "time": "09:05", "timezone": "Asia/Shanghai"}},
    ]
    provider = CacheHitProvider()
    logger = XiaomingLogger.create_worker(tmp_path, "worker-cache-test")
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="MAIN SYSTEM PROMPT",
        model="deepseek-v4-flash",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=1,
        stream=False,
        logger=logger,
        context_auto_compact=False,
    )
    worker_session = Session(session_id="worker-cache-test")
    worker_session.input_items.extend(copy.deepcopy(forked_prefix))

    answer = loop.run("<worker_protocol>\nGoal: continue the forked task.\n</worker_protocol>", session=worker_session)

    assert answer == "Worker done."
    request = provider.requests[0]
    assert _canonical_bytes(request.input_items[: len(forked_prefix)]) == _canonical_bytes(forked_prefix)
    assert any(
        item.get("role") == "user" and "Goal: continue the forked task." in str(item.get("content") or "")
        for item in request.input_items[len(forked_prefix) :]
    )
    assert worker_session.last_token_usage is not None
    assert worker_session.last_token_usage.cached_tokens == 10_800
    assert worker_session.last_token_usage.input_tokens == 12_000
    assert worker_session.last_token_usage.cached_tokens / worker_session.last_token_usage.input_tokens >= 0.8
    log_text = logger.path.read_text()
    assert '"cached_tokens": 10800' in log_text


def test_agent_loop_allows_background_status_snapshot_once_per_turn():
    provider = RepeatingStatusProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([BackgroundStatusTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=10,
    )

    answer = loop.run("查看后台状态")

    assert len(provider.requests) == 2
    assert "本轮已经查询过 background_tasks_status" in answer
    assert "查看项目文件结构  running  worker started" in answer


def test_agent_loop_exposes_prompt_snapshot_while_tool_runs():
    captured: dict[str, Any] = {}
    session = Session(session_id="session-1")

    class SnapshotTool(DummyTool):
        def run(self, args: dict[str, Any]) -> ToolResult:
            captured["prompt_items"] = copy.deepcopy(session.last_prompt_input_items)
            captured["model_output_items"] = copy.deepcopy(session.last_model_output_items)
            return super().run(args)

    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([SnapshotTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    loop.run("do work", session=session)

    assert captured["prompt_items"] == provider.requests[0].input_items
    output_items = captured["model_output_items"]
    assert len(output_items) == 1
    assert output_items[0]["type"] == "function_call"
    assert output_items[0]["call_id"] == "call_1"
    assert output_items[0]["name"] == "dummy"
    assert output_items[0]["arguments"] == "{}"
    assert output_items[0]["xiaoming"]["time"]


def test_agent_loop_preserves_rendered_fork_snapshot_as_prompt_prefix():
    forked_items = [
        {"role": "developer", "content": "<bootstrap_context>stable</bootstrap_context>", "xiaoming": {"kind": "bootstrap_context"}},
        {"role": "user", "content": "[date=2026-06-09 tz=Asia/Shanghai]\n[@10:00] 安装 skill", "xiaoming": {"id": "msg-1", "date": "2026-06-09", "time": "10:00", "timezone": "Asia/Shanghai"}},
        {"role": "assistant", "content": "我会安排。", "tool_calls": [{"id": "call_1", "type": "function"}]},
        {"type": "function_call_output", "call_id": "call_1", "output": "Fork started - processing in background"},
    ]
    provider = CapturingTextProvider()
    session = Session(session_id="worker-session")
    session.input_items.extend(copy.deepcopy(forked_items))
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    loop.run("worker task directive", session=session)

    assert provider.requests[0].input_items[: len(forked_items)] == forked_items


def test_agent_loop_refreshes_instructions_provider_between_turns():
    provider = CapturingTextProvider()
    current = {"instructions": "first instructions"}
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="fallback instructions",
        instructions_provider=lambda: current["instructions"],
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )
    session = Session(session_id="session-1")

    loop.run("first", session=session)
    current["instructions"] = "second instructions"
    loop.run("second", session=session)

    assert "first instructions" in provider.requests[0].instructions
    assert "fallback instructions" not in provider.requests[0].instructions
    assert "second instructions" in provider.requests[1].instructions
    assert "first instructions" not in provider.requests[1].instructions


def test_agent_loop_user_prompt_submit_hook_can_update_input():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"UserPromptSubmit": [lambda payload: {"updated_input": payload["user_input"] + " with hook"}]}),
    )

    loop.run("do work")

    user_messages = [item["content"] for item in provider.requests[0].input_items if item.get("role") == "user" and item.get("xiaoming", {}).get("kind") == "user_message"]
    assert len(user_messages) == 1
    assert user_messages[0].endswith("do work with hook")


def test_agent_loop_user_prompt_submit_hook_can_block_model_call():
    provider = CapturingTextProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"UserPromptSubmit": [lambda payload: {"continue": False, "reason": "blocked input"}]}),
    )

    answer = loop.run("do work")

    assert answer == "Stopped by UserPromptSubmit hook: blocked input"
    assert provider.requests == []


def test_agent_loop_session_start_hook_can_block_model_call():
    provider = CapturingTextProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"SessionStart": [lambda payload: {"continue": False, "reason": "blocked session"}]}),
    )

    answer = loop.run("do work", session=Session(session_id="session-1"))

    assert answer == "Stopped by SessionStart hook: blocked session"
    assert provider.requests == []


def test_agent_loop_user_prompt_submit_context_is_current_turn_only():
    provider = CapturingTextProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"UserPromptSubmit": [lambda payload: {"additional_context": "current hook context"}]}),
    )
    session = Session()

    loop.run("first", session=session)
    loop.run("second", session=session)

    first_rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[0].input_items)
    second_rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[1].input_items)
    assert "current hook context" in first_rendered
    assert second_rendered.count("current hook context") == 1
    assert not any(item.get("xiaoming", {}).get("kind") == "hook_context" for item in session.input_items)


def test_agent_loop_pre_tool_use_hook_can_deny_tool():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"PreToolUse": [lambda payload: {"decision": "deny", "reason": "blocked for test"}]}),
    )

    answer = loop.run("do work")

    assert answer == "Done."
    tool_output = next(item for item in provider.requests[1].input_items if item.get("type") == "function_call_output")
    assert "Status: denied" in tool_output["output"]
    assert "blocked for test" in tool_output["output"]


def test_agent_loop_post_tool_use_context_is_sent_to_followup():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"PostToolUse": [lambda payload: {"additional_context": "post tool hint"}]}),
    )

    loop.run("do work")

    second_rendered = "\n".join(str(item.get("content") or item.get("output") or "") for item in provider.requests[1].input_items)
    assert "post tool hint" in second_rendered


def test_agent_loop_post_tool_use_hook_can_stop_after_tool_result():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"PostToolUse": [lambda payload: {"continue": False, "reason": "stop after tool"}]}),
    )

    answer = loop.run("do work")

    assert answer == "Stopped by PostToolUse hook: stop after tool"
    assert len(provider.requests) == 1


def test_agent_loop_stop_hook_can_suppress_output():
    provider = CapturingTextProvider("final answer")
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager({"Stop": [lambda payload: {"suppress_output": True}]}),
    )

    assert loop.run("answer") == ""


def test_agent_loop_runs_session_start_post_tool_and_stop_hooks():
    events = []
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        hooks=HookManager(
            {
                "SessionStart": [lambda payload: events.append(("SessionStart", payload["session_id"]))],
                "PostToolUse": [lambda payload: events.append(("PostToolUse", payload["tool"], payload["status"]))],
                "Stop": [lambda payload: events.append(("Stop", payload["message"]))],
            }
        ),
    )
    session = Session(session_id="session-1")

    loop.run("do work", session=session)
    loop.run("more", session=session)

    assert events.count(("SessionStart", "session-1")) == 1
    assert ("PostToolUse", "dummy", "success") in events
    assert ("Stop", "Done.") in events


def test_agent_loop_records_interrupted_tool_output_before_reraising():
    class InterruptProvider:
        def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(
                message=None,
                tool_calls=[ToolCall(id="call_1", name="interrupting_tool", args={})],
                output_items=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "call_1", "type": "function"}],
                    }
                ],
                raw=None,
            )

    recorder = RecordingSessionEvents()
    session = Session(session_id="session-1")
    loop = AgentLoop(
        provider=InterruptProvider(),
        registry=ToolRegistry([InterruptingTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        session_recorder=recorder,
    )

    try:
        loop.run("do work", session=session)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert session.input_items[-2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "Tool: interrupting_tool\nStatus: interrupted\nError:\nTool execution interrupted by user before completion.",
    }
    assert session.input_items[-1]["role"] == "user"
    assert "<turn_aborted>" in session.input_items[-1]["content"]
    event_types = [event[1] for event in recorder.events]
    assert event_types[-3:] == ["tool_result", "tool_output", "turn_aborted"]
    assert recorder.events[-3][2]["status"] == "interrupted"
    assert recorder.events[-1][2]["reason"] == "user_interrupted"


def test_agent_loop_records_session_events_for_resume():
    recorder = RecordingSessionEvents()
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        session_recorder=recorder,
    )
    session = Session(session_id="session-1")

    answer = loop.run("do work", session=session)

    assert answer == "Done."
    event_types = [event[1] for event in recorder.events]
    assert event_types == [
        "base_instructions",
        "turn_started",
        "prompt_item",
        "prompt_item",
        "prompt_item",
        "user_message",
        "turn_context",
        "assistant_output",
        "tool_call",
        "tool_result",
        "tool_output",
        "assistant_message",
        "turn_completed",
    ]
    assert recorder.events[0][0] == "session-1"
    assert recorder.events[5][2]["content"] == "do work"
    assert recorder.events[8][2]["tool"] == "dummy"
    assert recorder.events[9][2]["status"] == "success"


def test_agent_loop_reuses_session_history_between_user_turns():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )
    session = Session()

    loop.run("first", session=session)
    loop.run("second", session=session)

    second_turn_items = provider.requests[2].input_items
    user_messages = [item["content"] for item in second_turn_items if item.get("role") == "user" and item.get("xiaoming", {}).get("kind") == "user_message"]

    assert [message.endswith(text) for message, text in zip(user_messages, ["first", "second"], strict=True)] == [True, True]
    assert user_messages[0].count("[@") == 1
    assert second_turn_items[-1]["content"].endswith("second")
    assert any(item.get("type") == "function_call_output" for item in second_turn_items)


def test_agent_loop_auto_compacts_large_history_before_turn():
    provider = CompactingProvider()
    recorder = RecordingSessionEvents()
    session = Session(session_id="session-1")
    session.input_items.extend(
        [
            {"role": "user", "content": "old request " * 200},
            {"role": "assistant", "content": "old answer " * 200},
        ]
    )
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        session_recorder=recorder,
        context_compact_threshold_tokens=20,
    )

    answer = loop.run("continue", session=session)

    assert answer == "Done."
    assert len(provider.requests) == 2
    assert provider.requests[0].tools == []
    assert "Create a compact checkpoint summary" in provider.requests[0].input_items[-1]["content"]
    assert any(item.get("xiaoming", {}).get("kind") == "context_summary" for item in session.input_items)
    assert session.reference_turn_context is not None
    assert session.last_token_usage is not None
    assert session.last_token_usage.cached_tokens == 3
    assert "context_compaction_completed" in [event[1] for event in recorder.events]


def test_agent_loop_dream_context_runs_dream_runner(monkeypatch):
    provider = CapturingTextProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="deepseek-v4-flash",
        temperature=0.0,
        max_output_tokens=4096,
        max_turns=3,
    )
    session = Session(session_id="session-1")

    class FakeDreamRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, session):
            from xiaoming.memory.dream_runner import DreamResult

            return DreamResult(accepted=True, reason="ok", draft_count=1)

    monkeypatch.setattr("xiaoming.agent_loop.DreamRunner", FakeDreamRunner)

    assert loop.dream_context(session) == "Dream accepted: 1 diary draft(s). Reason: ok"


def test_agent_loop_pre_compact_hook_can_skip_auto_compaction():
    provider = CapturingTextProvider()
    session = Session(session_id="session-1")
    session.input_items.append({"role": "user", "content": "large " * 200})
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        context_compact_threshold_tokens=20,
        hooks=HookManager({"PreCompact": [lambda payload: {"decision": "deny", "reason": "keep history"}]}),
    )

    loop.run("continue", session=session)

    assert len(provider.requests) == 1
    assert not any(item.get("xiaoming", {}).get("kind") == "context_summary" for item in session.input_items)


def test_agent_loop_post_compact_context_is_visible_on_compacted_turn():
    provider = CompactingProvider()
    session = Session(session_id="session-1")
    session.input_items.append({"role": "user", "content": "large " * 200})
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        context_compact_threshold_tokens=20,
        hooks=HookManager({"PostCompact": [lambda payload: {"additional_context": "after compact hint"}]}),
    )

    loop.run("continue", session=session)

    rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[1].input_items)
    assert "after compact hint" in rendered


def test_agent_loop_background_context_is_visible_and_compacted():
    provider = CompactingProvider()
    session = Session(session_id="session-1")
    session.input_items.append({"role": "user", "content": "large " * 200})
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        context_compact_threshold_tokens=20,
        runtime_context_provider=lambda: "Background tasks:\n- 写 README running",
    )

    loop.run("continue", session=session)

    compaction_rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[0].input_items)
    turn_rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[1].input_items)
    assert "Background tasks" in compaction_rendered
    assert "写 README running" in compaction_rendered
    assert "Background tasks" in turn_rendered
    assert "写 README running" in turn_rendered


def test_agent_loop_sends_recoverable_tool_errors_back_to_model():
    provider = RecoverableErrorProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    answer = loop.run("do work")

    assert answer == "Recovered."
    retry_items = provider.requests[1].input_items
    assert retry_items[-1]["type"] == "function_call_output"
    assert retry_items[-1]["call_id"] == "call_1"
    assert "failed to parse tool arguments" in retry_items[-1]["output"]
    assert "Retry with a smaller patch." in retry_items[-1]["output"]


def test_agent_loop_emits_recoverable_tool_error_detail():
    provider = RecoverableErrorProvider()
    events = []
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    loop.run("do work", on_event=events.append)

    assert "Recovering from tool-call error: apply_patch - failed to parse tool arguments" in events


def test_agent_loop_returns_fatal_turn_error_without_polluting_session():
    session = Session()
    loop = AgentLoop(
        provider=FatalErrorProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    answer = loop.run("do work", session=session)

    assert "model output was truncated" in answer
    assert session.input_items == []
    assert session.pending_turn is not None
    assert session.pending_turn.status == "failed"


def test_agent_loop_emits_progress_before_model_and_tools():
    provider = FakeProvider()
    events = []
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    answer = loop.run("do work", on_event=events.append)

    assert answer == "Done."
    assert events[0] == "Thinking about the next step..."
    assert "Running tool: dummy" in events
    assert "Tool completed: dummy (success)" in events


def test_agent_loop_emits_tool_error_detail_in_progress():
    class ErrorTool:
        name = "error_tool"
        description = "Error."
        input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

        @property
        def spec(self):
            from xiaoming.llm.types import ToolSpec

            return ToolSpec(self.name, self.description, self.input_schema)

        def run(self, args):
            return ToolResult(self.name, "error", error="unknown skill: writing-plans")

    class ErrorProvider:
        def __init__(self):
            self.calls = 0

        def complete(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    message=None,
                    tool_calls=[ToolCall(id="call_1", name="error_tool", args={})],
                    output_items=[{"type": "function_call", "call_id": "call_1", "name": "error_tool", "arguments": "{}"}],
                    raw=None,
                )
            return LLMResponse(message="Done.", tool_calls=[], output_items=[], raw=None)

    events = []
    loop = AgentLoop(
        provider=ErrorProvider(),
        registry=ToolRegistry([ErrorTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    loop.run("do work", on_event=events.append)

    assert "Tool completed: error_tool (error: unknown skill: writing-plans)" in events


def test_agent_loop_includes_tool_intent_when_registry_can_describe_call():
    class DescribingRegistry(ToolRegistry):
        def describe_call(self, name, args):
            return "create file hello.txt"

    provider = FakeProvider()
    events = []
    loop = AgentLoop(
        provider=provider,
        registry=DescribingRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    answer = loop.run("do work", on_event=events.append)

    assert answer == "Done."
    assert "Running tool: dummy - create file hello.txt" in events


def test_agent_loop_emits_waiting_progress_during_slow_model_call():
    events = []
    loop = AgentLoop(
        provider=SlowProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        progress_interval_seconds=0.01,
    )

    answer = loop.run("do work", on_event=events.append)

    assert answer == "Done."
    assert "Still waiting for model response..." in events


def test_agent_loop_times_out_hanging_model_calls():
    events = []
    loop = AgentLoop(
        provider=HangingProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        progress_interval_seconds=0.01,
        model_timeout_seconds=0.02,
    )

    answer = loop.run("do work", on_event=events.append)

    assert "model response timed out after 0.02 seconds" in answer
    assert "Still waiting for model response..." in events


def test_agent_loop_retries_retryable_provider_errors():
    events = []
    provider = TransientProvider(fail_times=2)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        retry_base_delay_seconds=0,
    )

    answer = loop.run("do work", on_event=events.append)

    assert answer == "Recovered."
    assert len(provider.requests) == 3
    assert "Model call failed; retrying 1/3: server overloaded" in events
    assert "Model call failed; retrying 2/3: server overloaded" in events


def test_agent_loop_does_not_retry_non_retryable_provider_errors():
    provider = FatalProviderCallProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        retry_base_delay_seconds=0,
    )

    answer = loop.run("do work")

    assert "invalid schema" in answer


def test_agent_loop_streams_text_deltas_without_repeating_final_message():
    events = []
    loop = AgentLoop(
        provider=StreamingTextProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        stream=True,
    )

    answer = loop.run("say hello", on_event=events.append)

    assert answer == ""
    assert events == [ProgressEvent("status", "Thinking about the next step..."), ProgressEvent("text_delta", "Hel", end=""), ProgressEvent("text_delta", "lo", end="")]


def test_agent_loop_logs_streaming_usage(tmp_path):
    logger = XiaomingLogger.create(tmp_path)
    loop = AgentLoop(
        provider=StreamingUsageProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        stream=True,
        logger=logger,
    )

    answer = loop.run("say hello")

    assert answer == ""
    log_text = logger.path.read_text()
    assert '"event": "model_stream_usage"' in log_text
    assert '"input_tokens": 10' in log_text
    assert '"cached_tokens": 4' in log_text
    assert '"event": "model_call_finished"' in log_text


def test_agent_loop_streaming_tool_call_executes_after_arguments_complete():
    events = []
    provider = StreamingToolProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        stream=True,
    )

    answer = loop.run("use tool", on_event=events.append)

    assert answer == ""
    assert len(provider.requests) == 2
    assert any(event == ProgressEvent("tool_started", "Running tool: dummy") for event in events)
    assert any(event == ProgressEvent("tool_finished", "Tool completed: dummy (success)") for event in events)


def test_agent_loop_streaming_bad_tool_json_falls_back_to_non_stream_for_current_call():
    provider = StreamingBadJsonFallbackProvider()
    events = []
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        stream=True,
    )

    answer = loop.run("use tool", on_event=events.append)

    assert answer == "Done."
    assert len(provider.stream_requests) == 2
    assert len(provider.complete_requests) == 2
    assert loop.stream is True
    assert ProgressEvent("status", "Streaming tool arguments were incomplete; retrying this model call without streaming.") in events


def test_agent_loop_streaming_incomplete_tool_call_is_fatal():
    loop = AgentLoop(
        provider=StreamingIncompleteToolProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        stream=True,
    )

    answer = loop.run("use tool")

    assert "incomplete streamed tool call" in answer


def test_agent_loop_stream_idle_timeout():
    loop = AgentLoop(
        provider=HangingStreamProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        stream=True,
        stream_idle_timeout_seconds=0.01,
    )

    answer = loop.run("hang")

    assert "model stream idle timed out after 0.01 seconds" in answer


def test_agent_loop_emits_model_tool_preface_before_running_tool():
    events = []
    loop = AgentLoop(
        provider=ToolPrefaceProvider(),
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
    )

    answer = loop.run("do work", on_event=events.append)

    assert answer == "Done."
    assert events.index("I will create the file with write_file.") < events.index("Running tool: dummy")


def test_agent_loop_logs_turns_model_calls_and_tools(tmp_path):
    provider = FakeProvider()
    logger = XiaomingLogger.create(tmp_path)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()], logger=logger),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        logger=logger,
    )

    loop.run("do work")

    log_text = logger.path.read_text()
    assert '"event": "turn_started"' in log_text
    assert '"event": "model_call_started"' in log_text
    assert '"event": "model_call_finished"' in log_text
    assert '"event": "tool_call_started"' in log_text
    assert '"event": "tool_call_finished"' in log_text


def test_agent_loop_logs_fatal_model_errors(tmp_path):
    logger = XiaomingLogger.create(tmp_path)
    loop = AgentLoop(
        provider=FatalErrorProvider(),
        registry=ToolRegistry([DummyTool()], logger=logger),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        logger=logger,
    )

    loop.run("do work")

    log_text = logger.path.read_text()
    assert '"event": "model_fatal_error"' in log_text
    assert "model output was truncated" in log_text


def test_agent_loop_injects_explicitly_mentioned_skill_instructions():
    provider = FakeProvider()
    events = []
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        skill_library=SkillLibrary([Skill(name="frontend", description="Build UI.", content="Use semantic markup.")]),
    )

    loop.run("用 $frontend 写页面", on_event=events.append)

    assert "Loaded skill: frontend" in events
    rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[0].input_items)
    assert "Active skills:" in rendered
    assert "Use semantic markup." in rendered


def test_agent_loop_auto_injects_explicit_plain_skill_mentions():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        skill_library=SkillLibrary([Skill(name="brainstorming", description="Explore requirements.", content="Ask before implementation.")]),
    )

    loop.run("用 brainstorming 讨论需求")

    rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[0].input_items)
    assert "<name>brainstorming</name>" in rendered
    assert "Ask before implementation." in rendered


def test_agent_loop_does_not_auto_inject_unmentioned_semantic_skill_matches():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        skill_library=SkillLibrary([Skill(name="brainstorming", description="Use before creative work.", content="Ask before implementation.")]),
    )

    loop.run("帮我写个网页")

    rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[0].input_items)
    assert "<name>brainstorming</name>" not in rendered
    assert "Ask before implementation." not in rendered


def test_agent_loop_advertises_available_skills_without_full_content():
    provider = FakeProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([DummyTool()]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        skill_library=SkillLibrary([Skill(name="frontend", description="Build UI.", content="Use semantic markup.")]),
    )

    loop.run("写页面")

    rendered = "\n".join(str(item.get("content") or "") for item in provider.requests[0].input_items)
    assert "Available skills:" in rendered
    assert "frontend - Build UI." in rendered
    assert "load_skill" in rendered
    assert "Use semantic markup." not in rendered


def test_agent_loop_can_load_skill_through_tool():
    library = SkillLibrary([Skill(name="frontend", description="Build UI.", content="Use semantic markup.")])
    provider = LoadSkillProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([LoadSkillTool(library)]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        skill_library=library,
    )

    session = Session()

    answer = loop.run("写页面", session=session)

    assert answer == "Done."
    tool_output = next(item["output"] for item in provider.requests[1].input_items if item.get("type") == "function_call_output")
    assert "<skill>" in tool_output
    assert "<name>frontend</name>" in tool_output
    assert "Use semantic markup." in tool_output
    assert "frontend" in session.loaded_skills
    rendered = "\n".join(str(item.get("content") or item.get("output") or "") for item in provider.requests[1].input_items)
    assert "<skill>" in rendered
    assert "<name>frontend</name>" in rendered
    assert "Use semantic markup." in rendered
    assert "active workflow instructions" not in rendered
    assert "skill controls HOW" not in rendered


def test_agent_loop_logs_loaded_skill_retention(tmp_path):
    library = SkillLibrary([Skill(name="frontend", description="Build UI.", content="Use semantic markup.")])
    provider = LoadSkillProvider()
    logger = XiaomingLogger.create(tmp_path)
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([LoadSkillTool(library)], logger=logger),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        skill_library=library,
        logger=logger,
    )

    loop.run("写页面", session=Session())

    log_text = logger.path.read_text()
    assert '"event": "loaded_skill_remembered"' in log_text
    assert '"skill": "frontend"' in log_text


def test_agent_loop_retains_loaded_skill_context_across_user_turns():
    library = SkillLibrary([Skill(name="frontend", description="Build UI.", content="Use semantic markup.")])
    provider = LoadSkillProvider()
    loop = AgentLoop(
        provider=provider,
        registry=ToolRegistry([LoadSkillTool(library)]),
        instructions="rules",
        model="gpt-5",
        temperature=0.2,
        max_output_tokens=4096,
        max_turns=3,
        skill_library=library,
    )
    session = Session()

    loop.run("写页面", session=session)
    loop.run("继续", session=session)

    rendered = "\n".join(str(item.get("content") or item.get("output") or "") for item in provider.requests[2].input_items)
    assert '<skill>\n<name>frontend</name>' in rendered
    assert "Use semantic markup." in rendered


def test_load_skill_tool_refreshes_library_on_miss(tmp_path):
    library = SkillLibrary.discover(tmp_path)
    skill_dir = tmp_path / ".agents" / "skills" / "writing-plans"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: writing-plans\ndescription: Write plans.\n---\nBody\n")

    result = LoadSkillTool(library, workspace=tmp_path).run({"name": "writing-plans"})

    assert result.status == "success"
    assert "<name>writing-plans</name>" in result.output
    assert library.load("writing-plans") is not None


def test_install_skill_tool_refreshes_library_after_install(tmp_path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _skill_repo_zip(
            {"skills/frontend/SKILL.md": b"---\nname: frontend\ndescription: Build UI.\n---\nBody\n"}
        ),
    }
    library = SkillLibrary.discover(tmp_path)
    tool = InstallSkillTool(tmp_path, library, approval_mode="auto_edit", approve=lambda action: True, fetch=lambda url: responses[url])

    result = tool.run({"url": "https://github.com/acme/skills/tree/main/skills/frontend"})

    assert result.status == "success"
    assert "Installed skill: frontend" in result.output
    assert library.load("frontend") is not None


def test_install_skill_tool_accepts_repo_and_paths(tmp_path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _skill_repo_zip(
            {
                "skills/frontend/SKILL.md": b"---\nname: frontend\n---\nFrontend\n",
                "skills/backend/SKILL.md": b"---\nname: backend\n---\nBackend\n",
            }
        ),
    }
    library = SkillLibrary.discover(tmp_path)
    tool = InstallSkillTool(tmp_path, library, approval_mode="auto_edit", approve=lambda action: True, fetch=lambda url: responses[url])

    result = tool.run({"repo": "acme/skills", "paths": ["skills/frontend", "skills/backend"]})

    assert result.status == "success"
    assert "Installed skill: frontend" in result.output
    assert "Installed skill: backend" in result.output
    assert library.load("frontend") is not None
    assert library.load("backend") is not None


def test_install_skill_tool_rejects_string_paths(tmp_path):
    library = SkillLibrary.discover(tmp_path)
    tool = InstallSkillTool(tmp_path, library, approval_mode="auto_edit", approve=lambda action: True, fetch=lambda url: b"")

    result = tool.run({"repo": "acme/skills", "paths": "skills/frontend"})

    assert result.status == "error"
    assert "paths must be an array" in result.error


def _skill_repo_zip(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            archive.writestr(f"skills-main/{path}", content)
    return buffer.getvalue()
