from argparse import Namespace
import time

import pytest

from xiaoming.async_runtime.events import CoordinatorNotice
from xiaoming.async_runtime.scheduler import SchedulerDecision
from xiaoming.cli import AsyncNoticeBuffer, ChatRuntime, build_loop, run_chat
from xiaoming.llm.types import LLMResponse, ToolCall


class FakeCoordinator:
    def __init__(self, config, on_notice):
        self.config = config
        self.on_notice = on_notice
        self.started = False
        self.stopped = False
        self.quiet = False
        self.messages = []
        self.pending_question = False

    def start(self):
        self.started = True
        self.on_notice(CoordinatorNotice("后台 coordinator 已启动。"))

    def stop(self):
        self.stopped = True

    def submit_user_message(self, text):
        self.messages.append(text)
        return f"动态回应：{text}"

    def schedule_background_task(self, text):
        self.messages.append(text)
        from xiaoming.tools.base import ToolResult

        return ToolResult("schedule_background_task", "success", output=f"scheduled: {text}")

    def status_text(self):
        return "后台任务: 1\n待确认: 0"

    def has_active_tasks(self):
        return bool(self.messages)

    def has_pending_question(self):
        return self.pending_question

    def pending_questions_text(self):
        return "question_id: q1; prompt: 允许写 README.md？" if self.pending_question else None

    def context_summary(self):
        return "Background tasks:\n- fake running" if self.messages or self.pending_question else None

    def reply_mailbox_message(self, **kwargs):
        self.messages.append(kwargs)
        from xiaoming.tools.base import ToolResult

        return ToolResult("reply_mailbox_message", "success", output="sent")

    def tasks_text(self):
        return "异步任务  running  正在工作"

    def cancel_current(self):
        return "已请求取消“异步任务”。"

    def cancel_all(self):
        return "已请求取消 1 个任务。"

    def set_quiet(self, quiet):
        self.quiet = quiet


class FakeLoop:
    def __init__(self):
        self.tasks = []

    def run(self, task, session=None, on_event=None):
        self.tasks.append(task)
        return f"主LLM回应：{task}"


class InterruptibleLoop:
    def __init__(self):
        self.tasks = []

    def run(self, task, session=None, on_event=None, should_cancel=None):
        self.tasks.append(task)
        if task == "前台任务":
            while should_cancel is not None and not should_cancel():
                time.sleep(0.01)
            return ""
        return f"主LLM回应：{task}"


class ToolCallingProvider:
    def __init__(self, calls):
        self.calls = list(calls)
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        call = self.calls.pop(0)
        return call(request)


class FakeCoordinatorScheduler:
    def schedule(self, user_message, registry):
        return SchedulerDecision(action="start_new_task", user_intent=user_message, task_title=user_message, visible_message="")


class FakeCoordinatorResponder:
    def user_reply(self, user_message, registry, mode, question=None):
        return ""

    def worker_notice(self, event, task, registry):
        return ""

    def command_reply(self, command, payload, registry):
        return ""


class NoticingLoop:
    def __init__(self, coordinator_getter):
        self.coordinator_getter = coordinator_getter

    def run(self, task, session=None, on_event=None):
        self.coordinator_getter().on_notice(CoordinatorNotice("worker 需要确认"))
        return f"主LLM回应：{task}"


def _args():
    return Namespace(
        task=None,
        provider=None,
        model=None,
        approval_mode=None,
        permission_mode=None,
        max_turns=None,
        model_timeout_seconds=None,
        stream=None,
        continue_session=False,
        resume_session_id=None,
        new_session=True,
    )


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_uses_async_coordinator_for_default_runtime(monkeypatch, capsys, tmp_path):
    coordinators = []

    def factory(config, on_notice):
        coordinator = FakeCoordinator(config, on_notice)
        coordinators.append(coordinator)
        return coordinator

    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=factory)
    fake_loop = FakeLoop()
    runtime.loop = fake_loop
    inputs = iter(["帮我写 README", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert coordinators[0].started is True
    assert coordinators[0].stopped is True
    assert coordinators[0].messages == []
    assert fake_loop.tasks == ["帮我写 README"]
    assert "主LLM回应：帮我写 README" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_flushes_async_notice_after_current_response(monkeypatch, capsys, tmp_path):
    coordinators = []

    def factory(config, on_notice):
        coordinator = FakeCoordinator(config, on_notice)
        coordinators.append(coordinator)
        return coordinator

    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=factory)
    runtime.loop = NoticingLoop(lambda: coordinators[0])
    inputs = iter(["我正在输入的内容", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert output.index("主LLM回应：我正在输入的内容") < output.index("worker 需要确认")


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_async_notice_buffer_flushes_when_input_is_deleted(monkeypatch, capsys):
    monkeypatch.setattr(AsyncNoticeBuffer, "POLL_SECONDS", 0.01)
    monkeypatch.setattr("xiaoming.cli._readline_buffer_empty", lambda: True)
    buffer = AsyncNoticeBuffer()
    buffer.start()
    buffer.set_input_active(True)

    buffer.enqueue(CoordinatorNotice("worker 需要确认"))

    import time

    deadline = time.time() + 1
    output = ""
    while time.time() < deadline:
        output += capsys.readouterr().out
        if "worker 需要确认" in output:
            break
        time.sleep(0.02)

    buffer.stop()
    assert "worker 需要确认" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_keeps_pending_worker_question_in_main_loop_context(monkeypatch, capsys, tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    coordinator.pending_question = True
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    fake_loop = FakeLoop()
    runtime.loop = fake_loop
    inputs = iter(["可以", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert coordinator.messages == []
    assert fake_loop.tasks == ["可以"]
    assert "主LLM回应：可以" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_does_not_auto_move_normal_turn_to_background(monkeypatch, capsys, tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    fake_loop = FakeLoop()
    runtime.loop = fake_loop
    inputs = iter(["普通问题", "exit"])
    extra_inputs = iter(["新的问题", None])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr("xiaoming.cli._read_user_input_until_done", lambda prompt, async_notices, done: next(extra_inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert fake_loop.tasks == ["普通问题", "新的问题"]
    assert coordinator.messages == []
    assert "当前任务已转为后台继续处理" not in output
    assert "主LLM回应：新的问题" in output


def test_default_runtime_provides_async_context_summary_to_main_loop(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    coordinator.messages.append("active")
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    runtime.async_coordinator = coordinator

    assert runtime._async_context_summary() == "Background tasks:\n- fake running"


def test_cli_smoke_schedules_background_task_and_answers_worker_question(tmp_path):
    workers = []

    from xiaoming.async_runtime.coordinator import AsyncCoordinator, CoordinatorConfig
    from xiaoming.async_runtime.events import WorkerEvent

    class Worker:
        def __init__(self, config, on_event):
            self.config = config
            self.on_event = on_event
            self.pid = 123
            self.sent = []

        def start(self):
            workers.append(self)
            self.on_event(WorkerEvent(self.config.task_id, "started", "worker started"))
            self.on_event(WorkerEvent(self.config.task_id, "approval_request", "允许写 README.md？", {"request_id": "q1"}))

        def send(self, kind, **payload):
            self.sent.append((kind, payload))

        def terminate(self, timeout_seconds=5):
            pass

    runtime_ref = {}
    provider = ToolCallingProvider(
        [
            lambda request: LLMResponse(
                message="我会放到后台处理。",
                tool_calls=[
                    ToolCall(
                        id="schedule-1",
                        name="schedule_background_task",
                        args={
                            "message": "写 README",
                            "task_name": "写 README",
                        },
                    )
                ],
                output_items=[{"type": "function_call", "call_id": "schedule-1", "name": "schedule_background_task", "arguments": "{}"}],
                raw=None,
            ),
            lambda request: LLMResponse(message="已安排。", tool_calls=[], output_items=[], raw=None),
            lambda request: LLMResponse(
                message="我把确认转给后台。",
                tool_calls=[
                    ToolCall(
                        id="answer-1",
                        name="reply_mailbox_message",
                        args={
                            "message_id": runtime_ref["runtime"].async_coordinator.mailbox.pending_reply_messages()[0].message_id,
                            "normalized_answer": "允许写 README.md",
                            "decision": "approved",
                            "message_to_user": "已批准。",
                            "authorization_note": "",
                        },
                    )
                ],
                output_items=[{"type": "function_call", "call_id": "answer-1", "name": "reply_mailbox_message", "arguments": "{}"}],
                raw=None,
            ),
            lambda request: LLMResponse(message="已批准。", tool_calls=[], output_items=[], raw=None),
        ]
    )

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=_args(),
        coordinator_factory=lambda config, on_notice: AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeCoordinatorScheduler(), responder=FakeCoordinatorResponder(), worker_factory=lambda config, on_event: Worker(config, on_event), on_notice=on_notice),
    )
    runtime.loop.provider = provider
    runtime_ref["runtime"] = runtime
    runtime.async_coordinator = runtime.build_async_coordinator(lambda notice: None)
    runtime.async_coordinator.start()
    try:
        first = runtime.loop.run("帮我写 README", session=runtime.session)
        _eventually(lambda: len(workers) == 1 and runtime.async_coordinator.has_pending_question())
        second = runtime.loop.run("同意", session=runtime.session)
        _eventually(lambda: bool(workers[0].sent))
    finally:
        runtime.async_coordinator.stop()

    assert first == "已安排。"
    assert second == "已批准。"
    assert workers[0].sent[-1] == ("answer_question", {"request_id": "q1", "answer": "允许写 README.md", "decision": "approved"})


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_async_task_commands(monkeypatch, capsys, tmp_path):
    coordinators = []

    def factory(config, on_notice):
        coordinator = FakeCoordinator(config, on_notice)
        coordinators.append(coordinator)
        return coordinator

    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=factory)
    inputs = iter(["/tasks", "/cancel", "/quiet", "/verbose", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "异步任务  running" in output
    assert "已请求取消" in output
    assert "Background notices reduced." in output
    assert "Background notices restored." in output


def test_default_runtime_exposes_universal_tool_schema(tmp_path):
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: FakeCoordinator(config, on_notice))

    tool_names = {tool.name for tool in runtime.loop.registry.specs()}

    assert "schedule_background_task" in tool_names
    assert "background_tasks_status" in tool_names
    assert "follow_background_task" in tool_names
    assert "answer_worker_question" not in tool_names
    assert "reply_mailbox_message" in tool_names
    assert "cancel_background_task" in tool_names
    assert "web_search" in tool_names
    assert "talk" in tool_names
    assert "list_files" in tool_names
    assert "read_file" in tool_names
    assert "search_code" in tool_names
    assert "web_fetch" in tool_names
    assert "git_status" in tool_names
    assert "load_skill" in tool_names
    assert "install_skill" in tool_names
    assert "shell" in tool_names
    assert "write_file" in tool_names
    assert "append_file" in tool_names
    assert "edit_file" in tool_names
    assert "apply_patch" in tool_names
    assert "write_file" not in runtime.loop.instructions
    assert "load_skill" not in runtime.loop.instructions
    assert runtime.loop.skill_library is None


def test_worker_loop_keeps_full_tool_surface(tmp_path):
    loop = build_loop(tmp_path, _args())

    tool_names = {tool.name for tool in loop.registry.specs()}

    assert "install_skill" in tool_names
    assert "shell" in tool_names
    assert "write_file" in tool_names
    assert "cancel_background_task" not in tool_names


def test_build_loop_passes_custom_approval_callback_to_install_skill(tmp_path):
    approvals = []
    args = _args()
    args.approval_mode = "suggest"
    loop = build_loop(tmp_path, args, approve=lambda action: approvals.append(action) or False)

    result = loop.registry.run("install_skill", {"url": "https://github.com/acme/skills/tree/main/skills/frontend"})

    assert result.status == "denied"
    assert approvals
    assert "Tool: install_skill" in approvals[0]


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_default_runtime_denies_file_inspection_tools(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    coordinator.messages.append("active")
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    runtime.async_coordinator = coordinator

    result = runtime.loop.registry.run("list_files", {"path": ".", "pattern": "*"})

    assert result.status == "denied"
    assert "schedule_background_task" in result.error


def test_runtime_foreground_task_allows_file_inspection_with_same_tool_schema(tmp_path):
    (tmp_path / "README.md").write_text("hello\n")
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: FakeCoordinator(config, on_notice))
    before_specs = [(spec.name, spec.input_schema) for spec in runtime.loop.registry.specs()]

    runtime.begin_foreground_task("查看当前目录")
    result = runtime.loop.registry.run("list_files", {"path": ".", "pattern": None})
    after_specs = [(spec.name, spec.input_schema) for spec in runtime.loop.registry.specs()]

    assert result.status == "success"
    assert "README.md" in result.output
    assert before_specs == after_specs


def test_runtime_moves_foreground_task_to_background_with_existing_coordinator(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    runtime.async_coordinator = coordinator
    runtime.begin_foreground_task("帮我修复 README")

    message = runtime.move_foreground_task_to_background()

    assert coordinator.messages
    assert "帮我修复 README" in str(coordinator.messages[0])
    assert "转为后台" in message


def test_default_runtime_allows_web_search_while_background_task_active(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    coordinator.messages.append("active")
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    runtime.async_coordinator = coordinator

    assert "web_search" in {tool.name for tool in runtime.loop.registry.specs()}
    assert runtime.loop.registry._tools["web_search"].__class__.__name__ == "CapabilityGuardedTool"


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_default_runtime_denies_web_fetch(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    coordinator.messages.append("active")
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    runtime.async_coordinator = coordinator

    result = runtime.loop.registry.run("web_fetch", {"url": "https://github.com/obra/superpowers"})

    assert result.status == "denied"
    assert "schedule_background_task" in result.error


def test_pending_worker_questions_are_visible_for_new_requests_so_main_llm_can_judge(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    coordinator.pending_question = True
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    runtime.async_coordinator = coordinator

    assert "question_id: q1" in runtime._pending_worker_questions_for_input("帮我写个五子棋小网页")


def test_pending_worker_questions_are_visible_for_short_answers(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    coordinator.pending_question = True
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: coordinator)
    runtime.async_coordinator = coordinator

    assert "question_id: q1" in runtime._pending_worker_questions_for_input("同意")


def test_answer_worker_question_tool_is_not_model_visible(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: FakeCoordinator(config, on_notice))
    runtime.async_coordinator = coordinator

    result = runtime.loop.registry.run(
        "answer_worker_question",
        {
            "question_id": "q1",
            "action": "ask_clarification",
            "normalized_answer": "",
            "decision": "none",
            "message_to_user": "请确认。",
        },
    )

    assert result.status == "error"
    assert "unknown tool" in result.error


def test_reply_mailbox_message_tool_routes_to_coordinator(tmp_path):
    coordinator = FakeCoordinator(None, lambda notice: None)
    runtime = ChatRuntime(workspace=tmp_path, args=_args(), coordinator_factory=lambda config, on_notice: FakeCoordinator(config, on_notice))
    runtime.async_coordinator = coordinator

    result = runtime.loop.registry.run(
        "reply_mailbox_message",
        {
            "message_id": "m1",
            "normalized_answer": "允许",
            "decision": "approved",
            "message_to_user": "已批准。",
            "authorization_note": "",
        },
    )

    assert result.status == "success"
    assert coordinator.messages[-1]["message_id"] == "m1"


def _eventually(predicate, timeout=2):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()
