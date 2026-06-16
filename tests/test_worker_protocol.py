from io import StringIO
from pathlib import Path

import pytest

from xiaoming.async_runtime.worker_process import WorkerConfig, WorkerProcess
from xiaoming.async_runtime.agents import builtin_agent_registry
from xiaoming.async_runtime.context_packets import WorkerContextPacket
from xiaoming.async_runtime.events import WorkerEvent
from xiaoming.async_runtime.protocol import ProtocolError, decode_message, encode_message, write_message
from xiaoming.logging import XiaomingLogger
from xiaoming.progress import ProgressEvent
from xiaoming.tools.talk import TalkTool
from xiaoming import worker_main
from xiaoming.worker_main import WorkerInbox


def test_protocol_round_trips_jsonl_message():
    encoded = encode_message("progress", task_id="t1", message="working")

    decoded = decode_message(encoded)

    assert decoded == {"kind": "progress", "message": "working", "task_id": "t1"}


def test_protocol_rejects_non_object_messages():
    with pytest.raises(ProtocolError):
        decode_message("[]")


def test_write_message_flushes_stream():
    stream = StringIO()

    write_message(stream, "completed", task_id="t1", message="done")

    assert decode_message(stream.getvalue())["kind"] == "completed"


def test_worker_process_starts_in_separate_session(monkeypatch, tmp_path):
    captured = {}

    class FakeProcess:
        def __init__(self):
            self.stdin = StringIO()
            self.stdout = StringIO()
            self.stderr = StringIO()
            self.pid = 123

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("xiaoming.async_runtime.worker_process.subprocess.Popen", fake_popen)
    worker = WorkerProcess(WorkerConfig(task_id="t1", task="task", workspace=tmp_path))

    worker.start()

    assert captured["kwargs"]["start_new_session"] is True
    start_message = decode_message(worker.process.stdin.getvalue().splitlines()[0])
    assert start_message["kind"] == "start_task"
    assert start_message["task_id"] == "t1"


def test_worker_event_from_json():
    event = WorkerEvent.from_json({"kind": "progress", "task_id": "t1", "message": "working"})

    assert event.task_id == "t1"
    assert event.kind == "progress"
    assert event.message == "working"


def test_worker_inbox_drains_append_messages_and_marks_cancelled():
    inbox = WorkerInbox()
    inbox.queue.put({"kind": "append_user_message", "message": "补充要求"})
    inbox.queue.put({"kind": "cancel"})

    assert inbox.drain_appends() == ["补充要求"]
    assert inbox.cancelled is True


def test_worker_approval_uses_structured_decision():
    inbox = WorkerInbox()
    stdout = StringIO()
    result = []

    import threading
    import time

    thread = threading.Thread(target=lambda: result.append(worker_main._approve_via_parent(stdout, "task-1", "install skill", inbox)))
    thread.start()
    started = time.monotonic()
    request = None
    while time.monotonic() - started < 1:
        lines = stdout.getvalue().splitlines()
        if lines:
            request = decode_message(lines[0])
            break
        time.sleep(0.01)
    assert request is not None

    inbox.queue.put({"kind": "answer_question", "request_id": request["data"]["request_id"], "answer": "好，安装完告诉我", "decision": "approved"})
    thread.join(timeout=1)

    assert result == [True]


def test_talk_tool_sends_clarification_request_and_waits_for_answer():
    inbox = WorkerInbox()
    stdout = StringIO()
    tool = TalkTool(lambda purpose, message, context, options: worker_main._talk_via_parent(stdout, "task-1", inbox, purpose=purpose, message=message, context=context, options=options))
    result = []

    import threading
    import time

    thread = threading.Thread(
        target=lambda: result.append(
            tool.run(
                {
                    "purpose": "clarify",
                    "message": "棋盘大小选哪个？",
                    "context": "brainstorming",
                    "options": ["19x19", "9x9"],
                }
            )
        )
    )
    thread.start()
    started = time.monotonic()
    request = None
    while time.monotonic() - started < 1:
        lines = stdout.getvalue().splitlines()
        if lines:
            request = decode_message(lines[0])
            break
        time.sleep(0.01)
    assert request is not None

    assert request["kind"] == "clarification_request"
    assert request["message"] == "棋盘大小选哪个？"
    assert request["data"]["purpose"] == "clarify"
    assert request["data"]["context"] == "brainstorming"
    assert request["data"]["options"] == ["19x19", "9x9"]

    inbox.queue.put({"kind": "answer_question", "request_id": request["data"]["request_id"], "answer": "选 9x9", "decision": "none"})
    thread.join(timeout=1)

    assert result[0].status == "success"
    assert result[0].output == "选 9x9"


def test_talk_tool_sends_decision_request():
    inbox = WorkerInbox()
    stdout = StringIO()
    tool = TalkTool(lambda purpose, message, context, options: worker_main._talk_via_parent(stdout, "task-1", inbox, purpose=purpose, message=message, context=context, options=options))
    result = []

    import threading
    import time

    thread = threading.Thread(target=lambda: result.append(tool.run({"purpose": "decision", "message": "是否写设计文档？"})))
    thread.start()
    started = time.monotonic()
    request = None
    while time.monotonic() - started < 1:
        lines = stdout.getvalue().splitlines()
        if lines:
            request = decode_message(lines[0])
            break
        time.sleep(0.01)
    assert request is not None

    assert request["kind"] == "decision_request"
    inbox.queue.put({"kind": "answer_question", "request_id": request["data"]["request_id"], "answer": "先写设计文档", "decision": "none"})
    thread.join(timeout=1)

    assert result[0].status == "success"
    assert result[0].output == "先写设计文档"


def test_worker_task_prompt_treats_worker_as_skill_capable_not_subagent():
    prompt = worker_main._worker_task_prompt(worker_main.TaskSpec(title="t", goal="g"))

    assert "Title: t" in prompt
    assert "Goal: g" in prompt
    assert "independent, fully capable coding agent" not in prompt
    assert "call load_skill before acting" not in prompt
    assert "report_task_result tool exactly once" not in prompt


def test_forked_worker_directive_includes_operating_rules_after_task_contract():
    agent = builtin_agent_registry().get("worker")
    prompt = worker_main._forked_worker_directive(agent, worker_main.TaskSpec(title="Build", goal="Create a page"), None)

    assert "<task_contract>" in prompt
    assert "</task_contract>" in prompt
    assert "<worker_operating_rules>" in prompt
    assert 'source="codex-derived"' not in prompt
    assert "Do not guess file contents" in prompt
    assert "Respect AGENTS.md" in prompt
    assert "Final response must summarize" in prompt
    assert prompt.index("</task_contract>") < prompt.index("<worker_operating_rules>")
    assert prompt.index("</worker_operating_rules>") < prompt.index("Start working on the task contract now")


def test_worker_protocol_context_treats_worker_as_skill_capable_not_subagent():
    prompt = worker_main._worker_protocol_context_item()["content"]

    assert "independent, fully capable coding agent" in prompt
    assert "Your user is the coordinator" in prompt
    assert "The coordinator represents the human user" in prompt
    assert "full access to the repository tools and skill system" in prompt
    assert "normal interactive coding session" in prompt
    assert "responsible for completing the assignment end to end" in prompt
    assert "skill instructions mention subagents or background agents" in prompt
    assert "Use the native install_skill tool" in prompt
    assert "Before inspecting files" in prompt
    assert "call load_skill before acting" in prompt
    assert "Loaded skill instructions guide how you approach the task" in prompt
    assert "If a loaded skill says to ask the user" in prompt
    assert "ask your user by calling the talk tool" in prompt
    assert "calling the talk tool" in prompt
    assert "Normal assistant text is progress only" in prompt
    assert "cannot receive a user reply" in prompt
    assert "not a full implementation plan" in prompt
    assert "You may choose filenames" in prompt
    assert "Do not assume openai/skills" in prompt
    assert "If a skill install source is unknown" in prompt
    assert "report_task_result" not in prompt


def test_worker_agent_context_includes_agent_definition():
    agent = builtin_agent_registry().get("worker")
    prompt = worker_main._worker_agent_context(agent)

    assert "<name>worker</name>" in prompt
    assert "<tool_profile>full</tool_profile>" in prompt
    assert "complete the assigned task" in prompt.lower()


def test_worker_context_packet_injected_as_bootstrap_context():
    session = worker_main.Session(session_id="task-1")
    packet = WorkerContextPacket(
        task_id="task-1",
        session_id="session-1",
        agent_type="general-worker",
        context_policy="briefed",
        workspace="/repo",
        handoff_summary="Build gomoku.",
    )

    worker_main._inject_worker_context_packet(session, packet)

    context = session.bootstrap_contexts["runtime:worker-context-packet"]
    assert "<worker_context_packet>" in context.content
    assert "<handoff_summary>" in context.content
    assert "Build gomoku." in context.content


def test_worker_injects_bootstrap_contexts(monkeypatch, tmp_path):
    skill_dir = tmp_path / ".agents" / "skills" / "superpowers" / "skills" / "using-superpowers"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: using-superpowers\n---\nUse skills first.\n")
    session = worker_main.Session(session_id="task-1")
    logger = XiaomingLogger.create_worker(tmp_path, "task-1")

    worker_main._inject_worker_bootstrap_contexts(tmp_path, session, logger)

    assert "superpowers:using-superpowers" in session.bootstrap_contexts
    assert "worker_bootstrap_context_injected" in logger.path.read_text()


def test_worker_protocol_is_injected_as_bootstrap_context_not_history():
    session = worker_main.Session(session_id="task-1")

    worker_main._inject_worker_protocol_context(session)

    assert session.input_items == []
    assert "runtime:worker-protocol" in session.bootstrap_contexts
    assert "report_task_result" not in session.bootstrap_contexts["runtime:worker-protocol"].content


def test_worker_task_contract_is_injected_as_bootstrap_context_not_history():
    session = worker_main.Session(session_id="task-1")
    spec = worker_main.TaskSpec(title="Build", goal="Create a page")

    worker_main._inject_worker_task_contract(session, spec)

    assert session.input_items == []
    assert "runtime:task-contract" in session.bootstrap_contexts
    assert "Goal: Create a page" in session.bootstrap_contexts["runtime:task-contract"].content


def test_worker_applies_forked_parent_history_before_running(monkeypatch, tmp_path):
    captured = {}

    class FakeRegistry:
        def specs(self):
            return []

    class FakeLoop:
        registry = FakeRegistry()
        skill_library = None

        def run(self, task, session=None, on_event=None):
            captured["session"] = session
            captured["task"] = task
            return "done"

    def fake_build_loop(*args, **kwargs):
        captured["build_loop_kwargs"] = kwargs
        return FakeLoop()

    forked_items = [
        {"role": "user", "content": "之前确定使用独立页面", "xiaoming": {"id": "msg-1"}},
        {"role": "assistant", "content": "收到", "xiaoming": {"id": "msg-2"}},
    ]
    monkeypatch.setattr(worker_main, "build_loop", fake_build_loop)
    stdin = StringIO(
        encode_message(
            "start_task",
            task_id="task-1",
            task="开发五子棋",
            workspace=str(tmp_path),
            forked_instructions="MAIN SYSTEM PROMPT",
            forked_input_items=forked_items,
        )
    )
    stdout = StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    result = worker_main.main()

    assert result == 0
    session = captured["session"]
    assert session.input_items[:2] == forked_items
    assert captured["build_loop_kwargs"]["instructions_override"] == "MAIN SYSTEM PROMPT"
    assert "runtime:worker-protocol" not in session.bootstrap_contexts
    assert "runtime:task-contract" not in session.bootstrap_contexts
    assert "<worker_protocol>" in captured["task"]
    assert "Goal: 开发五子棋" in captured["task"]


def test_worker_protocol_repair_prompt_requires_talk():
    prompt = worker_main._worker_protocol_repair_prompt()

    assert "normal assistant text is progress only" in prompt
    assert "call the talk tool" in prompt
    assert "report_task_result" not in prompt


def test_worker_process_invalid_stdout_fails_once_and_ignores_later_terminal_event():
    events = []

    class FakeProcess:
        stdout = StringIO("\n" + encode_message("completed", task_id="t1", message="done"))

    class TestWorkerProcess(WorkerProcess):
        def terminate(self, timeout_seconds: float = 5) -> None:
            events.append(WorkerEvent(self.config.task_id, "terminated", ""))

    worker = TestWorkerProcess(WorkerConfig(task_id="t1", task="task", workspace=Path(".")), on_event=events.append)
    worker.process = FakeProcess()  # type: ignore[assignment]

    worker._read_stdout()

    failures = [event for event in events if event.kind == "failed"]
    assert len(failures) == 1
    assert failures[0].data["error_kind"] == "invalid_worker_event"


def test_worker_process_drains_stderr_to_worker_log(tmp_path):
    class FakeProcess:
        stderr = StringIO("debug line sk-1234567890\n")

    worker = WorkerProcess(WorkerConfig(task_id="t1", task="task", workspace=tmp_path))
    worker.process = FakeProcess()  # type: ignore[assignment]

    worker._read_stderr()

    assert (tmp_path / ".xiaoming" / "logs" / "workers" / "t1.stderr.log").read_text() == "debug line [REDACTED]\n"


def test_worker_session_recorder_keeps_compaction_replacement_items(tmp_path):
    from xiaoming.worker_diagnostics import WorkerSessionRecorder

    recorder = WorkerSessionRecorder(tmp_path, "task-1")
    replacement_items = [{"role": "user", "content": "x" * 50_000}]

    recorder.append("task-1", "context_compaction_completed", {"replacement_items": replacement_items})

    text = recorder.path.read_text()
    assert "replacement_items" in text
    assert "truncated" not in text


def test_worker_main_emits_completed_from_final_answer(monkeypatch, tmp_path):
    captured = {}

    class FakeRegistry:
        def specs(self):
            return []

    class FakeLoop:
        registry = FakeRegistry()
        skill_library = None

        def run(self, task, session=None, on_event=None):
            return "done"

    def fake_build_loop(*args, **kwargs):
        captured.update(kwargs)
        return FakeLoop()

    monkeypatch.setattr(worker_main, "build_loop", fake_build_loop)
    stdin = StringIO(encode_message("start_task", task_id="task-1", task="do work", workspace=str(tmp_path)))
    stdout = StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    result = worker_main.main()

    events = [decode_message(line) for line in stdout.getvalue().splitlines()]
    assert result == 0
    assert events[-1]["kind"] == "completed"
    assert events[-1]["message"] == "done"
    assert captured["logger"].path == tmp_path / ".xiaoming" / "logs" / "workers" / "task-1.log"
    assert captured["session_recorder"].path == tmp_path / ".xiaoming" / "worker_sessions" / "task-1.jsonl"
    assert "talk" in [tool.name for tool in captured["extra_tools"]]
    assert "report_task_result" not in [tool.name for tool in captured["extra_tools"]]
    assert '"event": "worker_tools_available"' in captured["logger"].path.read_text()
    assert '"event": "worker_skills_available"' in captured["logger"].path.read_text()


def test_worker_main_continues_after_review_feedback(monkeypatch, tmp_path):
    tasks = []

    class FakeRegistry:
        def specs(self):
            return []

    class FakeLoop:
        registry = FakeRegistry()
        skill_library = None

        def run(self, task, session=None, on_event=None):
            tasks.append(task)
            return f"answer-{len(tasks)}"

    def fake_build_loop(*args, **kwargs):
        return FakeLoop()

    monkeypatch.setattr(worker_main, "build_loop", fake_build_loop)
    stdin = StringIO(
        encode_message("start_task", task_id="task-1", task="do work", workspace=str(tmp_path))
        + encode_message("review_feedback", feedback="missing README")
        + encode_message("review_accepted", message="accepted")
    )
    stdout = StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    result = worker_main.main()

    events = [decode_message(line) for line in stdout.getvalue().splitlines()]
    completed = [event for event in events if event["kind"] == "completed"]
    assert result == 0
    assert [event["message"] for event in completed] == ["answer-1", "answer-2"]
    assert len(tasks) == 2
    assert "missing README" in tasks[1]


def test_worker_main_uses_streamed_text_when_loop_return_is_suppressed(monkeypatch, tmp_path):
    class FakeRegistry:
        def specs(self):
            return []

    class FakeLoop:
        registry = FakeRegistry()
        skill_library = None

        def run(self, task, session=None, on_event=None):
            on_event(ProgressEvent("text_delta", "Review-Status: "))
            on_event(ProgressEvent("text_delta", "ACCEPTED\n\nSummary:\nDone."))
            return ""

    monkeypatch.setattr(worker_main, "build_loop", lambda *args, **kwargs: FakeLoop())
    stdin = StringIO(encode_message("start_task", task_id="task-1", task="review work", workspace=str(tmp_path)))
    stdout = StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    result = worker_main.main()

    events = [decode_message(line) for line in stdout.getvalue().splitlines()]
    completed = [event for event in events if event["kind"] == "completed"]
    assert result == 0
    assert completed[-1]["message"] == "Review-Status: ACCEPTED\n\nSummary:\nDone."


def test_worker_main_streamed_fallback_uses_text_after_last_tool(monkeypatch, tmp_path):
    class FakeRegistry:
        def specs(self):
            return []

    class FakeLoop:
        registry = FakeRegistry()
        skill_library = None

        def run(self, task, session=None, on_event=None):
            on_event(ProgressEvent("text_delta", "I will inspect the file."))
            on_event(ProgressEvent("tool_started", "Running tool: read_file"))
            on_event(ProgressEvent("tool_finished", "Tool completed: read_file (success)"))
            on_event(ProgressEvent("text_delta", "Review-Status: "))
            on_event(ProgressEvent("text_delta", "ACCEPTED\n\nSummary:\nDone."))
            return ""

    monkeypatch.setattr(worker_main, "build_loop", lambda *args, **kwargs: FakeLoop())
    stdin = StringIO(encode_message("start_task", task_id="task-1", task="review work", workspace=str(tmp_path)))
    stdout = StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    result = worker_main.main()

    events = [decode_message(line) for line in stdout.getvalue().splitlines()]
    completed = [event for event in events if event["kind"] == "completed"]
    assert result == 0
    assert completed[-1]["message"] == "Review-Status: ACCEPTED\n\nSummary:\nDone."


def test_worker_main_applies_tool_profile_to_build_loop(monkeypatch, tmp_path):
    captured = {}

    class FakeRegistry:
        def specs(self):
            return []

    class FakeLoop:
        registry = FakeRegistry()
        skill_library = None

        def run(self, task, session=None, on_event=None):
            return "reported"

    def fake_build_loop(*args, **kwargs):
        captured.update(kwargs)
        return FakeLoop()

    monkeypatch.setattr(worker_main, "build_loop", fake_build_loop)
    stdin = StringIO(
        encode_message(
            "start_task",
            task_id="task-1",
            task="install skill",
            workspace=str(tmp_path),
            agent_type="skill-installer-worker",
        )
    )
    stdout = StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)

    result = worker_main.main()

    assert result == 0
    assert captured["include_write_tools"] is True
    assert captured["include_shell_tool"] is True
    assert captured["include_skill_install_tool"] is True
    assert captured["include_load_skill_tool"] is True
    assert captured["capability_profile"] == "full"
    assert "report_task_result" not in [tool.name for tool in captured["extra_tools"]]
