import json
import time
import threading
from pathlib import Path

from xiaoming.async_runtime.coordinator import AsyncCoordinator, CoordinatorConfig, MAX_REVISION_ATTEMPTS
from xiaoming.async_runtime.events import WorkerEvent
from xiaoming.async_runtime.external_sessions import ExternalSessionRecord
from xiaoming.async_runtime.local_thread_worker import ForegroundWorkerHandle
from xiaoming.async_runtime.mailbox import MailboxStore
from xiaoming.async_runtime.question_decider import WorkerQuestionDecision
from xiaoming.async_runtime.responder import ResponderError
from xiaoming.async_runtime.scheduler import SchedulerDecision, SchedulerError
from xiaoming.async_runtime.task_store import TaskStore
from xiaoming.async_runtime.tasks import TaskRecord, TaskRegistry, TaskSpec, VerificationResult
from xiaoming.session import LoadedSkill, Session
from xiaoming.tools.background_task import ScheduleBackgroundTaskTool


class RecordingHandle:
    def __init__(self, task_id="task-1"):
        self.task_id = task_id
        self.pid = None
        self.started = False
        self.sent = []
        self.terminated = False

    def start(self):
        self.started = True

    def send(self, kind, **payload):
        self.sent.append((kind, payload))

    def terminate(self, timeout_seconds=5):
        self.terminated = True


class ReplyingHandle(RecordingHandle):
    def __init__(self, task_id, on_event):
        super().__init__(task_id)
        self.on_event = on_event

    def send(self, kind, **payload):
        super().send(kind, **payload)
        if kind == "talk":
            self.on_event(WorkerEvent(self.task_id, "peer_reply", "收到，继续处理。"))


def _canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class FakeWorker:
    def __init__(self, config, on_event):
        self.config = config
        self.on_event = on_event
        self.pid = 123
        self.sent = []
        self.terminated = False

    def start(self):
        self.on_event(WorkerEvent(self.config.task_id, "started", "worker started"))

    def send(self, kind, **payload):
        self.sent.append((kind, payload))

    def terminate(self, timeout_seconds=5):
        self.terminated = True
        self.on_event(WorkerEvent(self.config.task_id, "cancelled", "cancelled"))


class FakeResponder:
    def user_reply(self, user_message, registry, mode, question=None):
        return f"动态回应：{mode}：{user_message}"

    def worker_notice(self, event, task, registry):
        return f"动态通知：{task.title}：{event.kind}"

    def command_reply(self, command, payload, registry):
        return f"动态命令：{command}"


class FailingResponder:
    def user_reply(self, user_message, registry, mode, question=None):
        raise ResponderError("model failed after 3 attempts")

    def worker_notice(self, event, task, registry):
        raise ResponderError("model failed after 3 attempts")

    def command_reply(self, command, payload, registry):
        raise ResponderError("model failed after 3 attempts")


class FakeScheduler:
    def schedule(self, user_message, registry):
        if "继续修改" in user_message:
            running = registry.active()[0]
            return SchedulerDecision(
                action="queue_task",
                user_intent=user_message,
                task_title=user_message,
                visible_message="动态调度：排队等待",
                conflict_task_ids={running.task_id},
                affected_files={"src/xiaoming/cli.py"},
            )
        return SchedulerDecision(
            action="start_new_task",
            user_intent=user_message,
            task_title=user_message,
            visible_message="动态调度：启动任务",
            affected_files={"src/xiaoming/cli.py"} if "src/xiaoming/cli.py" in user_message else set(),
        )


class FakeQuestionDecider:
    def __init__(self):
        self.calls = []

    def decide(self, task, question):
        self.calls.append((task, question))
        if task.authorization_note and "自动批准" in task.authorization_note:
            return WorkerQuestionDecision("approved", "根据授权自动批准。", "authorization note covers request")
        return WorkerQuestionDecision("ask_user", "", "authorization note does not cover request")


class FailingScheduler:
    def schedule(self, user_message, registry):
        raise SchedulerError("scheduler failed after 3 attempts")


class RecordingVerifier:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.calls = []

    def verify(self, spec, report):
        self.calls.append((spec, report))
        return VerificationResult(self.accepted, [] if self.accepted else ["not good enough"])


class SequenceVerifier:
    def __init__(self, *accepted_values):
        self.accepted_values = list(accepted_values)
        self.calls = []

    def verify(self, spec, report):
        self.calls.append((spec, report))
        accepted = self.accepted_values.pop(0) if self.accepted_values else False
        return VerificationResult(accepted, [] if accepted else ["artifact missing"])


def _pending_mailbox_message_id(coordinator: AsyncCoordinator) -> str:
    return coordinator.mailbox.pending_reply_messages()[0].message_id


def _reply_pending_mailbox(
    coordinator: AsyncCoordinator,
    answer: str,
    decision: str = "none",
    message_to_user: str = "已转给 worker。",
    authorization_note: str = "",
):
    return coordinator.reply_mailbox_message(
        _pending_mailbox_message_id(coordinator),
        answer,
        decision,
        message_to_user,
        authorization_note,
    )


def _review_report(status: str, summary: str = "", evidence: str = "", issues: str = "", feedback: str = "", summary_for_main: str = "") -> str:
    return (
        f"Review-Status: {status}\n\n"
        f"Summary:\n{summary or '(none)'}\n\n"
        f"Evidence:\n{evidence or '(none)'}\n\n"
        f"Issues:\n{issues or '(none)'}\n\n"
        f"Feedback-To-Worker:\n{feedback or '(none)'}\n\n"
        f"Summary-For-Main:\n{summary_for_main or '(none)'}\n"
    )


def test_coordinator_schedules_normal_message_without_immediate_reply(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        reply = coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
    finally:
        coordinator.stop()

    assert reply == ""
    assert workers[0].config.task == "写 README"
    assert any("动态调度：启动任务" in notice.message for notice in notices)


def test_coordinator_talks_to_internal_worker_without_completing_task(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, verifier=RecordingVerifier())
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]

        def reply():
            _eventually(lambda: workers[0].sent == [("talk", {"message": "怎么运行？"})])
            workers[0].on_event(WorkerEvent(task.task_id, "peer_reply", "运行 pytest"))

        thread = threading.Thread(target=reply, daemon=True)
        thread.start()
        result = coordinator.talk_to_peer(task.task_id, "怎么运行？")
        thread.join(timeout=1)
        status_before_stop = task.status
        submissions_before_stop = list(task.worker_submissions)
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert result.output == "运行 pytest"
    assert status_before_stop == "running"
    assert submissions_before_stop == []


def test_coordinator_talk_accepts_visible_short_task_id(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, verifier=RecordingVerifier())
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        short_id = task.task_id[:8]

        def reply():
            _eventually(lambda: workers[0].sent == [("talk", {"message": "进度？"})])
            workers[0].on_event(WorkerEvent(task.task_id, "peer_reply", "正在处理"))

        thread = threading.Thread(target=reply, daemon=True)
        thread.start()
        result = coordinator.talk_to_peer(short_id, "进度？")
        thread.join(timeout=1)
        status_before_stop = task.status
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert result.output == "正在处理"
    assert status_before_stop == "running"


def test_coordinator_talk_to_needs_user_decision_resumes_worker_without_waiting(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, verifier=RecordingVerifier())
    coordinator.start()
    try:
        coordinator.submit_user_message("写围棋")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        task.needs_user_decision_summary = "CLI or web GUI?"
        task.transition("needs_user_decision", task.needs_user_decision_summary)

        result = coordinator.talk_to_peer(task.task_id[:8], "简单的网页吧")
        sent = list(workers[0].sent)
        status = task.status
        progress = task.last_progress
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert "已转交" in result.output
    assert status == "running"
    assert progress == "user decision sent to worker"
    assert sent[-1][0] == "review_feedback"
    assert "简单的网页吧" in sent[-1][1]["feedback"]
    assert "CLI or web GUI?" in sent[-1][1]["feedback"]


def test_coordinator_routes_explicit_codex_task_to_external_worker(monkeypatch, tmp_path: Path):
    started = []

    class FakeCodexWorker:
        def __init__(self, config, on_event):
            self.config = config
            self.on_event = on_event
            self.pid = None

        def start(self):
            started.append(self.config.task)
            self.on_event(
                WorkerEvent(
                    self.config.task_id,
                    "completed",
                    "codex done",
                    {"external_provider": "codex", "external_session_id": "codex-thread-1"},
                )
            )

        def send(self, kind, **payload):
            pass

        def terminate(self, timeout_seconds=5):
            pass

    monkeypatch.setattr("xiaoming.async_runtime.coordinator.CodexWorkerProcess", FakeCodexWorker)
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), verifier=RecordingVerifier())
    coordinator.start()
    try:
        result = coordinator.schedule_background_task("请使用 Codex 写 README")
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
        task = coordinator.registry.list()[0]
        external = coordinator.external_sessions[task.task_id]
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert started == ["请使用 Codex 写 README"]
    assert task.agent_type == "codex"
    assert external.provider == "codex"
    assert external.session_id == "codex-thread-1"


def test_coordinator_routes_codex_task_from_internal_route_note(monkeypatch, tmp_path: Path):
    started = []

    class FakeCodexWorker:
        def __init__(self, config, on_event):
            self.config = config
            self.on_event = on_event
            self.pid = None

        def start(self):
            started.append(self.config.task)
            self.on_event(
                WorkerEvent(
                    self.config.task_id,
                    "completed",
                    "codex done",
                    {"external_provider": "codex", "external_session_id": "codex-thread-1"},
                )
            )

        def send(self, kind, **payload):
            pass

        def terminate(self, timeout_seconds=5):
            pass

    monkeypatch.setattr("xiaoming.async_runtime.coordinator.CodexWorkerProcess", FakeCodexWorker)
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), verifier=RecordingVerifier())
    coordinator.start()
    try:
        spec = TaskSpec(title="开发象棋网页", goal="开发一个简单的中国象棋网页", notes="requested_executor=codex")
        result = coordinator.schedule_background_task(spec)
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
        task = coordinator.registry.list()[0]
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert started == ["开发一个简单的中国象棋网页"]
    assert task.agent_type == "codex"


def test_schedule_background_task_tool_routes_rewritten_hello_world_request_to_codex(monkeypatch, tmp_path: Path):
    started = []

    class FakeCodexWorker:
        def __init__(self, config, on_event):
            self.config = config
            self.on_event = on_event
            self.pid = None

        def start(self):
            started.append(self.config.task)
            self.on_event(
                WorkerEvent(
                    self.config.task_id,
                    "completed",
                    "codex done",
                    {"external_provider": "codex", "external_session_id": "codex-thread-hello-world"},
                )
            )

        def send(self, kind, **payload):
            pass

        def terminate(self, timeout_seconds=5):
            pass

    monkeypatch.setattr("xiaoming.async_runtime.coordinator.CodexWorkerProcess", FakeCodexWorker)
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), verifier=RecordingVerifier())
    tool = ScheduleBackgroundTaskTool(
        lambda: coordinator,
        turn_context_getter=lambda: "在后台使用codex开发一个简单的hello world网页",
    )
    coordinator.start()
    try:
        result = tool.run(
            {
                "task_name": "Hello World 网页",
                "message": "在 workspace 中创建一个简单的 Hello World 网页。使用 HTML + CSS，设计简洁美观。创建一个 index.html 文件。",
            }
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
        task = coordinator.registry.list()[0]
        external = coordinator.external_sessions[task.task_id]
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert task.agent_type == "codex"
    assert task.title == "Hello World 网页"
    assert started == ["在 workspace 中创建一个简单的 Hello World 网页。使用 HTML + CSS，设计简洁美观。创建一个 index.html 文件。"]
    assert external.provider == "codex"
    assert external.session_id == "codex-thread-hello-world"


def test_coordinator_records_external_session_from_codex_progress(monkeypatch, tmp_path: Path):
    class FakeCodexWorker:
        def __init__(self, config, on_event):
            self.config = config
            self.on_event = on_event
            self.pid = None

        def start(self):
            self.on_event(
                WorkerEvent(
                    self.config.task_id,
                    "progress",
                    "Codex is still working; waiting for the next event.",
                    {"external_provider": "codex", "external_session_id": "codex-thread-1"},
                )
            )

        def send(self, kind, **payload):
            pass

        def terminate(self, timeout_seconds=5):
            pass

    monkeypatch.setattr("xiaoming.async_runtime.coordinator.CodexWorkerProcess", FakeCodexWorker)
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), verifier=RecordingVerifier())
    coordinator.start()
    try:
        spec = TaskSpec(title="开发象棋网页", goal="开发一个简单的中国象棋网页", notes="requested_executor=codex")
        result = coordinator.schedule_background_task(spec)
        _eventually(lambda: coordinator.registry.list()[0].last_progress == "Codex is still working; waiting for the next event.")
        task = coordinator.registry.list()[0]
        external = coordinator.external_sessions[task.task_id]
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert task.status == "running"
    assert external.session_id == "codex-thread-1"
    assert external.status == "active"


def test_coordinator_talks_to_completed_external_codex_session(monkeypatch, tmp_path: Path):
    calls = []

    class FakeCodexSession:
        def __init__(self, workspace, session_id="", timeout_seconds=900):
            self.workspace = workspace
            self.session_id = session_id
            self.timeout_seconds = timeout_seconds

        def send(self, message):
            calls.append((self.session_id, message))
            return "打开 index.html", self.session_id

    monkeypatch.setattr("xiaoming.async_runtime.coordinator.CodexRemoteControlSession", FakeCodexSession)
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder())
    task = TaskRecord(title="Codex 象棋", original_request="用 codex 写象棋", current_goal="用 codex 写象棋", status="accepted", agent_type="codex")
    coordinator.registry.add(task)
    coordinator.external_sessions[task.task_id] = ExternalSessionRecord(
        peer_id=task.task_id,
        provider="codex",
        title=task.title,
        workspace=str(tmp_path),
        session_id="codex-thread-1",
    )

    result = coordinator.talk_to_peer(task.task_id[:8], "怎么运行？")

    assert result.status == "success"
    assert result.output == "打开 index.html"
    assert calls == [("codex-thread-1", "怎么运行？")]


def test_coordinator_sends_needs_user_decision_to_external_codex_without_waiting(monkeypatch, tmp_path: Path):
    calls = []

    class FakeCodexSession:
        def __init__(self, workspace, session_id="", timeout_seconds=900):
            self.workspace = workspace
            self.session_id = session_id
            self.timeout_seconds = timeout_seconds

        def send(self, message, on_progress=None):
            calls.append((self.session_id, message))
            if on_progress is not None:
                on_progress("Codex is still working; waiting for the next event.")
            return "created web go", self.session_id

    monkeypatch.setattr("xiaoming.async_runtime.coordinator.CodexRemoteControlSession", FakeCodexSession)
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), verifier=RecordingVerifier())
    task = TaskRecord(title="围棋开发", original_request="使用 codex 写围棋", current_goal="使用 codex 写围棋", status="needs_user_decision", agent_type="codex")
    task.needs_user_decision_summary = "CLI or web GUI?"
    coordinator.registry.add(task)
    coordinator.external_sessions[task.task_id] = ExternalSessionRecord(
        peer_id=task.task_id,
        provider="codex",
        title=task.title,
        workspace=str(tmp_path),
        session_id="codex-thread-1",
    )
    coordinator.start()
    try:
        result = coordinator.talk_to_peer(task.task_id[:8], "简单的网页吧")
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
        final_task = coordinator.registry.list()[0]
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert "已转交" in result.output
    assert calls
    assert calls[0][0] == "codex-thread-1"
    assert "简单的网页吧" in calls[0][1]
    assert "CLI or web GUI?" in calls[0][1]
    assert final_task.last_progress == "created web go"


def test_schedule_background_task_starts_worker_without_scheduler_notice(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        result = coordinator.schedule_background_task("写 README")
        _eventually(lambda: len(workers) == 1)
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert "action: start_new_task" in result.output
    assert workers[0].config.task == "写 README"
    assert workers[0].config.task_spec is not None
    assert workers[0].config.task_spec.goal == "写 README"
    assert not any("动态调度：启动任务" in notice.message for notice in notices)


def test_schedule_background_task_ignores_legacy_agent_context_fields(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        spec = TaskSpec(
            title="Install superpowers",
            goal="Install the superpowers skill",
            agent_type="skill-installer-worker",
            context_policy="isolated",
            skills_to_preload=["skill-installer"],
        )
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
    finally:
        coordinator.stop()

    assert workers[0].config.agent_type == "worker"
    assert workers[0].config.context_policy == "forked"
    assert workers[0].config.skills_to_preload == []


def test_schedule_background_task_applies_agent_defaults_and_forks_parent_context(tmp_path: Path):
    workers = []
    session = Session(session_id="main-session")
    session.base_instructions = "MAIN SYSTEM PROMPT"
    session.last_prompt_input_items = [
        {"role": "developer", "content": "<bootstrap_context>stable rendered context</bootstrap_context>", "xiaoming": {"kind": "bootstrap_context"}},
        {"role": "user", "content": "[@10:00] 安装 superpowers skill", "xiaoming": {"id": "msg-1", "date": "2026-06-09", "time": "10:00", "timezone": "Asia/Shanghai"}},
    ]
    session.last_model_output_items = [
        {
            "role": "assistant",
            "content": "我会安排安装。",
            "tool_calls": [
                {
                    "id": "call-schedule",
                    "type": "function",
                    "function": {"name": "schedule_background_task", "arguments": "{}"},
                }
            ],
        }
    ]
    session.input_items.append({"role": "user", "content": "安装 superpowers skill", "xiaoming": {"id": "msg-1"}})
    session.input_items.append({"role": "assistant", "content": "我会安排安装。", "xiaoming": {"id": "msg-2"}})
    session.input_items.append({"type": "function_call", "call_id": "call_pending", "name": "schedule_background_task", "arguments": "{}"})
    session.remember_loaded_skill(LoadedSkill.create(name="brainstorming", description="Discuss requirements.", content="Ask before implementation."))

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        session_provider=lambda: session,
    )
    coordinator.start()
    try:
        spec = TaskSpec(
            title="Install superpowers",
            goal="Install the superpowers skill from https://github.com/obra/superpowers/tree/main/skills/brainstorming",
        )
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
    finally:
        coordinator.stop()

    config = workers[0].config
    assert config.agent_type == "worker"
    assert config.context_policy == "forked"
    assert config.skills_to_preload == []
    assert config.context_packet is not None
    assert config.context_packet.session_id == "main-session"
    assert config.context_packet.selected_skills == []
    assert config.context_packet.relevant_messages == []
    assert config.forked_instructions == "MAIN SYSTEM PROMPT"
    assert config.forked_input_items == [
        {"role": "developer", "content": "<bootstrap_context>stable rendered context</bootstrap_context>", "xiaoming": {"kind": "bootstrap_context"}},
        {"role": "user", "content": "[@10:00] 安装 superpowers skill", "xiaoming": {"id": "msg-1", "date": "2026-06-09", "time": "10:00", "timezone": "Asia/Shanghai"}},
        {
            "role": "assistant",
            "content": "我会安排安装。",
            "tool_calls": [
                {
                    "id": "call-schedule",
                    "type": "function",
                    "function": {"name": "schedule_background_task", "arguments": "{}"},
                }
            ],
        },
        {"type": "function_call_output", "call_id": "call-schedule", "output": "Fork started - processing in background"},
    ]
    assert config.forked_loaded_skills == []
    session.input_items[0]["content"] = "mutated after scheduling"
    session.loaded_skills["brainstorming"].content = "mutated after scheduling"
    assert config.forked_input_items[1]["content"] == "[@10:00] 安装 superpowers skill"


def test_forked_worker_context_prefix_matches_main_prompt_snapshot_bytes_after_multiple_turns(tmp_path: Path):
    workers = []
    session = Session(session_id="main-session")
    session.base_instructions = "MAIN SYSTEM PROMPT"
    main_prompt_prefix = [
        {"role": "developer", "content": "<bootstrap_context>stable rendered context</bootstrap_context>", "xiaoming": {"kind": "bootstrap_context"}},
        {"role": "user", "content": "[@10:00] 帮我安装 superpowers", "xiaoming": {"id": "msg-1", "date": "2026-06-09", "time": "10:00", "timezone": "Asia/Shanghai"}},
        {"role": "assistant", "content": "[@10:00] 我会安排后台安装。", "xiaoming": {"id": "msg-2", "date": "2026-06-09", "time": "10:00", "timezone": "Asia/Shanghai"}},
        {"role": "user", "content": "[@10:02] 帮我开发五子棋网页", "xiaoming": {"id": "msg-3", "date": "2026-06-09", "time": "10:02", "timezone": "Asia/Shanghai"}},
        {"role": "assistant", "content": "[@10:02] 这个开发任务我会放到后台。", "xiaoming": {"id": "msg-4", "date": "2026-06-09", "time": "10:02", "timezone": "Asia/Shanghai"}},
        {"role": "user", "content": "[@10:05] 再看看当前后台任务", "xiaoming": {"id": "msg-5", "date": "2026-06-09", "time": "10:05", "timezone": "Asia/Shanghai"}},
    ]
    schedule_output = {
        "role": "assistant",
        "content": "我会安排第三个后台任务。",
        "tool_calls": [
            {
                "id": "call-schedule-third",
                "type": "function",
                "function": {"name": "schedule_background_task", "arguments": "{}"},
            }
        ],
    }
    session.last_prompt_input_items = main_prompt_prefix
    session.last_model_output_items = [schedule_output]

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        session_provider=lambda: session,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="第三个任务", goal="继续完成第三个后台任务"))
        _eventually(lambda: len(workers) == 1)
    finally:
        coordinator.stop()

    forked_items = workers[0].config.forked_input_items
    expected_items = [
        *main_prompt_prefix,
        schedule_output,
        {"type": "function_call_output", "call_id": "call-schedule-third", "output": "Fork started - processing in background"},
    ]
    assert _canonical_bytes(forked_items) == _canonical_bytes(expected_items)
    assert _canonical_bytes(forked_items[: len(main_prompt_prefix)]) == _canonical_bytes(main_prompt_prefix)


def test_schedule_background_task_uses_single_worker_for_skill_install_messages(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        spec = TaskSpec(
            title="Install brainstorming skill",
            goal="Install skill from https://github.com/obra/superpowers/tree/main/skills/brainstorming",
            agent_type="general",
        )
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
    finally:
        coordinator.stop()

    assert workers[0].config.agent_type == "worker"
    assert workers[0].config.context_policy == "forked"
    assert workers[0].config.skills_to_preload == []


def test_context_summary_includes_active_tasks_and_pending_questions(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.schedule_background_task("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        summary = coordinator.context_summary()
    finally:
        coordinator.stop()

    assert "Background tasks:" in summary
    assert "写 README" in summary
    assert "needs_user" in summary
    assert "Pending worker questions:" in summary
    assert "approve-1" in summary
    assert "允许写 README.md" in summary


def test_reported_completed_task_is_accepted_after_verification(tmp_path: Path):
    notices = []
    workers = []
    artifact = tmp_path / "README.md"

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        spec = TaskSpec(title="写 README", goal="写 README", expected_artifacts=["README.md"], allowed_write_paths=["README.md"])
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
        artifact.write_text("ok\n")
        workers[0].on_event(
            WorkerEvent(
                workers[0].config.task_id,
                "reported",
                "done",
                {"report": {"status": "completed", "summary": "done", "changed_files": [], "created_files": ["README.md"], "artifacts": ["README.md"], "verification": [], "blockers": [], "evidence": ["README.md exists"]}},
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
    finally:
        coordinator.stop()

    assert any("已提交结果，正在验收" in notice.message for notice in notices)
    assert any("已完成" in notice.message for notice in notices)


def test_completed_task_final_answer_is_verified_before_acceptance(tmp_path: Path):
    notices = []
    workers = []
    verifier = RecordingVerifier()

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
        verifier=verifier,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "done in final answer"))
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
    finally:
        coordinator.stop()

    assert len(verifier.calls) == 1
    assert verifier.calls[0][1].status == "completed"
    assert verifier.calls[0][1].summary == "done in final answer"
    assert workers[0].sent[-1] == ("review_accepted", {"message": "accepted"})
    assert any("已提交结果，正在验收" in notice.message for notice in notices)
    assert any("已完成" in notice.message for notice in notices)


def test_rejected_completion_sends_review_feedback_to_same_worker(tmp_path: Path):
    notices = []
    workers = []
    verifier = SequenceVerifier(False)

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
        verifier=verifier,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "done but incomplete"))
        _eventually(lambda: bool(workers[0].sent))
    finally:
        coordinator.stop()

    task = coordinator.registry.list()[0]
    assert task.revision_attempts == 1
    assert len(workers) == 1
    assert workers[0].sent[-1][0] == "review_feedback"
    assert "artifact missing" in workers[0].sent[-1][1]["feedback"]
    assert any("需要修正" in notice.message for notice in notices)


def test_completed_task_uses_internal_verifier_worker_when_model_is_configured(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path, provider="deepseek", model="fake-model"),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "created nothing"))
        _eventually(lambda: len(workers) == 2)
        verifier = workers[1]
        verifier.on_event(
            WorkerEvent(
                verifier.config.task_id,
                "completed",
                _review_report(
                    "NEEDS_REVISION",
                    summary="README is missing.",
                    evidence="I inspected the workspace and did not find README.md.",
                    issues="README.md missing.",
                    feedback="Create README.md and report the file path.",
                    summary_for_main="The worker has not created README.md yet.",
                ),
            )
        )
        _eventually(lambda: bool(workers[0].sent))
        task = coordinator.registry.list()[0]
        assert task.status == "needs_revision"
        assert task.verifier_task_ids == [workers[1].config.task_id]
        assert task.worker_submissions[0].summary == "created nothing"
        assert task.review_reports[0].status == "NEEDS_REVISION"
        assert task.review_reports[0].feedback_to_worker == "Create README.md and report the file path."
        assert workers[0].sent[-1][0] == "review_feedback"
        assert "Create README.md" in workers[0].sent[-1][1]["feedback"]
        assert workers[1].sent[-1] == ("review_accepted", {"message": "review result received"})
    finally:
        coordinator.stop()

    assert workers[1].config.agent_type == "verifier"
    assert workers[1].config.context_policy == "forked"
    assert workers[1].config.forked_input_items == workers[0].config.forked_input_items
    assert workers[1].config.forked_instructions == workers[0].config.forked_instructions
    assert "created nothing" in workers[1].config.task


def test_internal_verifier_prompt_includes_worker_question_answers(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path, provider="deepseek", model="fake-model"),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "clarification_request", "README 使用中文吗？", {"request_id": "q1", "purpose": "clarify"}))
        _eventually(lambda: coordinator.has_pending_question())
        _reply_pending_mailbox(coordinator, "使用中文", "none", "已转给 worker。")
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "created README.md in Chinese"))
        _eventually(lambda: len(workers) == 2)
    finally:
        coordinator.stop()

    assert "README 使用中文吗？" in workers[1].config.task
    assert "使用中文" in workers[1].config.task


def test_internal_verifier_revision_with_missing_worker_escalates_to_main_without_repair_worker(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path, provider="deepseek", model="fake-model"),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        task_id = workers[0].config.task_id
        workers[0].on_event(WorkerEvent(task_id, "completed", "created nothing"))
        _eventually(lambda: len(workers) == 2)
        coordinator._workers.pop(task_id)
        workers[1].on_event(
            WorkerEvent(
                workers[1].config.task_id,
                "completed",
                _review_report(
                    "NEEDS_REVISION",
                    summary="README is missing.",
                    issues="README.md missing.",
                    feedback="Create README.md.",
                    summary_for_main="The original worker is unavailable.",
                ),
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "needs_user_decision")
    finally:
        coordinator.stop()

    assert len(workers) == 2
    assert coordinator.registry.list()[0].needs_user_decision_summary
    assert any("需要用户决策" in notice.message for notice in notices)


def test_internal_verifier_worker_accepts_with_review_report(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path, provider="deepseek", model="fake-model"),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "created README.md"))
        _eventually(lambda: len(workers) == 2)
        workers[1].on_event(
            WorkerEvent(
                workers[1].config.task_id,
                "completed",
                _review_report(
                    "ACCEPTED",
                    summary="README satisfies the request.",
                    evidence="README.md exists and contains project description.",
                    issues="(none)",
                ),
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
        task = coordinator.registry.list()[0]
        assert task.review_reports[0].status == "ACCEPTED"
        assert task.verification_result is not None
        assert task.verification_result.accepted is True
        assert workers[0].sent[-1] == ("review_accepted", {"message": "accepted"})
    finally:
        coordinator.stop()

    assert any("已完成" in notice.message for notice in notices)


def test_internal_verifier_accepts_markdown_section_labels(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path, provider="deepseek", model="fake-model"),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写文件", goal="创建 ok.txt"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "created ok.txt"))
        _eventually(lambda: len(workers) == 2)
        workers[1].on_event(
            WorkerEvent(
                workers[1].config.task_id,
                "completed",
                "文件存在且内容正确。\n\n"
                "**Review-Status:** ACCEPTED\n\n"
                "**Summary:** 文件已成功创建。\n\n"
                "**Evidence:** read_file confirmed ok.txt.\n\n"
                "**Issues:** (none)\n\n"
                "**Feedback-To-Worker:** (none)\n\n"
                "**Summary-For-Main:** (none)",
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
        task = coordinator.registry.list()[0]
        assert task.review_reports[0].status == "ACCEPTED"
        assert task.review_reports[0].summary == "文件已成功创建。"
        assert task.needs_user_decision_summary == ""
    finally:
        coordinator.stop()

    assert any("已完成" in notice.message for notice in notices)


def test_invalid_internal_verifier_report_needs_user_decision(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path, provider="deepseek", model="fake-model"),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "created README.md"))
        _eventually(lambda: len(workers) == 2)
        workers[1].on_event(WorkerEvent(workers[1].config.task_id, "completed", "looks good"))
        _eventually(lambda: coordinator.registry.list()[0].status == "needs_user_decision")
        task = coordinator.registry.list()[0]
        assert "missing Review-Status" in task.needs_user_decision_summary
        assert task.active_verifier_id == ""
    finally:
        coordinator.stop()

    assert any("无法解析" in notice.message for notice in notices)


def test_internal_verifier_revision_limit_escalates_to_main(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path, provider="deepseek", model="fake-model"),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        task.revision_attempts = MAX_REVISION_ATTEMPTS
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "still incomplete"))
        _eventually(lambda: len(workers) == 2)
        workers[1].on_event(
            WorkerEvent(
                workers[1].config.task_id,
                "completed",
                _review_report(
                    "NEEDS_REVISION",
                    summary="README is still missing.",
                    issues="README.md missing.",
                    feedback="Create README.md.",
                    summary_for_main="The worker did not create README.md after revisions.",
                ),
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "needs_user_decision")
        task = coordinator.registry.list()[0]
        assert task.review_reports[0].status == "NEEDS_REVISION"
        assert "Review feedback loop exhausted" in task.needs_user_decision_summary
        assert "Create README.md" in task.needs_user_decision_summary
    finally:
        coordinator.stop()

    assert any("需要用户决策" in notice.message for notice in notices)


def test_rejected_completion_needs_user_decision_after_revision_limit(tmp_path: Path):
    notices = []
    workers = []
    verifier = SequenceVerifier(False)

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
        verifier=verifier,
    )
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="写 README", goal="写 README"))
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        task.revision_attempts = MAX_REVISION_ATTEMPTS
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "still incomplete"))
        _eventually(lambda: coordinator.registry.list()[0].status == "needs_user_decision")
    finally:
        coordinator.stop()

    assert len(workers) == 1
    assert not workers[0].sent
    assert "Review feedback loop exhausted" in coordinator.registry.list()[0].needs_user_decision_summary
    assert any("需要用户决策" in notice.message for notice in notices)


def test_context_summary_excludes_accepted_tasks_but_status_keeps_history(tmp_path: Path):
    workers = []
    artifact = tmp_path / "README.md"

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        spec = TaskSpec(title="写 README", goal="写 README", expected_artifacts=["README.md"], allowed_write_paths=["README.md"])
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
        artifact.write_text("ok\n")
        workers[0].on_event(
            WorkerEvent(
                workers[0].config.task_id,
                "reported",
                "done",
                {"report": {"status": "completed", "summary": "done", "changed_files": [], "created_files": ["README.md"], "artifacts": ["README.md"], "verification": [], "blockers": [], "evidence": ["README.md exists"]}},
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
        summary = coordinator.context_summary()
        status = coordinator.tasks_text()
    finally:
        coordinator.stop()

    assert summary is None
    assert "写 README" in status
    assert "accepted" in status


def test_current_tasks_text_excludes_cancelled_history(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.schedule_background_task("安装 superpowers skill")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        coordinator.cancel_task(task.task_id)
        current = coordinator.current_tasks_text()
        history = coordinator.tasks_text()
    finally:
        coordinator.stop()

    assert current == "当前没有后台任务。"
    assert "cancelled" in history


def test_talk_to_waiting_task_prefix_returns_status_instead_of_unknown(tmp_path: Path):
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder())
    task = TaskRecord(title="安装 superpowers skill", original_request="安装 skill", current_goal="安装 skill", status="waiting")
    coordinator.registry.add(task)

    result = coordinator.talk_to_peer(task.task_id[:8], "现在怎么样了？")

    assert result.status == "success"
    assert "安装 superpowers skill" in result.output
    assert "waiting" in result.output


def test_register_running_worker_task_adds_local_thread_worker(tmp_path: Path):
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder())
    task = TaskRecord(title="前台任务", original_request="long task", current_goal="long task", agent_type="local_thread")
    handle = RecordingHandle(task.task_id)

    task_id = coordinator.register_running_worker_task(task, handle)

    assert task_id == task.task_id
    assert coordinator.registry.get(task_id) is task
    assert coordinator.registry.active() == [task]
    assert coordinator._workers[task_id] is handle
    assert "local_thread" in coordinator.tasks_text()


def test_local_thread_worker_approval_uses_question_decider(tmp_path: Path):
    decider = FakeQuestionDecider()
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), question_decider=decider)
    task = TaskRecord(title="前台任务", original_request="install", current_goal="install", agent_type="local_thread", authorization_note="自动批准合理安装请求")
    result = []
    handle = ForegroundWorkerHandle(task.task_id, coordinator.submit_worker_event, threading.Event())
    coordinator.register_running_worker_task(task, handle)
    coordinator.start()
    try:
        thread = threading.Thread(target=lambda: result.append(handle.request_approval("install skill?")))
        thread.start()
        thread.join(timeout=2)
    finally:
        coordinator.stop()

    assert result == [True]
    assert decider.calls
    assert coordinator.mailbox.pending_reply_messages() == []
    assert task.last_worker_answer_decision == "approved"


def test_local_thread_worker_approval_waits_for_user_reply(tmp_path: Path):
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), question_decider=FakeQuestionDecider())
    task = TaskRecord(title="前台任务", original_request="write", current_goal="write", agent_type="local_thread")
    result = []
    handle = ForegroundWorkerHandle(task.task_id, coordinator.submit_worker_event, threading.Event())
    coordinator.register_running_worker_task(task, handle)
    coordinator.start()
    try:
        thread = threading.Thread(target=lambda: result.append(handle.request_approval("write file?")))
        thread.start()
        _eventually(lambda: bool(coordinator.mailbox.pending_reply_messages()))
        message = coordinator.mailbox.pending_reply_messages()[0]
        reply = coordinator.reply_mailbox_message(message.message_id, "允许写文件", decision="approved")
        thread.join(timeout=2)
    finally:
        coordinator.stop()

    assert reply.status == "success"
    assert result == [True]
    assert task.last_worker_answer_decision == "approved"


def test_talk_to_local_thread_worker_uses_worker_send(tmp_path: Path):
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder())
    task = TaskRecord(title="前台任务", original_request="long task", current_goal="long task", agent_type="local_thread")
    handle = ReplyingHandle(task.task_id, coordinator.submit_worker_event)
    coordinator.register_running_worker_task(task, handle)
    coordinator.start()
    try:
        result = coordinator.talk_to_peer(task.task_id, "进展如何？")
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert result.output == "收到，继续处理。"
    assert handle.sent == [("talk", {"message": "进展如何？"})]


def test_cancel_local_thread_worker_uses_terminate(tmp_path: Path):
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder())
    task = TaskRecord(title="前台任务", original_request="long task", current_goal="long task", agent_type="local_thread")
    handle = RecordingHandle(task.task_id)
    coordinator.register_running_worker_task(task, handle)

    result = coordinator.cancel_task(task.task_id)

    assert result.status == "success"
    assert handle.terminated is True
    assert "cancelled" in coordinator.tasks_text()


def test_reported_completed_task_accepts_directory_scope_and_described_artifact(tmp_path: Path):
    notices = []
    workers = []
    skill_dir = tmp_path / ".agents" / "skills" / "superpowers"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("ok\n")

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        spec = TaskSpec(title="安装 skill", goal="安装 skill", expected_artifacts=[".agents/skills/superpowers/ 目录及其中的 skill 文件"], allowed_write_paths=[".agents/skills/superpowers/"], verification_commands=["ls -la .agents/skills/superpowers/"])
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(
            WorkerEvent(
                workers[0].config.task_id,
                "reported",
                "done",
                {
                    "report": {
                        "status": "completed",
                        "summary": "done",
                        "changed_files": [],
                        "created_files": [".agents/skills/superpowers/SKILL.md"],
                        "artifacts": [".agents/skills/superpowers/ directory with SKILL.md"],
                        "verification": [],
                        "blockers": [],
                        "evidence": ["Files successfully created and verified via ls -la"],
                    }
                },
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
    finally:
        coordinator.stop()

    assert any("已完成" in notice.message for notice in notices)


def test_reported_completed_task_is_rejected_when_artifact_missing(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path, max_revision_attempts=0), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        spec = TaskSpec(title="安装 skill", goal="安装 skill", expected_artifacts=[".agents/skills/superpowers/SKILL.md"], allowed_write_paths=[".agents/skills/**"])
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(
            WorkerEvent(
                workers[0].config.task_id,
                "reported",
                "done",
                {"report": {"status": "completed", "summary": "done", "changed_files": [], "created_files": [], "artifacts": [], "verification": [], "blockers": [], "evidence": []}},
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "rejected")
    finally:
        coordinator.stop()

    task = coordinator.registry.list()[0]
    assert "expected artifact missing" in task.last_progress
    assert any("未通过验收" in notice.message for notice in notices)


def test_reported_completed_task_accepts_absolute_allowed_path_and_relative_report(tmp_path: Path):
    notices = []
    workers = []
    artifact = tmp_path / "gomoku.html"

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        spec = TaskSpec(title="开发五子棋网页", goal="开发五子棋网页", expected_artifacts=["一个独立的 HTML 文件：gomoku.html"], allowed_write_paths=[str(artifact)])
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
        artifact.write_text("<!doctype html>\n")
        workers[0].on_event(
            WorkerEvent(
                workers[0].config.task_id,
                "reported",
                "done",
                {"report": {"status": "completed", "summary": "done", "changed_files": ["gomoku.html"], "created_files": ["gomoku.html"], "artifacts": ["gomoku.html"], "verification": [], "blockers": [], "evidence": ["gomoku.html exists"]}},
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
    finally:
        coordinator.stop()

    assert any("已完成" in notice.message for notice in notices)


def test_reported_completed_task_accepts_described_directory_artifact_from_report(tmp_path: Path):
    notices = []
    workers = []
    artifact = tmp_path / "superpowers"

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        spec = TaskSpec(title="安装 superpowers skill", goal="安装 superpowers skill", expected_artifacts=["克隆到本地的 superpowers 仓库目录，安装好的依赖和配置文件。"], allowed_write_paths=[str(artifact)])
        coordinator.schedule_background_task(spec)
        _eventually(lambda: len(workers) == 1)
        artifact.mkdir()
        (artifact / "README.md").write_text("ok\n")
        workers[0].on_event(
            WorkerEvent(
                workers[0].config.task_id,
                "reported",
                "done",
                {"report": {"status": "completed", "summary": "done", "changed_files": [], "created_files": [str(artifact)], "artifacts": [str(artifact)], "verification": [], "blockers": [], "evidence": ["superpowers exists"]}},
            )
        )
        _eventually(lambda: coordinator.registry.list()[0].status == "accepted")
    finally:
        coordinator.stop()

    assert any("已完成" in notice.message for notice in notices)


def test_reported_failed_and_blocked_are_not_accepted(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.schedule_background_task(TaskSpec(title="失败任务", goal="失败任务"))
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "reported", "failed", {"report": {"status": "failed", "summary": "network down", "changed_files": [], "created_files": [], "artifacts": [], "verification": [], "blockers": [], "evidence": []}}))
        _eventually(lambda: coordinator.registry.list()[0].status == "failed")

        coordinator.schedule_background_task(TaskSpec(title="阻塞任务", goal="阻塞任务"))
        _eventually(lambda: len(workers) == 2)
        workers[1].on_event(WorkerEvent(workers[1].config.task_id, "reported", "blocked", {"report": {"status": "blocked", "summary": "need URL", "changed_files": [], "created_files": [], "artifacts": [], "verification": [], "blockers": ["need URL"], "evidence": []}}))
        _eventually(lambda: coordinator.registry.list()[1].status == "blocked")
    finally:
        coordinator.stop()

    assert any("失败：network down" in notice.message for notice in notices)
    assert any("需要你确认：need URL" in notice.message for notice in notices)


def test_coordinator_queues_conflicting_task_until_first_completes(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.submit_user_message("修改 src/xiaoming/cli.py")
        _eventually(lambda: len(workers) == 1)
        coordinator.submit_user_message("继续修改 src/xiaoming/cli.py")
        _eventually(lambda: len(coordinator.registry.waiting()) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "done"))
        _eventually(lambda: len(workers) == 2)
    finally:
        coordinator.stop()

    assert any("动态调度：排队等待" in notice.message for notice in notices)
    assert workers[1].config.task == "继续修改 src/xiaoming/cli.py"


def test_coordinator_routes_question_answer_to_worker(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "clarification_request", "使用中文吗？", {"request_id": "q1"}))
        _eventually(lambda: coordinator.has_pending_question())
        ack = _reply_pending_mailbox(coordinator, "用中文", "none", "好的，已转给后台任务。")
        status_after_answer = coordinator.registry.list()[0].status
    finally:
        coordinator.stop()

    assert ack.status == "success"
    assert workers[0].sent[-1] == ("answer_question", {"request_id": "q1", "answer": "用中文", "decision": "none"})
    assert status_after_answer == "running"


def test_coordinator_exposes_worker_question_context_and_options(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("讨论围棋需求")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(
            WorkerEvent(
                workers[0].config.task_id,
                "clarification_request",
                "棋盘大小选哪个？",
                {"request_id": "q1", "purpose": "clarify", "context": "brainstorming first question", "options": ["19x19", "9x9"]},
            )
        )
        _eventually(lambda: coordinator.has_pending_question())
        pending = coordinator.pending_questions_text() or ""
    finally:
        coordinator.stop()

    assert "context: brainstorming first question" in pending
    assert "options: 19x19 | 9x9" in pending


def test_tasks_text_includes_task_id_and_pending_question_id(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task_id = workers[0].config.task_id
        workers[0].on_event(WorkerEvent(task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        text = coordinator.tasks_text()
    finally:
        coordinator.stop()

    assert task_id[:8] in text
    assert "needs_user" in text
    assert "approve-1" in text
    assert "允许写 README.md" in text


def test_tasks_text_includes_pending_mailbox_message_without_legacy_question(tmp_path: Path):
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=lambda config, on_event: FakeWorker(config, on_event))
    task = coordinator.registry.add(TaskRecord(title="写 README", original_request="写 README", current_goal="写 README", task_id="task-1", status="needs_user"))
    coordinator.mailbox.create_message(
        task_id=task.task_id,
        worker_id=task.task_id,
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许写 README.md？",
        requires_reply=True,
        metadata={"request_id": "approve-1"},
    )

    text = coordinator.tasks_text()

    assert "approve-1" in text
    assert "允许写 README.md" in text


def test_task_record_no_longer_has_legacy_pending_questions_queue():
    task = TaskRecord(title="任务", original_request="任务", current_goal="任务")

    assert not hasattr(task, "pending_questions")


def test_coordinator_routes_approval_request_to_user_and_answer_back_to_worker(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        reply = _reply_pending_mailbox(coordinator, "允许写 README.md", "approved", "好的，已批准。")
        status_after_answer = coordinator.registry.list()[0].status
    finally:
        coordinator.stop()

    assert any("允许写 README.md" in notice.message for notice in notices)
    assert reply.status == "success"
    assert workers[0].sent[-1] == ("answer_question", {"request_id": "approve-1", "answer": "允许写 README.md", "decision": "approved"})
    assert status_after_answer == "running"


def test_coordinator_records_worker_question_in_mailbox_and_answers_by_mailbox_id(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        pending_messages = coordinator.mailbox.pending_for_main()
        pending_status = pending_messages[0].status
        pending_request_id = pending_messages[0].metadata["request_id"]
        pending_requires_reply = pending_messages[0].requires_reply
        result = _reply_pending_mailbox(coordinator, "允许写 README.md", "approved", "好的，已批准。")
        answered_messages = coordinator.mailbox.list()
    finally:
        coordinator.stop()

    assert len(pending_messages) == 1
    assert pending_status == "pending"
    assert pending_request_id == "approve-1"
    assert pending_requires_reply is True
    assert result.status == "success"
    assert answered_messages[0].status == "answered"
    assert answered_messages[0].reply == "允许写 README.md"
    assert workers[0].sent[-1] == ("answer_question", {"request_id": "approve-1", "answer": "允许写 README.md", "decision": "approved"})


def test_coordinator_replies_to_worker_question_by_mailbox_message_id(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.mailbox.pending_reply_messages())
        message_id = coordinator.mailbox.pending_reply_messages()[0].message_id
        result = coordinator.reply_mailbox_message(message_id, "允许写 README.md", "approved", "好的，已批准。")
        answered_messages = coordinator.mailbox.list()
        status_after_answer = coordinator.registry.list()[0].status
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert answered_messages[0].status == "answered"
    assert answered_messages[0].reply == "允许写 README.md"
    assert workers[0].sent[-1] == ("answer_question", {"request_id": "approve-1", "answer": "允许写 README.md", "decision": "approved"})
    assert status_after_answer == "running"


def test_coordinator_does_not_mark_mailbox_question_presented_until_ui_ack(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.mailbox.pending_for_main())
        message = coordinator.mailbox.pending_for_main()[0]
    finally:
        coordinator.stop()

    assert notices
    question_notice = next(notice for notice in notices if notice.message_id == message.message_id)
    assert question_notice.task_id == message.task_id
    assert message.presented_count == 0
    assert message.last_presented_at is None

    coordinator.mark_notice_presented(message.message_id)

    assert message.presented_count == 1
    assert message.last_presented_at


def test_coordinator_question_notice_is_driven_by_mailbox_candidate(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task_id = workers[0].config.task_id
        workers[0].on_event(WorkerEvent(task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.mailbox.pending_reply_messages())
        message = coordinator.mailbox.pending_reply_messages()[0]
    finally:
        coordinator.stop()

    assert any(notice.message == "写 README 需要你确认：写 README：允许写 README.md？" for notice in notices)
    assert message.presented_count == 0


def test_coordinator_can_replay_presented_pending_questions_on_cli_start(tmp_path: Path):
    notices = []
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=lambda config, on_event: FakeWorker(config, on_event), on_notice=notices.append)
    task = coordinator.registry.add(TaskRecord(title="旧任务", original_request="写 README", current_goal="写 README", task_id="task-1", status="needs_user"))
    message = coordinator.mailbox.create_message(
        task_id=task.task_id,
        worker_id=task.task_id,
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许写 README.md？",
        requires_reply=True,
        metadata={"request_id": "approve-1", "task_title": task.title},
    )
    coordinator.mailbox.mark_presented(message.message_id)

    replayed = coordinator.replay_pending_questions()

    assert replayed == 1
    assert notices[0].message_id == message.message_id
    assert notices[0].message == "旧任务 需要你确认：允许写 README.md？"


def test_coordinator_persists_worker_question_mailbox(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: TaskStore(tmp_path).load_mailbox().pending_for_main())
        restored = TaskStore(tmp_path).load_mailbox()
    finally:
        coordinator.stop()

    assert restored.pending_for_main()[0].metadata["request_id"] == "approve-1"


def test_coordinator_cancels_stale_pending_mailbox_for_terminal_recovered_task(tmp_path: Path):
    registry = TaskRegistry()
    task = registry.add(TaskRecord(title="写 README", original_request="写 README", current_goal="写 README", task_id="task-1", status="failed"))
    mailbox = MailboxStore()
    mailbox.create_message(
        task_id=task.task_id,
        worker_id=task.task_id,
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许写 README.md？",
        requires_reply=True,
        metadata={"request_id": "q1"},
    )
    store = TaskStore(tmp_path)
    store.save_registry(registry)
    store.save_mailbox(mailbox)

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=lambda config, on_event: FakeWorker(config, on_event))

    assert coordinator.mailbox.list()[0].status == "cancelled"
    assert coordinator.has_pending_question() is False


def test_coordinator_cancels_pending_mailbox_question_for_recovered_active_task(tmp_path: Path):
    registry = TaskRegistry()
    task = registry.add(TaskRecord(title="写 README", original_request="写 README", current_goal="写 README", task_id="task-1", status="needs_user"))
    mailbox = MailboxStore()
    message = mailbox.create_message(
        task_id=task.task_id,
        worker_id=task.task_id,
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许写 README.md？",
        requires_reply=True,
        metadata={"request_id": "q1"},
    )
    store = TaskStore(tmp_path)
    store.save_registry(registry)
    store.save_mailbox(mailbox)

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=lambda config, on_event: FakeWorker(config, on_event))

    result = coordinator.reply_mailbox_message(message.message_id, "允许写 README.md", "approved", "已批准。")

    assert result.status == "error"
    assert "unknown pending mailbox message" in result.error
    assert coordinator.mailbox.list()[0].status == "cancelled"
    assert coordinator.has_pending_question() is False


def test_coordinator_keeps_non_reply_mailbox_history_for_terminal_recovered_task(tmp_path: Path):
    registry = TaskRegistry()
    task = registry.add(TaskRecord(title="写 README", original_request="写 README", current_goal="写 README", task_id="task-1", status="accepted"))
    mailbox = MailboxStore()
    mailbox.create_message(
        task_id=task.task_id,
        worker_id=task.task_id,
        from_role="worker",
        to_role="main",
        kind="result",
        content="README 已创建",
        requires_reply=False,
    )
    store = TaskStore(tmp_path)
    store.save_registry(registry)
    store.save_mailbox(mailbox)

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=lambda config, on_event: FakeWorker(config, on_event))

    assert coordinator.mailbox.list()[0].status == "pending"
    assert coordinator.mailbox.pending_for_main() == []


def test_coordinator_cancels_pending_mailbox_messages_when_task_is_cancelled(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task_id = workers[0].config.task_id
        workers[0].on_event(WorkerEvent(task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        coordinator.cancel_task(task_id)
        messages = coordinator.mailbox.list()
    finally:
        coordinator.stop()

    assert messages
    assert messages[0].status == "cancelled"
    assert coordinator.has_pending_question() is False


def test_coordinator_records_worker_progress_and_result_in_mailbox_without_pending_reply(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        verifier=RecordingVerifier(True),
    )
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task_id = workers[0].config.task_id
        workers[0].on_event(WorkerEvent(task_id, "progress", "正在写 README"))
        workers[0].on_event(WorkerEvent(task_id, "completed", "README 已创建"))
        _eventually(lambda: coordinator.registry.get(task_id).status == "accepted")
        messages = coordinator.mailbox.list()
        pending = coordinator.mailbox.pending_for_main()
    finally:
        coordinator.stop()

    assert [message.kind for message in messages] == ["progress", "result"]
    assert [message.content for message in messages] == ["正在写 README", "README 已创建"]
    assert all(message.requires_reply is False for message in messages)
    assert pending == []


def test_coordinator_saves_authorization_note_from_worker_answer(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("安装 superpowers")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "是否允许安装 brainstorming？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        _reply_pending_mailbox(
            coordinator,
            "同意安装 brainstorming，后续 superpowers 安装请求由小明判断。",
            "approved",
            "已批准。",
            authorization_note="用户授权小明在当前 superpowers 安装任务中自动批准合理的低风险安装请求。",
        )
        task = coordinator.registry.list()[0]
    finally:
        coordinator.stop()

    assert "自动批准合理的低风险安装请求" in task.authorization_note


def test_coordinator_auto_answers_worker_question_when_authorization_note_covers_it(tmp_path: Path):
    notices = []
    workers = []
    decider = FakeQuestionDecider()

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FakeResponder(),
        worker_factory=factory,
        on_notice=notices.append,
        question_decider=decider,
    )
    coordinator.start()
    try:
        coordinator.submit_user_message("安装 superpowers")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        task.authorization_note = "用户授权小明在当前任务内自动批准合理的 superpowers 安装请求。"
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "是否允许安装 writing-plans？", {"request_id": "approve-2"}))
        _eventually(lambda: workers[0].sent)
        mailbox_messages = coordinator.mailbox.list()
    finally:
        coordinator.stop()

    assert workers[0].sent[-1] == ("answer_question", {"request_id": "approve-2", "answer": "根据授权自动批准。", "decision": "approved"})
    assert coordinator.has_pending_question() is False
    assert mailbox_messages[0].status == "answered"
    assert mailbox_messages[0].reply == "根据授权自动批准。"
    assert any("根据授权" in notice.message for notice in notices)
    assert decider.calls


def test_coordinator_keeps_worker_question_visible_when_responder_fails(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(
        CoordinatorConfig(tmp_path),
        scheduler=FakeScheduler(),
        responder=FailingResponder(),
        worker_factory=factory,
        on_notice=notices.append,
    )
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        task = coordinator.registry.list()[0]
        status_before_stop = task.status
        worker_terminated_before_stop = workers[0].terminated
    finally:
        coordinator.stop()

    assert status_before_stop == "needs_user"
    assert worker_terminated_before_stop is False
    assert any("允许写 README.md" in notice.message for notice in notices)
    assert not any("Error:" in notice.message for notice in notices)


def test_coordinator_does_not_treat_unrelated_message_as_worker_answer(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        coordinator.submit_user_message("好，完成后告诉我")
        time.sleep(0.05)
    finally:
        coordinator.stop()

    assert coordinator.has_pending_question()
    assert workers[0].sent == []


def test_coordinator_rejects_invalid_mailbox_reply_decision(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        result = coordinator.reply_mailbox_message(_pending_mailbox_message_id(coordinator), "允许", "maybe", "请明确是否批准。")
    finally:
        coordinator.stop()

    assert result.status == "error"
    assert "invalid decision" in result.error
    assert coordinator.has_pending_question()
    assert workers[0].sent == []


def test_coordinator_rejects_reused_mailbox_message_id(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "approval_request", "允许写 README.md？", {"request_id": "approve-1"}))
        _eventually(lambda: coordinator.has_pending_question())
        message_id = _pending_mailbox_message_id(coordinator)
        first = coordinator.reply_mailbox_message(message_id, "允许", "approved", "已批准。")
        second = coordinator.reply_mailbox_message(message_id, "允许", "approved", "已批准。")
    finally:
        coordinator.stop()

    assert first.status == "success"
    assert second.status == "error"
    assert "unknown pending mailbox message" in second.error
    assert workers[0].sent == [("answer_question", {"request_id": "approve-1", "answer": "允许", "decision": "approved"})]


def test_coordinator_keeps_worker_progress_quiet_by_default(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    coordinator.submit_user_message("写 README")
    _eventually(lambda: len(workers) == 1)
    notices.clear()
    workers[0].on_event(WorkerEvent(workers[0].config.task_id, "tool_started", "write README.md"))
    workers[0].on_event(WorkerEvent(workers[0].config.task_id, "progress", "Thinking about the next step..."))
    _eventually(lambda: coordinator.registry.list()[0].last_progress == "Thinking about the next step...")
    assert notices == []
    coordinator.stop()


def test_coordinator_verbose_mode_prints_worker_progress(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path, quiet=False), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    coordinator.submit_user_message("写 README")
    _eventually(lambda: len(workers) == 1)
    notices.clear()
    workers[0].on_event(WorkerEvent(workers[0].config.task_id, "tool_started", "write README.md"))
    _eventually(lambda: bool(notices))
    assert "write README.md" in notices[-1].message
    coordinator.stop()


def test_coordinator_deduplicates_repeated_task_notice(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path, quiet=False), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        notices.clear()
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "tool_started", "write README.md"))
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "tool_started", "write README.md"))
        _eventually(lambda: bool(notices))
        time.sleep(0.05)
    finally:
        coordinator.stop()

    write_notices = [notice.message for notice in notices if notice.message == "写 README：write README.md"]
    assert write_notices == ["写 README：write README.md"]


def test_coordinator_follow_task_returns_after_status_change(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task_id = workers[0].config.task_id
        results = []
        follower = threading.Thread(target=lambda: results.append(coordinator.follow_task(task_id, timeout_seconds=1).output))
        follower.start()
        time.sleep(0.05)
        workers[0].on_event(WorkerEvent(task_id, "progress", "正在写 README"))
        follower.join(timeout=2)
    finally:
        coordinator.stop()

    assert results
    assert "正在写 README" in results[0]


def test_coordinator_follow_task_times_out_without_stopping_task(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        task_id = workers[0].config.task_id
        result = coordinator.follow_task(task_id, timeout_seconds=0.01)
        status_after_follow = coordinator.registry.get(task_id).status
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert "仍在运行" in result.output
    assert status_after_follow == "running"


def test_coordinator_keeps_question_pending_when_answer_reply_generation_fails(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "clarification_request", "使用中文吗？", {"request_id": "q1"}))
        _eventually(lambda: coordinator.has_pending_question())
        coordinator.responder = FailingResponder()
        reply = coordinator.submit_user_message("用中文")
    finally:
        coordinator.stop()

    assert reply == ""
    assert coordinator.has_pending_question()
    assert workers[0].sent == []


def test_coordinator_reports_worker_failure_once_with_deterministic_notice(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "failed", "invalid worker event: bad json"))
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "failed", "invalid worker event: bad json"))
        _eventually(lambda: coordinator.registry.list()[0].status == "failed")
    finally:
        coordinator.stop()

    failure_notices = [notice.message for notice in notices if "失败" in notice.message]
    assert failure_notices == ["写 README 失败：invalid worker event: bad json"]


def test_cancel_current_ignores_completed_task(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.schedule_background_task("写 README")
        _eventually(lambda: len(workers) == 1)
        workers[0].on_event(WorkerEvent(workers[0].config.task_id, "completed", "done"))
        _eventually(lambda: coordinator.registry.current_task_id is None)
        reply = coordinator.cancel_current()
    finally:
        coordinator.stop()

    assert reply == "当前没有正在运行的后台任务。"


def test_cancel_task_by_id_cancels_running_worker(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.schedule_background_task("写 README")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        result = coordinator.cancel_task(task.task_id)
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert "已请求取消后台任务" in result.output
    assert task.status == "cancelled"
    assert workers[0].terminated is True


def test_cancel_task_by_id_cancels_waiting_task(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 src/xiaoming/cli.py")
        _eventually(lambda: len(workers) == 1)
        coordinator.submit_user_message("继续修改 src/xiaoming/cli.py")
        _eventually(lambda: len(coordinator.registry.waiting()) == 1)
        waiting = coordinator.registry.waiting()[0]
        result = coordinator.cancel_task(waiting.task_id)
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert waiting.status == "cancelled"
    assert len(workers) == 1


def test_cancel_task_clears_pending_worker_questions(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.schedule_background_task("写 README")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        workers[0].on_event(WorkerEvent(task.task_id, "approval_request", "允许写 README.md？", {"request_id": "q1"}))
        _eventually(lambda: coordinator.has_pending_question())
        result = coordinator.cancel_task(task.task_id)
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert coordinator.has_pending_question() is False


def test_cancel_task_unknown_id_returns_error(tmp_path: Path):
    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder())

    result = coordinator.cancel_task("missing-task")

    assert result.status == "error"
    assert "unknown task_id" in result.error


def test_cancel_task_completed_task_is_idempotent(tmp_path: Path):
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory)
    coordinator.start()
    try:
        coordinator.schedule_background_task("写 README")
        _eventually(lambda: len(workers) == 1)
        task = coordinator.registry.list()[0]
        workers[0].on_event(WorkerEvent(task.task_id, "completed", "done"))
        _eventually(lambda: task.status == "accepted")
        result = coordinator.cancel_task(task.task_id)
    finally:
        coordinator.stop()

    assert result.status == "success"
    assert "任务已经结束，无需取消" in result.output
    assert task.status == "accepted"


def test_coordinator_reports_scheduler_failure_without_starting_worker(tmp_path: Path):
    notices = []
    workers = []

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FailingScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        coordinator.submit_user_message("写 README")
        _eventually(lambda: bool(notices))
    finally:
        coordinator.stop()

    assert workers == []
    assert any("Error: scheduler failed after 3 attempts" in notice.message for notice in notices)


def _eventually(predicate, timeout=2):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()
