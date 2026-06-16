from pathlib import Path
import time

from xiaoming.async_runtime.events import WorkerEvent
from xiaoming.async_runtime.context_packets import ResourceClaim, WorkerContextPacket
from xiaoming.async_runtime.mailbox import MailboxStore
from xiaoming.async_runtime.task_store import TaskStore
from xiaoming.async_runtime.tasks import ReviewReport, TaskRecord, TaskRegistry, TaskResultReport, TaskSpec, VerificationResult, WorkerSubmission


def test_task_store_round_trips_task_report_and_verification(tmp_path: Path):
    registry = TaskRegistry()
    task = TaskRecord(
        title="写 README",
        original_request="写 README",
        current_goal="写 README",
        task_spec=TaskSpec(title="写 README", goal="写 README", expected_artifacts=["README.md"]),
        status="rejected",
        last_progress="expected artifact missing: README.md",
    )
    task.result_report = TaskResultReport(status="completed", summary="done", artifacts=[])
    task.verification_result = VerificationResult(accepted=False, reasons=["expected artifact missing: README.md"])
    registry.add(task)

    store = TaskStore(tmp_path)
    store.save_registry(registry)
    restored = store.load_registry()

    restored_task = restored.list()[0]
    assert restored_task.status == "rejected"
    assert restored_task.task_spec is not None
    assert restored_task.task_spec.expected_artifacts == ["README.md"]
    assert restored_task.result_report is not None
    assert restored_task.result_report.status == "completed"
    assert restored_task.verification_result is not None
    assert restored_task.verification_result.reasons == ["expected artifact missing: README.md"]


def test_task_store_round_trips_worker_submissions_and_review_reports(tmp_path: Path):
    registry = TaskRegistry()
    task = TaskRecord(
        title="写 README",
        original_request="写 README",
        current_goal="写 README",
        task_id="task-1",
        status="needs_user_decision",
        worker_submissions=[WorkerSubmission(round=1, summary="created nothing", report={"status": "completed", "summary": "created nothing"})],
        review_reports=[
            ReviewReport(
                round=1,
                verifier_id="task-1-verifier-1",
                status="NEEDS_REVISION",
                summary="README missing.",
                feedback_to_worker="Create README.md.",
                full_text="Review-Status: NEEDS_REVISION\n\nSummary:\nREADME missing.",
            )
        ],
        active_verifier_id="",
        needs_user_decision_summary="Review loop exhausted.",
    )
    registry.add(task)

    store = TaskStore(tmp_path)
    store.save_registry(registry)
    restored = store.load_registry().get("task-1")

    assert restored is not None
    assert len(restored.worker_submissions) == 1
    assert restored.worker_submissions[0].summary == "created nothing"
    assert len(restored.review_reports) == 1
    assert restored.review_reports[0].status == "NEEDS_REVISION"
    assert restored.review_reports[0].feedback_to_worker == "Create README.md."
    assert restored.needs_user_decision_summary == "Review loop exhausted."


def test_task_store_round_trips_mailbox_snapshot(tmp_path: Path):
    mailbox = MailboxStore()
    first = mailbox.create_message(
        task_id="task-1",
        worker_id="task-1",
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许写 README.md？",
        requires_reply=True,
        metadata={"request_id": "q1"},
    )
    mailbox.create_message(
        task_id="task-1",
        worker_id="task-1",
        from_role="worker",
        to_role="main",
        kind="progress",
        content="正在写 README",
        requires_reply=False,
    )
    mailbox.reply(first.message_id, "允许")

    store = TaskStore(tmp_path)
    store.save_mailbox(mailbox)
    restored = store.load_mailbox()

    messages = restored.list()
    assert [message.kind for message in messages] == ["approval_request", "progress"]
    assert messages[0].status == "answered"
    assert messages[0].reply == "允许"
    assert messages[0].metadata["request_id"] == "q1"
    assert messages[1].content == "正在写 README"


def test_task_store_persists_worker_spawn_context(tmp_path: Path):
    packet = WorkerContextPacket(
        task_id="task-1",
        session_id="session-1",
        agent_type="general-worker",
        context_policy="briefed",
        workspace=str(tmp_path),
        selected_skills=["brainstorming"],
        active_tasks_summary="none",
        source_item_ids=["item-1"],
        resource_claims=[ResourceClaim("gomoku.html", "file", "explicit")],
        handoff_summary="Build gomoku.",
        decisions_already_made=["Standalone page."],
        relevant_messages=[],
    )
    task = TaskRecord(
        title="Build gomoku",
        original_request="帮我写五子棋",
        current_goal="Build gomoku",
        task_spec=TaskSpec(title="Build gomoku", goal="Build gomoku"),
        task_id="task-1",
        agent_type="general-worker",
        context_policy="briefed",
        skills_to_preload=["brainstorming"],
        context_packet=packet,
        forked_instructions="MAIN SYSTEM PROMPT",
        forked_input_items=[{"role": "user", "content": "Use standalone page", "xiaoming": {"id": "msg-1"}}],
        forked_loaded_skills=[{"name": "brainstorming", "description": "Discuss requirements.", "content": "Ask first.", "path": "", "content_hash": "hash"}],
    )
    registry = TaskRegistry()
    registry.add(task)
    store = TaskStore(tmp_path)

    store.save_registry(registry)
    restored = store.load_registry().get("task-1")

    assert restored is not None
    assert restored.agent_type == "general-worker"
    assert restored.skills_to_preload == ["brainstorming"]
    assert restored.context_packet is not None
    assert restored.context_packet.content_hash == packet.content_hash
    assert restored.forked_instructions == "MAIN SYSTEM PROMPT"
    assert restored.forked_input_items == [{"role": "user", "content": "Use standalone page", "xiaoming": {"id": "msg-1"}}]
    assert restored.forked_loaded_skills[0]["name"] == "brainstorming"


def test_task_store_marks_in_process_tasks_failed_on_recovery(tmp_path: Path):
    registry = TaskRegistry()
    running = registry.add(TaskRecord(title="运行中", original_request="运行中", current_goal="运行中", status="running"))
    needs_user = registry.add(TaskRecord(title="待确认", original_request="待确认", current_goal="待确认", status="needs_user"))
    waiting = registry.add(TaskRecord(title="排队中", original_request="排队中", current_goal="排队中", status="waiting", conflicts_with={running.task_id}))

    store = TaskStore(tmp_path)
    store.save_registry(registry)
    restored = store.load_registry()

    tasks = {task.title: task for task in restored.list()}
    assert tasks["运行中"].status == "failed"
    assert "coordinator restarted" in tasks["运行中"].last_progress
    assert tasks["待确认"].status == "failed"
    assert tasks["排队中"].status == "waiting"
    assert running.task_id not in tasks["排队中"].conflicts_with


def test_coordinator_starts_recovered_waiting_task_without_conflicts(tmp_path: Path):
    notices = []
    workers = []
    registry = TaskRegistry()
    registry.add(TaskRecord(title="排队中", original_request="排队中", current_goal="排队中", status="waiting"))
    TaskStore(tmp_path).save_registry(registry)

    from xiaoming.async_runtime.coordinator import AsyncCoordinator, CoordinatorConfig

    def factory(config, on_event):
        worker = FakeWorker(config, on_event)
        workers.append(worker)
        return worker

    coordinator = AsyncCoordinator(CoordinatorConfig(tmp_path), scheduler=FakeScheduler(), responder=FakeResponder(), worker_factory=factory, on_notice=notices.append)
    coordinator.start()
    try:
        _eventually(lambda: len(workers) == 1)
    finally:
        coordinator.stop()

    assert workers[0].config.task == "排队中"


class FakeWorker:
    def __init__(self, config, on_event):
        self.config = config
        self.on_event = on_event
        self.pid = 123

    def start(self):
        self.on_event(WorkerEvent(self.config.task_id, "started", "worker started"))

    def send(self, kind, **payload):
        pass

    def terminate(self, timeout_seconds=5):
        pass


class FakeResponder:
    def user_reply(self, user_message, registry, mode, question=None):
        return ""

    def worker_notice(self, event, task, registry):
        return ""

    def command_reply(self, command, payload, registry):
        return ""


class FakeScheduler:
    def schedule(self, user_message, registry):
        raise AssertionError("recovered waiting task should not call scheduler")


def _eventually(predicate, timeout=2):
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()
