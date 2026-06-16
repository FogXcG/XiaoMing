from xiaoming.async_runtime.mailbox import MailboxStore
from xiaoming.async_runtime.tasks import TaskRecord, TaskRegistry


def test_mailbox_snapshot_round_trips_order_status_and_metadata():
    mailbox = MailboxStore()
    first = mailbox.create_message(
        task_id="task-1",
        worker_id="worker-1",
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许写 README.md？",
        requires_reply=True,
        metadata={"request_id": "q1", "options": ["yes", "no"]},
    )
    second = mailbox.create_message(
        task_id="task-1",
        worker_id="worker-1",
        from_role="worker",
        to_role="main",
        kind="progress",
        content="正在写 README",
        requires_reply=False,
    )

    mailbox.reply(first.message_id, "允许")
    mailbox.cancel_task_messages("task-1")

    restored = MailboxStore.from_snapshot(mailbox.snapshot())
    messages = restored.list()

    assert [message.message_id for message in messages] == [first.message_id, second.message_id]
    assert messages[0].status == "answered"
    assert messages[0].reply == "允许"
    assert messages[0].metadata == {"request_id": "q1", "options": ["yes", "no"]}
    assert messages[1].status == "cancelled"
    assert restored.pending_for_main() == []


def test_mailbox_marks_pending_message_as_presented_without_changing_reply_status():
    mailbox = MailboxStore()
    message = mailbox.create_message(
        task_id="task-1",
        from_role="worker",
        to_role="main",
        kind="clarification_request",
        content="使用中文吗？",
        requires_reply=True,
    )

    mailbox.mark_presented(message.message_id, at="2026-06-12T00:00:00+00:00")
    mailbox.mark_presented(message.message_id, at="2026-06-12T00:01:00+00:00")

    restored = MailboxStore.from_snapshot(mailbox.snapshot())
    restored_message = restored.list()[0]

    assert restored_message.status == "pending"
    assert restored_message.presented_count == 2
    assert restored_message.last_presented_at == "2026-06-12T00:01:00+00:00"


def test_mailbox_pending_reply_digest_includes_only_pending_reply_messages():
    registry = TaskRegistry()
    registry.add(TaskRecord(title="安装 skill", original_request="", current_goal="", task_id="task-1"))
    mailbox = MailboxStore()
    pending = mailbox.create_message(
        task_id="task-1",
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="是否允许 git clone？",
        requires_reply=True,
        metadata={"request_id": "q1", "purpose": "permission", "options": ["approve", "deny"]},
    )
    mailbox.create_message(
        task_id="task-1",
        from_role="worker",
        to_role="main",
        kind="progress",
        content="正在下载",
        requires_reply=False,
    )
    answered = mailbox.create_message(
        task_id="task-1",
        from_role="worker",
        to_role="main",
        kind="clarification_request",
        content="使用哪个分支？",
        requires_reply=True,
        metadata={"request_id": "q2"},
    )
    mailbox.reply(answered.message_id, "main")

    digest = mailbox.pending_reply_digest(registry)

    assert pending in mailbox.pending_reply_messages()
    assert digest is not None
    assert "question_id: q1" in digest
    assert "task: 安装 skill" in digest
    assert "purpose: permission" in digest
    assert "options: approve | deny" in digest
    assert "正在下载" not in digest
    assert "使用哪个分支" not in digest


def test_mailbox_latest_task_updates_keeps_only_latest_progress_or_result_per_task():
    registry = TaskRegistry()
    registry.add(TaskRecord(title="任务 A", original_request="", current_goal="", task_id="task-a"))
    registry.add(TaskRecord(title="任务 B", original_request="", current_goal="", task_id="task-b"))
    mailbox = MailboxStore()
    mailbox.create_message(task_id="task-a", from_role="worker", to_role="main", kind="progress", content="A 10%", requires_reply=False)
    latest_a = mailbox.create_message(task_id="task-a", from_role="worker", to_role="main", kind="result", content="A done", requires_reply=False)
    latest_b = mailbox.create_message(task_id="task-b", from_role="worker", to_role="main", kind="progress", content="B 20%", requires_reply=False)
    mailbox.create_message(task_id="task-b", from_role="worker", to_role="main", kind="approval_request", content="B approve?", requires_reply=True)

    updates = mailbox.latest_task_updates(registry)

    assert [(update.task_id, update.task_title, update.message_id, update.content) for update in updates] == [
        ("task-a", "任务 A", latest_a.message_id, "A done"),
        ("task-b", "任务 B", latest_b.message_id, "B 20%"),
    ]


def test_mailbox_notice_candidates_skip_presented_answered_and_cancelled_messages():
    registry = TaskRegistry()
    registry.add(TaskRecord(title="任务 A", original_request="", current_goal="", task_id="task-a"))
    mailbox = MailboxStore()
    pending = mailbox.create_message(
        task_id="task-a",
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许写文件？",
        requires_reply=True,
        metadata={"request_id": "q1"},
    )
    already_presented = mailbox.create_message(
        task_id="task-a",
        from_role="worker",
        to_role="main",
        kind="clarification_request",
        content="使用中文吗？",
        requires_reply=True,
        metadata={"request_id": "q2"},
    )
    mailbox.mark_presented(already_presented.message_id)
    answered = mailbox.create_message(
        task_id="task-a",
        from_role="worker",
        to_role="main",
        kind="decision_request",
        content="选 A 还是 B？",
        requires_reply=True,
        metadata={"request_id": "q3"},
    )
    mailbox.reply(answered.message_id, "A")
    cancelled = mailbox.create_message(
        task_id="task-a",
        from_role="worker",
        to_role="main",
        kind="approval_request",
        content="允许删除？",
        requires_reply=True,
        metadata={"request_id": "q4"},
    )
    cancelled.cancel()

    candidates = mailbox.notice_candidates(registry)

    assert [(candidate.message_id, candidate.text) for candidate in candidates] == [
        (pending.message_id, "任务 A 需要你确认：允许写文件？")
    ]
