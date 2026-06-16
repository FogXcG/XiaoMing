from __future__ import annotations

import json
from pathlib import Path

from xiaoming.async_runtime.context_packets import WorkerContextPacket
from xiaoming.async_runtime.mailbox import MailboxStore
from xiaoming.async_runtime.tasks import ReviewReport, TaskRecord, TaskRegistry, TaskResultReport, TaskSpec, VerificationResult, WorkerSubmission


class TaskStore:
    def __init__(self, workspace: Path):
        self.path = workspace / ".xiaoming" / "tasks" / "index.json"
        self.mailbox_path = workspace / ".xiaoming" / "tasks" / "mailbox.json"

    def load_registry(self) -> TaskRegistry:
        registry = TaskRegistry()
        if not self.path.exists():
            return registry
        try:
            data = json.loads(self.path.read_text())
        except Exception:
            return registry
        if not isinstance(data, list):
            return registry
        for item in data:
            if not isinstance(item, dict):
                continue
            task = _record_from_dict(item)
            if task is not None:
                registry.add(task)
        registry.current_task_id = None
        _recover_loaded_registry(registry)
        return registry

    def save_registry(self, registry: TaskRegistry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(registry.snapshot(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def load_mailbox(self) -> MailboxStore:
        if not self.mailbox_path.exists():
            return MailboxStore()
        try:
            data = json.loads(self.mailbox_path.read_text())
        except Exception:
            return MailboxStore()
        return MailboxStore.from_snapshot(data)

    def save_mailbox(self, mailbox: MailboxStore) -> None:
        self.mailbox_path.parent.mkdir(parents=True, exist_ok=True)
        self.mailbox_path.write_text(json.dumps(mailbox.snapshot(), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _record_from_dict(data: dict) -> TaskRecord | None:
    try:
        task_spec = TaskSpec.from_dict(data["task_spec"]) if isinstance(data.get("task_spec"), dict) else None
        task = TaskRecord(
            title=str(data.get("title") or "后台任务"),
            original_request=str(data.get("original_request") or data.get("current_goal") or ""),
            current_goal=str(data.get("current_goal") or ""),
            task_spec=task_spec,
            status=data.get("status") or "failed",
            task_id=str(data.get("task_id") or ""),
            affected_files=set(data.get("affected_files") or []),
            affected_modules=set(data.get("affected_modules") or []),
            domains=set(data.get("domains") or []),
            conflicts_with=set(data.get("conflicts_with") or []),
            last_progress=str(data.get("last_progress") or ""),
            authorization_note=str(data.get("authorization_note") or ""),
            agent_type=str(data.get("agent_type") or "worker"),
            context_policy=str(data.get("context_policy") or "forked"),
            skills_to_preload=[str(item) for item in data.get("skills_to_preload") or []],
            forked_instructions=str(data.get("forked_instructions") or ""),
            forked_input_items=[dict(item) for item in data.get("forked_input_items") or [] if isinstance(item, dict)],
            forked_loaded_skills=[dict(item) for item in data.get("forked_loaded_skills") or [] if isinstance(item, dict)],
            task_decision_log=[str(item) for item in data.get("task_decision_log") or []],
            worker_question_log=[str(item) for item in data.get("worker_question_log") or []],
            authorization_log=[str(item) for item in data.get("authorization_log") or []],
            revision_attempts=int(data.get("revision_attempts") or 0),
            parent_task_id=str(data.get("parent_task_id")) if data.get("parent_task_id") is not None else None,
            verifier_task_ids=[str(item) for item in data.get("verifier_task_ids") or []],
            worker_submissions=_worker_submissions_from_list(data.get("worker_submissions")),
            review_reports=_review_reports_from_list(data.get("review_reports")),
            active_verifier_id=str(data.get("active_verifier_id") or ""),
            needs_user_decision_summary=str(data.get("needs_user_decision_summary") or ""),
        )
        if isinstance(data.get("context_packet"), dict):
            task.context_packet = WorkerContextPacket.from_dict(data["context_packet"])
        if isinstance(data.get("result_report"), dict):
            task.result_report = TaskResultReport.from_dict(data["result_report"])
        if isinstance(data.get("verification_result"), dict):
            raw_verification = data["verification_result"]
            task.verification_result = VerificationResult(accepted=bool(raw_verification.get("accepted")), reasons=[str(item) for item in raw_verification.get("reasons") or []])
        return task if task.task_id else None
    except Exception:
        return None


def _worker_submissions_from_list(raw: object) -> list[WorkerSubmission]:
    submissions: list[WorkerSubmission] = []
    if not isinstance(raw, list):
        return submissions
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            submissions.append(WorkerSubmission.from_dict(item))
        except Exception:
            continue
    return submissions


def _review_reports_from_list(raw: object) -> list[ReviewReport]:
    reports: list[ReviewReport] = []
    if not isinstance(raw, list):
        return reports
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            reports.append(ReviewReport.from_dict(item))
        except Exception:
            continue
    return reports


def _recover_loaded_registry(registry: TaskRegistry) -> None:
    stale_active_ids: set[str] = set()
    for task in registry.list():
        if task.status in {"planning", "running", "needs_user", "reported_completed", "verifying", "needs_revision"}:
            stale_active_ids.add(task.task_id)
            task.transition("failed", "coordinator restarted before this task finished; reschedule the task to continue.")
    if not stale_active_ids:
        return
    for task in registry.list():
        if task.status == "waiting":
            task.conflicts_with.difference_update(stale_active_ids)
