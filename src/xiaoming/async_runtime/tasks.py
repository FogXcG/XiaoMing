from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import ast
import json
from typing import Any, Literal
from uuid import uuid4

from xiaoming.async_runtime.context_packets import ResourceClaim, WorkerContextPacket


TaskStatus = Literal[
    "planning",
    "waiting",
    "running",
    "needs_user",
    "reported_completed",
    "verifying",
    "needs_revision",
    "needs_user_decision",
    "accepted",
    "rejected",
    "failed",
    "blocked",
    "cancelled",
]
ExecutionMode = Literal["background", "foreground"]
ReviewStatus = Literal["ACCEPTED", "NEEDS_REVISION", "NEEDS_USER_DECISION"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TaskSpec:
    title: str
    goal: str
    success_criteria: list[str] = field(default_factory=list)
    expected_artifacts: list[str] = field(default_factory=list)
    allowed_write_paths: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    agent_type: str = ""
    execution_mode: ExecutionMode = "background"
    context_policy: str = ""
    skills_to_preload: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    resource_claims: list[ResourceClaim] = field(default_factory=list)
    verification_required: bool = True
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        return cls(
            title=_required_string(data, "title"),
            goal=_required_string(data, "goal"),
            success_criteria=_string_list(data.get("success_criteria")),
            expected_artifacts=_string_list(data.get("expected_artifacts")),
            allowed_write_paths=_string_list(data.get("allowed_write_paths")),
            verification_commands=_string_list(data.get("verification_commands")),
            agent_type=str(data.get("agent_type") or ""),
            execution_mode=_execution_mode(data.get("execution_mode")),
            context_policy=str(data.get("context_policy") or ""),
            skills_to_preload=_string_list(data.get("skills_to_preload")),
            constraints=_string_list(data.get("constraints")),
            resource_claims=_resource_claims(data.get("resource_claims")),
            verification_required=bool(data.get("verification_required", True)),
            notes=str(data.get("notes") or ""),
        )

    @classmethod
    def from_request(cls, request: str, title: str | None = None) -> "TaskSpec":
        stripped = request.strip()
        return cls(title=(title or stripped[:80] or "后台任务").strip(), goal=stripped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "goal": self.goal,
            "success_criteria": list(self.success_criteria),
            "expected_artifacts": list(self.expected_artifacts),
            "allowed_write_paths": list(self.allowed_write_paths),
            "verification_commands": list(self.verification_commands),
            "agent_type": self.agent_type,
            "execution_mode": self.execution_mode,
            "context_policy": self.context_policy,
            "skills_to_preload": list(self.skills_to_preload),
            "constraints": list(self.constraints),
            "resource_claims": [claim.to_dict() for claim in self.resource_claims],
            "verification_required": self.verification_required,
            "notes": self.notes,
        }


@dataclass
class TaskVerificationRecord:
    command: str
    status: str
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskVerificationRecord":
        return cls(command=str(data.get("command") or ""), status=str(data.get("status") or ""), reason=str(data.get("reason") or ""))

    def to_dict(self) -> dict[str, str]:
        return {"command": self.command, "status": self.status, "reason": self.reason}


@dataclass
class TaskResultReport:
    status: Literal["completed", "failed", "blocked"]
    summary: str
    changed_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    verification: list[TaskVerificationRecord] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResultReport":
        status = data.get("status")
        if status not in {"completed", "failed", "blocked"}:
            raise ValueError(f"invalid report status: {status}")
        verification = [TaskVerificationRecord.from_dict(item) for item in _dict_list(data.get("verification"))]
        return cls(
            status=status,
            summary=str(data.get("summary") or ""),
            changed_files=_string_list(data.get("changed_files")),
            created_files=_string_list(data.get("created_files")),
            artifacts=_string_list(data.get("artifacts")),
            verification=verification,
            blockers=_string_list(data.get("blockers")),
            evidence=_string_list(data.get("evidence")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "changed_files": list(self.changed_files),
            "created_files": list(self.created_files),
            "artifacts": list(self.artifacts),
            "verification": [item.to_dict() for item in self.verification],
            "blockers": list(self.blockers),
            "evidence": list(self.evidence),
        }


@dataclass
class VerificationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"accepted": self.accepted, "reasons": list(self.reasons)}


@dataclass
class WorkerSubmission:
    round: int
    summary: str
    report: dict[str, Any]
    created_at: str = field(default_factory=_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerSubmission":
        report = data.get("report")
        return cls(
            round=int(data.get("round") or 0),
            summary=str(data.get("summary") or ""),
            report=dict(report) if isinstance(report, dict) else {},
            created_at=str(data.get("created_at") or _now()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "summary": self.summary,
            "report": dict(self.report),
            "created_at": self.created_at,
        }


@dataclass
class ReviewReport:
    round: int
    verifier_id: str
    status: ReviewStatus
    summary: str = ""
    evidence: str = ""
    issues: str = ""
    feedback_to_worker: str = ""
    summary_for_main: str = ""
    full_text: str = ""
    created_at: str = field(default_factory=_now)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewReport":
        status = str(data.get("status") or "")
        if status not in {"ACCEPTED", "NEEDS_REVISION", "NEEDS_USER_DECISION"}:
            raise ValueError(f"invalid review status: {status}")
        return cls(
            round=int(data.get("round") or 0),
            verifier_id=str(data.get("verifier_id") or ""),
            status=status,  # type: ignore[arg-type]
            summary=str(data.get("summary") or ""),
            evidence=str(data.get("evidence") or ""),
            issues=str(data.get("issues") or ""),
            feedback_to_worker=str(data.get("feedback_to_worker") or ""),
            summary_for_main=str(data.get("summary_for_main") or ""),
            full_text=str(data.get("full_text") or ""),
            created_at=str(data.get("created_at") or _now()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "verifier_id": self.verifier_id,
            "status": self.status,
            "summary": self.summary,
            "evidence": self.evidence,
            "issues": self.issues,
            "feedback_to_worker": self.feedback_to_worker,
            "summary_for_main": self.summary_for_main,
            "full_text": self.full_text,
            "created_at": self.created_at,
        }


@dataclass
class TaskRecord:
    title: str
    original_request: str
    current_goal: str
    task_spec: TaskSpec | None = None
    status: TaskStatus = "planning"
    task_id: str = field(default_factory=lambda: str(uuid4()))
    affected_files: set[str] = field(default_factory=set)
    affected_modules: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    conflicts_with: set[str] = field(default_factory=set)
    worker_pid: int | None = None
    last_progress: str = ""
    authorization_note: str = ""
    result_report: TaskResultReport | None = None
    verification_result: VerificationResult | None = None
    agent_type: str = "worker"
    context_policy: str = "forked"
    skills_to_preload: list[str] = field(default_factory=list)
    context_packet: WorkerContextPacket | None = None
    forked_instructions: str = ""
    forked_input_items: list[dict[str, Any]] = field(default_factory=list)
    forked_loaded_skills: list[dict[str, str]] = field(default_factory=list)
    task_decision_log: list[str] = field(default_factory=list)
    worker_question_log: list[str] = field(default_factory=list)
    authorization_log: list[str] = field(default_factory=list)
    revision_attempts: int = 0
    parent_task_id: str | None = None
    verifier_task_ids: list[str] = field(default_factory=list)
    worker_submissions: list[WorkerSubmission] = field(default_factory=list)
    review_reports: list[ReviewReport] = field(default_factory=list)
    active_verifier_id: str = ""
    needs_user_decision_summary: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def transition(self, status: TaskStatus, progress: str | None = None) -> None:
        self.status = status
        if progress is not None:
            self.last_progress = progress
        self.updated_at = _now()

    def snapshot(self) -> dict:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "original_request": self.original_request,
            "current_goal": self.current_goal,
            "status": self.status,
            "affected_files": sorted(self.affected_files),
            "affected_modules": sorted(self.affected_modules),
            "domains": sorted(self.domains),
            "conflicts_with": sorted(self.conflicts_with),
            "last_progress": self.last_progress,
            "authorization_note": self.authorization_note,
            "agent_type": self.agent_type,
            "context_policy": self.context_policy,
            "skills_to_preload": list(self.skills_to_preload),
            "context_packet": self.context_packet.to_dict() if self.context_packet is not None else None,
            "forked_instructions": self.forked_instructions,
            "forked_input_items": list(self.forked_input_items),
            "forked_loaded_skills": list(self.forked_loaded_skills),
            "task_decision_log": list(self.task_decision_log),
            "worker_question_log": list(self.worker_question_log),
            "authorization_log": list(self.authorization_log),
            "revision_attempts": self.revision_attempts,
            "parent_task_id": self.parent_task_id,
            "verifier_task_ids": list(self.verifier_task_ids),
            "worker_submissions": [submission.to_dict() for submission in self.worker_submissions],
            "review_reports": [review.to_dict() for review in self.review_reports],
            "active_verifier_id": self.active_verifier_id,
            "needs_user_decision_summary": self.needs_user_decision_summary,
            "task_spec": self.task_spec.to_dict() if self.task_spec is not None else None,
            "result_report": self.result_report.to_dict() if self.result_report is not None else None,
            "verification_result": self.verification_result.to_dict() if self.verification_result is not None else None,
        }


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self.current_task_id: str | None = None

    def add(self, task: TaskRecord) -> TaskRecord:
        self._tasks[task.task_id] = task
        self.current_task_id = task.task_id
        return task

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def current(self) -> TaskRecord | None:
        if not self.current_task_id:
            return None
        task = self._tasks.get(self.current_task_id)
        if task is None or task.status not in {"planning", "waiting", "running", "needs_user", "reported_completed", "verifying", "needs_revision", "needs_user_decision"}:
            return None
        return task

    def clear_current_if(self, task_id: str) -> None:
        if self.current_task_id != task_id:
            return
        active = self.active()
        self.current_task_id = active[-1].task_id if active else None

    def list(self) -> list[TaskRecord]:
        return list(self._tasks.values())

    def active(self) -> list[TaskRecord]:
        return [task for task in self._tasks.values() if task.status in {"planning", "waiting", "running", "needs_user", "reported_completed", "verifying", "needs_revision", "needs_user_decision"}]

    def waiting(self) -> list[TaskRecord]:
        return [task for task in self._tasks.values() if task.status == "waiting"]

    def find_duplicate(self, title: str, goal: str) -> TaskRecord | None:
        normalized_title = _normalize(title)
        normalized_goal = _normalize(goal)
        for task in self.active():
            if _normalize(task.title) == normalized_title or _normalize(task.current_goal) == normalized_goal:
                return task
        return None

    def conflicts_for(self, files: set[str], modules: set[str], domains: set[str]) -> list[TaskRecord]:
        conflicts: list[TaskRecord] = []
        for task in self.active():
            if task.status == "waiting":
                continue
            if files and task.affected_files.intersection(files):
                conflicts.append(task)
                continue
            if modules and task.affected_modules.intersection(modules):
                conflicts.append(task)
                continue
            if domains and task.domains.intersection(domains):
                conflicts.append(task)
        return conflicts

    def snapshot(self) -> list[dict]:
        return [task.snapshot() for task in self.list()]

    def active_snapshot(self) -> list[dict]:
        return [task.snapshot() for task in self.active()]


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required task spec field: {key}")
    return value.strip()


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(stripped)
                except (SyntaxError, ValueError):
                    parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            separators = "\n" if "\n" in stripped else ","
            return [item.strip() for item in stripped.split(separators) if item.strip()]
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _resource_claims(value: Any) -> list[ResourceClaim]:
    return [ResourceClaim.from_dict(item) for item in _dict_list(value)]


def _execution_mode(value: Any) -> ExecutionMode:
    return "foreground" if str(value or "").strip() == "foreground" else "background"
