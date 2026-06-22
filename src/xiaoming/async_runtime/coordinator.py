from __future__ import annotations

from dataclasses import dataclass
import copy
import json
from pathlib import Path
import queue
import re
import threading
import time
from typing import Callable

from xiaoming.async_runtime.agents import builtin_agent_registry
from xiaoming.async_runtime.context_packets import ContextPacketBuilder
from xiaoming.async_runtime.events import CoordinatorNotice, Question, UserMessage, WorkerEvent
from xiaoming.async_runtime.external_sessions import CodexRemoteControlSession, CodexWorkerProcess, ExternalSessionRecord, ExternalSessionStore
from xiaoming.async_runtime.question_decider import WorkerQuestionDecider, WorkerQuestionDecision
from xiaoming.async_runtime.responder import CoordinatorResponder, ResponderError
from xiaoming.async_runtime.scheduler import Scheduler, SchedulerDecision, SchedulerError
from xiaoming.async_runtime.task_store import TaskStore
from xiaoming.async_runtime.tasks import ReviewReport, TaskRecord, TaskRegistry, TaskSpec, TaskResultReport, VerificationResult, WorkerSubmission
from xiaoming.async_runtime.verifier import TaskVerifier, Verifier
from xiaoming.async_runtime.worker_process import WorkerConfig, WorkerProcess
from xiaoming.session import Session
from xiaoming.tools.base import ToolResult


MAX_REVISION_ATTEMPTS = 3
TERMINAL_TASK_STATUSES = {"accepted", "rejected", "failed", "blocked", "cancelled"}
FORK_PLACEHOLDER_RESULT = "Fork started - processing in background"
NOTICE_DEDUPE_WINDOW_SECONDS = 30.0
FOLLOW_TASK_TIMEOUT_SECONDS = 30.0


@dataclass
class CoordinatorConfig:
    workspace: Path
    provider: str | None = None
    model: str | None = None
    approval_mode: str | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    model_timeout_seconds: float | None = None
    stream: bool | None = None
    quiet: bool = True
    max_revision_attempts: int = MAX_REVISION_ATTEMPTS


class AsyncCoordinator:
    def __init__(
        self,
        config: CoordinatorConfig,
        scheduler: Scheduler | None = None,
        responder: CoordinatorResponder | None = None,
        worker_factory: Callable[[WorkerConfig, Callable[[WorkerEvent], None]], WorkerProcess] | None = None,
        on_notice: Callable[[CoordinatorNotice], None] | None = None,
        verifier: Verifier | None = None,
        question_decider: WorkerQuestionDecider | None = None,
        session_provider: Callable[[], Session] | None = None,
    ):
        self.config = config
        if scheduler is None:
            raise ValueError("AsyncCoordinator requires an explicit scheduler")
        if responder is None:
            raise ValueError("AsyncCoordinator requires an explicit responder")
        self.scheduler = scheduler
        self.responder = responder
        self._uses_default_worker_factory = worker_factory is None
        self.worker_factory = worker_factory or (lambda worker_config, on_event: WorkerProcess(worker_config, on_event))
        self.on_notice = on_notice
        self.task_store = TaskStore(config.workspace)
        self.registry = self.task_store.load_registry()
        self.external_session_store = ExternalSessionStore(config.workspace)
        self.external_sessions: dict[str, ExternalSessionRecord] = {record.peer_id: record for record in self.external_session_store.load()}
        self.mailbox = self.task_store.load_mailbox()
        self._cancel_mailbox_messages_for_terminal_tasks()
        self._input_queue: queue.Queue[UserMessage] = queue.Queue()
        self._worker_events: queue.Queue[WorkerEvent] = queue.Queue()
        self._workers: dict[str, WorkerProcess] = {}
        self._peer_reply_queues: dict[str, queue.Queue[str]] = {}
        self._verifier_workers: dict[str, WorkerProcess] = {}
        self._verifier_parent_ids: dict[str, str] = {}
        self._stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._event_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._state_changed = threading.Condition(self._lock)
        self._terminal_notices: set[tuple[str, str]] = set()
        self._notice_cache: dict[tuple[str, str, str], float] = {}
        self.verifier = verifier or TaskVerifier(config.workspace)
        self._use_verifier_worker = verifier is None and bool(config.provider and config.model)
        self.question_decider = question_decider
        self.agent_registry = builtin_agent_registry()
        self.context_packet_builder = ContextPacketBuilder(config.workspace)
        self.session_provider = session_provider

    def start(self) -> None:
        self._stop.clear()
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._event_thread = threading.Thread(target=self._run_worker_events, daemon=True)
        self._scheduler_thread.start()
        self._event_thread.start()
        with self._lock:
            self._try_start_waiting_locked()
            self._persist()

    def stop(self) -> None:
        self._stop.set()
        for worker in list(self._workers.values()):
            worker.terminate(timeout_seconds=3)
        for worker in list(self._verifier_workers.values()):
            worker.terminate(timeout_seconds=3)
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=3)
        if self._event_thread is not None:
            self._event_thread.join(timeout=3)

    def submit_user_message(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        self._input_queue.put(UserMessage(stripped))
        return ""

    def reply_mailbox_message(self, message_id: str, normalized_answer: str, decision: str = "none", message_to_user: str = "", authorization_note: str = "") -> ToolResult:
        if decision not in {"approved", "denied", "none"}:
            return ToolResult("reply_mailbox_message", "error", error=f"invalid decision: {decision}")
        with self._lock:
            message = self.mailbox.get(message_id)
            if message is None or message.status != "pending" or not message.requires_reply or message.to_role != "main":
                return ToolResult("reply_mailbox_message", "error", error=f"unknown pending mailbox message: {message_id}")
            question = self._question_from_mailbox_message_locked(message.message_id)
            if question is None:
                return ToolResult("reply_mailbox_message", "error", error=f"unknown pending mailbox message: {message_id}")
            self.mailbox.reply(message.message_id, normalized_answer)
            self._mark_mailbox_question_answered_locked(message.task_id)
            self._record_worker_answer_locked(question, normalized_answer, decision)
            if authorization_note.strip():
                self._update_authorization_note_locked(question.task_id, authorization_note)
            self._persist()
        self._send_to_worker(question.task_id, "answer_question", request_id=question.request_id, answer=normalized_answer, decision=decision)
        return ToolResult("reply_mailbox_message", "success", output=message_to_user or "Answer sent to worker.")

    def schedule_background_task(self, task: str | TaskSpec) -> ToolResult:
        task_spec = TaskSpec.from_request(task) if isinstance(task, str) else task
        if not task_spec.goal.strip():
            return ToolResult("schedule_background_task", "error", error="request is empty")
        with self._lock:
            try:
                decision = self.scheduler.schedule(task_spec.goal, self.registry)
            except SchedulerError as exc:
                return ToolResult("schedule_background_task", "error", error=str(exc))
            self._apply_decision_locked(UserMessage(task_spec.goal), decision, emit_visible=False, task_spec=task_spec)
            self._persist()
            return ToolResult("schedule_background_task", "success", output=_decision_summary(decision))

    def tasks_text(self) -> str:
        with self._lock:
            tasks = self.registry.list()
            if not tasks:
                return "当前没有后台任务。"
            return "\n".join(self._task_status_line_locked(task) for task in tasks)

    def current_tasks_text(self) -> str:
        with self._lock:
            tasks = self.registry.active()
            if not tasks:
                return "当前没有后台任务。"
            return "\n".join(self._task_status_line_locked(task) for task in tasks)

    def talk_to_peer(self, peer_id: str, message: str) -> ToolResult:
        requested_peer_id = peer_id.strip()
        text = message.strip()
        if not requested_peer_id:
            return ToolResult("talk_to_peer", "error", error="peer_id is required")
        if not text:
            return ToolResult("talk_to_peer", "error", error="message is required")
        external: ExternalSessionRecord | None = None
        reply_queue: queue.Queue[str] | None = None
        with self._lock:
            normalized_peer_id, resolve_error = self._resolve_peer_id_locked(requested_peer_id)
            if resolve_error:
                return ToolResult("talk_to_peer", "error", error=resolve_error)
            worker = self._workers.get(normalized_peer_id)
            if worker is None:
                external = self.external_sessions.get(normalized_peer_id)
                if external is None:
                    return ToolResult("talk_to_peer", "error", error=f"unknown peer: {normalized_peer_id}")
                task = self.registry.get(normalized_peer_id)
                if task is not None and task.status == "needs_user_decision":
                    return self._send_external_user_decision_locked(task, external, text)
                external.status = "active"
                external.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._persist_external_sessions_locked()
                self._persist()
            else:
                task = self.registry.get(normalized_peer_id)
                if task is not None and task.status == "needs_user_decision":
                    feedback = _user_decision_feedback(task, text)
                    task.needs_user_decision_summary = ""
                    task.transition("running", "user decision sent to worker")
                    worker.send("review_feedback", feedback=feedback, reasons=[feedback], attempt=task.revision_attempts, max_attempts=self.config.max_revision_attempts)
                    self._persist()
                    return ToolResult("talk_to_peer", "success", output=f"已转交给 {task.title}，后台会继续处理。")
                reply_queue = queue.Queue(maxsize=1)
                self._peer_reply_queues[normalized_peer_id] = reply_queue
                worker.send("talk", message=text)
                if task is not None:
                    task.transition(task.status, "talk sent to worker")
                    self._persist()
        if external is not None:
            return self._talk_to_external_peer(normalized_peer_id, external, text)
        assert reply_queue is not None
        return self._wait_for_internal_peer_reply(normalized_peer_id, reply_queue)

    def _send_external_user_decision_locked(self, task: TaskRecord, external: ExternalSessionRecord, message: str) -> ToolResult:
        feedback = _user_decision_feedback(task, message)
        task.needs_user_decision_summary = ""
        task.transition("running", "user decision sent to external peer")
        external.status = "active"
        external.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._persist_external_sessions_locked()
        self._persist()

        def _run() -> None:
            data = {"external_provider": external.provider, "external_session_id": external.session_id}
            if external.provider != "codex":
                self._worker_events.put(WorkerEvent(task.task_id, "failed", f"unsupported external peer provider: {external.provider}", data))
                return
            session = CodexRemoteControlSession(self.config.workspace, session_id=external.session_id, timeout_seconds=float(self.config.model_timeout_seconds or 900))
            try:
                answer, session_id = session.send(
                    feedback,
                    on_progress=lambda progress: self._worker_events.put(
                        WorkerEvent(task.task_id, "progress", progress, {"external_provider": "codex", "external_session_id": session.session_id})
                    ),
                )
            except Exception as exc:
                self._worker_events.put(WorkerEvent(task.task_id, "failed", f"external Codex worker failed: {exc}", data))
                return
            self._worker_events.put(WorkerEvent(task.task_id, "completed", answer, {"external_provider": "codex", "external_session_id": session_id}))

        threading.Thread(target=_run, daemon=True).start()
        return ToolResult("talk_to_peer", "success", output=f"已转交给 {task.title}，后台会继续处理。")

    def _resolve_peer_id_locked(self, peer_id: str) -> tuple[str, str]:
        if peer_id in self._workers:
            return peer_id, ""
        if peer_id in self.external_sessions:
            return peer_id, ""
        candidates = set(self._workers) | set(self.external_sessions)
        matches = sorted(task_id for task_id in candidates if task_id.startswith(peer_id))
        if len(matches) == 1:
            return matches[0], ""
        if len(matches) > 1:
            return "", f"ambiguous peer_id prefix {peer_id}: {', '.join(match[:8] for match in matches)}"
        return peer_id, ""

    def _talk_to_external_peer(self, peer_id: str, external: ExternalSessionRecord, message: str) -> ToolResult:
        if external.provider != "codex":
            return ToolResult("talk_to_peer", "error", error=f"unsupported external peer provider: {external.provider}")
        session = CodexRemoteControlSession(self.config.workspace, session_id=external.session_id, timeout_seconds=float(self.config.model_timeout_seconds or 900))
        try:
            answer, session_id = session.send(message)
        except Exception as exc:
            with self._lock:
                external.status = "failed"
                external.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                self._persist_external_sessions_locked()
            return ToolResult("talk_to_peer", "error", error=f"external Codex talk failed: {exc}")
        with self._lock:
            external.session_id = session_id or external.session_id
            external.status = "active"
            external.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self._persist_external_sessions_locked()
            task = self.registry.get(peer_id)
            if task is not None:
                task.transition(task.status, "external peer replied")
                self._persist()
        return ToolResult("talk_to_peer", "success", output=answer)

    def _wait_for_internal_peer_reply(self, task_id: str, reply_queue: queue.Queue[str], timeout_seconds: float = 120) -> ToolResult:
        try:
            reply = reply_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            with self._lock:
                self._peer_reply_queues.pop(task_id, None)
            return ToolResult("talk_to_peer", "error", error=f"peer reply timed out after {timeout_seconds:g}s")
        with self._lock:
            self._peer_reply_queues.pop(task_id, None)
        return ToolResult("talk_to_peer", "success", output=reply)

    def follow_task(self, task_id: str, timeout_seconds: float = FOLLOW_TASK_TIMEOUT_SECONDS) -> ToolResult:
        normalized_task_id = task_id.strip()
        if not normalized_task_id:
            return ToolResult("follow_background_task", "error", error="task_id is required")
        deadline = time.monotonic() + max(timeout_seconds, 0)
        with self._state_changed:
            task = self.registry.get(normalized_task_id)
            if task is None:
                return ToolResult("follow_background_task", "error", error=f"unknown task_id: {normalized_task_id}")
            initial_status = task.status
            initial_progress = task.last_progress
            if task.status in TERMINAL_TASK_STATUSES or task.status in {"needs_user", "needs_user_decision"}:
                return ToolResult("follow_background_task", "success", output=_follow_task_snapshot(task, changed=True))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    task = self.registry.get(normalized_task_id)
                    if task is None:
                        return ToolResult("follow_background_task", "error", error=f"unknown task_id: {normalized_task_id}")
                    return ToolResult("follow_background_task", "success", output=_follow_task_snapshot(task, changed=False))
                self._state_changed.wait(timeout=remaining)
                task = self.registry.get(normalized_task_id)
                if task is None:
                    return ToolResult("follow_background_task", "error", error=f"unknown task_id: {normalized_task_id}")
                if task.status != initial_status or task.last_progress != initial_progress:
                    return ToolResult("follow_background_task", "success", output=_follow_task_snapshot(task, changed=True))

    def status_text(self) -> str:
        with self._lock:
            active = self.registry.active()
            pending = self.mailbox.pending_reply_messages()
            return f"后台任务: {len(active)}\n待确认: {len(pending)}"

    def context_summary(self) -> str | None:
        with self._lock:
            tasks = self.registry.active()
            question_lines = self._pending_question_lines_locked()
            if not tasks and not question_lines:
                return None
            lines = ["Background tasks:"]
            if tasks:
                for task in tasks:
                    progress = f"; progress: {task.last_progress}" if task.last_progress else ""
                    lines.append(f"- id: {task.task_id}; title: {task.title}; status: {task.status}{progress}")
            else:
                lines.append("- none")
            if question_lines:
                lines.append("Pending worker questions:")
                lines.extend(question_lines)
            return "\n".join(lines)

    def has_active_tasks(self) -> bool:
        with self._lock:
            return bool(self.registry.active())

    def has_pending_question(self) -> bool:
        with self._lock:
            return bool(self.mailbox.pending_reply_messages())

    def pending_questions_text(self) -> str | None:
        with self._lock:
            question_lines = self._pending_question_lines_locked()
            if not question_lines:
                return None
            lines = [
                "There are pending background worker questions. Decide whether the user's message clearly answers one of them.",
                "Only call reply_mailbox_message when the user's message is a clear answer to a pending mailbox message. Translate the answer into normalized_answer. For approval questions, set decision=approved or decision=denied.",
                "If the message is a new request, status question, unrelated message, or ambiguous answer, do not call reply_mailbox_message. Answer the user's current message normally, and ask a concise clarification only if the user appears to be trying to answer the worker question.",
                "For approval questions, only approve if the user clearly grants permission for the specific requested action. Generic acknowledgements such as ok, 好的, 知道了, or 完成后告诉我 are not approval by themselves.",
                "Do not treat silence, status questions, or unrelated messages as denial.",
                "Pending questions:",
            ]
            lines.extend(question_lines)
            return "\n".join(lines)

    def cancel_current(self) -> str:
        with self._lock:
            task = self.registry.current()
            if task is None:
                return "当前没有正在运行的后台任务。"
            self._cancel_task_locked(task.task_id)
            self._try_start_waiting_locked()
            return f"已请求取消后台任务：{task.title}"

    def cancel_all(self) -> str:
        with self._lock:
            task_ids = [task.task_id for task in self.registry.active()]
            for task_id in task_ids:
                self._cancel_task_locked(task_id)
            self._try_start_waiting_locked()
            if not task_ids:
                return "当前没有正在运行的后台任务。"
            return f"已请求取消 {len(task_ids)} 个后台任务。"

    def cancel_task(self, task_id: str) -> ToolResult:
        normalized_task_id = task_id.strip()
        if not normalized_task_id:
            return ToolResult("cancel_background_task", "error", error="task_id is required")
        with self._lock:
            task = self.registry.get(normalized_task_id)
            if task is None:
                return ToolResult("cancel_background_task", "error", error=f"unknown task_id: {normalized_task_id}")
            if task.status in TERMINAL_TASK_STATUSES:
                return ToolResult("cancel_background_task", "success", output=f"任务已经结束，无需取消：{task.status}")
            self._cancel_task_locked(normalized_task_id)
            self._try_start_waiting_locked()
            return ToolResult("cancel_background_task", "success", output=f"已请求取消后台任务：{task.title}")

    def set_quiet(self, quiet: bool) -> None:
        self.config.quiet = quiet

    def _run_scheduler(self) -> None:
        while not self._stop.is_set():
            try:
                message = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                try:
                    decision = self.scheduler.schedule(message.content, self.registry)
                except SchedulerError as exc:
                    self._notice(f"Error: {exc}")
                    self._persist()
                    continue
                self._apply_decision_locked(message, decision, emit_visible=True)
                self._persist()

    def _apply_decision_locked(self, message: UserMessage, decision: SchedulerDecision, emit_visible: bool, task_spec: TaskSpec | None = None) -> None:
        if emit_visible:
            self._notice(decision.visible_message)
        if decision.action == "start_new_task":
            task = TaskRecord(
                title=task_spec.title if task_spec is not None else decision.task_title,
                original_request=message.content,
                current_goal=task_spec.goal if task_spec is not None else decision.user_intent,
                task_spec=task_spec or TaskSpec(title=decision.task_title, goal=decision.user_intent),
                status="running",
                affected_files=set(decision.affected_files),
                affected_modules=set(decision.affected_modules),
                domains=set(decision.domains),
            )
            self._prepare_worker_task_locked(task)
            self.registry.add(task)
            self._start_worker_locked(task)
            self._persist()
            return
        if decision.action == "queue_task":
            task = TaskRecord(
                title=task_spec.title if task_spec is not None else decision.task_title,
                original_request=message.content,
                current_goal=task_spec.goal if task_spec is not None else decision.user_intent,
                task_spec=task_spec or TaskSpec(title=decision.task_title, goal=decision.user_intent),
                status="waiting",
                affected_files=set(decision.affected_files),
                affected_modules=set(decision.affected_modules),
                domains=set(decision.domains),
                conflicts_with=set(decision.conflict_task_ids),
            )
            self._prepare_worker_task_locked(task)
            self.registry.add(task)
            self._persist()
            return
        if decision.action == "attach_to_task" and decision.target_task_id:
            self._send_to_worker(decision.target_task_id, "append_user_message", message=message.content)
            task = self.registry.get(decision.target_task_id)
            if task is not None:
                task.transition(task.status, "attached user message")
                self._persist()
            return
        if decision.action == "cancel_and_restart" and decision.target_task_id:
            self._cancel_task_locked(decision.target_task_id)
            replacement = TaskRecord(title=decision.task_title, original_request=message.content, current_goal=decision.user_intent, task_spec=TaskSpec(title=decision.task_title, goal=decision.user_intent), status="running")
            self._prepare_worker_task_locked(replacement)
            self.registry.add(replacement)
            self._start_worker_locked(replacement)
            self._persist()

    def _start_worker_locked(self, task: TaskRecord) -> None:
        if task.context_packet is None:
            self._prepare_worker_task_locked(task)
        worker_config = WorkerConfig(
            task_id=task.task_id,
            task=task.current_goal,
            workspace=self.config.workspace,
            provider=self.config.provider,
            model=self.config.model,
            approval_mode=self.config.approval_mode,
            permission_mode=self.config.permission_mode,
            max_turns=self.config.max_turns,
            model_timeout_seconds=self.config.model_timeout_seconds,
            stream=self.config.stream,
            task_spec=task.task_spec,
            agent_type=task.agent_type,
            context_policy=task.context_policy,
            skills_to_preload=list(task.skills_to_preload),
            context_packet=task.context_packet,
            forked_instructions=task.forked_instructions,
            forked_input_items=list(task.forked_input_items),
            forked_loaded_skills=list(task.forked_loaded_skills),
        )
        if self._uses_default_worker_factory and _should_use_codex(task):
            task.agent_type = "codex"
            worker = CodexWorkerProcess(worker_config, self._worker_events.put)
            self.external_sessions[task.task_id] = ExternalSessionRecord(
                peer_id=task.task_id,
                provider="codex",
                title=task.title,
                workspace=str(self.config.workspace),
                status="running",
            )
            self._persist_external_sessions_locked()
        else:
            worker = self.worker_factory(worker_config, self._worker_events.put)
        self._workers[task.task_id] = worker
        worker.start()
        task.worker_pid = worker.pid
        task.transition("running", "worker started")

    def _mark_mailbox_question_answered_locked(self, task_id: str) -> None:
        task = self.registry.get(task_id)
        if task is None:
            return
        has_pending = any(message.task_id == task_id for message in self.mailbox.pending_reply_messages())
        if not has_pending and task.status == "needs_user":
            task.transition("running", "user answered question")

    def _record_mailbox_message_for_question_locked(self, task: TaskRecord, question: Question):
        return self.mailbox.create_message(
            task_id=question.task_id,
            worker_id=question.task_id,
            from_role="worker",
            to_role="main",
            kind=question.kind,
            content=question.prompt,
            requires_reply=True,
            metadata={
                "request_id": question.request_id,
                "purpose": question.purpose,
                "context": question.context,
                "options": list(question.options),
                "task_title": task.title,
            },
        )

    def _pending_question_lines_locked(self) -> list[str]:
        return self.mailbox.pending_reply_lines(self.registry)

    def _question_from_mailbox_message_locked(self, message_id: str) -> Question | None:
        message = self.mailbox.get(message_id)
        if message is None:
            return None
        request_id = str(message.metadata.get("request_id") or message.message_id)
        if message.kind not in {"approval_request", "clarification_request", "decision_request"}:
            return None
        options = [str(option) for option in message.metadata.get("options") or []]
        return Question(
            task_id=message.task_id,
            kind=message.kind,
            prompt=message.content,
            request_id=request_id,
            purpose=str(message.metadata.get("purpose") or ""),
            context=str(message.metadata.get("context") or ""),
            options=options,
        )

    def _update_authorization_note_locked(self, task_id: str, note: str) -> None:
        task = self.registry.get(task_id)
        if task is None:
            return
        task.authorization_note = note.strip()
        task.authorization_log.append(note.strip())
        task.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _record_worker_question_locked(self, task: TaskRecord, question: Question) -> None:
        detail = f"Question {question.request_id} ({question.kind}; purpose={question.purpose or 'none'}): {question.prompt}"
        if question.context:
            detail += f"\nContext: {question.context}"
        if question.options:
            detail += "\nOptions: " + " | ".join(question.options)
        task.worker_question_log.append(detail)

    def _record_worker_answer_locked(self, question: Question, answer: str, decision: str) -> None:
        task = self.registry.get(question.task_id)
        if task is None:
            return
        task.worker_question_log.append(f"Answer to {question.request_id}: {answer} (decision={decision})")

    def _run_worker_events(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._worker_events.get(timeout=0.1)
            except queue.Empty:
                continue
            with self._lock:
                self._handle_worker_event_locked(event)

    def _handle_worker_event_locked(self, event: WorkerEvent) -> None:
        task = self.registry.get(event.task_id)
        if task is None:
            parent_id = self._verifier_parent_ids.get(event.task_id)
            if parent_id:
                parent = self.registry.get(parent_id)
                if parent is not None:
                    self._handle_verifier_worker_event_locked(parent, event)
            return
        self._record_external_session_from_event_locked(task, event)
        if task.status in {"accepted", "rejected", "failed", "blocked", "cancelled"} and event.kind in {"reported", "completed", "failed", "cancelled"}:
            return
        if event.kind == "peer_reply":
            reply_queue = self._peer_reply_queues.pop(event.task_id, None)
            if reply_queue is not None:
                try:
                    reply_queue.put_nowait(event.message)
                except queue.Full:
                    pass
            task.transition(task.status, "peer replied")
            self._persist()
            return
        if event.kind in {"approval_request", "clarification_request", "decision_request"}:
            question = Question(
                task_id=event.task_id,
                kind=event.kind,
                prompt=f"{task.title}：{event.message}",
                request_id=str(event.data.get("request_id") or ""),
                purpose=str(event.data.get("purpose") or ""),
                context=str(event.data.get("context") or ""),
                options=[str(option) for option in event.data.get("options") or []],
            )
            message = self._record_mailbox_message_for_question_locked(task, question)
            self._record_worker_question_locked(task, question)
            decision = self._decide_worker_question_locked(task, question)
            if decision is not None and decision.decision in {"approved", "denied"}:
                self.mailbox.reply(message.message_id, decision.answer)
                task.worker_question_log.append(
                    f"Auto-answer to {question.request_id}: {decision.answer} (decision={decision.decision}; reason={decision.reason})"
                )
                self._send_to_worker(task.task_id, "answer_question", request_id=question.request_id, answer=decision.answer, decision=decision.decision)
                task.transition(task.status, decision.reason or "worker question handled by authorization note")
                self._notice(_auto_answer_notice(task, question, decision), task.task_id)
                self._persist()
                return
            task.worker_question_log.append(f"Question {question.request_id} requires user answer")
            task.transition("needs_user", question.prompt)
            self._notice_pending_mailbox_question_locked(task.task_id)
            self._persist()
            return
        if event.kind == "reported":
            self._record_mailbox_status_locked(task, event, kind="result")
            self._handle_reported_task_locked(task, event)
            return
        if event.kind == "completed":
            self._record_mailbox_status_locked(task, event, kind="result")
            self._handle_completed_task_locked(task, event.message)
            return
        if event.kind in {"failed", "cancelled"}:
            self._mark_external_session_terminal_locked(task, "failed" if event.kind == "failed" else "cancelled")
            self._record_mailbox_status_locked(task, event, kind="result")
            self._workers.pop(task.task_id, None)
            task.transition("failed" if event.kind == "failed" else "cancelled", event.message)
            self.registry.clear_current_if(task.task_id)
            notice = self._worker_notice(task, event)
            if notice:
                self._notice(notice)
            self._try_start_waiting_locked()
            self._persist()
            return
        if event.kind == "assistant_delta":
            task.transition(task.status, event.message)
            self._persist()
            return
        if event.kind in {"started", "heartbeat", "progress", "tool_started", "tool_finished"}:
            if event.kind == "progress":
                self._record_mailbox_status_locked(task, event, kind="progress")
            task.transition(task.status, event.message)
            if not self.config.quiet and event.message:
                self._notice(f"{task.title}：{event.message}", task.task_id)
            self._persist()
            return
        task.transition(task.status, event.message)
        if not self.config.quiet and event.message:
            self._notice(f"{task.title}：{event.message}", task.task_id)
        self._persist()

    def _record_mailbox_status_locked(self, task: TaskRecord, event: WorkerEvent, kind: str) -> None:
        content = event.message.strip()
        if not content:
            return
        self.mailbox.create_message(
            task_id=task.task_id,
            worker_id=event.task_id,
            from_role="worker",
            to_role="main",
            kind=kind,  # type: ignore[arg-type]
            content=content,
            requires_reply=False,
            metadata={
                "event_kind": event.kind,
                "task_title": task.title,
            },
        )

    def _try_start_waiting_locked(self) -> None:
        for task in self.registry.waiting():
            conflicts = [self.registry.get(conflict_id) for conflict_id in task.conflicts_with]
            if any(conflict is not None and conflict.status in {"planning", "running", "needs_user"} for conflict in conflicts):
                continue
            task.conflicts_with.clear()
            task.transition("running", "等待结束，重新启动。")
            notice = self._worker_notice(task, WorkerEvent(task.task_id, "progress", "waiting task is ready to start"))
            if notice:
                self._notice(notice)
            self._prepare_worker_task_locked(task)
            self._start_worker_locked(task)
            self._persist()
            break

    def _cancel_task_locked(self, task_id: str) -> str:
        task = self.registry.get(task_id)
        if task is None:
            return ""
        self._clear_questions_for_task_locked(task_id)
        worker = self._workers.get(task_id)
        if worker is not None:
            worker.terminate(timeout_seconds=5)
        self._terminate_verifiers_for_task_locked(task_id)
        task.transition("cancelled", "用户请求取消。")
        self.registry.clear_current_if(task_id)
        self._persist()
        return ""

    def _terminate_verifiers_for_task_locked(self, task_id: str) -> None:
        verifier_ids = [verifier_id for verifier_id, parent_id in self._verifier_parent_ids.items() if parent_id == task_id]
        for verifier_id in verifier_ids:
            verifier = self._verifier_workers.pop(verifier_id, None)
            self._verifier_parent_ids.pop(verifier_id, None)
            if verifier is not None:
                verifier.terminate(timeout_seconds=3)

    def _clear_questions_for_task_locked(self, task_id: str) -> None:
        self.mailbox.cancel_task_messages(task_id)

    def _handle_reported_task_locked(self, task: TaskRecord, event: WorkerEvent) -> None:
        report_data = event.data.get("report")
        if not isinstance(report_data, dict):
            self._workers.pop(task.task_id, None)
            task.transition("failed", "worker reported invalid task result")
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 失败：worker reported invalid task result")
            self._try_start_waiting_locked()
            self._persist()
            return
        try:
            report = TaskResultReport.from_dict(report_data)
        except ValueError as exc:
            self._workers.pop(task.task_id, None)
            task.transition("failed", str(exc))
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 失败：{exc}")
            self._try_start_waiting_locked()
            self._persist()
            return
        task.result_report = report
        if report.status == "failed":
            self._workers.pop(task.task_id, None)
            task.transition("failed", report.summary or "worker reported failure")
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 失败：{task.last_progress}")
            self._try_start_waiting_locked()
            self._persist()
            return
        if report.status == "blocked":
            self._workers.pop(task.task_id, None)
            detail = report.summary or "; ".join(report.blockers) or "worker is blocked"
            task.transition("blocked", detail)
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 需要你确认：{detail}")
            self._try_start_waiting_locked()
            self._persist()
            return
        self._verify_completed_report_locked(task, report)

    def _handle_completed_task_locked(self, task: TaskRecord, final_answer: str) -> None:
        report = TaskResultReport(
            status="completed",
            summary=final_answer.strip(),
            evidence=[f"Worker final answer:\n{final_answer.strip()}"] if final_answer.strip() else [],
        )
        self._verify_completed_report_locked(task, report)

    def _verify_completed_report_locked(self, task: TaskRecord, report: TaskResultReport) -> None:
        task.result_report = report
        task.worker_submissions.append(WorkerSubmission(round=len(task.worker_submissions) + 1, summary=report.summary, report=report.to_dict()))
        task.transition("reported_completed", report.summary or "worker final answer")
        self._notice(f"{task.title} 已提交结果，正在验收。")
        spec = task.task_spec or TaskSpec(title=task.title, goal=task.current_goal)
        task.transition("verifying", "verifying worker result")
        if self._use_verifier_worker:
            self._start_verifier_worker_locked(task, report)
            self._persist()
            return
        verification = self.verifier.verify(spec, report)
        task.verification_result = verification
        self._apply_verification_result_locked(task, report, verification)
        self._try_start_waiting_locked()
        self._persist()

    def _apply_verification_result_locked(self, task: TaskRecord, report: TaskResultReport, verification) -> None:
        task.verification_result = verification
        if verification.accepted:
            task.transition("accepted", report.summary or "accepted")
            self._send_to_worker(task.task_id, "review_accepted", message="accepted")
            self._workers.pop(task.task_id, None)
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 已完成：{task.last_progress}")
        else:
            if self._start_revision_if_available_locked(task, report, verification.reasons):
                self._persist()
                return
            task.transition("rejected", "; ".join(verification.reasons))
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 未通过验收：{task.last_progress}")

    def _start_verifier_worker_locked(self, task: TaskRecord, report: TaskResultReport) -> None:
        verifier_id = f"{task.task_id}-verifier-{len(task.verifier_task_ids) + 1}"
        task.verifier_task_ids.append(verifier_id)
        verifier_spec = TaskSpec(
            title=f"Review {task.title}",
            goal=_verifier_goal(task, report),
            agent_type="verifier",
            context_policy="forked",
            verification_required=False,
        )
        context_packet = self.context_packet_builder.build(
            session=self._session_for_context_packet(),
            task_id=verifier_id,
            agent_type="verifier",
            context_policy="forked",
            task_spec=verifier_spec,
            registry=self.registry,
            selected_skills=[],
        )
        worker_config = WorkerConfig(
            task_id=verifier_id,
            task=verifier_spec.goal,
            workspace=self.config.workspace,
            provider=self.config.provider,
            model=self.config.model,
            approval_mode=self.config.approval_mode,
            permission_mode=self.config.permission_mode,
            max_turns=self.config.max_turns,
            model_timeout_seconds=self.config.model_timeout_seconds,
            stream=self.config.stream,
            task_spec=verifier_spec,
            agent_type="verifier",
            context_policy="forked",
            skills_to_preload=[],
            context_packet=context_packet,
            forked_instructions=task.forked_instructions,
            forked_input_items=list(task.forked_input_items),
            forked_loaded_skills=list(task.forked_loaded_skills),
        )
        verifier = self.worker_factory(worker_config, self._worker_events.put)
        self._verifier_workers[verifier_id] = verifier
        self._verifier_parent_ids[verifier_id] = task.task_id
        task.active_verifier_id = verifier_id
        verifier.start()

    def _handle_verifier_worker_event_locked(self, task: TaskRecord, event: WorkerEvent) -> None:
        if event.kind == "completed":
            verifier = self._verifier_workers.pop(event.task_id, None)
            self._verifier_parent_ids.pop(event.task_id, None)
            task.active_verifier_id = ""
            if verifier is not None:
                verifier.send("review_accepted", message="review result received")
            review_round = task.worker_submissions[-1].round if task.worker_submissions else len(task.review_reports) + 1
            try:
                review = _review_report_from_worker_message(event.message, round=review_round, verifier_id=event.task_id)
            except ValueError as exc:
                summary = f"verifier worker returned invalid review result: {exc}"
                task.needs_user_decision_summary = summary
                task.transition("needs_user_decision", summary)
                self._notice(f"{task.title} 验收器返回了无法解析的结果，需要用户决策：{exc}")
                self._persist()
                return
            report = task.result_report or TaskResultReport(status="completed", summary="")
            self._apply_review_report_locked(task, report, review)
            self._try_start_waiting_locked()
            self._persist()
            return
        if event.kind in {"failed", "cancelled"}:
            self._verifier_workers.pop(event.task_id, None)
            self._verifier_parent_ids.pop(event.task_id, None)
            task.active_verifier_id = ""
            summary = f"verifier worker {event.kind}: {event.message}"
            task.needs_user_decision_summary = summary
            task.transition("needs_user_decision", summary)
            self._notice(f"{task.title} 验收器{ '取消' if event.kind == 'cancelled' else '失败' }，需要用户决策：{event.message}")
            self._persist()
            return
        if event.kind in {"progress", "assistant_delta", "tool_started", "tool_finished", "heartbeat", "started"}:
            task.transition(task.status, f"verifier: {event.message}" if event.message else "verifier running")
            self._persist()
            return

    def _apply_review_report_locked(self, task: TaskRecord, report: TaskResultReport, review: ReviewReport) -> None:
        task.review_reports.append(review)
        if review.status == "ACCEPTED":
            task.verification_result = VerificationResult(accepted=True, reasons=[review.summary] if review.summary else [])
            task.transition("accepted", review.summary or report.summary or "accepted")
            self._send_to_worker(task.task_id, "review_accepted", message="accepted")
            self._workers.pop(task.task_id, None)
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 已完成：{task.last_progress}")
            return
        if review.status == "NEEDS_REVISION":
            reasons = _review_feedback_items(review)
            task.verification_result = VerificationResult(accepted=False, reasons=reasons)
            if self._start_revision_if_available_locked(task, report, reasons):
                return
            summary = _review_summary_for_main(review)
            task.needs_user_decision_summary = summary
            task.transition("needs_user_decision", summary)
            self.registry.clear_current_if(task.task_id)
            self._notice(f"{task.title} 未通过验收，需要用户决策：{summary}")
            return
        summary = _review_summary_for_main(review)
        task.verification_result = VerificationResult(accepted=False, reasons=[summary])
        task.needs_user_decision_summary = summary
        task.transition("needs_user_decision", summary)
        self.registry.clear_current_if(task.task_id)
        self._notice(f"{task.title} 需要用户决策：{summary}")

    def _start_revision_if_available_locked(self, task: TaskRecord, report: TaskResultReport, reasons: list[str]) -> bool:
        max_attempts = max(self.config.max_revision_attempts, 0)
        if max_attempts == 0:
            return False
        if task.revision_attempts >= max_attempts:
            feedback = _revision_feedback(reasons)
            summary = _review_exhausted_summary(task, report, reasons)
            task.needs_user_decision_summary = summary
            task.transition("needs_user_decision", summary)
            self._notice(f"{task.title} 自动修正 {max_attempts} 次后仍未通过验收，需要用户决策：{feedback}")
            return True
        task.revision_attempts += 1
        feedback = _revision_feedback(reasons)
        task.transition("needs_revision", feedback)
        self._notice(f"{task.title} 未通过验收，需要修正（第 {task.revision_attempts}/{max_attempts} 次）：{feedback}")
        worker = self._workers.get(task.task_id)
        if worker is not None:
            self._send_to_worker(
                task.task_id,
                "review_feedback",
                feedback=feedback,
                reasons=[str(reason) for reason in reasons],
                attempt=task.revision_attempts,
                max_attempts=max_attempts,
            )
            return True
        summary = _review_exhausted_summary(task, report, reasons)
        task.needs_user_decision_summary = summary
        task.transition("needs_user_decision", summary)
        self._notice(f"{task.title} 原 worker 不可用，需要用户决策：{_revision_feedback(reasons)}")
        return True

    def _send_to_worker(self, task_id: str, kind: str, **payload) -> None:
        worker = self._workers.get(task_id)
        if worker is not None:
            worker.send(kind, **payload)

    def _decide_worker_question_locked(self, task: TaskRecord, question: Question) -> WorkerQuestionDecision | None:
        if self.question_decider is None:
            return None
        try:
            decision = self.question_decider.decide(task, question)
        except Exception as exc:
            task.transition(task.status, f"worker question decision failed: {exc}")
            return None
        if decision.decision == "ask_user":
            task.worker_question_log.append(f"Question {question.request_id} asks user: {decision.reason}")
            return None
        return decision

    def _notice_pending_mailbox_question_locked(self, task_id: str | None = None) -> bool:
        candidates = self.mailbox.notice_candidates(self.registry)
        if task_id is not None:
            candidate = next((item for item in candidates if item.task_id == task_id), None)
        else:
            candidate = candidates[0] if candidates else None
        if candidate is None:
            return False
        if not self._notice(candidate.text, candidate.task_id):
            return False
        self.mailbox.mark_presented(candidate.message_id)
        return True

    def _worker_notice(self, task: TaskRecord, event: WorkerEvent) -> str:
        if event.kind in {"completed", "failed", "cancelled"}:
            key = (task.task_id, event.kind)
            if key in self._terminal_notices:
                return ""
            self._terminal_notices.add(key)
        if event.kind == "completed":
            detail = f"：{event.message}" if event.message else ""
            return f"{task.title} 已完成{detail}"
        if event.kind == "failed":
            detail = event.message or "后台任务失败。"
            return f"{task.title} 失败：{detail}"
        if event.kind == "cancelled":
            detail = f"：{event.message}" if event.message else ""
            return f"{task.title} 已取消{detail}"
        if event.message:
            return f"{task.title}：{event.message}"
        return ""

    def _task_status_line_locked(self, task: TaskRecord) -> str:
        pending = []
        for message in self.mailbox.pending_reply_messages():
            if message.task_id != task.task_id:
                continue
            request_id = str(message.metadata.get("request_id") or message.message_id)
            pending.append(f"{request_id}: {message.content}")
        return _task_status_line(task, pending_questions=pending)

    def _command_reply_or_error(self, command: str, payload: dict) -> str:
        try:
            return self.responder.command_reply(command, payload, self.registry)
        except ResponderError as exc:
            return f"Error: {exc}"

    def _notice(self, message: str, task_id: str | None = None) -> bool:
        if self.on_notice is None or not message:
            return False
        if task_id is not None and self._is_duplicate_notice_locked(task_id, message):
            return False
        self.on_notice(CoordinatorNotice(message=message, task_id=task_id))
        return True

    def _persist(self) -> None:
        self.task_store.save_registry(self.registry)
        self.task_store.save_mailbox(self.mailbox)
        try:
            self._state_changed.notify_all()
        except RuntimeError:
            pass

    def _persist_external_sessions_locked(self) -> None:
        self.external_session_store.save(list(self.external_sessions.values()))

    def _record_external_session_from_event_locked(self, task: TaskRecord, event: WorkerEvent) -> None:
        if event.data.get("external_provider") != "codex":
            return
        record = self.external_sessions.get(task.task_id)
        if record is None:
            record = ExternalSessionRecord(peer_id=task.task_id, provider="codex", title=task.title, workspace=str(self.config.workspace))
            self.external_sessions[task.task_id] = record
        session_id = str(event.data.get("external_session_id") or "")
        if session_id:
            record.session_id = session_id
        record.status = "active"
        record.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._persist_external_sessions_locked()

    def _mark_external_session_terminal_locked(self, task: TaskRecord, status: str) -> None:
        record = self.external_sessions.get(task.task_id)
        if record is None:
            return
        record.status = status
        record.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._persist_external_sessions_locked()

    def _mark_pending_mailbox_presented_for_task_locked(self, task_id: str) -> None:
        for message in self.mailbox.pending_reply_messages():
            if message.task_id == task_id:
                message.mark_presented()

    def _cancel_mailbox_messages_for_terminal_tasks(self) -> None:
        for task in self.registry.list():
            if task.status in TERMINAL_TASK_STATUSES:
                for message in self.mailbox.pending_reply_messages():
                    if message.task_id == task.task_id:
                        message.cancel()

    def _is_duplicate_notice_locked(self, task_id: str, message: str) -> bool:
        task = self.registry.get(task_id)
        status = task.status if task is not None else ""
        key = (task_id, status, message)
        now = time.monotonic()
        stale = [cache_key for cache_key, seen_at in self._notice_cache.items() if now - seen_at > NOTICE_DEDUPE_WINDOW_SECONDS]
        for cache_key in stale:
            self._notice_cache.pop(cache_key, None)
        last_seen = self._notice_cache.get(key)
        if last_seen is not None and now - last_seen <= NOTICE_DEDUPE_WINDOW_SECONDS:
            return True
        self._notice_cache[key] = now
        return False

    def _prepare_worker_task_locked(self, task: TaskRecord) -> None:
        agent = self.agent_registry.get("worker")
        spec = task.task_spec
        task.agent_type = agent.name
        task.context_policy = agent.default_context_policy
        task.skills_to_preload = []
        if spec is None:
            spec = TaskSpec(title=task.title, goal=task.current_goal)
            task.task_spec = spec
        parent_session = self._session_for_context_packet()
        task.forked_instructions = _forked_instructions(parent_session, task.context_policy)
        task.forked_input_items = _forked_input_items(parent_session, task.context_policy)
        task.forked_loaded_skills = _forked_loaded_skills(parent_session, task.context_policy)
        task.context_packet = self.context_packet_builder.build(
            session=parent_session,
            task_id=task.task_id,
            agent_type=task.agent_type,
            context_policy=task.context_policy,
            task_spec=spec,
            registry=self.registry,
            selected_skills=task.skills_to_preload,
        )

    def _session_for_context_packet(self) -> Session:
        if self.session_provider is None:
            return Session()
        try:
            return self.session_provider()
        except Exception:
            return Session()


def _forked_input_items(session: Session, context_policy: str) -> list[dict]:
    if context_policy in {"isolated", "resume_worker"}:
        return []
    prompt_snapshot = _forked_prompt_snapshot_items(session)
    if prompt_snapshot:
        return prompt_snapshot
    return _complete_prompt_history(session.input_items)


def _forked_instructions(session: Session, context_policy: str) -> str:
    if context_policy in {"isolated", "resume_worker"}:
        return ""
    return str(session.last_prompt_instructions or session.base_instructions or "")


def _forked_loaded_skills(session: Session, context_policy: str) -> list[dict[str, str]]:
    if context_policy in {"isolated", "resume_worker"}:
        return []
    if _has_forked_prompt_snapshot(session):
        return []
    return [copy.deepcopy(skill.to_payload()) for skill in sorted(session.loaded_skills.values(), key=lambda item: item.name)]


_TOOL_OUTPUT_TYPES = {"function_call_output", "custom_tool_call_output"}


def _has_forked_prompt_snapshot(session: Session) -> bool:
    return bool(getattr(session, "last_prompt_input_items", None))


def _forked_prompt_snapshot_items(session: Session) -> list[dict]:
    prompt_items = getattr(session, "last_prompt_input_items", None)
    if not isinstance(prompt_items, list) or not prompt_items:
        return []
    output_items = getattr(session, "last_model_output_items", None)
    forked = copy.deepcopy(prompt_items)
    if isinstance(output_items, list) and output_items:
        forked.extend(copy.deepcopy(output_items))
        forked.extend(_placeholder_tool_outputs(output_items))
    return forked


def _placeholder_tool_outputs(items: list[dict]) -> list[dict]:
    outputs: list[dict] = []
    for item in items:
        for call_id in sorted(_tool_call_ids(item)):
            outputs.append({"type": "function_call_output", "call_id": call_id, "output": FORK_PLACEHOLDER_RESULT})
    return outputs


def _complete_prompt_history(items: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    index = 0
    while index < len(items):
        item = items[index]
        if item.get("type") in _TOOL_OUTPUT_TYPES:
            index += 1
            continue
        call_ids = _tool_call_ids(item)
        if not call_ids:
            cleaned.append(copy.deepcopy(item))
            index += 1
            continue
        next_index = index + 1
        outputs: list[dict] = []
        output_ids: set[str] = set()
        while next_index < len(items) and items[next_index].get("type") in _TOOL_OUTPUT_TYPES:
            output = items[next_index]
            call_id = str(output.get("call_id") or "")
            if call_id:
                output_ids.add(call_id)
            outputs.append(output)
            next_index += 1
        if call_ids.issubset(output_ids):
            cleaned.append(copy.deepcopy(item))
            cleaned.extend(copy.deepcopy(output) for output in outputs)
        elif item.get("role") == "assistant" and str(item.get("content") or "").strip():
            text_only = copy.deepcopy(item)
            text_only.pop("tool_calls", None)
            cleaned.append(text_only)
        index = next_index
    return cleaned


def _tool_call_ids(item: dict) -> set[str]:
    if item.get("type") in {"function_call", "custom_tool_call"} and item.get("call_id"):
        return {str(item["call_id"])}
    if item.get("role") != "assistant":
        return set()
    calls = item.get("tool_calls")
    if not isinstance(calls, list):
        return set()
    return {str(call.get("id") or "") for call in calls if isinstance(call, dict) and call.get("id")}


def _revision_feedback(reasons: list[str]) -> str:
    cleaned = [str(reason).strip() for reason in reasons if str(reason).strip()]
    return "; ".join(cleaned) if cleaned else "verifier did not accept the result"


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- (none)\n"
    return "".join(f"- {item}\n" for item in items)


def _verifier_goal(task: TaskRecord, report: TaskResultReport) -> str:
    spec = task.task_spec or TaskSpec(title=task.title, goal=task.current_goal)
    return (
        "Review the worker's submitted result for the original user task.\n\n"
        "You are a read-only verifier. Inspect the workspace with read-only tools when needed. "
        "Do not modify files. Do not trust the worker report without checking relevant evidence. "
        "Your job is to produce a review report for Main Xiaoming and, if needed, actionable feedback for the worker.\n\n"
        f"Original task title:\n{spec.title}\n\n"
        f"Original task goal:\n{spec.goal}\n\n"
        "Original success criteria:\n"
        + _bullet_list(spec.success_criteria)
        + "\nExpected artifacts:\n"
        + _bullet_list(spec.expected_artifacts)
        + "\nVerification commands requested by the user or planner:\n"
        + _bullet_list(spec.verification_commands)
        + "\nTask decisions and worker/user communication during execution:\n"
        + _bullet_list(_task_execution_facts(task))
        + "\nWorker submitted report:\n"
        + json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n\nReturn only this review report format:\n"
        + "Review-Status: ACCEPTED | NEEDS_REVISION | NEEDS_USER_DECISION\n\n"
        + "Summary:\n"
        + "Concise review outcome.\n\n"
        + "Evidence:\n"
        + "What you inspected and what you found.\n\n"
        + "Issues:\n"
        + "Concrete issues, or '(none)' if accepted.\n\n"
        + "Feedback-To-Worker:\n"
        + "Only include actionable revision instructions when Review-Status is NEEDS_REVISION. Otherwise write '(none)'.\n\n"
        + "Summary-For-Main:\n"
        + "What Main Xiaoming should tell the user if the task needs user decision, or '(none)'.\n\n"
        + "Use ACCEPTED only if the work satisfies the original goal. "
        + "Use NEEDS_REVISION when the same worker can fix the issues. "
        + "Use NEEDS_USER_DECISION when the task is ambiguous, unsafe to proceed, blocked by user intent, or cannot be fixed by worker feedback alone."
    )


def _task_execution_facts(task: TaskRecord) -> list[str]:
    facts: list[str] = []
    facts.extend(task.task_decision_log)
    facts.extend(task.worker_question_log)
    facts.extend(f"Authorization: {item}" for item in task.authorization_log)
    return facts


def _review_report_from_worker_message(message: str, round: int, verifier_id: str) -> ReviewReport:
    text = message.strip()
    status_match = re.search(r"(?im)^\s*\**\s*Review-Status\s*\**\s*:\s*\**\s*([A-Za-z_ -]+?)\s*\**\s*$", text)
    if status_match is None:
        raise ValueError("missing Review-Status")
    status = status_match.group(1).strip().upper().replace("-", "_").replace(" ", "_")
    if status not in {"ACCEPTED", "NEEDS_REVISION", "NEEDS_USER_DECISION"}:
        raise ValueError(f"invalid Review-Status: {status_match.group(1).strip()}")
    sections = _parse_review_sections(text)
    return ReviewReport(
        round=round,
        verifier_id=verifier_id,
        status=status,  # type: ignore[arg-type]
        summary=sections.get("summary", ""),
        evidence=sections.get("evidence", ""),
        issues=sections.get("issues", ""),
        feedback_to_worker=sections.get("feedback_to_worker", ""),
        summary_for_main=sections.get("summary_for_main", ""),
        full_text=text,
    )


_REVIEW_SECTION_KEYS = {
    "summary": "summary",
    "evidence": "evidence",
    "issues": "issues",
    "feedback-to-worker": "feedback_to_worker",
    "summary-for-main": "summary_for_main",
}


def _parse_review_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {value: [] for value in _REVIEW_SECTION_KEYS.values()}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^\s*\**\s*(Summary|Evidence|Issues|Feedback-To-Worker|Summary-For-Main)\s*\**\s*:\s*\**\s*(.*)$", line, re.IGNORECASE)
        if match:
            current = _REVIEW_SECTION_KEYS[match.group(1).strip().lower()]
            remainder = match.group(2).strip()
            if remainder:
                sections[current].append(remainder)
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items()}


def _review_feedback_items(review: ReviewReport) -> list[str]:
    for value in (review.feedback_to_worker, review.issues, review.summary, review.full_text):
        cleaned = _none_if_placeholder(value)
        if cleaned:
            return [cleaned]
    return ["verifier requested revision"]


def _review_summary_for_main(review: ReviewReport) -> str:
    for value in (review.summary_for_main, review.summary, review.issues, review.feedback_to_worker, review.full_text):
        cleaned = _none_if_placeholder(value)
        if cleaned:
            return cleaned
    return "verifier requested main decision"


def _none_if_placeholder(value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned.lower() in {"(none)", "none", "无", "无。", "n/a"}:
        return ""
    return cleaned


def _should_use_codex(task: TaskRecord) -> bool:
    if _requested_executor(task) == "codex":
        return True
    text = f"{task.title}\n{task.current_goal}\n{task.original_request}".lower()
    return "codex" in text


def _requested_executor(task: TaskRecord) -> str:
    if task.task_spec is None:
        return ""
    for line in task.task_spec.notes.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "requested_executor":
            return value.strip().lower()
    return ""


def _user_decision_feedback(task: TaskRecord, user_message: str) -> str:
    previous = task.needs_user_decision_summary.strip()
    if previous:
        return f"User decision/input:\n{user_message.strip()}\n\nPrevious decision request:\n{previous}\n\nContinue the original task using this user decision."
    return f"User decision/input:\n{user_message.strip()}\n\nContinue the original task using this user decision."


def _review_exhausted_summary(task: TaskRecord, report: TaskResultReport, reasons: list[str]) -> str:
    base = task.task_spec or TaskSpec(title=task.title, goal=task.current_goal)
    return (
        "Review feedback loop exhausted.\n\n"
        f"Original goal:\n{base.goal}\n\n"
        f"Latest verifier feedback:\n{_revision_feedback(reasons)}\n\n"
        f"Latest worker answer:\n{report.summary or '(empty)'}\n\n"
        "Main Xiaoming should explain the current task state to the user and decide, based on the user's response, "
        "whether to continue revising, adjust the goal, accept the current result with concerns, or stop the task."
    )


def _auto_answer_notice(task: TaskRecord, question: Question, decision: WorkerQuestionDecision) -> str:
    action = "批准" if decision.decision == "approved" else "拒绝"
    reason = f"：{decision.reason}" if decision.reason else ""
    return f"{task.title} 已根据授权自动{action}后台任务请求{reason}"


def _decision_summary(decision: SchedulerDecision) -> str:
    parts = [
        f"action: {decision.action}",
        f"title: {decision.task_title}",
    ]
    if decision.target_task_id:
        parts.append(f"target_task_id: {decision.target_task_id}")
    if decision.conflict_task_ids:
        parts.append("conflict_task_ids: " + ", ".join(sorted(decision.conflict_task_ids)))
    if decision.duplicate_task_ids:
        parts.append("duplicate_task_ids: " + ", ".join(sorted(decision.duplicate_task_ids)))
    if decision.reason:
        parts.append(f"reason: {decision.reason}")
    return "\n".join(parts)


def _task_status_line(task: TaskRecord, pending_questions: list[str] | None = None) -> str:
    detail = task.last_progress
    if task.verification_result is not None and task.verification_result.reasons:
        detail = "; ".join(task.verification_result.reasons)
    questions = ""
    if pending_questions is None:
        pending_questions = []
    if pending_questions:
        questions = "  pending: " + " | ".join(pending_questions)
    agent = f" [{task.agent_type}]" if task.agent_type and task.agent_type != "worker" else ""
    return f"{task.task_id[:8]}  {task.title}{agent}  {task.status}  {detail}{questions}"


def _follow_task_snapshot(task: TaskRecord, changed: bool) -> str:
    prefix = "任务状态已变化。" if changed else "任务仍在运行，我先停止等待，后续状态变化会主动通知。"
    return f"{prefix}\n{_task_status_line(task)}"
