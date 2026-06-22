from __future__ import annotations

import argparse
import copy
import contextlib
import queue
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from xiaoming.async_runtime.leases import FileWriteLeaseClient
from xiaoming.async_runtime.agents import AgentDefinition, builtin_agent_registry
from xiaoming.async_runtime.context_packets import WorkerContextPacket
from xiaoming.async_runtime.protocol import ProtocolError, decode_message, write_message
from xiaoming.async_runtime.tasks import TaskSpec
from xiaoming.bootstrap import discover_bootstrap_contexts
from xiaoming.cli import build_loop, build_universal_runtime_tools
from xiaoming.logging import XiaomingLogger
from xiaoming.progress import ProgressEvent
from xiaoming.session import BootstrapContext, LoadedSkill, Session
from xiaoming.worker_diagnostics import WorkerSessionRecorder


class WorkerInbox:
    def __init__(self) -> None:
        self.queue: queue.Queue[dict] = queue.Queue()
        self.cancelled = False
        self.closed = False
        self._reader = threading.Thread(target=self._read_stdin, daemon=True)

    def start(self) -> None:
        self._reader.start()

    def get(self, timeout: float | None = None) -> dict | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_appends(self) -> list[str]:
        appends: list[str] = []
        retained: list[dict] = []
        while True:
            message = self.get(timeout=0)
            if message is None:
                break
            if message.get("kind") == "cancel":
                self.cancelled = True
            elif message.get("kind") == "append_user_message":
                appends.append(str(message.get("message") or ""))
            else:
                retained.append(message)
        for message in retained:
            self.queue.put(message)
        return [message for message in appends if message.strip()]

    def drain_queued_work(self) -> tuple[list[str], list[str]]:
        appends: list[str] = []
        talks: list[str] = []
        retained: list[dict] = []
        while True:
            message = self.get(timeout=0)
            if message is None:
                break
            kind = message.get("kind")
            if kind == "cancel":
                self.cancelled = True
            elif kind == "append_user_message":
                appends.append(str(message.get("message") or ""))
            elif kind == "talk":
                talks.append(str(message.get("message") or ""))
            else:
                retained.append(message)
        for message in retained:
            self.queue.put(message)
        return [message for message in appends if message.strip()], [message for message in talks if message.strip()]

    def _read_stdin(self) -> None:
        try:
            for line in sys.stdin:
                try:
                    payload = decode_message(line)
                except ProtocolError:
                    continue
                if payload.get("kind") == "cancel":
                    self.cancelled = True
                self.queue.put(payload)
        finally:
            self.closed = True


def main() -> int:
    protocol_out = sys.stdout
    try:
        start = decode_message(sys.stdin.readline())
    except ProtocolError as exc:
        write_message(protocol_out, "failed", task_id="", message=f"invalid start message: {exc}", data={"error_kind": "invalid_start_message"})
        return 1
    if start.get("kind") != "start_task":
        write_message(protocol_out, "failed", task_id="", message="first worker message must be start_task", data={"error_kind": "invalid_start_message"})
        return 1
    task_id = str(start["task_id"])
    task = str(start["task"])
    task_spec = _task_spec_from_start(start, task)
    agent_type = _normalized_agent_type(str(start.get("agent_type") or task_spec.agent_type or "worker"))
    agent = _agent_for_type(agent_type)
    skills_to_preload = [str(item) for item in start.get("skills_to_preload") or task_spec.skills_to_preload]
    context_packet = _context_packet_from_start(start)
    forked_instructions = str(start.get("forked_instructions") or "")
    forked_input_items = _forked_input_items_from_start(start)
    forked_loaded_skills = _forked_loaded_skills_from_start(start)
    workspace = Path(str(start["workspace"]))
    inbox = WorkerInbox()
    inbox.start()
    write_message(protocol_out, "started", task_id=task_id, message=f"Started {task}")
    args = SimpleNamespace(
        task=task,
        provider=start.get("provider"),
        model=start.get("model"),
        approval_mode=start.get("approval_mode") or "full_auto",
        permission_mode=start.get("permission_mode"),
        max_turns=start.get("max_turns"),
        model_timeout_seconds=start.get("model_timeout_seconds"),
        stream=start.get("stream"),
        continue_session=False,
        resume_session_id=None,
        new_session=True,
    )
    try:
        logger = XiaomingLogger.create_worker(workspace, task_id)
        session_recorder = WorkerSessionRecorder(workspace, task_id)
        logger.info(
            "worker_started",
            task_id=task_id,
            task=task,
            workspace=str(workspace),
            provider=start.get("provider"),
            model=start.get("model"),
            approval_mode=start.get("approval_mode") or "full_auto",
            permission_mode=start.get("permission_mode"),
            stream=start.get("stream"),
        )
        lease_client = FileWriteLeaseClient(workspace / ".xiaoming" / "leases", task_id)

        with contextlib.redirect_stdout(sys.stderr):
            tool_flags = _tool_flags_for_profile(agent.tool_profile)
            loop = build_loop(
                workspace,
                args,
                approve=lambda action: _approve_via_parent(protocol_out, task_id, action, inbox),
                lease_callback=lease_client.acquire,
                extra_tools=build_universal_runtime_tools(
                    coordinator_getter=lambda: None,
                    talk_callback=lambda purpose, message, context, options: _talk_via_parent(protocol_out, task_id, inbox, purpose=purpose, message=message, context=context, options=options),
                ),
                logger=logger,
                session_recorder=session_recorder,
                instructions_override=forked_instructions or None,
                capability_profile=agent.tool_profile,
                **tool_flags,
            )
            _log_worker_capabilities(logger, loop)
            session = Session(session_id=task_id)
            byte_fork = bool(forked_instructions and forked_input_items)
            _apply_forked_parent_context(session, forked_input_items, [] if byte_fork else forked_loaded_skills, logger)
            if byte_fork:
                logger.info("worker_byte_fork_prompt_applied", input_items=len(forked_input_items))
            else:
                _inject_worker_bootstrap_contexts(workspace, session, logger)
                _inject_preloaded_skills(session, loop.skill_library, skills_to_preload, logger)
            if forked_instructions:
                pending_tasks = [(_forked_worker_directive(agent, task_spec, context_packet), False)]
            else:
                _inject_worker_agent_context(session, agent)
                _inject_worker_protocol_context(session)
                _inject_worker_context_packet(session, context_packet)
                _inject_worker_task_contract(session, task_spec)
                pending_tasks = [("Start working on the task contract provided in context.", False)]
            pending_completion: str | None = None
            while pending_tasks or pending_completion is not None:
                if inbox.cancelled:
                    write_message(protocol_out, "cancelled", task_id=task_id, message="Task cancelled.")
                    return 1
                if not pending_tasks and pending_completion is not None:
                    answer = pending_completion
                    pending_completion = None
                    write_message(protocol_out, "completed", task_id=task_id, message=answer)
                    followup = _wait_for_review_followup(inbox)
                    if followup is None:
                        return 0
                    if followup == ("__cancelled__", False):
                        write_message(protocol_out, "cancelled", task_id=task_id, message="Task cancelled.")
                        return 1
                    pending_tasks.append(followup)
                    continue
                current, is_talk = pending_tasks.pop(0)
                streamed_text: list[str] = []

                def on_progress(event: str | ProgressEvent) -> None:
                    if isinstance(event, ProgressEvent):
                        if event.kind in {"tool_started", "tool_finished"}:
                            streamed_text.clear()
                        elif event.kind == "text_delta":
                            streamed_text.append(event.message)
                    _emit_progress(protocol_out, task_id, event)

                answer = loop.run(current, session=session, on_event=on_progress)
                if not answer.strip() and streamed_text:
                    answer = "".join(streamed_text).strip()
                if answer.startswith("Error:"):
                    write_message(protocol_out, "failed", task_id=task_id, message=answer, data={"error_kind": "agent_loop_error"})
                    return 1
                appended, queued_talks = inbox.drain_queued_work()
                if appended or (queued_talks and not is_talk):
                    if answer:
                        if is_talk:
                            write_message(protocol_out, "peer_reply", task_id=task_id, message=answer)
                        else:
                            pending_completion = answer
                    pending_tasks.extend((message, False) for message in appended)
                    pending_tasks.extend((message, True) for message in queued_talks)
                    continue
                if is_talk:
                    write_message(protocol_out, "peer_reply", task_id=task_id, message=answer)
                    if pending_tasks or pending_completion is not None:
                        continue
                    followup = _wait_for_review_followup(inbox)
                    if followup is None:
                        return 0
                    if followup == ("__cancelled__", False):
                        write_message(protocol_out, "cancelled", task_id=task_id, message="Task cancelled.")
                        return 1
                    pending_tasks.append(followup)
                    continue
                write_message(protocol_out, "completed", task_id=task_id, message=answer)
                followup = _wait_for_review_followup(inbox)
                if followup is None:
                    return 0
                if followup == ("__cancelled__", False):
                    write_message(protocol_out, "cancelled", task_id=task_id, message="Task cancelled.")
                    return 1
                pending_tasks.append(followup)
        return 0
    except BaseException as exc:
        if "logger" in locals():
            logger.error("worker_crashed", exc=exc, task_id=task_id)
        write_message(protocol_out, "failed", task_id=task_id, message=str(exc), data={"error_kind": "worker_crashed"})
        return 1
    finally:
        if "lease_client" in locals():
            lease_client.release_all()


def _emit_progress(protocol_out, task_id: str, event: str | ProgressEvent) -> None:
    if isinstance(event, ProgressEvent):
        kind = "assistant_delta" if event.kind == "text_delta" else event.kind
        write_message(protocol_out, kind, task_id=task_id, message=event.message)
        return
    write_message(protocol_out, "progress", task_id=task_id, message=event)


def _wait_for_review_followup(inbox: WorkerInbox) -> tuple[str, bool] | None:
    while True:
        if inbox.cancelled:
            return ("__cancelled__", False)
        payload = inbox.get(timeout=0.1)
        if payload is None:
            if inbox.closed:
                return None
            continue
        kind = payload.get("kind")
        if kind == "cancel":
            return ("__cancelled__", False)
        if kind == "review_accepted":
            return None
        if kind == "review_feedback":
            return (_review_feedback_prompt(str(payload.get("feedback") or "")), False)
        if kind == "append_user_message":
            message = str(payload.get("message") or "").strip()
            if message:
                return (message, False)
        if kind == "talk":
            message = str(payload.get("message") or "").strip()
            if message:
                return (message, True)


def _review_feedback_prompt(feedback: str) -> str:
    return (
        "The verifier reviewed your submitted result and requested revision.\n\n"
        f"Verifier feedback:\n{feedback or '(no details provided)'}\n\n"
        "Continue from your current task context. Fix the feedback while staying scoped to the original user goal. "
        "Run reasonable verification, then submit the revised final answer."
    )


def _log_worker_capabilities(logger: XiaomingLogger, loop) -> None:
    logger.info(
        "worker_tools_available",
        tools=[tool.name for tool in loop.registry.specs()],
    )
    skills = []
    if loop.skill_library is not None:
        skills = [
            {
                "name": skill.name,
                "description": skill.description,
                "path": str(skill.path) if skill.path is not None else "",
            }
            for skill in loop.skill_library.skills
        ]
    logger.info("worker_skills_available", skills=skills)


def _inject_worker_bootstrap_contexts(workspace: Path, session: Session, logger: XiaomingLogger) -> None:
    for context in discover_bootstrap_contexts(workspace):
        session.remember_bootstrap_context(context)
        logger.info(
            "worker_bootstrap_context_injected",
            plugin=context.plugin_name,
            source=context.source,
            path=context.path,
            content_hash=context.content_hash,
        )


def _inject_worker_protocol_context(session: Session) -> None:
    session.remember_bootstrap_context(
        BootstrapContext.create(
            plugin_name="runtime",
            source="runtime:worker-protocol",
            content=_worker_protocol_context_item()["content"],
        )
    )


def _inject_worker_task_contract(session: Session, task_spec: TaskSpec) -> None:
    session.remember_bootstrap_context(
        BootstrapContext.create(
            plugin_name="runtime",
            source="runtime:task-contract",
            content=_worker_task_prompt(task_spec),
        )
    )


def _approve_via_parent(protocol_out, task_id: str, action: str, inbox: WorkerInbox) -> bool:
    request_id = str(uuid4())
    write_message(protocol_out, "approval_request", task_id=task_id, message=action, data={"request_id": request_id})
    while True:
        payload = inbox.get(timeout=0.1)
        if inbox.cancelled:
            raise KeyboardInterrupt()
        if payload is None:
            continue
        if payload.get("kind") == "cancel":
            raise KeyboardInterrupt()
        if payload.get("kind") != "answer_question":
            continue
        if payload.get("request_id") != request_id:
            continue
        decision = str(payload.get("decision") or "none").strip().lower()
        if decision == "approved":
            return True
        if decision == "denied":
            return False
        continue
    return False


def _talk_via_parent(protocol_out, task_id: str, inbox: WorkerInbox, *, purpose: str, message: str, context: str = "", options: list[str] | None = None) -> str:
    request_id = str(uuid4())
    kind = "decision_request" if purpose == "decision" else "clarification_request"
    write_message(
        protocol_out,
        kind,
        task_id=task_id,
        message=message,
        data={
            "request_id": request_id,
            "purpose": purpose,
            "context": context,
            "options": list(options or []),
        },
    )
    while True:
        payload = inbox.get(timeout=0.1)
        if inbox.cancelled:
            raise KeyboardInterrupt()
        if payload is None:
            continue
        if payload.get("kind") == "cancel":
            raise KeyboardInterrupt()
        if payload.get("kind") != "answer_question":
            continue
        if payload.get("request_id") != request_id:
            continue
        return str(payload.get("answer") or "")


def _task_spec_from_start(start: dict, fallback_task: str) -> TaskSpec:
    raw = start.get("task_spec")
    if isinstance(raw, dict):
        try:
            return TaskSpec.from_dict(raw)
        except ValueError:
            pass
    return TaskSpec.from_request(fallback_task)


def _context_packet_from_start(start: dict) -> WorkerContextPacket | None:
    raw = start.get("context_packet")
    if not isinstance(raw, dict):
        return None
    try:
        return WorkerContextPacket.from_dict(raw)
    except Exception:
        return None


def _forked_input_items_from_start(start: dict) -> list[dict]:
    raw = start.get("forked_input_items")
    if not isinstance(raw, list):
        return []
    return [copy.deepcopy(item) for item in raw if isinstance(item, dict)]


def _forked_loaded_skills_from_start(start: dict) -> list[dict[str, str]]:
    raw = start.get("forked_loaded_skills")
    if not isinstance(raw, list):
        return []
    return [{str(key): str(value) for key, value in item.items()} for item in raw if isinstance(item, dict)]


def _apply_forked_parent_context(session: Session, input_items: list[dict], loaded_skills: list[dict[str, str]], logger: XiaomingLogger) -> None:
    session.input_items.extend(copy.deepcopy(input_items))
    for payload in loaded_skills:
        session.remember_loaded_skill(LoadedSkill.from_payload(payload))
    logger.info(
        "worker_parent_context_forked",
        input_items=len(input_items),
        loaded_skills=[str(payload.get("name") or "") for payload in loaded_skills if payload.get("name")],
    )


def _agent_for_type(agent_type: str) -> AgentDefinition:
    registry = builtin_agent_registry()
    try:
        return registry.get(_normalized_agent_type(agent_type))
    except KeyError:
        return registry.get("worker")


def _normalized_agent_type(agent_type: str) -> str:
    return "verifier" if str(agent_type or "").strip() == "verifier" else "worker"


def _inject_worker_agent_context(session: Session, agent_or_type: AgentDefinition | str) -> None:
    agent = agent_or_type if isinstance(agent_or_type, AgentDefinition) else _agent_for_type(agent_or_type)
    session.remember_bootstrap_context(
        BootstrapContext.create(
            plugin_name="runtime",
            source="runtime:worker-agent-definition",
            content=_worker_agent_context(agent),
        )
    )


def _tool_flags_for_profile(profile: str) -> dict[str, bool]:
    return {
        "include_workspace_tools": True,
        "include_write_tools": True,
        "include_shell_tool": True,
        "include_skill_install_tool": True,
        "include_load_skill_tool": True,
    }


def _worker_agent_context(agent: AgentDefinition) -> str:
    return "\n".join(
        [
            "<worker_agent_definition>",
            f"<name>{agent.name}</name>",
            f"<tool_profile>{agent.tool_profile}</tool_profile>",
            f"<default_context_policy>{agent.default_context_policy}</default_context_policy>",
            "<instructions>",
            agent.system_prompt,
            "</instructions>",
            "</worker_agent_definition>",
        ]
    )


def _inject_preloaded_skills(session: Session, skill_library, skills_to_preload: list[str], logger: XiaomingLogger) -> None:
    loaded: list[dict[str, str]] = []
    if skill_library is None:
        return
    for skill_name in skills_to_preload:
        skill = skill_library.load(skill_name)
        if skill is None:
            loaded.append({"name": skill_name, "status": "missing"})
            continue
        loaded_skill = LoadedSkill.create(
            name=skill.name,
            description=skill.description,
            content=skill.content,
            path=str(skill.path) if skill.path is not None else "",
        )
        session.remember_loaded_skill(loaded_skill)
        loaded.append({"name": loaded_skill.name, "status": "loaded", "path": loaded_skill.path, "content_hash": loaded_skill.content_hash})
    logger.info("worker_preloaded_skills", skills=loaded)


def _inject_worker_context_packet(session: Session, packet: WorkerContextPacket | None) -> None:
    if packet is None:
        return
    session.remember_bootstrap_context(
        BootstrapContext.create(
            plugin_name="runtime",
            source="runtime:worker-context-packet",
            content=packet.render_for_worker(),
        )
    )


def _forked_worker_directive(agent: AgentDefinition, task_spec: TaskSpec, packet: WorkerContextPacket | None) -> str:
    parts = [
        "<forked_worker_task>",
        "STOP. READ THIS FIRST.",
        "You are a forked worker running with the coordinator's inherited system prompt and conversation history.",
        "The inherited prompt may describe the main interactive assistant. For this turn, follow the worker protocol and task contract below.",
        _worker_agent_context(agent),
        _worker_protocol_context_item()["content"],
    ]
    if packet is not None:
        parts.extend(["<worker_context>", packet.render_for_worker(), "</worker_context>"])
    parts.extend(
        [
            "<task_contract>",
            _worker_task_prompt(task_spec),
            "</task_contract>",
            _worker_operating_rules(),
            "Start working on the task contract now. Use tools directly. Ask the coordinator with talk when input is required.",
            "</forked_worker_task>",
        ]
    )
    return "\n".join(parts)


def _worker_task_prompt(spec: TaskSpec) -> str:
    return (
        f"Title: {spec.title}\n"
        f"Goal: {spec.goal}\n"
        "User-stated success criteria:\n"
        + _bullet_list(spec.success_criteria)
        + "\nExplicitly requested artifacts:\n"
        + _bullet_list(spec.expected_artifacts)
        + "\nExplicit write path constraints:\n"
        + _bullet_list(spec.allowed_write_paths)
        + "\nExplicit verification commands:\n"
        + _bullet_list(spec.verification_commands)
        + f"\nNotes: {spec.notes or '(none)'}"
    )


def _worker_operating_rules() -> str:
    return (
        "<worker_operating_rules>\n"
        "- Continue until the assigned task is complete, clearly blocked, or requires coordinator/user input.\n"
        "- Do not guess file contents, command results, repository state, or user intent. Inspect or ask first.\n"
        "- Before editing, read the relevant files and search the codebase. Prefer rg-style search when available.\n"
        "- Respect AGENTS.md and project rules that apply to files you touch, including more specific nested rules.\n"
        "- Make minimal, focused changes. Do not refactor unrelated code or fix unrelated bugs.\n"
        "- Match the existing style and local patterns. Add abstractions only when they reduce real complexity.\n"
        "- Use the smallest reliable edit method. Split large writes or tool arguments into smaller chunks.\n"
        "- After editing, run the smallest relevant verification command. Broaden verification only when risk warrants it.\n"
        "- If verification fails, read the failure, decide whether it relates to your change, and iterate when appropriate.\n"
        "- If permission, product intent, design approval, or user preference is unclear, ask the coordinator with talk.\n"
        "- Do not claim completion until the work is done, verified when practical, or explicitly blocked.\n"
        "- Final response must summarize the outcome, changed files or artifacts, verification performed, and remaining risks.\n"
        "</worker_operating_rules>"
    )


def _worker_protocol_context_item() -> dict:
    return {
        "role": "developer",
        "content": (
            "<worker_protocol>\n"
            "You are an independent, fully capable coding agent working in this workspace.\n"
            "Your user is the coordinator. The coordinator represents the human user and relays messages between you and the human.\n"
            "Your conversation history was forked from the coordinator's session so you can see the user's prior context. Treat that history as background context, while the task contract below is your current assignment.\n"
            "You have full access to the repository tools and skill system. Follow loaded skills exactly as you would in a normal interactive coding session.\n"
            "Treat yourself as the agent responsible for completing the assignment end to end. If skill instructions mention subagents or background agents, interpret those references as applying to your own current agent session unless the coordinator explicitly tells you otherwise.\n"
            "Before inspecting files, writing files, running commands, or reporting progress, decide whether any available skill applies. If a skill applies, call load_skill before acting. If you skip an obvious skill, state the reason briefly.\n"
            "Loaded skill instructions guide how you approach the task; the task contract defines what must be delivered.\n"
            "If a loaded skill says to ask the user, present a design, wait for approval, or get review, ask your user by calling the talk tool. Wait for the coordinator's answer before continuing. Normal assistant text is progress only and cannot receive a user reply. Never place a question, approval request, design review request, or decision request only in normal assistant text. Do not treat missing input as denial.\n"
            "Use progress text only for status updates that do not require an answer. Permission approval must use the existing approval path, not talk.\n"
            "For skill installation, install only after the source is known. Use the native install_skill tool instead of recreating installer behavior with shell, git clone, curl, mkdir, cp, or write_file. If a skill install source is unknown, load find-skills to discover candidates or ask the coordinator with talk for the GitHub URL or repo/path. Do not assume openai/skills is the default source.\n"
            "The task contract is primarily the user's goal, not a full implementation plan. You may choose filenames, artifacts, implementation approach, and verification steps unless they are explicitly listed. Do not treat empty optional sections as missing requirements.\n"
            "When the work is complete, provide a concise final answer that states the outcome, important changed files or artifacts, and any verification performed. If you cannot complete the work, state the blocker clearly in your final answer or ask your user with talk when input is needed.\n"
            "</worker_protocol>"
        ),
        "xiaoming": {"kind": "worker_protocol"},
    }


def _worker_protocol_repair_prompt() -> str:
    return (
        "In this worker protocol, normal assistant text is progress only and cannot receive replies. "
        "If you need input from the coordinator or the human user, call the talk tool now. "
        "If no input is needed, continue the task. When the work is complete, provide a concise final answer."
    )


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- (none)\n"
    return "".join(f"- {item}\n" for item in items)


if __name__ == "__main__":
    raise SystemExit(main())
