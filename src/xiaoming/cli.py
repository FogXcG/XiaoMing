from __future__ import annotations

import argparse
import inspect
import os
import select
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from xiaoming.async_runtime.coordinator import AsyncCoordinator, CoordinatorConfig
from xiaoming.async_runtime.events import CoordinatorNotice
from xiaoming.async_runtime.leases import WriteLeaseCallback
from xiaoming.async_runtime.question_decider import LLMWorkerQuestionDecider
from xiaoming.async_runtime.responder import LLMMessagingResponder
from xiaoming.async_runtime.scheduler import LLMScheduler
from xiaoming.async_runtime.tasks import TaskSpec
from xiaoming.agent_loop import AgentLoop
from xiaoming.bootstrap import discover_bootstrap_contexts
from xiaoming.checkpoints import CheckpointStore
from xiaoming.config import load_config
from xiaoming.hook_config import load_workspace_hooks
from xiaoming.hooks import HookManager
from xiaoming.llm.deepseek_provider import DeepSeekProvider
from xiaoming.llm.openai_provider import OpenAIProvider
from xiaoming.logging import XiaomingLogger
from xiaoming.permissions.engine import PermissionEngine
from xiaoming.permissions.store import add_project_rule, load_project_rules, permissions_path
from xiaoming.permissions.types import PermissionBehavior, PermissionMode, PermissionRule
from xiaoming.progress import ProgressEvent
from xiaoming.session import Session
from xiaoming.sessions import SessionRecord, SessionStore, rehydrate_session
from xiaoming.skill_installer import SkillInstallError, install_skill_from_url
from xiaoming.skills import SkillLibrary
from xiaoming.tools.apply_patch import ApplyPatchTool
from xiaoming.tools.append_file import AppendFileTool
from xiaoming.tools.background_task import BackgroundTasksStatusTool, CancelBackgroundTaskTool, FollowBackgroundTaskTool, ReplyMailboxMessageTool, ScheduleBackgroundTaskTool, TalkToPeerTool
from xiaoming.tools.base import Tool, ToolResult
from xiaoming.tools.edit_file import EditFileTool
from xiaoming.tools.git_status import GitStatusTool
from xiaoming.tools.fetch_skill import FetchSkillTool
from xiaoming.tools.install_skill import InstallSkillTool
from xiaoming.tools.list_files import ListFilesTool
from xiaoming.tools.load_skill import LoadSkillTool
from xiaoming.tools.read_file import ReadFileTool
from xiaoming.tools.registry import ToolRegistry
from xiaoming.tools.search_code import SearchCodeTool
from xiaoming.tools.shell import ShellTool
from xiaoming.tools.talk import TalkCallback, TalkTool
from xiaoming.tools.web import WebFetchTool, WebSearchTool
from xiaoming.tools.write_file import WriteFileTool


def enable_line_editing(importer=__import__) -> bool:
    try:
        readline = importer("readline")
    except Exception:
        return False
    parse_and_bind = getattr(readline, "parse_and_bind", None)
    if parse_and_bind is not None:
        for binding in (
            "set input-meta on",
            "set output-meta on",
            "set convert-meta off",
            "set enable-bracketed-paste on",
        ):
            try:
                parse_and_bind(binding)
            except Exception:
                continue
    return True


def discard_pending_terminal_input() -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        return False
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="xiaoming")
    parser.add_argument("task", nargs="?")
    parser.add_argument("--provider", choices=["openai", "deepseek"])
    parser.add_argument("--model")
    parser.add_argument("--approval-mode", choices=["suggest", "auto_edit", "full_auto"])
    parser.add_argument("--permission-mode", choices=["default", "plan", "accept_edits", "auto", "bypass"])
    parser.add_argument("--continue", dest="continue_session", action="store_true", default=False)
    parser.add_argument("--resume", dest="resume_session_id")
    parser.add_argument("--new", dest="new_session", action="store_true", default=False)
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--model-timeout", dest="model_timeout_seconds", type=float)
    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument("--stream", dest="stream", action="store_true", default=None)
    stream_group.add_argument("--no-stream", dest="stream", action="store_false")
    return parser.parse_args(argv)


def approve_action(action: str) -> bool:
    enable_line_editing()
    print(action)
    if not sys.stdin.isatty():
        return False
    try:
        answer = input("Approve? [y/N] ").strip().lower()
    except EOFError:
        print()
        return False
    return answer in {"y", "yes"}


ORCHESTRATOR_ALLOWED_TOOLS = {
    "web_search",
    "schedule_background_task",
    "background_tasks_status",
    "follow_background_task",
    "cancel_background_task",
    "talk_to_peer",
    "reply_mailbox_message",
}
READ_ONLY_ALLOWED_TOOLS = {
    "web_search",
    "list_files",
    "read_file",
    "search_code",
    "web_fetch",
    "git_status",
    "load_skill",
    "talk",
}
SKILL_INSTALL_ALLOWED_TOOLS = READ_ONLY_ALLOWED_TOOLS | {"install_skill"}
CapabilityProfile = str | Callable[[], str]


@dataclass
class ForegroundTask:
    task_name: str
    original_message: str
    status: str = "running"
    worker_id: str | None = None


@dataclass
class ForegroundTurnResult:
    answer: str
    pending_user_input: str | None = None


def build_universal_runtime_tools(
    coordinator_getter: Callable[[], object | None],
    talk_callback: TalkCallback,
    turn_context_getter: Callable[[], str] | None = None,
) -> list[Tool]:
    return [
        ScheduleBackgroundTaskTool(coordinator_getter, turn_context_getter=turn_context_getter),
        BackgroundTasksStatusTool(coordinator_getter),
        FollowBackgroundTaskTool(coordinator_getter),
        CancelBackgroundTaskTool(coordinator_getter),
        TalkToPeerTool(coordinator_getter),
        ReplyMailboxMessageTool(coordinator_getter),
        TalkTool(talk_callback),
    ]


def unavailable_talk_callback(purpose: str, message: str, context: str, options: list[str]) -> str:
    return "talk is only available to background workers; answer the user directly or schedule a background task."


def tool_capability_hook(profile: CapabilityProfile) -> Callable[[dict], dict | None]:
    def hook(payload: dict) -> dict | None:
        normalized = _resolve_capability_profile(profile)
        tool = str(payload.get("tool") or "")
        if normalized == "foreground" or normalized == "orchestrator":
            return None
        if normalized in {"read_only", "verify"}:
            if tool in READ_ONLY_ALLOWED_TOOLS:
                return None
            return {"decision": "deny", "reason": "This worker profile is read-only; use read/search tools or talk, and do not modify files, run shell commands, install skills, or manage background tasks."}
        if normalized == "skill_install":
            if tool in SKILL_INSTALL_ALLOWED_TOOLS:
                return None
            return {"decision": "deny", "reason": "This worker installs skills through install_skill; do not use shell, git clone, write tools, or background task management tools."}
        if tool in {"schedule_background_task", "background_tasks_status", "follow_background_task", "cancel_background_task", "talk_to_peer", "reply_mailbox_message"}:
            return {"decision": "deny", "reason": "Background workers cannot manage the coordinator's task queue; use talk when you need coordinator or user input."}
        return None

    return hook


def _resolve_capability_profile(profile: CapabilityProfile) -> str:
    value = profile() if callable(profile) else profile
    return (value or "full").strip()


class CapabilityGuardedTool:
    def __init__(self, tool: Tool, profile: CapabilityProfile):
        self.tool = tool
        self.profile = profile
        self.name = tool.name
        self.description = tool.description
        self.input_schema = tool.input_schema
        self.supports_parallel_tool_calls = bool(getattr(tool, "supports_parallel_tool_calls", False))
        self.workspace = getattr(tool, "workspace", None)

    @property
    def spec(self):
        return self.tool.spec

    def run(self, args: dict) -> ToolResult:
        decision = tool_capability_hook(self.profile)({"tool": self.name, "arguments": args})
        if decision is not None and decision.get("decision") == "deny":
            return ToolResult(self.name, "denied", error=str(decision.get("reason") or "blocked by tool capability policy"))
        return self.tool.run(args)


def _with_tool_capability_hook(hooks: HookManager | None, profile: CapabilityProfile) -> HookManager:
    manager = hooks or HookManager()
    return manager.with_prepended("PreToolUse", tool_capability_hook(profile))


def build_instructions(workspace: Path, role: str = "worker") -> str:
    personality_prompt = (Path(__file__).parent / "prompts" / "personality.md").read_text()
    default_prompt = _orchestrator_prompt() if role == "orchestrator" else (Path(__file__).parent / "prompts" / "system.md").read_text()
    safety = "Safety policy: never execute destructive commands automatically; keep file access inside the workspace."
    background_tasks = """
Background task policy:
- In interactive chat, handle user-facing conversation and coordination. Answer simple questions directly.
- For simple greetings, reply naturally without introducing a fixed identity unless the user asks.
- For work that should continue in the background, first tell the user what will be handled in the background, then call schedule_background_task with a plain-language message for the worker.
- Once you decide work belongs in the background, call schedule_background_task immediately. Do not inspect files, run shell commands, or probe the workspace first; the worker will inspect the workspace.
- Do not call schedule_background_task for ordinary Q&A, explanations, or quick clarifications.
- Installing remote skills, cloning repositories, dependency setup, and other network/file-changing setup work must be scheduled with schedule_background_task during interactive chat.
- schedule_background_task means "arrange work"; it does not mean the work has completed.
- When scheduling work, pass only message and, when useful, a short task_name.
- Put user-stated constraints directly in message. Do not pass or invent internal fields such as worker type, context policy, skills, expected artifacts, allowed paths, or verification commands.
- When the user asks about background progress, call background_tasks_status at most once in that turn, then answer from that snapshot. Do not inspect session files, logs, or the filesystem to infer background task state or claim completion while any background task is active.
- If the user explicitly asks you to wait, follow, or watch a specific background task, use follow_background_task with the task_id. If the task_id is unknown, call background_tasks_status once first.
- When <pending_worker_questions> is present, decide whether the user's message clearly answers one of those questions before responding normally.
- If the user clearly answers a pending worker question, call reply_mailbox_message with a normalized answer suitable for the worker. For approval questions, set decision to approved or denied.
- For approval questions, only set decision=approved when the user clearly grants permission for the specific requested action. Generic acknowledgements such as "ok", "好的", "知道了", or "完成后告诉我" are not approval by themselves.
- If the user delegates future similar decisions, include a concise authorization_note describing the delegation scope for that worker. Otherwise set authorization_note to an empty string.
- If the user's message is related to a pending worker question but ambiguous, ask a short clarification in normal chat and do not call reply_mailbox_message.
- If the user's message is a new request, status question, unrelated message, or otherwise does not clearly answer the pending worker question, leave the question pending. Do not treat silence, status questions, or unrelated messages as denial.
- To cancel a background task, call cancel_background_task with the exact task_id. If the task_id is unknown, call background_tasks_status first.
- For multiple cancellations, call cancel_background_task once per task. Do not claim cancellation unless the tool succeeds.
- If replacing a background task, cancel the old task first, then schedule the replacement.
- If a background task failed, do not take over the same file-changing work in the foreground. Explain the failure, ask for direction if needed, or schedule a new background retry.
- After schedule_background_task returns, keep any follow-up brief and avoid repeating the same acknowledgement.
""".strip()
    agents = workspace / "AGENTS.md"
    project_rules = agents.read_text() if agents.exists() else ""
    return f"{safety}\n\n{personality_prompt}\n\n{default_prompt}\n\n{background_tasks}\n\nProject rules:\n{project_rules}"


def build_registry(
    workspace: Path,
    approval_mode: str,
    permission_engine: PermissionEngine | None = None,
    skill_library: SkillLibrary | None = None,
    logger: XiaomingLogger | None = None,
    checkpoint_store: CheckpointStore | None = None,
    checkpoint_id=None,
    lease_callback: WriteLeaseCallback | None = None,
    approve: Callable[[str], bool] = approve_action,
    extra_tools: list[Tool] | None = None,
    include_skill_install_tool: bool = True,
    include_load_skill_tool: bool = True,
    include_workspace_tools: bool = True,
    include_write_tools: bool = True,
    include_shell_tool: bool = True,
    tool_wrapper: Callable[[Tool], Tool] | None = None,
    hooks: HookManager | None = None,
    capability_profile: CapabilityProfile = "full",
) -> ToolRegistry:
    approve = _approval_with_permission_hook(approve, hooks)
    tools: list[Tool] = [WebSearchTool()]
    if include_workspace_tools:
        tools.extend(
            [
                ListFilesTool(workspace, permission_engine=permission_engine),
                ReadFileTool(workspace, permission_engine=permission_engine),
                SearchCodeTool(workspace, permission_engine=permission_engine),
                WebFetchTool(),
                GitStatusTool(workspace),
            ]
        )
    if include_write_tools:
        tools.extend(
            [
                WriteFileTool(workspace, approval_mode=approval_mode, approve=approve, permission_engine=permission_engine, checkpoint_store=checkpoint_store, checkpoint_id=checkpoint_id, lease_callback=lease_callback),
                AppendFileTool(workspace, approval_mode=approval_mode, approve=approve, permission_engine=permission_engine, checkpoint_store=checkpoint_store, checkpoint_id=checkpoint_id, lease_callback=lease_callback),
                EditFileTool(workspace, approval_mode=approval_mode, approve=approve, permission_engine=permission_engine, checkpoint_store=checkpoint_store, checkpoint_id=checkpoint_id, lease_callback=lease_callback),
                ApplyPatchTool(workspace, approval_mode=approval_mode, approve=approve, permission_engine=permission_engine, checkpoint_store=checkpoint_store, checkpoint_id=checkpoint_id, lease_callback=lease_callback),
            ]
        )
    if include_shell_tool:
        tools.append(ShellTool(workspace, approval_mode=approval_mode, approve=approve, permission_engine=permission_engine))
    if skill_library is not None and include_skill_install_tool:
        tools.append(InstallSkillTool(workspace, skill_library, approval_mode=approval_mode, approve=approve))
        tools.append(FetchSkillTool(workspace, skill_library))
    if include_load_skill_tool and skill_library is not None and skill_library.skills:
        tools.append(LoadSkillTool(skill_library, workspace=workspace))
    if extra_tools:
        tools.extend(extra_tools)
    if tool_wrapper is not None:
        tools = [tool_wrapper(tool) for tool in tools]
    tools = [CapabilityGuardedTool(tool, capability_profile) for tool in tools]
    return ToolRegistry(tools, logger=logger)


def build_loop(
    workspace: Path,
    args: argparse.Namespace,
    logger: XiaomingLogger | None = None,
    session_recorder: object | None = None,
    checkpoint_store: CheckpointStore | None = None,
    checkpoint_id=None,
    approve: Callable[[str], bool] = approve_action,
    lease_callback: WriteLeaseCallback | None = None,
    extra_tools: list[Tool] | None = None,
    include_skill_install_tool: bool = True,
    include_load_skill_tool: bool = True,
    include_workspace_tools: bool = True,
    include_write_tools: bool = True,
    include_shell_tool: bool = True,
    tool_wrapper: Callable[[Tool], Tool] | None = None,
    pending_worker_questions: Callable[[], str | None] | None = None,
    runtime_context_provider: Callable[[], str | None] | None = None,
    role: str = "worker",
    capability_profile: CapabilityProfile = "full",
    include_skill_context: bool = True,
    hooks: HookManager | None = None,
    instructions_override: str | None = None,
) -> AgentLoop:
    config = load_config(
        workspace,
        {
            "provider": args.provider,
            "model": args.model,
            "approval_mode": args.approval_mode,
            "permission_mode": getattr(args, "permission_mode", None),
            "max_turns": args.max_turns,
            "model_timeout_seconds": getattr(args, "model_timeout_seconds", None),
            "stream": getattr(args, "stream", None),
        },
    )
    if config.model.provider == "deepseek":
        provider = DeepSeekProvider(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("DEEPSEEK_BASE_URL"))
    else:
        provider = OpenAIProvider(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
    skill_library = SkillLibrary.discover(config.workspace.root)
    permission_engine = PermissionEngine(
        config.workspace.root,
        mode=PermissionMode(config.agent.permission_mode),
        rules=load_project_rules(config.workspace.root),
    )
    workspace_hooks = hooks if hooks is not None else load_workspace_hooks(config.workspace.root, logger=logger)
    effective_hooks = _with_tool_capability_hook(workspace_hooks, capability_profile)
    instructions = instructions_override or build_instructions(config.workspace.root, role=role)
    return AgentLoop(
        provider=provider,
        registry=build_registry(
            config.workspace.root,
            config.agent.approval_mode,
            permission_engine,
            skill_library,
            logger,
            checkpoint_store,
            checkpoint_id,
            lease_callback,
            approve=approve,
            extra_tools=extra_tools,
            include_skill_install_tool=include_skill_install_tool,
            include_load_skill_tool=include_load_skill_tool,
            include_workspace_tools=include_workspace_tools,
            include_write_tools=include_write_tools,
            include_shell_tool=include_shell_tool,
            tool_wrapper=tool_wrapper,
            hooks=effective_hooks,
            capability_profile=capability_profile,
        ),
        instructions=instructions,
        instructions_provider=(lambda: instructions_override) if instructions_override else (lambda: build_instructions(config.workspace.root, role=role)),
        model=config.model.model,
        temperature=config.model.temperature,
        max_output_tokens=config.model.max_output_tokens,
        max_turns=config.agent.max_turns,
        model_timeout_seconds=config.agent.model_timeout_seconds,
        stream=config.agent.stream,
        stream_idle_timeout_seconds=config.agent.stream_idle_timeout_seconds,
        skill_library=skill_library if include_skill_context else None,
        logger=logger,
        session_recorder=session_recorder,
        pending_worker_questions=pending_worker_questions,
        runtime_context_provider=runtime_context_provider,
        hooks=effective_hooks,
    )


def _approval_with_permission_hook(approve: Callable[[str], bool], hooks: HookManager | None) -> Callable[[str], bool]:
    if hooks is None or not hooks.has_hooks():
        return approve

    def wrapped(action: str) -> bool:
        result = hooks.run("PermissionRequest", {"action": action})
        if result.decision == "allow":
            return True
        if result.decision == "deny" or not result.continue_:
            return False
        return approve(action)

    return wrapped


@dataclass
class ChatRuntime:
    workspace: Path
    args: argparse.Namespace
    loop_factory: Callable = build_loop
    coordinator_factory: Callable | None = None

    def __post_init__(self) -> None:
        self.logger = XiaomingLogger.create(self.workspace)
        self.logger.info("cli_started", workspace=str(self.workspace))
        self.session_store = SessionStore(self.workspace)
        self.checkpoint_store = CheckpointStore(self.workspace)
        self.active_checkpoint_id: str | None = None
        self.async_coordinator: AsyncCoordinator | None = None
        self.async_notice_handler: Callable[[CoordinatorNotice], None] = _print_async_notice
        self.foreground_task: ForegroundTask | None = None
        self.active_user_input: str = ""
        self.session_record, self.resumed_existing_session = self._select_session()
        self.session = self._load_session(self.session_record)
        self._ensure_bootstrap_contexts()
        self.loop = self._build_loop()

    def _select_session(self) -> tuple[SessionRecord, bool]:
        config = self.config
        resume_id = getattr(self.args, "resume_session_id", None)
        if resume_id:
            record = self.session_store.get(resume_id)
            if record is None:
                raise RuntimeError(f"unknown session: {resume_id}")
            return record, True
        if not getattr(self.args, "new_session", False):
            record = self.session_store.latest_for_workspace()
            if record is not None:
                return record, True
        return self.session_store.create(title="New session", provider=config.model.provider, model=config.model.model), False

    def _load_session(self, record: SessionRecord) -> Session:
        return rehydrate_session(self.session_store.read_events(record.id), session_id=record.id)

    def _build_loop(self):
        try:
            if self.loop_factory is build_loop:
                return self.loop_factory(
                    self.workspace,
                    self.args,
                    self.logger,
                    self.session_store,
                    self.checkpoint_store,
                    lambda: self.active_checkpoint_id,
                    extra_tools=build_universal_runtime_tools(
                        coordinator_getter=lambda: self.async_coordinator,
                        talk_callback=unavailable_talk_callback,
                        turn_context_getter=lambda: self.active_user_input,
                    ),
                    pending_worker_questions=self._pending_worker_questions_for_input,
                    include_skill_context=False,
                    runtime_context_provider=self._async_context_summary,
                    role="orchestrator",
                    capability_profile=self.current_capability_profile,
                )
            return self.loop_factory(
                self.workspace,
                self.args,
                self.logger,
                self.session_store,
                self.checkpoint_store,
                lambda: self.active_checkpoint_id,
            )
        except TypeError:
            try:
                return self.loop_factory(self.workspace, self.args, self.logger)
            except TypeError:
                return self.loop_factory(self.workspace, self.args)

    @property
    def config(self):
        return load_config(
            self.workspace,
            {
                "provider": self.args.provider,
                "model": self.args.model,
                "approval_mode": self.args.approval_mode,
                "permission_mode": getattr(self.args, "permission_mode", None),
                "max_turns": self.args.max_turns,
                "model_timeout_seconds": getattr(self.args, "model_timeout_seconds", None),
                "stream": getattr(self.args, "stream", None),
            },
        )

    def rebuild(self) -> None:
        self.loop = self._build_loop()
        self.start_new_session()

    def reload_skills(self) -> None:
        self.loop = self._build_loop()

    def start_new_session(self) -> None:
        config = self.config
        self.session_record = self.session_store.create(title="New session", provider=config.model.provider, model=config.model.model)
        self.resumed_existing_session = False
        self.session = Session(session_id=self.session_record.id)
        self._ensure_bootstrap_contexts()

    def resume_session(self, session_id: str) -> bool:
        record = self.session_store.get(session_id)
        if record is None:
            return False
        self.session_record = record
        self.resumed_existing_session = True
        self.session = self._load_session(record)
        self._ensure_bootstrap_contexts()
        self.loop = self._build_loop()
        return True

    def _ensure_bootstrap_contexts(self) -> None:
        for context in discover_bootstrap_contexts(self.workspace):
            existing = self.session.bootstrap_contexts.get(context.source)
            if existing is not None and existing.content_hash == context.content_hash:
                continue
            self.session.remember_bootstrap_context(context)
            self.session_store.append(self.session.session_id, "bootstrap_context", context.to_payload())
            self.logger.info("bootstrap_context_injected", plugin=context.plugin_name, source=context.source, path=context.path, content_hash=context.content_hash)

    def sessions_text(self, limit: int = 10) -> str:
        records = self.session_store.list()[:limit]
        if not records:
            return "No sessions."
        return "\n".join(f"{record.id}  {record.updated_at}  {record.title}" for record in records)

    def session_text(self) -> str:
        record = self.session_record
        return (
            f"Session: {record.id}\n"
            f"Title: {record.title}\n"
            f"Turns: {record.turns}\n"
            f"Started: {record.created_at}\n"
            f"Updated: {record.updated_at}\n"
            f"Path: {record.path}"
        )

    def checkpoints_text(self, limit: int = 10) -> str:
        records = self.checkpoint_store.list()[:limit]
        if not records:
            return "No checkpoints."
        return "\n".join(f"{record.id}  {record.created_at}  {record.prompt}" for record in records)

    def restore_checkpoint(self, checkpoint_id: str | None = None) -> str:
        record = self.checkpoint_store.get(checkpoint_id) if checkpoint_id else self.checkpoint_store.latest()
        if record is None:
            return "No checkpoint to restore."
        result = self.checkpoint_store.restore(record.id)
        return f"Restored checkpoint: {record.id}\nRestored files: {len(result.restored)}\nDeleted files: {len(result.deleted)}"

    def startup_session_text(self) -> str:
        action = "Resumed session" if self.resumed_existing_session else "Started new session"
        return (
            f"{action}: {self.session_record.id}\n"
            f"Title: {self.session_record.title}\n"
            f"Session items: {self.session.item_count}"
        )

    def install_skill(self, url: str) -> str:
        result = install_skill_from_url(url, self.workspace)
        self.reload_skills()
        return (
            f"Installed skill: {result.name}\n"
            f"Destination: {result.destination}\n"
            f"Files: {result.files}\n"
            f"Bytes: {result.bytes_written}\n"
            "Skills reloaded."
        )

    def try_update(self, **updates: str | None) -> tuple[bool, str | None]:
        previous = {
            "provider": self.args.provider,
            "model": self.args.model,
            "approval_mode": self.args.approval_mode,
            "permission_mode": getattr(self.args, "permission_mode", None),
            "max_turns": self.args.max_turns,
            "model_timeout_seconds": getattr(self.args, "model_timeout_seconds", None),
            "stream": getattr(self.args, "stream", None),
        }
        for key, value in updates.items():
            setattr(self.args, key, value)
        try:
            self.rebuild()
        except Exception as exc:
            for key, value in previous.items():
                setattr(self.args, key, value)
            self.rebuild()
            return False, str(exc)
        return True, None

    def status_text(self) -> str:
        config = self.config
        return (
            f"Provider: {config.model.provider}\n"
            f"Model: {config.model.model}\n"
            f"Approval: {config.agent.approval_mode}\n"
            f"Permission mode: {config.agent.permission_mode}\n"
            f"Max turns: {config.agent.max_turns}\n"
            f"Model timeout: {config.agent.model_timeout_seconds:g}s\n"
            f"Stream: {'on' if config.agent.stream else 'off'}\n"
            f"Session: {self.session_record.id}\n"
            f"Session items: {self.session.item_count}\n"
            f"Log: {self.logger.path}"
        )

    def context_text(self) -> str:
        if isinstance(self.loop, AgentLoop):
            return self.loop.context_status(self.session)
        return f"Session items: {self.session.item_count}"

    def compact_context(self) -> str:
        if not isinstance(self.loop, AgentLoop):
            return "Context compaction is unavailable for this runtime."
        result = self.loop.compact_context(self.session, reason="manual")
        return result

    def dream_context(self) -> str:
        if not hasattr(self.loop, "dream_context"):
            return "Dream mode is unavailable for this runtime."
        return self.loop.dream_context(self.session)

    def skills_text(self) -> str:
        if isinstance(self.loop, AgentLoop) and self.loop.skill_library is not None:
            return self.loop.skill_library.list_text()
        return SkillLibrary.discover(self.workspace).list_text()

    def talk_to_peer(self, peer_id: str, message: str) -> str:
        if self.async_coordinator is None:
            return "Peer talk is only available in xiaoming-cli async runtime mode."
        result = self.async_coordinator.talk_to_peer(peer_id, message)
        if result.status == "success":
            return result.output
        return f"peer talk failed: {result.error or result.output}"

    def should_use_async_chat(self) -> bool:
        return self.loop_factory is build_loop

    def build_async_coordinator(self, on_notice: Callable[[CoordinatorNotice], None]) -> AsyncCoordinator:
        config = self.config
        coordinator_config = CoordinatorConfig(
            workspace=self.workspace,
            provider=config.model.provider,
            model=config.model.model,
            approval_mode=config.agent.approval_mode,
            permission_mode=config.agent.permission_mode,
            max_turns=config.agent.max_turns,
            model_timeout_seconds=config.agent.model_timeout_seconds,
            stream=config.agent.stream,
        )
        if self.coordinator_factory is not None:
            return self.coordinator_factory(coordinator_config, on_notice)
        scheduler_provider = DeepSeekProvider(api_key=os.environ.get("DEEPSEEK_API_KEY"), base_url=os.environ.get("DEEPSEEK_BASE_URL")) if config.model.provider == "deepseek" else OpenAIProvider(api_key=os.environ.get("OPENAI_API_KEY"), base_url=os.environ.get("OPENAI_BASE_URL"))
        scheduler = LLMScheduler(scheduler_provider, config.model.model)
        responder = LLMMessagingResponder(scheduler_provider, config.model.model)
        question_decider = LLMWorkerQuestionDecider(scheduler_provider, config.model.model)
        return AsyncCoordinator(
            coordinator_config,
            scheduler=scheduler,
            responder=responder,
            question_decider=question_decider,
            on_notice=on_notice,
            session_provider=lambda: self.session,
        )

    def _pending_worker_questions_for_input(self, user_input: str) -> str | None:
        if self.async_coordinator is None:
            return None
        return self.async_coordinator.pending_questions_text()

    def current_capability_profile(self) -> str:
        if self.foreground_task is not None and self.foreground_task.status == "running":
            return "foreground"
        return "orchestrator"

    def begin_foreground_task(self, user_input: str) -> ForegroundTask:
        task = ForegroundTask(task_name=_foreground_task_name(user_input), original_message=user_input)
        self.foreground_task = task
        self.logger.info("foreground_task_started", task_name=task.task_name)
        return task

    def finish_foreground_task(self, status: str = "completed") -> None:
        if self.foreground_task is None:
            return
        self.foreground_task.status = status
        self.logger.info("foreground_task_finished", task_name=self.foreground_task.task_name, status=status)
        self.foreground_task = None

    def move_foreground_task_to_background(self) -> str:
        if self.foreground_task is None:
            return "当前没有正在前台执行的任务。"
        if self.async_coordinator is None:
            self.finish_foreground_task("failed")
            return "后台 coordinator 未运行，无法将当前任务转为后台。"
        task = self.foreground_task
        message = _foreground_background_message(task.original_message)
        result = self.async_coordinator.schedule_background_task(TaskSpec.from_request(message, title=task.task_name))
        if result.status != "success":
            task.status = "failed"
            return f"当前任务转为后台失败：{result.error or result.output or 'unknown error'}"
        task.status = "moved_to_background"
        self.logger.info("foreground_task_moved_to_background", task_name=task.task_name, result=result.output)
        self.foreground_task = None
        return f"当前任务已转为后台继续处理：{task.task_name}"

    def _async_context_summary(self) -> str | None:
        if self.async_coordinator is None:
            return None
        context_summary = getattr(self.async_coordinator, "context_summary", None)
        if context_summary is None:
            return None
        return context_summary()


def _foreground_task_name(user_input: str) -> str:
    cleaned = " ".join(user_input.split())
    return cleaned[:40] or "前台任务"


def _foreground_background_message(user_input: str) -> str:
    return (
        "继续完成刚才的前台任务。用户在处理过程中提交了新输入，所以该任务转为后台执行。"
        "你拥有 forked conversation context，请从已有上下文和工作区状态继续；不要从头重做。"
        "需要确认时使用 talk。\n\n"
        f"原任务：{user_input}"
    )


def _orchestrator_prompt() -> str:
    return """
In this runtime, act as the user's primary conversation partner and task coordinator.

Rules:
- Answer simple questions directly.
- Handle lightweight tasks yourself — inspect files, make small edits, run quick commands, load skills. You have full tool access.
- For independent, large, long-running, multi-file, dependency/setup, skill installation, repository cloning, or otherwise substantial work, briefly tell the user it will run in the background and call schedule_background_task.
- Use your judgment: a single-file edit or a quick search is fine to do inline; a multi-step refactor or a new feature should go to a background worker.
- Once you decide to schedule background work, call schedule_background_task immediately. Let the worker inspect files, run commands, and choose implementation details.
- Do not claim that background work is complete until background task status or notices say so.
- If a foreground task is interrupted by a new user message, the runtime may move the original task to a background worker. Continue handling the new user message normally.
- When calling schedule_background_task, pass only message and optionally task_name. Put any user-stated constraints in message.
- For progress questions, call background_tasks_status at most once in that turn and answer from the snapshot.
- If the user explicitly asks you to wait, follow, or watch a specific background task, use follow_background_task with the task_id. If the task_id is unknown, call background_tasks_status once first.
- When the user asks a natural language question or follow-up for an existing worker or peer, call talk_to_peer with the exact peer/task id and the user's message. Talk is transparent and does not mean task completion.
- To cancel a background task, call cancel_background_task with the exact task_id. If the task_id is unknown, call background_tasks_status first.
- For multiple cancellations, call cancel_background_task once per task. Do not claim cancellation unless the tool succeeds.
- If replacing a background task, cancel the old task first, then schedule the replacement.
- For current web information such as news or weather, use web_search when needed.
- If a background worker asks a question, only call reply_mailbox_message when the user's message is a clear answer to that worker question.
- If the user delegates future similar decisions, include a concise authorization_note in reply_mailbox_message for that worker. Otherwise leave authorization_note empty.
- If the user's message is a new request, status question, unrelated message, or ambiguous answer, do not call reply_mailbox_message. Continue the conversation normally, and ask a concise clarification only if the user appears to be trying to answer the worker question.
- Before calling any tool, briefly state what you are about to do and why in one short sentence.
""".strip()


def run_chat(runtime_or_loop) -> int:
    enable_line_editing()
    async_notices = AsyncNoticeBuffer()
    if isinstance(runtime_or_loop, ChatRuntime):
        runtime = runtime_or_loop
        runtime.async_notice_handler = async_notices.enqueue
    else:
        runtime = None
        loop = runtime_or_loop
        session = Session()
    print("Xiaoming chat. Type 'exit' or 'quit' to leave. Type '/' for commands.")
    if runtime is not None:
        print(runtime.startup_session_text())
        if runtime.should_use_async_chat():
            runtime.async_coordinator = runtime.build_async_coordinator(runtime.async_notice_handler)
            runtime.async_coordinator.start()
    async_notices.start()
    queued_user_inputs: list[str] = []
    while True:
        async_notices.flush()
        if queued_user_inputs:
            user_input = queued_user_inputs.pop(0).strip()
        else:
            try:
                user_input = _read_user_input("xiaoming> ", async_notices).strip()
            except KeyboardInterrupt:
                async_notices.set_input_active(False)
                discard_pending_terminal_input()
                if runtime is not None and runtime.async_coordinator is not None:
                    print()
                    print(runtime.async_coordinator.cancel_current())
                    continue
                raise
            except EOFError:
                async_notices.set_input_active(False)
                async_notices.stop()
                print()
                if runtime is not None and runtime.async_coordinator is not None:
                    runtime.async_coordinator.stop()
                return 0
        if not user_input:
            continue
        if user_input in {"exit", "quit"}:
            async_notices.stop()
            if runtime is not None and runtime.async_coordinator is not None:
                runtime.async_coordinator.stop()
            return 0
        if user_input in {"/exit", "/quit"}:
            async_notices.stop()
            if runtime is not None and runtime.async_coordinator is not None:
                runtime.async_coordinator.stop()
            return 0
        if user_input in {"/", "/help"}:
            print(_help_text())
            continue
        if user_input == "/clear":
            if runtime is None:
                session.clear()
            else:
                runtime.session.clear()
            print("Context cleared.")
            continue
        if user_input == "/status":
            if runtime is None:
                print(f"Session items: {session.item_count}")
            elif runtime.async_coordinator is not None:
                print(runtime.status_text())
                print(runtime.async_coordinator.status_text())
            else:
                print(runtime.status_text())
            continue
        if user_input == "/context":
            if runtime is None:
                print(f"Session items: {session.item_count}")
            else:
                print(runtime.context_text())
            continue
        if user_input == "/compact":
            if runtime is None:
                print("Context compaction is only available in xiaoming-cli runtime mode.")
            else:
                try:
                    print(runtime.compact_context())
                except Exception as exc:
                    runtime.logger.error("manual_context_compaction_failed", exc=exc)
                    print(f"Context compaction failed: {exc}")
            continue
        if user_input == "/dream":
            if runtime is None:
                print("Dream mode is only available in xiaoming-cli runtime mode.")
            else:
                try:
                    print(runtime.dream_context())
                except Exception as exc:
                    runtime.logger.error("manual_dream_failed", exc=exc)
                    print(f"Dream failed: {exc}")
            continue
        if user_input == "/tasks":
            if runtime is None or runtime.async_coordinator is None:
                print("Async tasks are only available in xiaoming-cli async runtime mode.")
            else:
                print(runtime.async_coordinator.tasks_text())
            continue
        if user_input.startswith("/talk"):
            if runtime is None or runtime.async_coordinator is None:
                print("Peer talk is only available in xiaoming-cli async runtime mode.")
            else:
                parts = user_input.split(maxsplit=2)
                if len(parts) < 3:
                    print("Usage: /talk <peer-id> <message>")
                else:
                    print(runtime.talk_to_peer(parts[1], parts[2]))
            continue
        if user_input == "/quiet":
            if runtime is not None and runtime.async_coordinator is not None:
                runtime.async_coordinator.set_quiet(True)
                print("Background notices reduced.")
            else:
                print("Quiet mode is only available in xiaoming-cli async runtime mode.")
            continue
        if user_input == "/verbose":
            if runtime is not None and runtime.async_coordinator is not None:
                runtime.async_coordinator.set_quiet(False)
                print("Background notices restored.")
            else:
                print("Verbose mode is only available in xiaoming-cli async runtime mode.")
            continue
        if user_input.startswith("/cancel"):
            if runtime is None or runtime.async_coordinator is None:
                print("Cancel is only available in xiaoming-cli async runtime mode.")
            elif user_input == "/cancel all":
                print(runtime.async_coordinator.cancel_all())
            else:
                print(runtime.async_coordinator.cancel_current())
            continue
        if user_input == "/session":
            if runtime is None:
                print(f"Session items: {session.item_count}")
            else:
                print(runtime.session_text())
            continue
        if user_input == "/sessions":
            if runtime is None:
                print("Sessions are only available in xiaoming-cli runtime mode.")
            else:
                print(runtime.sessions_text())
            continue
        if user_input == "/checkpoints":
            if runtime is None:
                print("Checkpoints are only available in xiaoming-cli runtime mode.")
            else:
                print(runtime.checkpoints_text())
            continue
        if user_input.startswith("/rewind"):
            if runtime is None:
                print("Checkpoints are only available in xiaoming-cli runtime mode.")
            else:
                parts = user_input.split()
                print(runtime.restore_checkpoint(parts[1] if len(parts) > 1 else None))
            continue
        if user_input == "/new":
            if runtime is None:
                session.clear()
            else:
                runtime.start_new_session()
            print("Started new session.")
            continue
        if user_input == "/skills":
            if runtime is None:
                print("Skills are only available in xiaoming-cli runtime mode.")
            else:
                print(runtime.skills_text())
            continue
        if user_input == "/logs":
            if runtime is None:
                print("Logs are only available in xiaoming-cli runtime mode.")
            else:
                print(runtime.logger.path)
            continue
        if runtime is not None and _handle_skill_command(runtime, user_input):
            continue
        if runtime is not None and _handle_config_command(runtime, user_input):
            continue
        try:
            if runtime is None:
                _print_answer(run_loop_with_progress(loop, user_input, session=session))
            else:
                result = _run_foreground_turn(runtime, user_input, async_notices)
                _print_answer(result.answer)
                if result.pending_user_input:
                    queued_user_inputs.insert(0, result.pending_user_input)
        except KeyboardInterrupt:
            discard_pending_terminal_input()
            if runtime is not None:
                runtime.logger.error("cli_turn_interrupted", user_input=user_input)
            print("\nInterrupted current operation.")
            if runtime is not None:
                print("Run /rewind to restore changes from this turn.")
        except Exception as exc:
            if runtime is not None:
                runtime.logger.error("cli_turn_exception", exc=exc, user_input=user_input)
            print(f"Error: {exc}")
        finally:
            if runtime is not None:
                runtime.active_checkpoint_id = None
            async_notices.flush()


def _run_foreground_turn(runtime: ChatRuntime, user_input: str, async_notices: "AsyncNoticeBuffer") -> ForegroundTurnResult:
    checkpoint = runtime.checkpoint_store.create(runtime.session.session_id, user_input)
    runtime.active_checkpoint_id = checkpoint.id
    runtime.active_user_input = user_input
    cancel_requested = threading.Event()
    done = threading.Event()
    result: dict[str, object] = {}

    def _target() -> None:
        try:
            result["answer"] = run_loop_with_progress(
                runtime.loop,
                user_input,
                session=runtime.session,
                should_cancel=cancel_requested.is_set,
            )
        except BaseException as exc:
            result["exception"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    pending_user_input = _read_user_input_until_done("xiaoming> ", async_notices, done)
    if pending_user_input is not None and pending_user_input.strip() and runtime.foreground_task is not None:
        cancel_requested.set()
        print(runtime.move_foreground_task_to_background())
    done.wait()
    runtime.active_checkpoint_id = None
    runtime.active_user_input = ""
    if "exception" in result:
        runtime.finish_foreground_task("failed")
        raise result["exception"]  # type: ignore[misc]
    if pending_user_input is not None and pending_user_input.strip():
        return ForegroundTurnResult(str(result.get("answer") or ""), pending_user_input.strip())
    runtime.finish_foreground_task("completed")
    return ForegroundTurnResult(str(result.get("answer") or ""))


def run_loop_with_progress(loop, task: str, session: Session | None, should_cancel: Callable[[], bool] | None = None) -> str:
    signature = inspect.signature(loop.run)
    kwargs = {"session": session}
    if "on_event" in signature.parameters:
        kwargs["on_event"] = _print_progress
    if "should_cancel" in signature.parameters:
        kwargs["should_cancel"] = should_cancel
    return loop.run(task, **kwargs)


def _print_answer(answer: str) -> None:
    if answer:
        _safe_print(answer)


def _print_progress(message: str | ProgressEvent) -> None:
    if isinstance(message, ProgressEvent):
        if message.kind == "text_delta":
            # Stream deltas directly — they build up on the current line
            # and should not trigger prompt save/restore on each character.
            sys.stdout.write(message.message)
            sys.stdout.write(message.end)
            sys.stdout.flush()
            return
        _safe_print(f"[xiaoming] {message.message}")
        return
    _safe_print(f"[xiaoming] {message}")


def _print_async_notice(notice: CoordinatorNotice) -> None:
    _safe_print(f"[xiaoming] {notice.message}", prefix="\n")


def _safe_print(text: str, end: str = "\n", prefix: str = "") -> None:
    """Print output without disrupting the readline input prompt.

    When readline callback mode is active, this saves the current input buffer,
    clears the prompt area, prints the output, then redraws the prompt and
    tells readline to refresh its display.
    """
    if not sys.stdin.isatty():
        sys.stdout.write(f"{prefix}{text}{end}")
        sys.stdout.flush()
        return
    try:
        import readline
        saved = readline.get_line_buffer()
        # \r -> column 0, \033[0J -> clear cursor to end of screen
        sys.stdout.write(f"\r\033[0J{prefix}{text}{end}")
        # Always redraw the prompt line at the bottom
        sys.stdout.write(f"xiaoming> {saved}")
        sys.stdout.flush()
        readline.redisplay()
    except Exception:
        sys.stdout.write(f"{prefix}{text}{end}")
        sys.stdout.flush()


def _read_user_input(prompt: str, async_notices: "AsyncNoticeBuffer") -> str:
    if not sys.stdin.isatty():
        async_notices.set_input_active(True)
        try:
            return input(prompt)
        finally:
            async_notices.set_input_active(False)
    try:
        import readline
    except Exception:
        async_notices.set_input_active(True)
        try:
            return input(prompt)
        finally:
            async_notices.set_input_active(False)
    if not hasattr(readline, "callback_handler_install") or not hasattr(readline, "callback_read_char"):
        async_notices.set_input_active(True)
        try:
            return input(prompt)
        finally:
            async_notices.set_input_active(False)

    lines: list[str | None] = []

    def _line_ready(line: str | None) -> None:
        lines.append(line)

    readline.callback_handler_install(prompt, _line_ready)
    async_notices.set_input_active(True)
    try:
        while not lines:
            async_notices.flush_if_input_empty(redraw_prompt=True)
            readable, _, _ = select.select([sys.stdin], [], [], async_notices.POLL_SECONDS)
            if readable:
                readline.callback_read_char()
        if lines[0] is None:
            raise EOFError
        return lines[0] or ""
    finally:
        async_notices.set_input_active(False)
        try:
            readline.callback_handler_remove()
        except Exception:
            pass


def _read_user_input_until_done(prompt: str, async_notices: "AsyncNoticeBuffer", done: threading.Event) -> str | None:
    if not sys.stdin.isatty():
        done.wait()
        return None
    try:
        import readline
    except Exception:
        print(prompt, end="", flush=True)
        while not done.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], async_notices.POLL_SECONDS)
            if readable:
                line = sys.stdin.readline()
                if line == "":
                    raise EOFError
                return line.rstrip("\n")
        return None
    if not hasattr(readline, "callback_handler_install") or not hasattr(readline, "callback_read_char"):
        print(prompt, end="", flush=True)
        while not done.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], async_notices.POLL_SECONDS)
            if readable:
                line = sys.stdin.readline()
                if line == "":
                    raise EOFError
                return line.rstrip("\n")
        return None

    lines: list[str | None] = []

    def _line_ready(line: str | None) -> None:
        lines.append(line)

    readline.callback_handler_install(prompt, _line_ready)
    async_notices.set_input_active(True)
    try:
        while not lines and not done.is_set():
            async_notices.flush_if_input_empty(redraw_prompt=True)
            readable, _, _ = select.select([sys.stdin], [], [], async_notices.POLL_SECONDS)
            if readable:
                readline.callback_read_char()
        if not lines:
            return None
        if lines[0] is None:
            raise EOFError
        return lines[0] or ""
    finally:
        async_notices.set_input_active(False)
        try:
            readline.callback_handler_remove()
        except Exception:
            pass


class AsyncNoticeBuffer:
    POLL_SECONDS = 0.1

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._notices: list[CoordinatorNotice] = []
        self._input_active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def enqueue(self, notice: CoordinatorNotice) -> None:
        with self._lock:
            self._notices.append(notice)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._watch_empty_input, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def set_input_active(self, active: bool) -> None:
        with self._lock:
            self._input_active = active

    def flush(self, redraw_prompt: bool = False) -> None:
        with self._lock:
            notices = self._notices
            self._notices = []
        for notice in notices:
            _print_async_notice(notice)
        if notices and redraw_prompt:
            print("xiaoming> ", end="", flush=True)

    def flush_if_input_empty(self, redraw_prompt: bool = False) -> bool:
        with self._lock:
            should_try_flush = self._input_active and bool(self._notices)
        if should_try_flush and _readline_buffer_empty():
            self.flush(redraw_prompt=redraw_prompt)
            return True
        return False

    def _watch_empty_input(self) -> None:
        while not self._stop.wait(self.POLL_SECONDS):
            self.flush_if_input_empty(redraw_prompt=True)


def _readline_buffer_empty() -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        import readline

        return not readline.get_line_buffer()
    except Exception:
        return False


def _handle_config_command(runtime: ChatRuntime, user_input: str) -> bool:
    parts = user_input.split()
    if not parts:
        return False
    command = parts[0]
    if command == "/model":
        if len(parts) == 1:
            config = runtime.config
            print(f"Provider: {config.model.provider}\nModel: {config.model.model}")
            return True
        if len(parts) != 3:
            print("Usage: /model <openai|deepseek> <model>")
            return True
        provider, model = parts[1], parts[2]
        if provider not in {"openai", "deepseek"}:
            print("Provider must be openai or deepseek.")
            return True
        ok, error = runtime.try_update(provider=provider, model=model)
        if not ok:
            print(f"Error switching model: {error}")
            return True
        _restart_async_coordinator(runtime)
        print(f"Model switched to {provider} {model}. Context cleared.")
        return True
    if command == "/provider":
        if len(parts) != 2:
            print("Usage: /provider <openai|deepseek>")
            return True
        provider = parts[1]
        if provider not in {"openai", "deepseek"}:
            print("Provider must be openai or deepseek.")
            return True
        ok, error = runtime.try_update(provider=provider, model=None)
        if not ok:
            print(f"Error switching provider: {error}")
            return True
        _restart_async_coordinator(runtime)
        config = runtime.config
        print(f"Provider switched to {config.model.provider} {config.model.model}. Context cleared.")
        return True
    if command == "/approval":
        if len(parts) != 2:
            print("Usage: /approval <suggest|auto_edit|full_auto>")
            return True
        mode = parts[1]
        if mode not in {"suggest", "auto_edit", "full_auto"}:
            print("Approval mode must be suggest, auto_edit, or full_auto.")
            return True
        ok, error = runtime.try_update(approval_mode=mode)
        if not ok:
            print(f"Error switching approval mode: {error}")
            return True
        _restart_async_coordinator(runtime)
        print(f"Approval mode set to {mode}. Context cleared.")
        return True
    if command == "/permission-mode":
        if len(parts) == 1:
            print(f"Permission mode: {runtime.config.agent.permission_mode}")
            return True
        if len(parts) != 2:
            print("Usage: /permission-mode <default|plan|accept_edits|auto|bypass>")
            return True
        mode = parts[1]
        if mode not in {"default", "plan", "accept_edits", "auto", "bypass"}:
            print("Permission mode must be default, plan, accept_edits, auto, or bypass.")
            return True
        ok, error = runtime.try_update(permission_mode=mode)
        if not ok:
            print(f"Error switching permission mode: {error}")
            return True
        _restart_async_coordinator(runtime)
        print(f"Permission mode set to {mode}. Context cleared.")
        return True
    if command == "/permissions":
        rules = load_project_rules(runtime.workspace)
        if not rules:
            print(f"No project permission rules. File: {permissions_path(runtime.workspace)}")
            return True
        print("\n".join(f"{rule.behavior.value} {rule.tool}({rule.pattern}) [{rule.source}]" for rule in rules))
        return True
    if command in {"/allow", "/deny", "/ask"}:
        if len(parts) < 2:
            print(f"Usage: {command} Tool(pattern)")
            return True
        parsed = _parse_permission_rule(" ".join(parts[1:]), PermissionBehavior(command.removeprefix("/")))
        if parsed is None:
            print(f"Usage: {command} Tool(pattern)")
            return True
        add_project_rule(runtime.workspace, parsed)
        runtime.reload_skills()
        print(f"Added project rule: {parsed.behavior.value} {parsed.tool}({parsed.pattern})")
        return True
    if command == "/model-timeout":
        if len(parts) != 2:
            print("Usage: /model-timeout <seconds>")
            return True
        try:
            seconds = float(parts[1])
        except ValueError:
            print("Model timeout must be a number of seconds.")
            return True
        ok, error = runtime.try_update(model_timeout_seconds=seconds)
        if not ok:
            print(f"Error switching model timeout: {error}")
            return True
        _restart_async_coordinator(runtime)
        print(f"Model timeout set to {seconds:g}s. Context cleared.")
        return True
    if command == "/stream":
        if len(parts) == 1:
            print(f"Stream: {'on' if runtime.config.agent.stream else 'off'}")
            return True
        if len(parts) != 2 or parts[1] not in {"on", "off"}:
            print("Usage: /stream on|off")
            return True
        enabled = parts[1] == "on"
        ok, error = runtime.try_update(stream=enabled)
        if not ok:
            print(f"Error switching stream mode: {error}")
            return True
        _restart_async_coordinator(runtime)
        print(f"Stream {'enabled' if enabled else 'disabled'}. Context cleared.")
        return True
    if command == "/resume":
        if len(parts) != 2:
            print("Usage: /resume <session-id>")
            return True
        if not runtime.resume_session(parts[1]):
            print(f"Unknown session: {parts[1]}")
            return True
        print(f"Resumed session: {parts[1]}")
        return True
    return False


def _handle_skill_command(runtime: ChatRuntime, user_input: str) -> bool:
    parts = user_input.split(maxsplit=2)
    if not parts or parts[0] != "/skill":
        return False
    if len(parts) == 2 and parts[1] == "reload":
        runtime.reload_skills()
        print("Skills reloaded.")
        return True
    if len(parts) == 3 and parts[1] == "install":
        try:
            print(runtime.install_skill(parts[2]))
        except SkillInstallError as exc:
            print(f"Error installing skill: {exc}")
        return True
    print("Usage: /skill reload | /skill install <github-tree-url>")
    return True


def _restart_async_coordinator(runtime: ChatRuntime) -> None:
    if runtime.async_coordinator is None:
        return
    runtime.async_coordinator.stop()
    runtime.async_coordinator = runtime.build_async_coordinator(runtime.async_notice_handler)
    runtime.async_coordinator.start()


def _help_text() -> str:
    return (
        "Commands:\n"
        "/help\n"
        "/status\n"
        "/context\n"
        "/compact\n"
        "/dream\n"
        "/tasks\n"
        "/talk <peer-id> <message>\n"
        "/cancel\n"
        "/cancel all\n"
        "/quiet\n"
        "/verbose\n"
        "/skills\n"
        "/logs\n"
        "/session\n"
        "/sessions\n"
        "/checkpoints\n"
        "/new\n"
        "/rewind [checkpoint-id]\n"
        "/resume <session-id>\n"
        "/skill reload\n"
        "/skill install <github-tree-url>\n"
        "/model\n"
        "/model <openai|deepseek> <model>\n"
        "/provider <openai|deepseek>\n"
        "/approval <suggest|auto_edit|full_auto>\n"
        "/permission-mode <default|plan|accept_edits|auto|bypass>\n"
        "/permissions\n"
        "/allow Tool(pattern)\n"
        "/deny Tool(pattern)\n"
        "/ask Tool(pattern)\n"
        "/model-timeout <seconds>\n"
        "/stream on|off\n"
        "/clear\n"
        "/exit"
    )


def _parse_permission_rule(text: str, behavior: PermissionBehavior) -> PermissionRule | None:
    if "(" not in text or not text.endswith(")"):
        return None
    tool, pattern = text.split("(", 1)
    tool = tool.strip()
    pattern = pattern[:-1].strip()
    if not tool or not pattern:
        return None
    return PermissionRule(behavior=behavior, tool=tool, pattern=pattern, source="project")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.task is None or args.task == "chat":
        return run_chat(ChatRuntime(Path.cwd(), args))
    loop = build_loop(Path.cwd(), args)
    _print_answer(run_loop_with_progress(loop, args.task, session=None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
