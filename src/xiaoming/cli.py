from __future__ import annotations

import argparse
import asyncio
import copy
import inspect
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Float, FloatContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

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
from xiaoming.config import DEFAULT_DEEPSEEK_MODEL, DEFAULT_OPENAI_MODEL, api_key_env_name, api_key_present, global_config_path, load_config, save_global_config, secrets_env_path, workspace_config_path, write_secrets_env
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
from xiaoming.time_meta import now_iso
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
    parser.add_argument("--init", dest="init", action="store_true", default=False)
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
class SessionSnapshot:
    input_items: list[dict]
    loaded_skills: dict
    bootstrap_contexts: dict
    base_instructions: str | None = None
    reference_turn_context: object | None = None


@dataclass
class ForegroundTask:
    task_name: str
    original_message: str
    status: str = "running"
    worker_id: str | None = None
    session_id: str | None = None
    boundary_snapshot: SessionSnapshot | None = None
    cancel_event: threading.Event | None = None


@dataclass
class ForegroundTurnResult:
    answer: str
    pending_user_input: str | None = None


@dataclass
class _TuiApprovalRequest:
    action: str
    done: threading.Event
    approved: bool | None = None


class TuiApprovalController:
    """Route tool approvals through the prompt_toolkit input area."""

    def __init__(self, output: "TuiOutput", invalidate: Callable[[], None]):
        self.output = output
        self.invalidate = invalidate
        self._lock = threading.Lock()
        self._pending: _TuiApprovalRequest | None = None

    def request(self, action: str) -> bool:
        request = _TuiApprovalRequest(action=action, done=threading.Event())
        with self._lock:
            if self._pending is not None:
                return False
            self._pending = request
        self.output.write(_format_tui_approval_action(action))
        self.output.write("Approve? [y/N]")
        self.invalidate()
        request.done.wait()
        return bool(request.approved)

    def consume_answer(self, answer: str) -> bool | None:
        with self._lock:
            request = self._pending
            if request is None:
                return None
            self._pending = None
        approved = answer.strip().lower() in {"y", "yes"}
        request.approved = approved
        request.done.set()
        return approved

    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None


def _format_tui_approval_action(action: str, max_chars: int = 1200, max_lines: int = 30) -> str:
    compacted = _omit_content_preview(action)
    lines = compacted.splitlines()
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... omitted {len(lines) - max_lines} lines ..."]
        compacted = "\n".join(lines)
    if len(compacted) > max_chars:
        compacted = compacted[:max_chars].rstrip() + "\n... omitted long approval details ..."
    return compacted


def _omit_content_preview(action: str) -> str:
    marker = "Content preview:"
    if marker not in action:
        return action
    head = action.split(marker, 1)[0].rstrip()
    return f"{head}\nContent preview: [omitted in TUI; approve only if the file path and operation are expected]"


def build_universal_runtime_tools(
    coordinator_getter: Callable[[], object | None],
    talk_callback: TalkCallback,
    turn_context_getter: Callable[[], str] | None = None,
    promoted_task_getter: Callable[[], str] | None = None,
) -> list[Tool]:
    return [
        ScheduleBackgroundTaskTool(coordinator_getter, turn_context_getter=turn_context_getter, promoted_task_getter=promoted_task_getter),
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
    tool_use_policy = """
Tool use policy:
- When you need multiple independent read-only tool calls, issue them in the same assistant turn whenever possible so the runtime can execute them in parallel.
- This applies to web_search, web_fetch, search_code, read_file, and list_files.
- Only call read-only tools sequentially when a later query truly depends on an earlier result.
- Do not batch write, shell mutation, patch, approval, or background task management tools.
""".strip()
    agents = workspace / "AGENTS.md"
    project_rules = agents.read_text() if agents.exists() else ""
    return f"{safety}\n\n{personality_prompt}\n\n{default_prompt}\n\n{background_tasks}\n\n{tool_use_policy}\n\nProject rules:\n{project_rules}"


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
        self.direct_approve_callback: Callable[[str], bool] = approve_action
        self._foreground_tasks_by_thread: dict[int, ForegroundTask] = {}
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
                        promoted_task_getter=self.current_promoted_foreground_task_id,
                    ),
                    approve=self._route_approval,
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

    def config_text(self) -> str:
        config = self.config
        env_name = api_key_env_name(config.model.provider)
        key_status = "set" if api_key_present(config.model.provider) else "missing"
        return (
            f"Provider: {config.model.provider}\n"
            f"Model: {config.model.model}\n"
            f"API key: {env_name} ({key_status})\n"
            f"Global config: {global_config_path()}\n"
            f"Project config: {workspace_config_path(self.workspace)}\n"
            f"Secrets file: {secrets_env_path()}"
        )

    def doctor_text(self) -> str:
        config = self.config
        env_name = api_key_env_name(config.model.provider)
        key_ok = api_key_present(config.model.provider)
        lines = [
            "Xiaoming doctor:",
            f"- Provider: {config.model.provider}",
            f"- Model: {config.model.model}",
            f"- API key {env_name}: {'ok' if key_ok else 'missing'}",
            f"- Global config: {'present' if global_config_path().exists() else 'missing'}",
            f"- Project config: {'present' if workspace_config_path(self.workspace).exists() else 'missing'}",
            f"- Stream: {'on' if config.agent.stream else 'off'}",
        ]
        if not key_ok:
            lines.append(f"Run `xiaoming-cli --init` or set {env_name}.")
        return "\n".join(lines)

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

    def begin_foreground_task(self, user_input: str, cancel_event: threading.Event | None = None) -> ForegroundTask:
        task = ForegroundTask(
            task_name=_foreground_task_name(user_input),
            original_message=user_input,
            session_id=self.session.session_id,
            boundary_snapshot=_foreground_boundary_snapshot(self.session, user_input),
            cancel_event=cancel_event,
        )
        self.foreground_task = task
        self.logger.info("foreground_task_started", task_name=task.task_name)
        return task

    def bind_foreground_task_to_current_thread(self, task: ForegroundTask) -> None:
        self._foreground_tasks_by_thread[threading.get_ident()] = task

    def unbind_foreground_task_from_current_thread(self) -> None:
        self._foreground_tasks_by_thread.pop(threading.get_ident(), None)

    def _route_approval(self, action: str) -> bool:
        task = self._foreground_tasks_by_thread.get(threading.get_ident())
        if task is not None and task.status == "promoted" and task.worker_id and self.async_coordinator is not None:
            request = getattr(self.async_coordinator, "request_promoted_foreground_approval", None)
            if callable(request):
                return bool(request(task.worker_id, action))
        return self.direct_approve_callback(action)

    def current_promoted_foreground_task_id(self) -> str:
        task = self._foreground_tasks_by_thread.get(threading.get_ident())
        if task is None or task.status != "promoted":
            return ""
        return task.worker_id or ""

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

    def promote_foreground_to_worker(self) -> str:
        if self.foreground_task is None or self.foreground_task.status != "running":
            return ""
        if self.async_coordinator is None:
            return ""
        task = self.foreground_task
        register = getattr(self.async_coordinator, "register_promoted_foreground_task", None)
        if not callable(register):
            return ""
        worker_id = register(
            task.task_name,
            task.original_message,
            task.session_id or self.session.session_id or "",
            cancel_callback=task.cancel_event.set if task.cancel_event is not None else None,
        )
        task.worker_id = str(worker_id)
        task.status = "promoted"
        self._fork_main_session_from_foreground(task)
        self.logger.info("foreground_task_promoted", task_name=task.task_name, worker_id=task.worker_id)
        self.foreground_task = None
        return f"当前任务已转为后台继续处理：{task.task_name}"

    def complete_promoted_foreground_task(self, task_id: str, status: str, summary: str) -> None:
        if self.async_coordinator is None:
            return
        complete = getattr(self.async_coordinator, "complete_promoted_foreground_task", None)
        if callable(complete):
            complete(task_id, status, summary)

    def _fork_main_session_from_foreground(self, task: ForegroundTask) -> None:
        snapshot = task.boundary_snapshot or _foreground_boundary_snapshot(self.session, task.original_message)
        config = self.config
        record = self.session_store.create(title=self.session_record.title or "New session", provider=config.model.provider, model=config.model.model)
        new_session = _session_from_snapshot(snapshot, session_id=record.id)
        new_session.input_items.append(_foreground_promoted_context_item(task))
        self.session_store.append(record.id, "context_compaction_completed", {"replacement_items": copy.deepcopy(new_session.input_items), "created_at": now_iso()})
        self.session_record = record
        self.session = new_session

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


def _foreground_boundary_snapshot(session: Session, user_input: str) -> SessionSnapshot:
    input_items = copy.deepcopy(session.input_items)
    if not _ends_with_user_message(input_items, user_input):
        input_items.append({"role": "user", "content": user_input, "xiaoming": {"kind": "user_message", "durable": True}})
    return SessionSnapshot(
        input_items=input_items,
        loaded_skills=copy.deepcopy(session.loaded_skills),
        bootstrap_contexts=copy.deepcopy(session.bootstrap_contexts),
        base_instructions=session.base_instructions,
        reference_turn_context=copy.deepcopy(session.reference_turn_context),
    )


def _session_from_snapshot(snapshot: SessionSnapshot, session_id: str) -> Session:
    session = Session(
        session_id=session_id,
        input_items=copy.deepcopy(snapshot.input_items),
        base_instructions=snapshot.base_instructions,
        reference_turn_context=copy.deepcopy(snapshot.reference_turn_context),
        loaded_skills=copy.deepcopy(snapshot.loaded_skills),
        bootstrap_contexts=copy.deepcopy(snapshot.bootstrap_contexts),
    )
    return session


def _foreground_promoted_context_item(task: ForegroundTask) -> dict:
    return {
        "role": "developer",
        "content": (
            "<foreground_promoted>\n"
            "用户刚才的任务已在后台继续执行。\n"
            f"task_id: {task.worker_id or ''}\n"
            f"task_name: {task.task_name}\n"
            f"原任务: {task.original_message}\n"
            "不要继续执行原任务；如需进展，查询后台任务或 talk_to_peer。\n"
            "现在处理用户的新输入。\n"
            "</foreground_promoted>"
        ),
        "xiaoming": {"kind": "foreground_promoted", "durable": True, "task_id": task.worker_id or ""},
    }


def _ends_with_user_message(input_items: list[dict], user_input: str) -> bool:
    if not input_items:
        return False
    item = input_items[-1]
    return item.get("role") == "user" and str(item.get("content") or "") == user_input


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

class TuiOutput:
    """Thread-safe output buffer for the prompt_toolkit terminal UI."""

    def __init__(self):
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._streaming: str = ""

    def write(self, text: str, end: str = "\n") -> None:
        with self._lock:
            if end == "":
                self._streaming += text
            else:
                if self._streaming:
                    self._lines.append(self._streaming)
                    self._streaming = ""
                if text:
                    self._lines.append(text)

    def complete_streaming(self) -> None:
        with self._lock:
            if self._streaming:
                self._lines.append(self._streaming)
                self._streaming = ""

    def flush(self) -> str:
        """Return completed lines without touching streaming buffer."""
        with self._lock:
            lines = self._lines[:]
            self._lines.clear()
            return "\n".join(lines) if lines else ""

    def peek_streaming(self) -> str:
        """Return current streaming text without clearing it."""
        with self._lock:
            return self._streaming


def _append_completed_tui_output(text: str, completed: str, streaming_pos: int) -> tuple[str, int]:
    if not completed:
        return text, streaming_pos
    if streaming_pos >= 0:
        return text[:streaming_pos] + completed, -1
    return ((text + "\n" + completed) if text else completed), -1


def _scroll_buffer_lines(buffer: Buffer, direction: int, count: int = 8) -> None:
    for _ in range(max(1, count)):
        if direction < 0:
            buffer.cursor_up()
        else:
            buffer.cursor_down()


def _make_output_window(output_buffer: Buffer) -> Window:
    output_control = BufferControl(buffer=output_buffer, focusable=True)
    return Window(content=output_control, wrap_lines=True)


def _make_root_container(output_window: Window, input_area: TextArea) -> FloatContainer:
    body = HSplit([
        output_window,
        Window(height=1, char="─", style="class:separator"),
        input_area,
    ])
    return FloatContainer(
        content=body,
        floats=[
            Float(
                xcursor=True,
                ycursor=True,
                content=CompletionsMenu(max_height=12, scroll_offset=1),
            )
        ],
    )


@dataclass(frozen=True)
class SlashCommand:
    command: str
    usage: str
    description: str
    needs_argument: bool = False


SLASH_COMMANDS: tuple[SlashCommand, ...] = (
    SlashCommand("/help", "/help", "Show available commands."),
    SlashCommand("/status", "/status", "Show session and background task status."),
    SlashCommand("/config", "/config", "Show current runtime configuration."),
    SlashCommand("/doctor", "/doctor", "Check local setup."),
    SlashCommand("/init", "/init", "Reconfigure provider, model, and API key."),
    SlashCommand("/context", "/context", "Show context usage details."),
    SlashCommand("/compact", "/compact", "Compact the current context."),
    SlashCommand("/dream", "/dream", "Run dream-mode context organization."),
    SlashCommand("/tasks", "/tasks", "List background tasks."),
    SlashCommand("/talk", "/talk <peer-id> <message>", "Talk to a worker or external peer.", True),
    SlashCommand("/cancel", "/cancel", "Cancel the current background task."),
    SlashCommand("/cancel", "/cancel all", "Cancel all background tasks.", True),
    SlashCommand("/quiet", "/quiet", "Reduce runtime progress output."),
    SlashCommand("/verbose", "/verbose", "Show detailed runtime progress output."),
    SlashCommand("/skills", "/skills", "List available skills."),
    SlashCommand("/logs", "/logs", "Show log file location."),
    SlashCommand("/session", "/session", "Show current session details."),
    SlashCommand("/sessions", "/sessions", "List saved sessions."),
    SlashCommand("/checkpoints", "/checkpoints", "List checkpoints."),
    SlashCommand("/new", "/new", "Start a new session."),
    SlashCommand("/rewind", "/rewind [checkpoint-id]", "Restore a checkpoint.", True),
    SlashCommand("/resume", "/resume <session-id>", "Resume a saved session.", True),
    SlashCommand("/skill", "/skill reload", "Reload skills.", True),
    SlashCommand("/skill", "/skill install <github-tree-url>", "Install a skill from GitHub.", True),
    SlashCommand("/model", "/model", "Show current model."),
    SlashCommand("/model", "/model <openai|deepseek> <model>", "Switch provider and model.", True),
    SlashCommand("/provider", "/provider <openai|deepseek>", "Switch provider.", True),
    SlashCommand("/approval", "/approval <suggest|auto_edit|full_auto>", "Set approval mode.", True),
    SlashCommand("/permission", "/permission", "Choose permission mode."),
    SlashCommand("/permission-mode", "/permission-mode <default|plan|accept_edits|auto|bypass>", "Set permission mode.", True),
    SlashCommand("/permissions", "/permissions", "Show permission rules."),
    SlashCommand("/allow", "/allow Tool(pattern)", "Allow a permission rule.", True),
    SlashCommand("/deny", "/deny Tool(pattern)", "Deny a permission rule.", True),
    SlashCommand("/ask", "/ask Tool(pattern)", "Ask for matching permission rule.", True),
    SlashCommand("/model-timeout", "/model-timeout <seconds>", "Set model timeout.", True),
    SlashCommand("/stream", "/stream on|off", "Toggle streaming.", True),
    SlashCommand("/clear", "/clear", "Clear current context."),
    SlashCommand("/exit", "/exit", "Exit Xiaoming."),
)

PERMISSION_MODES: tuple[tuple[str, str], ...] = (
    ("default", "Use project defaults and configured rules."),
    ("plan", "Ask before edits and execution."),
    ("accept_edits", "Allow edits, keep asking for execution."),
    ("auto", "Allow routine work with fewer prompts."),
    ("bypass", "Bypass permission prompts."),
)


class SlashCommandCompleter(Completer):
    """Prompt-toolkit completer for top-level slash commands."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/permission "):
            yield from _permission_mode_completions(text)
            return
        if text == "/permission":
            yield from _permission_mode_completions("/permission ")
            return
        if not text.startswith("/") or any(char.isspace() for char in text):
            return
        prefix = text
        seen: set[str] = set()
        for entry in SLASH_COMMANDS:
            if not entry.command.startswith(prefix):
                continue
            key = entry.usage
            if key in seen:
                continue
            seen.add(key)
            completion_text = _slash_completion_text(entry)
            yield Completion(
                completion_text,
                start_position=-len(prefix),
                display=entry.usage,
                display_meta=entry.description,
            )


def _permission_mode_completions(text: str):
    prefix = text.removeprefix("/permission").strip()
    for mode, description in PERMISSION_MODES:
        if not mode.startswith(prefix):
            continue
        yield Completion(
            f"/permission-mode {mode}",
            start_position=-len(text),
            display=mode,
            display_meta=description,
        )


def _slash_completion_text(entry: SlashCommand) -> str:
    usage_parts = entry.usage.split()
    if len(usage_parts) > 1 and all(not part.startswith("<") and not part.startswith("[") for part in usage_parts[1:]):
        return entry.usage
    return entry.command


def _apply_selected_completion(buffer: Buffer) -> bool:
    state = buffer.complete_state
    if state is None:
        return False
    completion = state.current_completion
    if completion is None:
        return False
    buffer.apply_completion(completion)
    return True



def run_chat(runtime_or_loop) -> int:
    """Run the interactive chat loop with a prompt_toolkit terminal UI."""
    async_notices = AsyncNoticeBuffer()
    if isinstance(runtime_or_loop, ChatRuntime):
        runtime = runtime_or_loop
        runtime.async_notice_handler = async_notices.enqueue
        session = runtime.session
        loop = runtime.loop
    else:
        runtime = None
        loop = runtime_or_loop
        session = Session()
    startup_lines: list[str] = ["Xiaoming chat. Type 'exit' or 'quit' to leave. Type '/' for commands."]
    if runtime is not None:
        startup_lines.append(runtime.startup_session_text())
        if runtime.should_use_async_chat():
            runtime.async_coordinator = runtime.build_async_coordinator(runtime.async_notice_handler)
            runtime.async_coordinator.start()
    async_notices.start()

    output = TuiOutput()
    cancel_event = threading.Event()

    # --- Build prompt_toolkit UI ---
    output_buffer = Buffer(read_only=False)
    output_buffer.text = "\n".join(startup_lines)
    output_window = _make_output_window(output_buffer)

    input_area = TextArea(
        text="",
        prompt="xiaoming> ",
        multiline=False,
        completer=SlashCommandCompleter(),
        complete_while_typing=True,
        height=1,
        wrap_lines=False,
    )

    root_container = _make_root_container(output_window, input_area)

    kb = KeyBindings()

    style = Style.from_dict({"separator": "fg:#555555"})

    app = Application(
        layout=Layout(root_container, focused_element=input_area),
        key_bindings=kb,
        style=style,
        mouse_support=True,
        full_screen=True,
    )

    def _emit(text: str, end: str = "\n") -> None:
        """Write to output buffer (thread-safe). UI refresh via periodic timer."""
        output.write(text, end=end)

    @kb.add("c-c")
    def _(event):
        if runtime is not None and runtime.async_coordinator is not None:
            if runtime.foreground_task is not None and runtime.foreground_task.cancel_event is not None:
                runtime.foreground_task.cancel_event.set()
                _emit("Interrupted current operation.")
            else:
                _emit(runtime.async_coordinator.cancel_current())
            _invalidate_ui()
        else:
            cancel_event.set()

    @kb.add("pageup")
    @kb.add("c-up")
    def _(event):
        _scroll_buffer_lines(output_buffer, -1)

    @kb.add("pagedown")
    @kb.add("c-down")
    def _(event):
        _scroll_buffer_lines(output_buffer, 1)

    def _invalidate_ui() -> None:
        """Schedule a UI refresh. Called from agent thread after output."""
        try:
            app.invalidate()
        except Exception:
            pass

    async_notices.on_enqueue = _invalidate_ui

    approval_controller = TuiApprovalController(output, _invalidate_ui)
    if runtime is not None and runtime.loop_factory is build_loop:
        runtime.direct_approve_callback = approval_controller.request
        runtime.loop = runtime._build_loop()
        loop = runtime.loop
    if runtime is not None and runtime.async_coordinator is not None:
        runtime.async_coordinator.replay_pending_questions()

    _streaming_pos = -1  # position in output_buffer where streaming text starts

    def _refresh_ui():
        nonlocal _streaming_pos
        completed = output.flush()
        streaming = output.peek_streaming()

        if completed:
            output_buffer.text, _streaming_pos = _append_completed_tui_output(output_buffer.text, completed, _streaming_pos)
            output_buffer.cursor_position = len(output_buffer.text)

        if streaming:
            if _streaming_pos >= 0:
                # Update streaming text in-place
                output_buffer.text = output_buffer.text[:_streaming_pos] + streaming
            else:
                # First streaming chunk — append
                _streaming_pos = len(output_buffer.text) + 1  # +1 for \n
                output_buffer.text += "\n" + streaming
            output_buffer.cursor_position = len(output_buffer.text)

        notices = async_notices.pull()
        for notice in notices:
            _streaming_pos = -1
            output_buffer.text += f"\n[xiaoming] {notice.message}"
            if notice.message_id and runtime is not None and runtime.async_coordinator is not None:
                runtime.async_coordinator.mark_notice_presented(notice.message_id)

    def _before_render(_app):
        _refresh_ui()

    app.before_render += _before_render

    def _on_progress_emit(message: str | ProgressEvent) -> None:
        if isinstance(message, ProgressEvent):
            if message.kind == "text_delta":
                # Streaming deltas: write to buffer only, don't invalidate
                # on every character — periodic timer handles refresh.
                output.write(message.message, end="")
                return
            output.write(f"[xiaoming] {message.message}")
            _invalidate_ui()
            return
        output.write(f"[xiaoming] {message}")
        _invalidate_ui()

    def _run_agent_task(user_input: str):
        """Run agent in a daemon thread, emitting output via _on_progress_emit."""
        def _agent_worker():
            try:
                if runtime is None:
                    run_loop_with_progress(
                        loop, user_input, session=session,
                        on_event=_on_progress_emit,
                    )
                    output.complete_streaming()
                    _invalidate_ui()
                    return
                runner_cancel_event = threading.Event()
                foreground_task = runtime.begin_foreground_task(user_input, cancel_event=runner_cancel_event)
                runtime.bind_foreground_task_to_current_thread(foreground_task)
                active_session = runtime.session
                checkpoint = runtime.checkpoint_store.create(active_session.session_id, user_input)
                runtime.active_checkpoint_id = checkpoint.id
                try:
                    answer = run_loop_with_progress(
                        runtime.loop, user_input,
                        session=active_session,
                        on_event=lambda message: None if foreground_task.status == "promoted" else _on_progress_emit(message),
                        should_cancel=runner_cancel_event.is_set,
                    )
                    if foreground_task.status == "promoted" and foreground_task.worker_id:
                        runtime.complete_promoted_foreground_task(foreground_task.worker_id, "completed", answer or "Task completed.")
                    else:
                        runtime.finish_foreground_task("completed")
                        output.complete_streaming()
                        _invalidate_ui()
                except Exception:
                    if foreground_task.status == "promoted" and foreground_task.worker_id:
                        runtime.complete_promoted_foreground_task(foreground_task.worker_id, "failed", "后台任务失败。")
                    else:
                        runtime.finish_foreground_task("failed")
                    raise
                finally:
                    runtime.unbind_foreground_task_from_current_thread()
                    runtime.active_checkpoint_id = None
                    if runtime.active_user_input == user_input:
                        runtime.active_user_input = ""
            except Exception as exc:
                if runtime is not None:
                    runtime.logger.error("cli_turn_exception", exc=exc, user_input=user_input)
                _emit(f"Error: {exc}")
                _invalidate_ui()
        threading.Thread(target=_agent_worker, daemon=True).start()

    # Periodic refresh: keep UI alive during streaming
    _stop_refresh = threading.Event()

    def _periodic_invalidate():
        while not _stop_refresh.is_set():
            _stop_refresh.wait(0.1)
            try:
                app.invalidate()
            except Exception:
                pass

    threading.Thread(target=_periodic_invalidate, daemon=True).start()

    @kb.add("/")
    def _(event):
        buffer = input_area.buffer
        buffer.insert_text("/")
        if buffer.document.text_before_cursor == "/":
            buffer.start_completion(select_first=True)

    @kb.add("down")
    def _(event):
        buffer = input_area.buffer
        if buffer.complete_state is not None:
            buffer.complete_next()

    @kb.add("up")
    def _(event):
        buffer = input_area.buffer
        if buffer.complete_state is not None:
            buffer.complete_previous()

    # Enter key: echo input + run agent
    @kb.add("enter")
    def _(event):
        _apply_selected_completion(input_area.buffer)
        user_input = input_area.text.strip()
        if not user_input:
            if approval_controller.has_pending():
                input_area.text = ""
                approval_controller.consume_answer("")
                output.write("Denied.")
                _invalidate_ui()
            return
        input_area.text = ""
        approval = approval_controller.consume_answer(user_input)
        if approval is not None:
            output.write(f">>> {user_input}")
            output.write("Approved." if approval else "Denied.")
            _invalidate_ui()
            return
        output.write(f">>> {user_input}")
        _invalidate_ui()
        result = _handle_input(user_input, runtime, session, loop,
                               output, _emit, _run_agent_task,
                               cancel_event if runtime is not None else None,
                               async_notices)
        if result == "exit":
            _stop_refresh.set()
            event.app.exit()

    try:
        app.run()
    except Exception:
        pass
    finally:
        async_notices.stop()
        if runtime is not None and runtime.async_coordinator is not None:
            runtime.async_coordinator.stop()
    return 0


def _print_async_notice(notice: CoordinatorNotice) -> None:
    """Default handler for async notices (used outside prompt_toolkit UI)."""
    print(f"[xiaoming] {notice.message}", flush=True)


# Backward-compatible stubs (kept for existing test imports)
def _print_progress(message: str | ProgressEvent) -> None:
    """Print a progress message (stub; prompt_toolkit UI uses on_event callback)."""
    if isinstance(message, ProgressEvent):
        print(message.message, end=message.end, flush=True)
    else:
        print(message, flush=True)


def _print_answer(answer: str) -> None:
    if answer:
        print(answer)


def _read_user_input(prompt: str, async_notices=None) -> str:
    return input(prompt)


def _read_user_input_until_done(prompt: str, async_notices=None, done: threading.Event | None = None) -> str | None:
    if done is not None:
        done.wait()
    return None


class AsyncNoticeBuffer:
    POLL_SECONDS = 0.1

    def __init__(self, on_enqueue: Callable[[], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._notices: list[CoordinatorNotice] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.on_enqueue = on_enqueue

    def enqueue(self, notice: CoordinatorNotice) -> None:
        with self._lock:
            self._notices.append(notice)
        if self.on_enqueue is not None:
            self.on_enqueue()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._watch_empty_input, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def pull(self) -> list[CoordinatorNotice]:
        """Return and clear buffered notices. Used by prompt_toolkit UI."""
        with self._lock:
            notices = self._notices[:]
            self._notices.clear()
        return notices

    def flush(self, redraw_prompt: bool = False) -> None:
        with self._lock:
            notices = self._notices
            self._notices = []
        for notice in notices:
            _print_async_notice(notice)
        if notices and redraw_prompt:
            print("xiaoming> ", end="", flush=True)

    def flush_if_input_empty(self, redraw_prompt: bool = False) -> bool:
        # Kept for API compatibility; no-op in prompt_toolkit mode.
        return False

    def _watch_empty_input(self) -> None:
        while not self._stop.wait(self.POLL_SECONDS):
            pass


def _handle_input(user_input: str, runtime, session, loop, output: TuiOutput,
                  emit: Callable[[str, str], None],
                  run_agent_task: Callable[[str], None],
                  cancel_event: threading.Event | None,
                  async_notices) -> str | None:
    """Handle one line of user input. Returns 'exit' if the app should quit."""
    if user_input in {"exit", "quit", "/exit", "/quit"}:
        return "exit"

    if runtime is not None and runtime.foreground_task is not None and runtime.foreground_task.status == "running":
        promoted = runtime.promote_foreground_to_worker()
        if promoted:
            emit(promoted)

    if user_input in {"/", "/help"}:
        output.write(_help_text())
        return None

    if user_input == "/clear":
        if runtime is None:
            session.clear()
        else:
            runtime.session.clear()
        output.write("Context cleared.")
        return None

    if user_input == "/status":
        if runtime is None:
            emit(f"Session items: {session.item_count}")
        elif runtime.async_coordinator is not None:
            emit(runtime.status_text())
            emit(runtime.async_coordinator.status_text())
        else:
            emit(runtime.status_text())
        return None

    if user_input == "/config":
        if runtime is None:
            emit("Config is only available in xiaoming-cli runtime mode.")
        else:
            emit(runtime.config_text())
        return None

    if user_input == "/doctor":
        if runtime is None:
            emit("Doctor is only available in xiaoming-cli runtime mode.")
        else:
            emit(runtime.doctor_text())
        return None

    if user_input == "/init":
        emit("Run `xiaoming-cli --init` in a normal terminal to reconfigure provider, model, and API key.")
        return None

    if user_input == "/context":
        if runtime is None:
            emit(f"Session items: {session.item_count}")
        else:
            emit(runtime.context_text())
        return None

    if user_input == "/compact":
        if runtime is None:
            emit("Context compaction is only available in xiaoming-cli runtime mode.")
        else:
            try:
                emit(runtime.compact_context())
            except Exception as exc:
                runtime.logger.error("manual_context_compaction_failed", exc=exc)
                emit(f"Context compaction failed: {exc}")
        return None

    if user_input == "/dream":
        if runtime is None:
            emit("Dream mode is only available in xiaoming-cli runtime mode.")
        else:
            try:
                emit(runtime.dream_context())
            except Exception as exc:
                runtime.logger.error("manual_dream_failed", exc=exc)
                emit(f"Dream failed: {exc}")
        return None

    if user_input == "/tasks":
        if runtime is None or runtime.async_coordinator is None:
            emit("Async tasks are only available in xiaoming-cli async runtime mode.")
        else:
            emit(runtime.async_coordinator.tasks_text())
        return None

    if user_input.startswith("/talk"):
        if runtime is None or runtime.async_coordinator is None:
            emit("Peer talk is only available in xiaoming-cli async runtime mode.")
        else:
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                emit("Usage: /talk <peer-id> <message>")
            else:
                emit(runtime.talk_to_peer(parts[1], parts[2]))
        return None

    if user_input == "/quiet":
        if runtime is not None and runtime.async_coordinator is not None:
            runtime.async_coordinator.set_quiet(True)
            emit("Background notices reduced.")
        else:
            emit("Quiet mode is only available in xiaoming-cli async runtime mode.")
        return None

    if user_input == "/verbose":
        if runtime is not None and runtime.async_coordinator is not None:
            runtime.async_coordinator.set_quiet(False)
            emit("Background notices restored.")
        else:
            emit("Verbose mode is only available in xiaoming-cli async runtime mode.")
        return None

    if user_input.startswith("/cancel"):
        if runtime is None or runtime.async_coordinator is None:
            emit("Cancel is only available in xiaoming-cli async runtime mode.")
        elif user_input == "/cancel all":
            emit(runtime.async_coordinator.cancel_all())
        else:
            emit(runtime.async_coordinator.cancel_current())
        return None

    if user_input == "/session":
        if runtime is None:
            emit(f"Session items: {session.item_count}")
        else:
            emit(runtime.session_text())
        return None

    if user_input == "/sessions":
        if runtime is None:
            emit("Sessions are only available in xiaoming-cli runtime mode.")
        else:
            emit(runtime.sessions_text())
        return None

    if user_input == "/checkpoints":
        if runtime is None:
            emit("Checkpoints are only available in xiaoming-cli runtime mode.")
        else:
            emit(runtime.checkpoints_text())
        return None

    if user_input.startswith("/rewind"):
        if runtime is None:
            emit("Checkpoints are only available in xiaoming-cli runtime mode.")
        else:
            parts = user_input.split()
            emit(runtime.restore_checkpoint(parts[1] if len(parts) > 1 else None))
        return None

    if user_input == "/new":
        if runtime is None:
            session.clear()
        else:
            runtime.start_new_session()
        output.write("Started new session.")
        return None

    if user_input == "/skills":
        if runtime is None:
            emit("Skills are only available in xiaoming-cli runtime mode.")
        else:
            emit(runtime.skills_text())
        return None

    if user_input == "/logs":
        if runtime is None:
            emit("Logs are only available in xiaoming-cli runtime mode.")
        else:
            emit(str(runtime.logger.path))
        return None

    if runtime is not None:
        skill_output = _handle_skill_command(runtime, user_input)
        if skill_output is not None:
            emit(skill_output)
            return None

        config_output = _handle_config_command(runtime, user_input)
        if config_output is not None:
            emit(config_output)
            return None

    if runtime is not None:
        runtime.active_user_input = user_input

    # Run agent in background thread
    run_agent_task(user_input)
    return None


def should_run_initialization(workspace: Path) -> bool:
    config = load_config(workspace, {})
    return not api_key_present(config.model.provider)


def run_initialization_wizard(
    workspace: Path,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
    force: bool = False,
) -> bool:
    if not force and not should_run_initialization(workspace):
        return False
    output_fn("Xiaoming setup")
    output_fn("Choose a model provider: 1) DeepSeek (recommended)  2) OpenAI")
    provider_answer = input_fn("Provider [1]: ").strip().lower()
    provider = "openai" if provider_answer in {"2", "openai"} else "deepseek"
    default_model = DEFAULT_DEEPSEEK_MODEL if provider == "deepseek" else DEFAULT_OPENAI_MODEL
    model = input_fn(f"Model [{default_model}]: ").strip() or default_model
    save_global_config(provider=provider, model=model)
    env_name = api_key_env_name(provider)
    api_key = input_fn(f"{env_name} (leave empty to use existing environment variable): ").strip()
    if api_key:
        write_secrets_env(provider=provider, api_key=api_key)
        output_fn(f"Saved API key to {secrets_env_path()}")
    else:
        output_fn(f"Skipped API key storage. Set {env_name} in your shell before using Xiaoming.")
    output_fn(f"Saved config to {global_config_path()}")
    return True


def run_loop_with_progress(loop, task: str, session: Session | None, should_cancel: Callable[[], bool] | None = None, on_event=None) -> str:
    signature = inspect.signature(loop.run)
    kwargs = {"session": session}
    if "on_event" in signature.parameters:
        kwargs["on_event"] = on_event or (lambda msg: None)
    if "should_cancel" in signature.parameters:
        kwargs["should_cancel"] = should_cancel
    return loop.run(task, **kwargs)


def _handle_config_command(runtime: ChatRuntime, user_input: str) -> str | None:
    parts = user_input.split()
    if not parts:
        return None
    command = parts[0]
    if command == "/model":
        if len(parts) == 1:
            config = runtime.config
            return f"Provider: {config.model.provider}\nModel: {config.model.model}"
        if len(parts) != 3:
            return "Usage: /model <openai|deepseek> <model>"
        provider, model = parts[1], parts[2]
        if provider not in {"openai", "deepseek"}:
            return "Provider must be openai or deepseek."
        ok, error = runtime.try_update(provider=provider, model=model)
        if not ok:
            return f"Error switching model: {error}"
        _restart_async_coordinator(runtime)
        return f"Model switched to {provider} {model}. Context cleared."
    if command == "/provider":
        if len(parts) != 2:
            return "Usage: /provider <openai|deepseek>"
        provider = parts[1]
        if provider not in {"openai", "deepseek"}:
            return "Provider must be openai or deepseek."
        ok, error = runtime.try_update(provider=provider, model=None)
        if not ok:
            return f"Error switching provider: {error}"
        _restart_async_coordinator(runtime)
        config = runtime.config
        return f"Provider switched to {config.model.provider} {config.model.model}. Context cleared."
    if command == "/approval":
        if len(parts) != 2:
            return "Usage: /approval <suggest|auto_edit|full_auto>"
        mode = parts[1]
        if mode not in {"suggest", "auto_edit", "full_auto"}:
            return "Approval mode must be suggest, auto_edit, or full_auto."
        ok, error = runtime.try_update(approval_mode=mode)
        if not ok:
            return f"Error switching approval mode: {error}"
        _restart_async_coordinator(runtime)
        return f"Approval mode set to {mode}. Context cleared."
    if command in {"/permission", "/permission-mode"}:
        if len(parts) == 1:
            return f"Permission mode: {runtime.config.agent.permission_mode}"
        if len(parts) != 2:
            return "Usage: /permission-mode <default|plan|accept_edits|auto|bypass>"
        mode = parts[1]
        if mode not in {"default", "plan", "accept_edits", "auto", "bypass"}:
            return "Permission mode must be default, plan, accept_edits, auto, or bypass."
        ok, error = runtime.try_update(permission_mode=mode)
        if not ok:
            return f"Error switching permission mode: {error}"
        _restart_async_coordinator(runtime)
        return f"Permission mode set to {mode}. Context cleared."
    if command == "/permissions":
        rules = load_project_rules(runtime.workspace)
        if not rules:
            return f"No project permission rules. File: {permissions_path(runtime.workspace)}"
        return "\n".join(f"{rule.behavior.value} {rule.tool}({rule.pattern}) [{rule.source}]" for rule in rules)
    if command in {"/allow", "/deny", "/ask"}:
        if len(parts) < 2:
            return f"Usage: {command} Tool(pattern)"
        parsed = _parse_permission_rule(" ".join(parts[1:]), PermissionBehavior(command.removeprefix("/")))
        if parsed is None:
            return f"Usage: {command} Tool(pattern)"
        add_project_rule(runtime.workspace, parsed)
        runtime.reload_skills()
        return f"Added project rule: {parsed.behavior.value} {parsed.tool}({parsed.pattern})"
    if command == "/model-timeout":
        if len(parts) != 2:
            return "Usage: /model-timeout <seconds>"
        try:
            seconds = float(parts[1])
        except ValueError:
            return "Model timeout must be a number of seconds."
        ok, error = runtime.try_update(model_timeout_seconds=seconds)
        if not ok:
            return f"Error switching model timeout: {error}"
        _restart_async_coordinator(runtime)
        return f"Model timeout set to {seconds:g}s. Context cleared."
    if command == "/stream":
        if len(parts) == 1:
            return f"Stream: {'on' if runtime.config.agent.stream else 'off'}"
        if len(parts) != 2 or parts[1] not in {"on", "off"}:
            return "Usage: /stream on|off"
        enabled = parts[1] == "on"
        ok, error = runtime.try_update(stream=enabled)
        if not ok:
            return f"Error switching stream mode: {error}"
        _restart_async_coordinator(runtime)
        return f"Stream {'enabled' if enabled else 'disabled'}. Context cleared."
    if command == "/resume":
        if len(parts) != 2:
            return "Usage: /resume <session-id>"
        if not runtime.resume_session(parts[1]):
            return f"Unknown session: {parts[1]}"
        return f"Resumed session: {parts[1]}"
    return None


def _handle_skill_command(runtime: ChatRuntime, user_input: str) -> str | None:
    parts = user_input.split(maxsplit=2)
    if not parts or parts[0] != "/skill":
        return None
    if len(parts) == 2 and parts[1] == "reload":
        runtime.reload_skills()
        return "Skills reloaded."
    if len(parts) == 3 and parts[1] == "install":
        try:
            return runtime.install_skill(parts[2])
        except SkillInstallError as exc:
            return f"Error installing skill: {exc}"
    return "Usage: /skill reload | /skill install <github-tree-url>"


def _restart_async_coordinator(runtime: ChatRuntime) -> None:
    if runtime.async_coordinator is None:
        return
    runtime.async_coordinator.stop()
    runtime.async_coordinator = runtime.build_async_coordinator(runtime.async_notice_handler)
    runtime.async_coordinator.start()


def _help_text() -> str:
    lines = ["Commands:"]
    lines.extend(entry.usage for entry in SLASH_COMMANDS)
    return "\n".join(lines)


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
    if getattr(args, "init", False):
        run_initialization_wizard(Path.cwd(), force=True)
        return 0
    if args.task is None or args.task == "chat":
        if sys.stdin.isatty():
            run_initialization_wizard(Path.cwd(), force=False)
        return run_chat(ChatRuntime(Path.cwd(), args))
    loop = build_loop(Path.cwd(), args)
    answer = run_loop_with_progress(loop, args.task, session=None)
    if answer:
        print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
