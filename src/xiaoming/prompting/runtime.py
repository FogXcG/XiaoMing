from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from xiaoming.llm.types import ToolSpec
from xiaoming.context.manager import ContextManager, render_message_times
from xiaoming.memory.view import build_memory_view, protected_recent_items
from xiaoming.prompting.base import build_base_instructions
from xiaoming.prompting.context import ContextBuilder
from xiaoming.prompting.items import PendingTurn, PromptItem, TurnContext
from xiaoming.session import BootstrapContext, LoadedSkill, Session


@dataclass(frozen=True)
class RuntimeState:
    cwd: Path
    provider: str
    model: str
    stream: bool
    permission_mode: str = "default"
    approval_policy: str | None = None
    project_instructions_text: str | None = None
    skills_summary_text: str | None = None
    plugins_summary_text: str | None = None
    checkpoint_id: str | None = None
    last_error_id: str | None = None
    interrupted_turn_id: str | None = None
    current_date: str | None = None
    pending_worker_questions_text: str | None = None
    background_tasks_text: str | None = None


@dataclass(frozen=True)
class CompiledPrompt:
    instructions: str
    input_items: list[dict[str, Any]]
    tools: list[ToolSpec]
    pending_turn: PendingTurn


class PromptRuntime:
    def __init__(self, instructions: str, context_builder: ContextBuilder | None = None):
        self.base_instructions = build_base_instructions(instructions)
        self.context_builder = context_builder or ContextBuilder()

    def prepare_turn(self, session: Session, user_input: str, state: RuntimeState, tools: list[ToolSpec]) -> CompiledPrompt:
        if session.base_instructions != self.base_instructions:
            session.base_instructions = self.base_instructions
            session.base_instructions_recorded = False
        turn_id = str(uuid4())
        current = self._turn_context(session, state)
        previous = session.reference_turn_context
        durable = (
            self.context_builder.build_initial_durable_context(current, turn_id)
            if previous is None
            else self.context_builder.build_durable_diff(previous, current, turn_id)
        )
        ephemeral = self.context_builder.build_ephemeral_context(current, turn_id)
        staged_items = durable + ephemeral + [PromptItem.user(turn_id, user_input)]
        pending = PendingTurn(turn_id=turn_id, user_input=user_input, staged_items=staged_items, turn_context=current)
        session.pending_turn = pending
        prompt_context_items = session.consume_prompt_context_items()
        staged_input_items = [item.to_input_item() for item in staged_items]
        history_items = _history_input_items(session)
        input_items = (
            _bootstrap_context_input_items(session, dynamic=False)
            + _loaded_skill_input_items(session)
            + _bootstrap_context_input_items(session, dynamic=True)
            + prompt_context_items
            + history_items
            + staged_input_items
        )
        return CompiledPrompt(instructions=session.base_instructions, input_items=render_message_times(input_items), tools=tools, pending_turn=pending)

    def mark_sent(self, session: Session) -> None:
        if session.pending_turn is not None:
            session.pending_turn.status = "sent"

    def commit_turn(self, session: Session) -> list[PromptItem]:
        pending = session.pending_turn
        if pending is None or pending.status == "committed":
            return []
        committed: list[PromptItem] = []
        seen_ids = _history_item_ids(session.input_items)
        for item in pending.staged_items:
            if item.id in seen_ids:
                continue
            if not item.durable:
                item.consumed = True
            session.input_items.append(item.to_input_item())
            committed.append(item)
            seen_ids.add(item.id)
        session.reference_turn_context = pending.turn_context
        pending.status = "committed"
        return committed

    def fail_pending(self, session: Session, status: str = "failed") -> None:
        if session.pending_turn is not None and session.pending_turn.status != "committed":
            session.pending_turn.status = "interrupted" if status == "interrupted" else "failed"

    def _turn_context(self, session: Session, state: RuntimeState) -> TurnContext:
        return TurnContext(
            cwd=str(state.cwd.resolve()),
            current_date=state.current_date or datetime.now().date().isoformat(),
            provider=state.provider,
            model=state.model,
            stream=state.stream,
            permission_mode=state.permission_mode,
            approval_policy=state.approval_policy,
            project_instructions_hash=_stable_hash(state.project_instructions_text),
            project_instructions_text=state.project_instructions_text,
            skills_summary_hash=_stable_hash(state.skills_summary_text),
            skills_summary_text=state.skills_summary_text,
            plugins_summary_hash=_stable_hash(state.plugins_summary_text),
            plugins_summary_text=state.plugins_summary_text,
            session_id=session.session_id,
            resumed=session.resumed,
            checkpoint_id=state.checkpoint_id,
            last_error_id=state.last_error_id,
            interrupted_turn_id=state.interrupted_turn_id,
            pending_worker_questions_text=state.pending_worker_questions_text,
            background_tasks_text=state.background_tasks_text,
        )


def _history_item_ids(items: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for item in items:
        meta = item.get("xiaoming")
        if isinstance(meta, dict) and meta.get("id"):
            ids.add(str(meta["id"]))
    return ids


def _stable_hash(text: str | None) -> str | None:
    if not text:
        return None
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _history_input_items(session: Session) -> list[dict[str, Any]]:
    history = ContextManager(session.input_items).for_prompt()
    if not any(diary.status == "active" for diary in session.memory_diaries.values()):
        return history
    return build_memory_view(list(session.memory_diaries.values()), recent_items=protected_recent_items(history))


def _loaded_skill_input_items(session: Session) -> list[dict[str, Any]]:
    return [_loaded_skill_input_item(skill) for skill in sorted(session.loaded_skills.values(), key=lambda item: item.name)]


def _bootstrap_context_input_items(session: Session, dynamic: bool | None = None) -> list[dict[str, Any]]:
    contexts = sorted(session.bootstrap_contexts.values(), key=lambda item: item.source)
    if dynamic is not None:
        contexts = [context for context in contexts if _is_dynamic_bootstrap_context(context) is dynamic]
    return [_bootstrap_context_input_item(context) for context in contexts]


def _is_dynamic_bootstrap_context(context: BootstrapContext) -> bool:
    return context.source in {"runtime:task-contract", "runtime:worker-context-packet"}


def _bootstrap_context_input_item(context: BootstrapContext) -> dict[str, Any]:
    return {
        "role": "developer",
        "content": (
            f"<bootstrap_context source=\"{_xml(context.source)}\" plugin=\"{_xml(context.plugin_name)}\">\n"
            f"{context.content}\n"
            "</bootstrap_context>"
        ),
        "xiaoming": {
            "kind": "bootstrap_context",
            "plugin_name": context.plugin_name,
            "source": context.source,
            "path": context.path,
            "content_hash": context.content_hash,
            "durable": True,
        },
    }


def _loaded_skill_input_item(skill: LoadedSkill) -> dict[str, Any]:
    content = [
        "<skill>",
        f"<name>{_xml(skill.name)}</name>",
    ]
    if skill.description:
        content.append(f"<description>{_xml(skill.description)}</description>")
    if skill.path:
        content.append(f"<path>{_xml(skill.path)}</path>")
    content.extend(
        [
            "<instructions>",
            skill.content,
            "</instructions>",
            "</skill>",
        ]
    )
    return {
        "role": "user",
        "content": "\n".join(content),
        "xiaoming": {
            "kind": "loaded_skill",
            "name": skill.name,
            "path": skill.path,
            "content_hash": skill.content_hash,
            "durable": True,
        },
    }


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
