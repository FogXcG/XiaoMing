from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

from xiaoming.llm.types import TokenUsage
from xiaoming.memory.models import DreamRun, MemoryDiary
from xiaoming.prompting.items import PendingTurn, TurnContext


@dataclass
class LoadedSkill:
    name: str
    description: str
    content: str
    path: str = ""
    content_hash: str = ""

    @classmethod
    def create(cls, *, name: str, description: str, content: str, path: str = "") -> "LoadedSkill":
        return cls(
            name=name,
            description=description,
            content=content,
            path=path,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "path": self.path,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "LoadedSkill":
        content = str(payload.get("content") or "")
        content_hash = str(payload.get("content_hash") or hashlib.sha256(content.encode("utf-8")).hexdigest())
        return cls(
            name=str(payload.get("name") or ""),
            description=str(payload.get("description") or ""),
            content=content,
            path=str(payload.get("path") or ""),
            content_hash=content_hash,
        )


@dataclass
class BootstrapContext:
    plugin_name: str
    source: str
    content: str
    path: str = ""
    content_hash: str = ""

    @classmethod
    def create(cls, *, plugin_name: str, source: str, content: str, path: str = "") -> "BootstrapContext":
        return cls(
            plugin_name=plugin_name,
            source=source,
            content=content,
            path=path,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )

    def to_payload(self) -> dict[str, str]:
        return {
            "plugin_name": self.plugin_name,
            "source": self.source,
            "content": self.content,
            "path": self.path,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BootstrapContext":
        content = str(payload.get("content") or "")
        content_hash = str(payload.get("content_hash") or hashlib.sha256(content.encode("utf-8")).hexdigest())
        return cls(
            plugin_name=str(payload.get("plugin_name") or ""),
            source=str(payload.get("source") or ""),
            content=content,
            path=str(payload.get("path") or ""),
            content_hash=content_hash,
        )


@dataclass
class Session:
    input_items: list[dict[str, Any]] = field(default_factory=list)
    session_id: str | None = None
    base_instructions: str | None = None
    base_instructions_recorded: bool = False
    reference_turn_context: TurnContext | None = None
    pending_turn: PendingTurn | None = None
    resumed: bool = False
    loaded_skills: dict[str, LoadedSkill] = field(default_factory=dict)
    bootstrap_contexts: dict[str, BootstrapContext] = field(default_factory=dict)
    last_token_usage: TokenUsage | None = None
    compaction_count: int = 0
    last_compacted_at: str | None = None
    current_turn_context_items: list[dict[str, Any]] = field(default_factory=list)
    next_model_context_items: list[dict[str, Any]] = field(default_factory=list)
    memory_diaries: dict[str, MemoryDiary] = field(default_factory=dict)
    dream_runs: dict[str, DreamRun] = field(default_factory=dict)
    last_prompt_instructions: str | None = None
    last_prompt_input_items: list[dict[str, Any]] = field(default_factory=list)
    last_model_output_items: list[dict[str, Any]] = field(default_factory=list)

    def clear(self) -> None:
        self.input_items.clear()
        self.reference_turn_context = None
        self.pending_turn = None
        self.loaded_skills.clear()
        self.bootstrap_contexts.clear()
        self.last_token_usage = None
        self.compaction_count = 0
        self.last_compacted_at = None
        self.current_turn_context_items.clear()
        self.next_model_context_items.clear()
        self.memory_diaries.clear()
        self.dream_runs.clear()
        self.last_prompt_instructions = None
        self.last_prompt_input_items.clear()
        self.last_model_output_items.clear()

    @property
    def item_count(self) -> int:
        return len(self.input_items)

    def remember_loaded_skill(self, skill: LoadedSkill) -> None:
        if skill.name:
            self.loaded_skills[skill.name] = skill

    def remember_bootstrap_context(self, context: BootstrapContext) -> None:
        if context.source:
            self.bootstrap_contexts[context.source] = context

    def stage_current_turn_context(self, content: str, event: str) -> None:
        if content:
            self.current_turn_context_items.append(_hook_context_item(content, event))

    def stage_next_model_context(self, content: str, event: str) -> None:
        if content:
            self.next_model_context_items.append(_hook_context_item(content, event))

    def consume_prompt_context_items(self) -> list[dict[str, Any]]:
        items = self.next_model_context_items + self.current_turn_context_items
        self.next_model_context_items = []
        self.current_turn_context_items = []
        return items


def _hook_context_item(content: str, event: str) -> dict[str, Any]:
    escaped_event = str(event).replace('"', "&quot;")
    return {
        "role": "developer",
        "content": f'<hook_context event="{escaped_event}">\n{content}\n</hook_context>',
        "xiaoming": {"kind": "hook_context", "event": event, "durable": False},
    }
