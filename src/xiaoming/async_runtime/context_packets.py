from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Literal


ContextPolicyMode = Literal["isolated", "briefed", "filtered_context_packet", "forked", "resume_worker"]
ResourceKind = Literal["file", "directory", "module", "domain", "external"]
ResourceConfidence = Literal["explicit", "inferred", "unknown"]


@dataclass(frozen=True)
class ResourceClaim:
    resource: str
    kind: ResourceKind
    confidence: ResourceConfidence

    def to_dict(self) -> dict[str, str]:
        return {
            "resource": self.resource,
            "kind": self.kind,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResourceClaim":
        kind = str(data.get("kind") or "domain")
        confidence = str(data.get("confidence") or "unknown")
        return cls(
            resource=str(data.get("resource") or ""),
            kind=kind if kind in {"file", "directory", "module", "domain", "external"} else "domain",
            confidence=confidence if confidence in {"explicit", "inferred", "unknown"} else "unknown",
        )


@dataclass(frozen=True)
class WorkerContextPolicy:
    mode: ContextPolicyMode

    def __post_init__(self) -> None:
        if self.mode not in {"isolated", "briefed", "filtered_context_packet", "forked", "resume_worker"}:
            raise ValueError(f"invalid worker context policy: {self.mode}")

    def to_dict(self) -> dict[str, str]:
        return {"mode": self.mode}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerContextPolicy":
        return cls(mode=str(data.get("mode") or "briefed"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class WorkerContextPacket:
    task_id: str
    session_id: str
    agent_type: str
    context_policy: str
    workspace: str
    selected_skills: list[str] = field(default_factory=list)
    active_tasks_summary: str = ""
    source_item_ids: list[str] = field(default_factory=list)
    resource_claims: list[ResourceClaim] = field(default_factory=list)
    handoff_summary: str = ""
    decisions_already_made: list[str] = field(default_factory=list)
    relevant_messages: list[dict[str, Any]] = field(default_factory=list)
    prior_worker_summary: str | None = None
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_hash", _stable_hash(self._content_dict(include_hash=False)))

    def to_dict(self) -> dict[str, Any]:
        return self._content_dict(include_hash=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerContextPacket":
        return cls(
            task_id=str(data.get("task_id") or ""),
            session_id=str(data.get("session_id") or ""),
            agent_type=str(data.get("agent_type") or "worker"),
            context_policy=str(data.get("context_policy") or "forked"),
            workspace=str(data.get("workspace") or ""),
            selected_skills=[str(item) for item in data.get("selected_skills") or []],
            active_tasks_summary=str(data.get("active_tasks_summary") or ""),
            source_item_ids=[str(item) for item in data.get("source_item_ids") or []],
            resource_claims=[ResourceClaim.from_dict(item) for item in data.get("resource_claims") or [] if isinstance(item, dict)],
            handoff_summary=str(data.get("handoff_summary") or ""),
            decisions_already_made=[str(item) for item in data.get("decisions_already_made") or []],
            relevant_messages=[dict(item) for item in data.get("relevant_messages") or [] if isinstance(item, dict)],
            prior_worker_summary=str(data["prior_worker_summary"]) if data.get("prior_worker_summary") is not None else None,
        )

    def render_for_worker(self) -> str:
        decisions = "\n".join(f"- {_xml(item)}" for item in self.decisions_already_made) or "- (none)"
        skills = "\n".join(f"- {_xml(item)}" for item in self.selected_skills) or "- (none)"
        resources = "\n".join(
            f"- {_xml(claim.resource)} ({claim.kind}, {claim.confidence})" for claim in self.resource_claims
        ) or "- (none)"
        messages = "\n".join(
            f"- { _xml(str(message.get('role') or 'message')) }: {_xml(str(message.get('content') or ''))}"
            for message in self.relevant_messages
        ) or "- (none)"
        parts = [
            "<worker_context_packet>",
            f"<task_id>{_xml(self.task_id)}</task_id>",
            f"<session_id>{_xml(self.session_id)}</session_id>",
            f"<agent_type>{_xml(self.agent_type)}</agent_type>",
            f"<context_policy>{_xml(self.context_policy)}</context_policy>",
            f"<workspace>{_xml(self.workspace)}</workspace>",
            f"<content_hash>{_xml(self.content_hash)}</content_hash>",
            "<handoff_summary>",
            _xml(self.handoff_summary),
            "</handoff_summary>",
            "<decisions_already_made>",
            decisions,
            "</decisions_already_made>",
            "<selected_skills>",
            skills,
            "</selected_skills>",
            "<resource_claims>",
            resources,
            "</resource_claims>",
            "<active_tasks_summary>",
            _xml(self.active_tasks_summary),
            "</active_tasks_summary>",
            "<relevant_messages>",
            messages,
            "</relevant_messages>",
        ]
        if self.prior_worker_summary:
            parts.extend(["<prior_worker_summary>", _xml(self.prior_worker_summary), "</prior_worker_summary>"])
        parts.append("</worker_context_packet>")
        return "\n".join(parts)

    def _content_dict(self, *, include_hash: bool) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "agent_type": self.agent_type,
            "context_policy": self.context_policy,
            "workspace": self.workspace,
            "selected_skills": list(self.selected_skills),
            "active_tasks_summary": self.active_tasks_summary,
            "source_item_ids": list(self.source_item_ids),
            "resource_claims": [claim.to_dict() for claim in self.resource_claims],
            "handoff_summary": self.handoff_summary,
            "decisions_already_made": list(self.decisions_already_made),
            "relevant_messages": list(self.relevant_messages),
            "prior_worker_summary": self.prior_worker_summary,
        }
        if include_hash:
            payload["content_hash"] = self.content_hash
        return payload


class ContextPacketBuilder:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def build(
        self,
        *,
        session,
        task_id: str,
        agent_type: str,
        context_policy: str,
        task_spec,
        registry,
        selected_skills: list[str],
        prior_worker_summary: str | None = None,
    ) -> WorkerContextPacket:
        mode = WorkerContextPolicy(mode=context_policy or "briefed").mode
        relevant_messages = _relevant_messages(session.input_items, mode)
        source_item_ids = [_item_id(message) for message in relevant_messages]
        source_item_ids = [item_id for item_id in source_item_ids if item_id]
        decisions = _decisions_from_messages(relevant_messages)
        return WorkerContextPacket(
            task_id=task_id,
            session_id=str(session.session_id or ""),
            agent_type=agent_type,
            context_policy=mode,
            workspace=str(self.workspace),
            selected_skills=sorted(set(selected_skills)),
            active_tasks_summary=_active_tasks_summary(registry),
            source_item_ids=source_item_ids,
            resource_claims=list(getattr(task_spec, "resource_claims", []) or []),
            handoff_summary=_handoff_summary(task_spec, relevant_messages, mode),
            decisions_already_made=decisions,
            relevant_messages=[_compact_message(message) for message in relevant_messages],
            prior_worker_summary=prior_worker_summary,
        )


def _relevant_messages(items: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode in {"isolated", "forked", "resume_worker"}:
        return []
    selected: list[dict[str, Any]] = []
    for item in items:
        if not _is_context_message(item):
            continue
        if mode == "filtered_context_packet" and item.get("role") not in {"user", "assistant"}:
            continue
        selected.append(item)
    return selected if mode == "filtered_context_packet" else selected[-6:]


def _is_context_message(item: dict[str, Any]) -> bool:
    if item.get("type") in {"function_call", "function_call_output", "custom_tool_call", "custom_tool_call_output"}:
        return False
    if item.get("role") not in {"user", "assistant"}:
        return False
    content = item.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    meta = item.get("xiaoming")
    kind = meta.get("kind") if isinstance(meta, dict) else None
    return kind not in {
        "runtime_context",
        "bootstrap_context",
        "loaded_skill",
        "hook_context",
        "worker_protocol",
        "developer_context",
        "developer_context_diff",
        "environment_context",
        "ephemeral_context",
    }


def _compact_message(item: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "role": str(item.get("role") or ""),
        "content": _truncate(str(item.get("content") or ""), 1200),
    }
    item_id = _item_id(item)
    if item_id:
        compact["id"] = item_id
    return compact


def _item_id(item: dict[str, Any]) -> str:
    meta = item.get("xiaoming")
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("id") or "")


def _decisions_from_messages(messages: list[dict[str, Any]]) -> list[str]:
    decisions: list[str] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        decisions.append(_truncate(content, 240))
    return decisions[-4:]


def _handoff_summary(task_spec, messages: list[dict[str, Any]], mode: str) -> str:
    goal = str(getattr(task_spec, "goal", "") or "").strip()
    title = str(getattr(task_spec, "title", "") or "").strip()
    if mode in {"isolated", "forked"}:
        return goal or title
    snippets = []
    for message in messages[-3:]:
        snippets.append(f"{message.get('role')}: {_truncate(str(message.get('content') or ''), 220)}")
    suffix = "\nRelevant recent context:\n" + "\n".join(snippets) if snippets else ""
    return (goal or title or "Complete the assigned task.") + suffix


def _active_tasks_summary(registry) -> str:
    try:
        active = registry.active_snapshot()
    except AttributeError:
        return "none"
    if not active:
        return "none"
    return json.dumps(active, ensure_ascii=False, sort_keys=True)


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _xml(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
