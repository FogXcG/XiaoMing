from __future__ import annotations

import json
from typing import Any

from xiaoming.context.truncation import truncate_middle
from xiaoming.time_meta import ensure_time_metadata


DEFAULT_TOOL_OUTPUT_INLINE_CHARS = 12_000


class ContextManager:
    def __init__(self, items: list[dict[str, Any]]):
        self.items = items

    def for_prompt(self) -> list[dict[str, Any]]:
        return normalize_for_prompt(self.items)

    def replace(self, items: list[dict[str, Any]]) -> None:
        self.items[:] = list(items)

    def estimate_tokens(self, extra_items: list[dict[str, Any]] | None = None, instructions: str | None = None) -> int:
        return estimate_tokens(self.items, extra_items=extra_items, instructions=instructions)


def normalize_for_prompt(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    index = 0
    while index < len(items):
        item = _trim_tool_output(items[index])
        if item.get("type") == "function_call_output":
            index += 1
            continue
        normalized.append(item)
        tool_call_ids = _assistant_tool_call_ids(item)
        if not tool_call_ids:
            index += 1
            continue
        next_index = index + 1
        output_ids: set[str] = set()
        while next_index < len(items) and items[next_index].get("type") == "function_call_output":
            output = _trim_tool_output(items[next_index])
            call_id = str(output.get("call_id") or "")
            if call_id in tool_call_ids:
                output_ids.add(call_id)
                normalized.append(output)
            next_index += 1
        for call_id in sorted(tool_call_ids - output_ids):
            normalized.append(_interrupted_tool_output(call_id))
        index = next_index
    return render_message_times(normalized)


def estimate_tokens(items: list[dict[str, Any]], extra_items: list[dict[str, Any]] | None = None, instructions: str | None = None) -> int:
    chars = len(instructions or "")
    for item in items:
        chars += len(json.dumps(item, ensure_ascii=False, default=str))
    for item in extra_items or []:
        chars += len(json.dumps(item, ensure_ascii=False, default=str))
    return max(1, chars // 4)


def recent_user_messages(items: list[dict[str, Any]], token_budget: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used = 0
    for item in reversed(items):
        if item.get("role") != "user":
            continue
        if item.get("xiaoming", {}).get("kind") in {"bootstrap_context", "loaded_skill", "worker_protocol"}:
            continue
        cost = estimate_tokens([item])
        if selected and used + cost > token_budget:
            break
        selected.append(item)
        used += cost
    return list(reversed(selected))


def build_summary_item(summary: str) -> dict[str, Any]:
    return ensure_time_metadata({
        "role": "user",
        "content": f"<conversation_summary>\n{summary.strip()}\n</conversation_summary>",
        "xiaoming": {"kind": "context_summary", "durable": True},
    })


def render_message_times(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    current_date: str | None = None
    for item in items:
        meta = item.get("xiaoming")
        if not isinstance(meta, dict) or not _should_render_time(item):
            rendered.append(item)
            continue
        date = str(meta.get("date") or "")
        time = str(meta.get("time") or "")
        timezone = str(meta.get("timezone") or "")
        if not date or not time:
            rendered.append(item)
            continue
        if _has_rendered_time_prefix(str(item.get("content") or "")):
            current_date = date
            rendered.append(item)
            continue
        prefix = ""
        if date != current_date:
            prefix = f"[date={date} tz={timezone}]\n"
            current_date = date
        updated = dict(item)
        updated["content"] = f"{prefix}[@{time}] {item.get('content')}"
        rendered.append(updated)
    return rendered


def _assistant_tool_call_ids(item: dict[str, Any]) -> set[str]:
    if item.get("type") in {"function_call", "custom_tool_call"} and item.get("call_id"):
        return {str(item["call_id"])}
    if item.get("role") != "assistant":
        return set()
    tool_calls = item.get("tool_calls")
    if not isinstance(tool_calls, list):
        return set()
    return {str(call.get("id") or "") for call in tool_calls if isinstance(call, dict) and call.get("id")}


def _interrupted_tool_output(call_id: str) -> dict[str, str]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": "Tool: unknown\nStatus: interrupted\nError:\nTool execution was interrupted before completion.",
    }


def _trim_tool_output(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("type") != "function_call_output":
        return item
    output = item.get("output")
    if not isinstance(output, str) or len(output) <= DEFAULT_TOOL_OUTPUT_INLINE_CHARS:
        return item
    trimmed = dict(item)
    trimmed["output"] = truncate_middle(output, DEFAULT_TOOL_OUTPUT_INLINE_CHARS)
    return trimmed


def _should_render_time(item: dict[str, Any]) -> bool:
    if item.get("role") not in {"user", "assistant"}:
        return False
    content = item.get("content")
    if not isinstance(content, str) or not content:
        return False
    kind = item.get("xiaoming", {}).get("kind") if isinstance(item.get("xiaoming"), dict) else None
    return kind not in {"developer_context", "developer_context_diff", "environment_context", "ephemeral_context", "runtime_context", "bootstrap_context", "loaded_skill", "hook_context", "worker_protocol"}


def _has_rendered_time_prefix(content: str) -> bool:
    return content.startswith("[@") or content.startswith("[date=")
