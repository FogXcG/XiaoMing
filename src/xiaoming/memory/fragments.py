from __future__ import annotations

import json
from typing import Any

from xiaoming.context.manager import estimate_tokens
from xiaoming.memory.models import MemoryFragment
from xiaoming.time_meta import ensure_time_metadata


def fragments_from_items(items: list[dict[str, Any]]) -> list[MemoryFragment]:
    fragments: list[MemoryFragment] = []
    for index, raw_item in enumerate(items):
        item = ensure_time_metadata(raw_item)
        content = _content_text(item)
        if not content:
            continue
        meta = item.get("xiaoming") if isinstance(item.get("xiaoming"), dict) else {}
        source_id = _source_id(item, meta, index)
        fragments.append(
            MemoryFragment(
                id=f"fragment:{source_id}",
                source_event_id=source_id,
                role_or_type=_role_or_type(item, meta),
                created_at=str(meta.get("created_at") or ""),
                timezone=str(meta.get("timezone") or ""),
                token_estimate=estimate_tokens([{"content": content}]),
                content=content,
            )
        )
    return fragments


def _content_text(item: dict[str, Any]) -> str:
    content = item.get("content")
    if isinstance(content, str):
        return content
    output = item.get("output")
    if isinstance(output, str):
        return output
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, sort_keys=True)


def _source_id(item: dict[str, Any], meta: dict[str, Any], index: int) -> str:
    if meta.get("id"):
        return str(meta["id"])
    if item.get("call_id"):
        return str(item["call_id"])
    return f"item-{index}"


def _role_or_type(item: dict[str, Any], meta: dict[str, Any]) -> str:
    if meta.get("kind"):
        return str(meta["kind"])
    if item.get("type"):
        return str(item["type"])
    return str(item.get("role") or "unknown")
