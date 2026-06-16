from __future__ import annotations

from typing import Any

from xiaoming.llm.types import TokenUsage


def extract_token_usage(raw: Any) -> TokenUsage | None:
    usage = _get(raw, "usage")
    if usage is None:
        return None
    raw_usage = _to_plain_dict(usage)
    return TokenUsage(
        input_tokens=_as_int(_first_present(usage, "input_tokens", "prompt_tokens")),
        output_tokens=_as_int(_first_present(usage, "output_tokens", "completion_tokens")),
        total_tokens=_as_int(_get(usage, "total_tokens")),
        cached_tokens=_as_int(_first_present(_get(usage, "input_tokens_details"), "cached_tokens")),
        cache_hit_tokens=_as_int(_get(usage, "prompt_cache_hit_tokens")),
        cache_miss_tokens=_as_int(_get(usage, "prompt_cache_miss_tokens")),
        reasoning_tokens=_as_int(_first_present(_get(usage, "output_tokens_details"), "reasoning_tokens")),
        raw=raw_usage,
    )


def _first_present(item: Any, *names: str) -> Any:
    if item is None:
        return None
    for name in names:
        value = _get(item, name)
        if value is not None:
            return value
    return None


def _get(item: Any, name: str) -> Any:
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _plain(inner) for key, inner in value.items()}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, dict):
            return {str(key): _plain(inner) for key, inner in dumped.items()}
    if hasattr(value, "dict"):
        dumped = value.dict(exclude_none=True)
        if isinstance(dumped, dict):
            return {str(key): _plain(inner) for key, inner in dumped.items()}
    if hasattr(value, "__dict__"):
        return {str(key): _plain(inner) for key, inner in vars(value).items() if not key.startswith("_") and inner is not None}
    return {}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(exclude_none=True))
    if hasattr(value, "dict"):
        return _plain(value.dict(exclude_none=True))
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {str(key): _plain(inner) for key, inner in vars(value).items() if not key.startswith("_") and inner is not None}
    return value
