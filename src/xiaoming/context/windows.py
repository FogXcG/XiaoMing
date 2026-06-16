from __future__ import annotations


DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
DEFAULT_COMPACT_RATIO = 0.9


def model_context_window_tokens(model: str | None) -> int:
    normalized = (model or "").lower()
    if not normalized:
        return DEFAULT_CONTEXT_WINDOW_TOKENS
    if normalized.startswith("deepseek-v4"):
        return 1_000_000
    if normalized.startswith("deepseek-v3.2"):
        return 1_000_000
    if normalized in {"deepseek-chat", "deepseek-reasoner"}:
        return 1_000_000
    if normalized.startswith("deepseek"):
        return 128_000
    if normalized.startswith("gpt-4.1"):
        return 1_047_576
    if normalized.startswith("gpt-5"):
        return 400_000
    if normalized.startswith("o1") or normalized.startswith("o3") or normalized.startswith("o4"):
        return 128_000
    if normalized.startswith("gpt-4o"):
        return 128_000
    return DEFAULT_CONTEXT_WINDOW_TOKENS


def compact_threshold_tokens(model: str | None, max_output_tokens: int | None) -> int:
    window = model_context_window_tokens(model)
    return max(4_000, int(window * DEFAULT_COMPACT_RATIO))
