from __future__ import annotations


def truncate_middle(text: str, limit: int) -> str:
    if limit < 1:
        return ""
    if len(text) <= limit:
        return text
    marker = "\n... truncated ...\n"
    if limit <= len(marker):
        return marker[:limit]
    remaining = limit - len(marker)
    start_len = remaining // 2
    end_len = remaining - start_len
    return text[:start_len] + marker + text[-end_len:]
