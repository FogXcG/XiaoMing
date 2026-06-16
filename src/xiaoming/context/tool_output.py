from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from xiaoming.context.truncation import truncate_middle


def maybe_spill_output(
    tool: str,
    output: str,
    workspace: Path | None,
    max_inline_chars: int,
    max_saved_chars: int,
) -> str:
    if workspace is None or len(output) <= max_inline_chars:
        return output

    safe_tool = re.sub(r"[^A-Za-z0-9_.-]+", "-", tool).strip("-") or "tool"
    directory = workspace / ".xiaoming" / "tool-outputs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path = directory / f"{stamp}-{safe_tool}.txt"
    saved = output[:max_saved_chars]
    path.write_text(saved)

    relative = path.relative_to(workspace)
    preview = truncate_middle(output, max_inline_chars)
    note = ""
    if len(output) > max_saved_chars:
        note = f"\nFull output exceeded the save cap; saved output truncated to {max_saved_chars} chars."
    return f"Output too large. Full output saved to:\n{relative}\n{note}\n\nPreview:\n{preview}"
