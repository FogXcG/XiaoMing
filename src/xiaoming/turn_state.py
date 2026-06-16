from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from xiaoming.llm.types import ToolCall


TurnStatus = Literal["running", "aborting", "aborted", "failed", "completed"]


@dataclass
class ActiveTurnState:
    turn_id: str
    status: TurnStatus = "running"
    pending_tool_calls: dict[str, ToolCall] = field(default_factory=dict)
    pending_worker_questions: set[str] = field(default_factory=set)
    pending_approvals: set[str] = field(default_factory=set)
    cancel_requested: bool = False

    def clear_pending(self) -> None:
        self.pending_tool_calls.clear()
        self.pending_worker_questions.clear()
        self.pending_approvals.clear()
