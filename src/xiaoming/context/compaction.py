from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from xiaoming.context.manager import ContextManager, build_summary_item, estimate_tokens, normalize_for_prompt, recent_user_messages
from xiaoming.llm.provider import LLMProvider
from xiaoming.llm.types import LLMRequest
from xiaoming.logging import XiaomingLogger
from xiaoming.session import Session


SUMMARIZATION_PROMPT = """
Create a compact checkpoint summary of this conversation for a coding agent that will continue the same session.

Preserve only durable, useful information:
- user goals and preferences
- decisions already made
- current implementation state
- files, paths, commands, errors, and verification results that matter
- background task or worker state
- pending questions, approvals, or unresolved risks

Do not include generic chatter. Do not invent completed work. Write the summary so the next model can continue without the full old transcript.
""".strip()


@dataclass(frozen=True)
class CompactionResult:
    tokens_before: int
    tokens_after: int
    replacement_items: list[dict[str, Any]]
    summary: str


class ContextCompactor:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        recent_user_budget_tokens: int = 8_000,
        logger: XiaomingLogger | None = None,
        recorder: object | None = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.recent_user_budget_tokens = recent_user_budget_tokens
        self.logger = logger
        self.recorder = recorder

    def compact(self, session: Session, instructions: str | None, reason: str = "manual", extra_items: list[dict[str, Any]] | None = None) -> CompactionResult:
        history = normalize_for_prompt(session.input_items)
        tokens_before = estimate_tokens(history, instructions=instructions)
        self._record(session, "context_compaction_started", {"reason": reason, "tokens_before": tokens_before})
        request = LLMRequest(
            instructions=instructions or "In this runtime, act as a coding agent.",
            input_items=history + list(extra_items or []) + [{"role": "user", "content": SUMMARIZATION_PROMPT}],
            tools=[],
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        try:
            response = self.provider.complete(request)
        except Exception as exc:
            self._record(session, "context_compaction_failed", {"reason": reason, "error": str(exc)})
            raise
        summary = (response.message or _message_from_output_items(response.output_items)).strip()
        if not summary:
            raise RuntimeError("context compaction produced an empty summary")
        replacement_items = [build_summary_item(summary)] + recent_user_messages(session.input_items, self.recent_user_budget_tokens)
        ContextManager(session.input_items).replace(replacement_items)
        session.reference_turn_context = None
        session.compaction_count += 1
        session.last_compacted_at = datetime.now(timezone.utc).isoformat()
        tokens_after = estimate_tokens(replacement_items, instructions=instructions)
        payload = {
            "reason": reason,
            "mode": "pre_turn",
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "summary": summary,
            "replacement_items": replacement_items,
            "created_at": session.last_compacted_at,
        }
        self._record(session, "context_compaction_completed", payload)
        if self.logger is not None:
            self.logger.info("context_compaction_completed", reason=reason, tokens_before=tokens_before, tokens_after=tokens_after)
        return CompactionResult(tokens_before=tokens_before, tokens_after=tokens_after, replacement_items=replacement_items, summary=summary)

    def _record(self, session: Session, event_type: str, payload: dict[str, Any]) -> None:
        if self.recorder is not None and hasattr(self.recorder, "append"):
            self.recorder.append(session.session_id, event_type, payload)


def _message_from_output_items(items: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for item in items:
        content = item.get("content")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
    return "\n".join(chunks)
