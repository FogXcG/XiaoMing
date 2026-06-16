from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from xiaoming.context.manager import estimate_tokens
from xiaoming.llm.provider import LLMProvider
from xiaoming.llm.types import LLMRequest
from xiaoming.memory.dream_tools import DreamToolState, dream_tool_registry
from xiaoming.memory.fragments import fragments_from_items
from xiaoming.memory.models import DreamRun, MemoryDiary
from xiaoming.memory.packetizer import packetize_fragments
from xiaoming.memory.store import MemoryStore
from xiaoming.session import Session
from xiaoming.time_meta import now_iso


DREAM_PROMPT = """
You are in dream mode.
You cannot respond to the user or perform external work.
Your task is to organize working memory by writing first-person diaries.
These diaries will replace some old context in future prompts.
The raw session log remains available outside the prompt.
Use the dream tools to inspect memory packets, write diary drafts, inspect the candidate view, and accept or reject the dream.
""".strip()


@dataclass(frozen=True)
class DreamResult:
    accepted: bool
    reason: str
    draft_count: int


class DreamRunner:
    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        max_output_tokens: int,
        recorder: object | None = None,
        packet_budget_tokens: int = 12000,
        max_turns: int = 8,
    ):
        self.provider = provider
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.recorder = recorder
        self.packet_budget_tokens = packet_budget_tokens
        self.max_turns = max_turns

    def run(self, session: Session) -> DreamResult:
        fragments = fragments_from_items(session.input_items)
        packets = packetize_fragments(fragments, self.packet_budget_tokens)
        state = DreamToolState(packets={packet.id: packet.fragments for packet in packets})
        registry = dream_tool_registry(state)
        store = MemoryStore(session, recorder=self.recorder)
        run = DreamRun(id=f"dream-{uuid4()}", started_at=now_iso(), status="running", tokens_before=estimate_tokens(session.input_items))
        store.save_dream_run(run)
        input_items = [{"role": "user", "content": DREAM_PROMPT}]
        for _ in range(self.max_turns):
            response = self.provider.complete(
                LLMRequest(
                    instructions="You are Xiaoming's dream-mode memory organizer.",
                    input_items=input_items,
                    tools=registry.specs(),
                    model=self.model,
                    temperature=0.0,
                    max_output_tokens=min(self.max_output_tokens, 4096),
                )
            )
            input_items.extend(response.output_items)
            for call in response.tool_calls:
                result = registry.run(call.name, call.args)
                input_items.append({"type": call.output_type, "call_id": call.id, "output": registry.format_result(result)})
            if state.accepted:
                accepted_at = now_iso()
                accepted = [
                    MemoryDiary(
                        id=diary.id,
                        scope=diary.scope,
                        start_time=diary.start_time,
                        end_time=diary.end_time,
                        timezone=diary.timezone,
                        status="active",
                        source_fragment_ids=list(diary.source_fragment_ids),
                        supersedes_diary_ids=list(diary.supersedes_diary_ids),
                        body=diary.body,
                        created_at=diary.created_at,
                        accepted_at=accepted_at,
                    )
                    for diary in state.draft_diaries
                ]
                for diary in accepted:
                    store.save_diary(diary)
                store.save_dream_run(
                    DreamRun(
                        id=run.id,
                        started_at=run.started_at,
                        ended_at=accepted_at,
                        status="accepted",
                        draft_diary_ids=[diary.id for diary in accepted],
                        reason=state.decision_reason,
                        tokens_before=run.tokens_before,
                        tokens_after=estimate_tokens(session.input_items),
                    )
                )
                return DreamResult(accepted=True, reason=state.decision_reason, draft_count=len(accepted))
            if state.rejected:
                ended_at = now_iso()
                store.save_dream_run(
                    DreamRun(
                        id=run.id,
                        started_at=run.started_at,
                        ended_at=ended_at,
                        status="rejected",
                        reason=state.decision_reason,
                        tokens_before=run.tokens_before,
                    )
                )
                return DreamResult(accepted=False, reason=state.decision_reason, draft_count=len(state.draft_diaries))
        ended_at = now_iso()
        store.save_dream_run(
            DreamRun(
                id=run.id,
                started_at=run.started_at,
                ended_at=ended_at,
                status="failed",
                reason="dream exceeded max turns",
                tokens_before=run.tokens_before,
            )
        )
        return DreamResult(accepted=False, reason="dream exceeded max turns", draft_count=len(state.draft_diaries))
