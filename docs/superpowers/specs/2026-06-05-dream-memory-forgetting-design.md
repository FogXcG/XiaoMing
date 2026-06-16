# Xiaoming Dream Memory Forgetting Design

## Goal

Add a dream-based forgetting mechanism for Xiaoming's working memory.

Forgetting means that old context stops being included in future LLM prompts. It does not delete the raw session event log. The raw log remains the audit trail and recovery source.

The desired prompt history becomes a hierarchical memory view:

```text
system prompt / personality / philosophy / objective facts
year diaries
month diaries
week diaries
day diaries
recent raw context
current turn
```

Older memory becomes coarser. Recent memory remains detailed. Xiaoming writes first-person diaries during dream mode, then wakes up by reloading the latest accepted memory view.

## Non-Goals

This design does not add vector memory, external databases, semantic retrieval, raw session deletion, or hard forgetting of private data. It also does not define strict diary body structure. Diary content is intentionally left to the dream model's judgment, while code manages metadata, source coverage, status, and prompt assembly.

Automatic idle dreaming is a later step. The first implementation should support manual `/dream` and safe prompt-view replacement.

## Current State

Xiaoming currently stores session events and rehydrates `Session.input_items` from those events. It has context compaction that asks an LLM to summarize history when token pressure is high, then replaces history with one summary item plus recent user messages.

That mechanism is useful, but it is flat. It does not preserve a time-layered sense of memory, and it cannot express daily, weekly, monthly, or yearly continuity.

Recent work added message time metadata:

```text
created_at
date
time
timezone
```

This makes time-based memory organization possible.

## Core Model

Dream memory introduces three persistent concepts.

### Memory Fragment

A memory fragment is the prompt-visible unit derived from raw session events.

```text
id
source_event_id
role_or_type
created_at
timezone
token_estimate
content or content_ref
visibility: working | archived
covered_by_diary_ids
```

Fragments can represent user messages, assistant messages, tool calls, tool outputs, summaries, worker state, or other prompt items.

### Memory Diary

A diary is a first-person memory artifact written by Xiaoming during dream mode.

```text
id
scope: day | week | month | year
start_time
end_time
timezone
status: draft | active | archived
source_fragment_ids
supersedes_diary_ids
body
created_at
accepted_at
```

The body is natural first-person writing. Code does not force fixed sections.

### Dream Run

A dream run records one sleep/dream/self-check/awaken cycle.

```text
id
started_at
ended_at
status: running | accepted | rejected | failed
snapshot_id
draft_diary_ids
reason
tokens_before
tokens_after
```

Drafts are committed only after the self-check accepts the candidate memory view.

## Working Memory View

The prompt builder should stop treating raw session history as the only source of truth. It should build a view from active diaries and recent raw context.

The default order is:

```text
base instructions
bootstrap context
loaded skills
runtime context
active year diaries
active month diaries
active week diaries
active day diaries
recent raw fragments
current user turn
```

Diary messages should be wrapped with explicit tags, for example:

```text
<memory_diary scope="day" start="2026-06-04T00:00:00+08:00" end="2026-06-05T00:00:00+08:00" timezone="Asia/Shanghai">
...
</memory_diary>
```

The diary role can be `user` initially, matching the existing loaded-skill and context-summary pattern. Personality, philosophy, and objective facts remain higher-priority prompt material.

## Recent Raw Context

The "last 24 hours" rule is a soft target, not a hard boundary.

Recent raw context should include the latest continuous context and any protected active state. If the last seven days are small, the builder may keep them raw. If the current task started more than 24 hours ago and is still active, it must not be cut in the middle.

Protected context includes:

```text
pending worker questions
pending approvals
active background tasks
current user turn
recent user constraints
unresolved implementation state
uncommitted change state when known
```

Protected context must remain visible even when older than the soft window.

## Dream Mode

Dream mode is a temporary runtime state.

During dream mode:

```text
Xiaoming does not answer user requests.
User input is queued until wake-up.
Normal tools are hidden.
Only dream memory tools are available.
Raw session events are not deleted.
Worker events created after the dream snapshot are not part of that dream.
```

The dream prompt explains:

```text
You are in dream mode.
You cannot respond to the user or perform external work.
Your task is to organize working memory by writing first-person diaries.
These diaries will replace some old context in future prompts.
The raw session log remains available outside the prompt.
Do not call normal work tools.
Write what you think future-you should remember.
When all drafts are ready, inspect the candidate memory view before accepting.
```

## Dream Tools

Dream mode should expose a small dedicated tool set:

```text
list_memory_packets()
read_memory_packet(packet_id)
write_diary_draft(scope, start_time, end_time, body, source_ids)
revise_diary_draft(diary_id, body, reason)
build_candidate_memory_view()
accept_dream(reason)
reject_dream(reason)
```

No shell, file editing, web search, skill installation, background scheduling, or worker management tools should be visible in dream mode.

## Source Packet Construction

Code should not perform semantic truncation. It should not decide that a tool output is important, that an error is repetitive, or that a topic changed. Those decisions belong to the dream LLM.

Code only controls non-semantic packet boundaries and budget.

Packetization uses time gaps:

```text
1. Start from the dream snapshot fragments.
2. If a packet fits the dream input budget, keep it intact.
3. If it is too large, find the best split point between fragments.
4. Prefer the largest time gap.
5. Add a night bonus for gaps around normal sleep hours, such as 00:00-08:00.
6. Penalize extremely unbalanced splits.
7. Recursively split until packets fit.
8. If no useful gap exists, split at a message boundary near the token midpoint.
9. Only if one fragment alone exceeds the budget, apply mechanical head/tail truncation and mark the omission.
```

Single-fragment truncation is a last-resort safety valve. It must be visible:

```text
[middle omitted due to size; original_chars=18342]
```

## Dream Flow

```text
1. Sleep
   Enter dream mode and queue new user input.

2. Snapshot
   Freeze the current memory inputs for this dream run.

3. Packetize
   Build source packets using time-gap splitting.

4. Draft
   The dream model reads packets and writes diary drafts.

5. Merge
   The model may merge lower-level drafts into week, month, or year drafts.

6. Candidate View
   Code assembles the memory view that would be used after wake-up.

7. Self-Check
   The dream model reads the candidate view and decides whether it can wake safely.

8. Accept or Reject
   `accept_dream` commits the memory patch. `reject_dream` discards drafts.

9. Awaken
   Xiaoming reloads the latest working memory view and then handles queued user input.
```

## Commit Semantics

Dream output is transactional.

Before acceptance:

```text
draft diaries exist
source fragment visibility is unchanged
superseded diaries remain active
normal prompt view is unchanged
```

After acceptance:

```text
draft diaries become active
covered source fragments become archived for prompt-view purposes
superseded diaries become archived
dream run is recorded as accepted
the next prompt uses the rebuilt memory view
```

If dream fails, times out, or self-check rejects:

```text
draft diaries are discarded or left as rejected diagnostics
working memory view remains unchanged
Xiaoming wakes with the old context
```

## Relationship With Workers

Dream is a main-Xiaoming memory state, not a worker task.

Active worker state is protected context. Dream must not archive pending questions, approvals, or active worker task status. Workers may continue running while Xiaoming dreams, but events produced after the dream snapshot are not included until the next wake cycle.

If a worker asks a question during dream mode, the notice can be queued and shown after wake-up.

## Manual First Implementation

The first version should implement manual dreaming only:

```text
/dream
```

Manual dreaming is easier to observe and debug. Automatic dreaming can be added later once diary quality and wake-up behavior are trusted.

## Testing Strategy

Tests should cover:

```text
packetizer chooses large time gaps
packetizer prefers night gaps when otherwise similar
packetizer avoids extreme imbalance
oversized single fragments are visibly truncated
dream mode exposes only dream tools
draft diaries do not affect prompt view before accept
accept commits active diaries and archives covered fragments
reject leaves prompt view unchanged
memory view orders year/month/week/day/raw context correctly
protected worker context remains visible
queued user input is processed after wake-up
```

## Open Implementation Order

Implement in these phases:

```text
1. Add memory fragment, diary, and dream run storage.
2. Add memory view builder without changing normal runtime behavior.
3. Add packetizer based on time gaps and budget.
4. Add dream-only tool registry.
5. Add manual `/dream`.
6. Add draft diary generation and candidate view.
7. Add self-check and transactional accept/reject.
8. Switch prompt history builder to hierarchical memory view.
9. Add automatic dream triggers later.
```

## Review Notes

The design intentionally avoids semantic code rules for forgetting. Code manages time, budgets, tool visibility, and transactions. The LLM writes first-person diaries and decides what future Xiaoming should remember within each packet.

The design also avoids deleting raw history. That keeps dream failures recoverable and makes future memory retrieval possible.
