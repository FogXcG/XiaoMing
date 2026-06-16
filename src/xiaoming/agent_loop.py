from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import random
import threading
import time
from typing import Callable

from xiaoming.agent_errors import AgentErrorInfo, ProviderCallError, format_fatal_turn_error, format_recoverable_tool_error
from xiaoming.hooks import HookManager, HookResult
from xiaoming.context.compaction import ContextCompactor
from xiaoming.context.manager import ContextManager, estimate_tokens
from xiaoming.context.windows import compact_threshold_tokens, model_context_window_tokens
from xiaoming.llm.provider import LLMProvider
from xiaoming.llm.streaming import StreamAccumulator, StreamDone, StreamError, StreamTextDelta, StreamToolCallDelta, StreamUsage
from xiaoming.llm.types import LLMRequest, LLMResponse, ToolCall
from xiaoming.logging import XiaomingLogger, summarize_text
from xiaoming.memory.dream_runner import DreamRunner
from xiaoming.progress import ProgressEvent
from xiaoming.prompting.runtime import PromptRuntime, RuntimeState
from xiaoming.session import BootstrapContext, LoadedSkill, Session
from xiaoming.skills import SkillLibrary
from xiaoming.time_meta import ensure_time_metadata
from xiaoming.tools.registry import ToolRegistry
from xiaoming.tools.base import ToolResult
from xiaoming.turn_state import ActiveTurnState


class MaxTurnsExceeded(RuntimeError):
    pass


class ModelResponseTimeout(RuntimeError):
    pass


REPEATED_UNCHANGED_TOOL_RESULT_LIMIT = 3


@dataclass
class _ToolCallState:
    call: ToolCall
    result: ToolResult | None = None
    stop_after_tool_message: str | None = None


@dataclass
class AgentLoop:
    provider: LLMProvider
    registry: ToolRegistry
    instructions: str
    model: str
    temperature: float
    max_output_tokens: int
    max_turns: int
    instructions_provider: Callable[[], str] | None = None
    progress_interval_seconds: float = 10.0
    model_timeout_seconds: float = 180
    stream: bool = True
    stream_idle_timeout_seconds: float = 60
    model_max_retries: int = 3
    retry_base_delay_seconds: float = 0.25
    skill_library: SkillLibrary | None = None
    logger: XiaomingLogger | None = None
    session_recorder: object | None = None
    pending_worker_questions: Callable[..., str | None] | None = None
    runtime_context_provider: Callable[[], str | None] | None = None
    hooks: HookManager | None = None
    context_auto_compact: bool = True
    context_compact_threshold_tokens: int | None = None
    context_recent_user_budget_tokens: int = 8_000
    max_parallel_tool_calls: int = 4

    def run(
        self,
        user_task: str,
        session: Session | None = None,
        on_event: Callable[[str | ProgressEvent], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> str:
        if session is None:
            session = Session()
        session_start_blocked = self._run_session_start_hook(session)
        if session_start_blocked:
            return session_start_blocked
        self._inject_explicit_skills(session, user_task)
        user_task, blocked = self._run_user_prompt_submit_hook(session, user_task)
        if blocked:
            return blocked
        streamed_text = False
        _log(self.logger, "info", "turn_started", user_task=summarize_text(user_task), session_items=session.item_count)
        prompt_runtime = PromptRuntime(self._current_instructions())
        runtime_context_text = self._runtime_context_text()
        self._maybe_auto_compact(session, prompt_runtime.base_instructions, user_task, runtime_context_text, on_event)
        skill_context = self._skill_context_for_task(user_task, on_event)
        compiled = prompt_runtime.prepare_turn(
            session,
            user_task,
                RuntimeState(
                    cwd=Path.cwd(),
                    provider=type(self.provider).__name__,
                    model=self.model,
                    stream=self.stream,
                    skills_summary_text=skill_context,
                    pending_worker_questions_text=_pending_worker_questions_text(self.pending_worker_questions, user_task),
                    background_tasks_text=runtime_context_text,
                ),
            self.registry.specs(),
        )
        if session.base_instructions and not session.base_instructions_recorded:
            _record(self.session_recorder, session.session_id, "base_instructions", {"text": session.base_instructions})
            session.base_instructions_recorded = True
        active_turn = ActiveTurnState(compiled.pending_turn.turn_id)
        last_tool_result_signature: tuple[str, str, str, str, str] | None = None
        repeated_tool_result_count = 0
        per_turn_tool_counts: dict[str, int] = {}
        background_status_snapshot_text: str | None = None
        _record(self.session_recorder, session.session_id, "turn_started", {"turn_id": active_turn.turn_id})
        for _turn in range(self.max_turns):
            if should_cancel is not None and should_cancel():
                prompt_runtime.fail_pending(session, "moved_to_background")
                active_turn.status = "aborted"
                active_turn.cancel_requested = True
                active_turn.clear_pending()
                _record_turn_aborted(self.session_recorder, session, active_turn.turn_id, "moved_to_background", "Foreground task moved to a background worker.")
                return ""
            use_stream = self.stream and _has_stream(self.provider)
            request = LLMRequest(
                instructions=compiled.instructions,
                input_items=list(compiled.input_items),
                tools=compiled.tools,
                model=self.model,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            session.last_prompt_instructions = request.instructions
            session.last_prompt_input_items = copy.deepcopy(request.input_items)
            session.last_model_output_items = []
            prompt_runtime.mark_sent(session)
            _emit(on_event, _status_event("Thinking about the next step...") if use_stream else "Thinking about the next step...")
            _log(
                self.logger,
                "info",
                "model_call_started",
                model=self.model,
                input_items=len(request.input_items),
                tools=[tool.name for tool in request.tools],
            )
            try:
                response, response_came_from_stream = self._call_model_with_recovery(request, use_stream, on_event)
                streamed_text = streamed_text or bool(response.message and response_came_from_stream)
            except ModelResponseTimeout as exc:
                prompt_runtime.fail_pending(session)
                active_turn.status = "failed"
                _emit(on_event, str(exc))
                _log(self.logger, "error", "model_call_timeout", model=self.model, timeout_seconds=self.model_timeout_seconds)
                _record_turn_failed(self.session_recorder, session, active_turn.turn_id, "model_timeout", str(exc))
                return f"Error: {exc}"
            except ProviderCallError as exc:
                prompt_runtime.fail_pending(session)
                active_turn.status = "failed"
                _emit(on_event, f"Model call failed: {exc.error_info.message}")
                _log(
                    self.logger,
                    "error",
                    "model_call_failed",
                    kind=exc.error_info.kind,
                    message=exc.error_info.message,
                    retryable=exc.error_info.retryable,
                )
                _record(
                    self.session_recorder,
                    session.session_id,
                    "error",
                    {"stage": "model_call", "kind": exc.error_info.kind, "message": exc.error_info.message, "recoverable": False},
                )
                _record_turn_failed(self.session_recorder, session, active_turn.turn_id, exc.error_info.kind, exc.error_info.message)
                return f"Error: {exc.error_info.message}"
            _log(
                self.logger,
                "info",
                "model_call_finished",
                tool_calls=[call.name for call in response.tool_calls],
                output_items=len(response.output_items),
                has_message=bool(response.message),
                usage=response.usage.to_dict() if response.usage else None,
            )
            if response.usage is not None:
                session.last_token_usage = response.usage
            if response.fatal_error is not None:
                prompt_runtime.fail_pending(session)
                active_turn.status = "failed"
                _emit(on_event, "Model returned an unrecoverable turn error.")
                _log(
                    self.logger,
                    "error",
                    "model_fatal_error",
                    source=response.fatal_error.source,
                    message=response.fatal_error.message,
                    hint=response.fatal_error.hint,
                )
                _record(
                    self.session_recorder,
                    session.session_id,
                    "error",
                    {"stage": "model_fatal_error", "source": response.fatal_error.source, "message": response.fatal_error.message, "recoverable": False},
                )
                _record_turn_failed(self.session_recorder, session, active_turn.turn_id, response.fatal_error.source, response.fatal_error.message)
                return format_fatal_turn_error(response.fatal_error)
            committed = prompt_runtime.commit_turn(session)
            for item in committed:
                if item.kind == "user_message":
                    _record(self.session_recorder, session.session_id, "user_message", {"content": item.content})
                else:
                    _record(self.session_recorder, session.session_id, "prompt_item", {"item": item.to_event_payload()})
            if committed:
                _record(self.session_recorder, session.session_id, "turn_context", {"turn_id": compiled.pending_turn.turn_id, "data": compiled.pending_turn.turn_context.to_event_payload()})
            output_items = [ensure_time_metadata(item) for item in response.output_items]
            session.last_model_output_items = copy.deepcopy(output_items)
            session.input_items.extend(output_items)
            if output_items:
                _record(self.session_recorder, session.session_id, "assistant_output", {"items": output_items})
            if response.recoverable_errors:
                for error in response.recoverable_errors:
                    _emit(on_event, f"Recovering from tool-call error: {error.tool_name} - {_short_error(error.message)}")
                    _log(self.logger, "error", "model_recoverable_tool_error", tool=error.tool_name, message=error.message, hint=error.retry_hint)
                    session.input_items.append(
                        ensure_time_metadata({
                            "type": "function_call_output",
                            "call_id": error.call_id,
                            "output": format_recoverable_tool_error(error),
                        })
                    )
                    _record(
                        self.session_recorder,
                        session.session_id,
                        "error",
                        {"stage": "tool_call_parse", "tool": error.tool_name, "message": error.message, "recoverable": True},
                    )
                compiled = type(compiled)(
                    instructions=session.base_instructions or compiled.instructions,
                    input_items=_session_context_items(session) + _history_items_for_prompt(session),
                    tools=self.registry.specs(),
                    pending_turn=compiled.pending_turn,
                )
                continue
            if not response.tool_calls:
                stop_result = self._run_stop_hook(session, response.message or "")
                if stop_result.additional_context:
                    session.stage_next_model_context(stop_result.additional_context, "Stop")
                if response.message and not response.output_items:
                    _record(self.session_recorder, session.session_id, "assistant_message", {"content": response.message})
                active_turn.status = "completed"
                _record(self.session_recorder, session.session_id, "turn_completed", {"turn_id": active_turn.turn_id})
                if not stop_result.continue_:
                    return f"Stopped by Stop hook: {stop_result.reason or 'no reason provided'}"
                if stop_result.suppress_output:
                    return ""
                if use_stream and streamed_text:
                    return ""
                return response.message or ""
            if response.message and not response_came_from_stream:
                _emit(on_event, response.message)
            pending_tool_calls = list(response.tool_calls)

            def prepare_tool_call(call: ToolCall) -> _ToolCallState:
                stop_after_tool_message: str | None = None
                active_turn.pending_tool_calls[call.id] = call
                description = self.registry.describe_call(call.name, call.args)
                message = f"Running tool: {call.name}"
                if description:
                    message += f" - {description}"
                _emit(on_event, ProgressEvent("tool_started", message) if use_stream else message)
                _log(self.logger, "info", "tool_call_started", tool=call.name, description=description, args=call.args)
                _record(self.session_recorder, session.session_id, "tool_call", {"call_id": call.id, "tool": call.name, "arguments": call.args, "description": description})
                pre_hook = self._run_pre_tool_use_hook(session, call)
                if call.name == "background_tasks_status" and per_turn_tool_counts.get(call.name, 0) >= 1:
                    result = ToolResult(
                        call.name,
                        "error",
                        error="本轮已经查询过 background_tasks_status；请基于上一次快照回答用户，不要重复轮询。",
                    )
                    stop_after_tool_message = _background_status_repeated_message(background_status_snapshot_text)
                elif pre_hook.decision == "deny" or not pre_hook.continue_:
                    result = ToolResult(call.name, "denied", error=pre_hook.reason or "blocked by PreToolUse hook")
                else:
                    result = None
                return _ToolCallState(call=call, result=result, stop_after_tool_message=stop_after_tool_message)

            def abort_interrupted_tool_call(call: ToolCall) -> None:
                if call in pending_tool_calls:
                    pending_tool_calls.remove(call)
                active_turn.status = "aborted"
                active_turn.cancel_requested = True
                active_turn.clear_pending()
                prompt_runtime.fail_pending(session, "interrupted")
                _record_turn_aborted(self.session_recorder, session, active_turn.turn_id, "user_interrupted", "Tool execution interrupted by user before completion.")

            def run_single_tool_call(state: _ToolCallState) -> None:
                if state.result is not None:
                    return
                try:
                    state.result = self.registry.run(state.call.name, state.call.args)
                except KeyboardInterrupt:
                    _record_interrupted_tool_call(self.session_recorder, session, state.call, self.registry)
                    abort_interrupted_tool_call(state.call)
                    raise

            def run_parallel_tool_calls(states: list[_ToolCallState]) -> None:
                runnable = [state for state in states if state.result is None]
                if not runnable:
                    return
                if len(runnable) == 1:
                    run_single_tool_call(runnable[0])
                    return
                group_id = f"{active_turn.turn_id}:parallel:{runnable[0].call.id}"
                _log(self.logger, "info", "parallel_tool_group_started", group_id=group_id, tools=[state.call.name for state in runnable], count=len(runnable))
                max_workers = max(1, min(self.max_parallel_tool_calls, len(runnable)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [(state, executor.submit(self.registry.run, state.call.name, state.call.args)) for state in runnable]
                    for state, future in futures:
                        try:
                            state.result = future.result()
                        except KeyboardInterrupt:
                            _record_interrupted_tool_call(self.session_recorder, session, state.call, self.registry)
                            abort_interrupted_tool_call(state.call)
                            raise
                _log(self.logger, "info", "parallel_tool_group_finished", group_id=group_id, tools=[state.call.name for state in runnable], count=len(runnable))

            def finalize_tool_call(state: _ToolCallState) -> str | None:
                nonlocal background_status_snapshot_text, last_tool_result_signature, repeated_tool_result_count
                call = state.call
                result = state.result or ToolResult(call.name, "error", error="tool did not produce a result")
                post_hook = self._run_post_tool_use_hook(session, call, result)
                if post_hook.additional_context:
                    session.stage_next_model_context(post_hook.additional_context, "PostToolUse")
                _log(self.logger, "info", "tool_call_finished", tool=call.name, status=result.status, error=result.error)
                _record(self.session_recorder, session.session_id, "tool_result", {"call_id": call.id, "tool": call.name, "status": result.status, "output": result.output, "error": result.error})
                status = result.status
                if result.error:
                    status += f": {_short_error(result.error)}"
                finished_message = f"Tool completed: {call.name} ({status})"
                _emit(on_event, ProgressEvent("tool_finished", finished_message) if use_stream else finished_message)
                formatted_tool_result = self.registry.format_result(result)
                per_turn_tool_counts[call.name] = per_turn_tool_counts.get(call.name, 0) + 1
                if call.name == "background_tasks_status" and background_status_snapshot_text is None:
                    background_status_snapshot_text = formatted_tool_result
                tool_output_item = ensure_time_metadata(
                    {
                        "type": call.output_type,
                        "call_id": call.id,
                        "output": formatted_tool_result,
                    }
                )
                session.input_items.append(tool_output_item)
                if call.name == "load_skill" and result.status == "success":
                    skill_name = str(call.args.get("name") or "").strip()
                    loaded = _loaded_skill_from_library(self.skill_library, skill_name)
                    if loaded is not None:
                        session.remember_loaded_skill(loaded)
                        _log(self.logger, "info", "loaded_skill_remembered", skill=loaded.name, path=loaded.path, content_hash=loaded.content_hash)
                        _record(self.session_recorder, session.session_id, "loaded_skill", loaded.to_payload())
                _record(
                    self.session_recorder,
                    session.session_id,
                    "tool_output",
                    {"item": tool_output_item},
                )
                if call in pending_tool_calls:
                    pending_tool_calls.remove(call)
                active_turn.pending_tool_calls.pop(call.id, None)
                if not post_hook.continue_:
                    active_turn.status = "completed"
                    _record(self.session_recorder, session.session_id, "turn_completed", {"turn_id": active_turn.turn_id})
                    return f"Stopped by PostToolUse hook: {post_hook.reason or 'no reason provided'}"
                if should_cancel is not None and should_cancel() and not pending_tool_calls:
                    active_turn.status = "aborted"
                    active_turn.cancel_requested = True
                    active_turn.clear_pending()
                    prompt_runtime.fail_pending(session, "moved_to_background")
                    _record_turn_aborted(self.session_recorder, session, active_turn.turn_id, "moved_to_background", "Foreground task moved to a background worker after tool completion.")
                    return ""
                if state.stop_after_tool_message is not None:
                    active_turn.status = "completed"
                    _log(self.logger, "warning", "duplicate_background_status_snapshot", tool=call.name)
                    _record(self.session_recorder, session.session_id, "turn_completed", {"turn_id": active_turn.turn_id, "reason": "duplicate_background_status_snapshot"})
                    return state.stop_after_tool_message
                signature = _tool_result_signature(call, result)
                if signature == last_tool_result_signature:
                    repeated_tool_result_count += 1
                else:
                    last_tool_result_signature = signature
                    repeated_tool_result_count = 1
                if repeated_tool_result_count >= REPEATED_UNCHANGED_TOOL_RESULT_LIMIT and not pending_tool_calls:
                    active_turn.status = "completed"
                    _log(self.logger, "warning", "repeated_unchanged_tool_result", tool=call.name, count=repeated_tool_result_count)
                    _record(self.session_recorder, session.session_id, "turn_completed", {"turn_id": active_turn.turn_id, "reason": "repeated_unchanged_tool_result"})
                    return _repeated_tool_result_message(call.name, formatted_tool_result)
                return None

            tool_calls = list(response.tool_calls)
            index = 0
            while index < len(tool_calls):
                call = tool_calls[index]
                if self.registry.supports_parallel_tool_calls(call.name):
                    group: list[ToolCall] = []
                    while index < len(tool_calls) and self.registry.supports_parallel_tool_calls(tool_calls[index].name) and len(group) < max(1, self.max_parallel_tool_calls):
                        group.append(tool_calls[index])
                        index += 1
                    states = [prepare_tool_call(group_call) for group_call in group]
                    run_parallel_tool_calls(states)
                    stop_answer: str | None = None
                    for state in states:
                        answer = finalize_tool_call(state)
                        if stop_answer is None and answer is not None:
                            stop_answer = answer
                    if stop_answer is not None:
                        return stop_answer
                    continue

                state = prepare_tool_call(call)
                index += 1
                run_single_tool_call(state)
                answer = finalize_tool_call(state)
                if answer is not None:
                    return answer
            compiled = type(compiled)(
                instructions=session.base_instructions or compiled.instructions,
                input_items=_session_context_items(session) + session.consume_prompt_context_items() + _history_items_for_prompt(session),
                tools=self.registry.specs(),
                pending_turn=compiled.pending_turn,
            )
        raise MaxTurnsExceeded(f"agent exceeded max_turns={self.max_turns}")

    def compact_context(self, session: Session, reason: str = "manual") -> str:
        result = self._compact_with_hooks(session, PromptRuntime(self._current_instructions()).base_instructions, reason=reason)
        return f"Context compacted: {result.tokens_before} -> {result.tokens_after} estimated tokens."

    def dream_context(self, session: Session) -> str:
        result = DreamRunner(
            provider=self.provider,
            model=self.model,
            max_output_tokens=self.max_output_tokens,
            recorder=self.session_recorder,
        ).run(session)
        if result.accepted:
            return f"Dream accepted: {result.draft_count} diary draft(s). Reason: {result.reason}"
        return f"Dream rejected: {result.reason}"

    def context_status(self, session: Session) -> str:
        usage = session.last_token_usage
        estimated = ContextManager(session.input_items).estimate_tokens(extra_items=_session_context_items(session), instructions=PromptRuntime(self._current_instructions()).base_instructions)
        threshold = self._context_compact_threshold()
        lines = [
            f"Estimated context tokens: {estimated}",
            f"Context window: {model_context_window_tokens(self.model)}",
            f"Compact threshold: {threshold}",
            f"Session items: {session.item_count}",
            f"Compactions: {session.compaction_count}",
        ]
        if session.last_compacted_at:
            lines.append(f"Last compacted: {session.last_compacted_at}")
        if usage is not None:
            if usage.input_tokens is not None:
                lines.append(f"Last input tokens: {usage.input_tokens}")
            cached = usage.cached_tokens if usage.cached_tokens is not None else usage.cache_hit_tokens
            if cached is not None:
                lines.append(f"Last cached tokens: {cached}")
            if usage.output_tokens is not None:
                lines.append(f"Last output tokens: {usage.output_tokens}")
        return "\n".join(lines)

    def _current_instructions(self) -> str:
        if self.instructions_provider is None:
            return self.instructions
        try:
            instructions = self.instructions_provider()
        except Exception as exc:
            _log(self.logger, "error", "instructions_provider_failed", exc=exc)
            return self.instructions
        if not instructions.strip():
            _log(self.logger, "error", "instructions_provider_empty")
            return self.instructions
        return instructions

    def _maybe_auto_compact(self, session: Session, instructions: str, user_task: str, runtime_context_text: str | None, on_event: Callable[[str | ProgressEvent], None] | None) -> None:
        if not self.context_auto_compact:
            return
        extra_items = _session_context_items(session) + [{"role": "user", "content": user_task}]
        if runtime_context_text:
            extra_items.append(_runtime_context_item(runtime_context_text))
        estimated = estimate_tokens(session.input_items, extra_items=extra_items, instructions=instructions)
        threshold = self._context_compact_threshold()
        if estimated < threshold:
            return
        _emit(on_event, _status_event("Context is large; compacting conversation history before continuing...") if self.stream else "Context is large; compacting conversation history before continuing...")
        _log(self.logger, "info", "context_auto_compaction_started", estimated_tokens=estimated, threshold=threshold, context_window=model_context_window_tokens(self.model))
        try:
            self._compact_with_hooks(session, instructions, reason="auto_threshold")
        except Exception as exc:
            _emit(on_event, _status_event(f"Context compaction failed; continuing with full history: {_short_error(str(exc))}") if self.stream else f"Context compaction failed; continuing with full history: {_short_error(str(exc))}")
            _log(self.logger, "error", "context_auto_compaction_failed", error=str(exc))

    def _compact_with_hooks(self, session: Session, instructions: str, reason: str):
        pre = self._run_pre_compact_hook(session, reason)
        if not pre.continue_ or pre.decision == "deny":
            raise RuntimeError(f"compaction skipped by PreCompact hook: {pre.reason or 'no reason provided'}")
        extra_items = []
        if pre.additional_context:
            extra_items.append({"role": "developer", "content": f'<hook_context event="PreCompact">\n{pre.additional_context}\n</hook_context>', "xiaoming": {"kind": "hook_context", "event": "PreCompact"}})
        runtime_context_text = self._runtime_context_text()
        if runtime_context_text:
            extra_items.append(_runtime_context_item(runtime_context_text))
        result = self._context_compactor().compact(session, instructions, reason=reason, extra_items=extra_items)
        post = self._run_post_compact_hook(session, reason, result.tokens_before, result.tokens_after)
        if post.additional_context:
            session.stage_next_model_context(post.additional_context, "PostCompact")
        if not post.continue_:
            raise RuntimeError(f"compaction stopped by PostCompact hook: {post.reason or 'no reason provided'}")
        return result

    def _runtime_context_text(self) -> str | None:
        if self.runtime_context_provider is None:
            return None
        try:
            return self.runtime_context_provider()
        except Exception as exc:
            _log(self.logger, "error", "runtime_context_provider_failed", exc=exc)
            return None

    def _context_compactor(self) -> ContextCompactor:
        return ContextCompactor(
            provider=self.provider,
            model=self.model,
            temperature=0.0,
            max_output_tokens=min(self.max_output_tokens, 4096),
            recent_user_budget_tokens=self.context_recent_user_budget_tokens,
            logger=self.logger,
            recorder=self.session_recorder,
        )

    def _context_compact_threshold(self) -> int:
        if self.context_compact_threshold_tokens is not None:
            return self.context_compact_threshold_tokens
        return compact_threshold_tokens(self.model, self.max_output_tokens)

    def _call_model_with_recovery(self, request: LLMRequest, use_stream: bool, on_event: Callable[[str | ProgressEvent], None] | None) -> tuple[LLMResponse, bool]:
        attempts = 0
        while True:
            try:
                response_came_from_stream = False
                if use_stream:
                    response = self._stream_with_progress(request, on_event)
                    response_came_from_stream = True
                    if _should_fallback_to_non_stream(response):
                        _emit(on_event, _status_event("Streaming tool arguments were incomplete; retrying this model call without streaming."))
                        _log(
                            self.logger,
                            "error",
                            "model_stream_fallback_to_complete",
                            source=response.fatal_error.source if response.fatal_error else "",
                            message=response.fatal_error.message if response.fatal_error else "",
                        )
                        response = self._complete_with_progress(request, on_event)
                        response_came_from_stream = False
                else:
                    response = self._complete_with_progress(request, on_event)
                return response, response_came_from_stream
            except KeyboardInterrupt:
                raise
            except ModelResponseTimeout as exc:
                error_info = AgentErrorInfo(kind="stream_timeout" if use_stream else "http_connection_failed", message=str(exc), retryable=True)
                attempts = self._maybe_retry(attempts, error_info, on_event)
            except ProviderCallError as exc:
                attempts = self._maybe_retry(attempts, exc.error_info, on_event)

    def _maybe_retry(self, attempts: int, error_info: AgentErrorInfo, on_event: Callable[[str | ProgressEvent], None] | None) -> int:
        if not error_info.retryable or attempts >= self.model_max_retries:
            raise ProviderCallError(error_info)
        next_attempt = attempts + 1
        _emit(on_event, f"Model call failed; retrying {next_attempt}/{self.model_max_retries}: {_short_error(error_info.message)}")
        _log(
            self.logger,
            "error",
            "model_call_retry_scheduled",
            kind=error_info.kind,
            message=error_info.message,
            attempt=next_attempt,
            max_attempts=self.model_max_retries,
        )
        delay = _retry_delay(self.retry_base_delay_seconds, next_attempt)
        if delay > 0:
            time.sleep(delay)
        return next_attempt

    def _run_session_start_hook(self, session: Session) -> str | None:
        if self.hooks is None or not self.hooks.has_hooks():
            return None
        if getattr(session, "_session_start_hook_ran", False):
            return None
        setattr(session, "_session_start_hook_ran", True)
        result = self.hooks.run("SessionStart", {"session_id": session.session_id, "resumed": session.resumed})
        _log(self.logger, "info", "hook_session_start", continue_=result.continue_, decision=result.decision, reason=result.reason)
        if result.additional_context:
            context = BootstrapContext.create(
                plugin_name="runtime",
                source=f"runtime:hook:SessionStart:{session.session_id or 'session'}",
                content=result.additional_context,
            )
            session.remember_bootstrap_context(context)
            _record(self.session_recorder, session.session_id, "bootstrap_context", context.to_payload())
        if not result.continue_ or result.decision == "deny":
            return f"Stopped by SessionStart hook: {result.reason or 'no reason provided'}"
        return None

    def _inject_explicit_skills(self, session: Session, user_task: str) -> None:
        if self.skill_library is None:
            return
        for skill in self.skill_library.explicit_skills_for_text(user_task):
            if skill.name in session.loaded_skills:
                continue
            loaded = LoadedSkill.create(
                name=skill.name,
                description=skill.description,
                content=skill.content,
                path=str(skill.path) if skill.path is not None else "",
            )
            session.remember_loaded_skill(loaded)
            _log(self.logger, "info", "explicit_skill_auto_injected", skill=loaded.name, path=loaded.path, content_hash=loaded.content_hash)
            _record(self.session_recorder, session.session_id, "loaded_skill", loaded.to_payload())

    def _run_user_prompt_submit_hook(self, session: Session, user_task: str) -> tuple[str, str | None]:
        if self.hooks is None or not self.hooks.has_hooks():
            return user_task, None
        result = self.hooks.run("UserPromptSubmit", {"session_id": session.session_id, "user_input": user_task})
        _log(self.logger, "info", "hook_user_prompt_submit", continue_=result.continue_, decision=result.decision, reason=result.reason)
        if not result.continue_ or result.decision == "deny":
            return user_task, f"Stopped by UserPromptSubmit hook: {result.reason or 'no reason provided'}"
        if result.additional_context:
            session.stage_current_turn_context(result.additional_context, "UserPromptSubmit")
        return result.updated_input if result.updated_input is not None else user_task, None

    def _run_pre_tool_use_hook(self, session: Session, call: ToolCall):
        if self.hooks is None or not self.hooks.has_hooks():
            return HookResult()
        result = self.hooks.run("PreToolUse", {"session_id": session.session_id, "tool": call.name, "arguments": call.args})
        _log(self.logger, "info", "hook_pre_tool_use", tool=call.name, continue_=result.continue_, decision=result.decision, reason=result.reason)
        return result

    def _run_post_tool_use_hook(self, session: Session, call: ToolCall, result: ToolResult) -> HookResult:
        if self.hooks is None or not self.hooks.has_hooks():
            return HookResult()
        hook_result = self.hooks.run("PostToolUse", {"session_id": session.session_id, "tool": call.name, "arguments": call.args, "status": result.status, "output": result.output, "error": result.error})
        _log(self.logger, "info", "hook_post_tool_use", tool=call.name, continue_=hook_result.continue_, decision=hook_result.decision, reason=hook_result.reason)
        return hook_result

    def _run_stop_hook(self, session: Session, message: str) -> HookResult:
        if self.hooks is None or not self.hooks.has_hooks():
            return HookResult()
        result = self.hooks.run("Stop", {"session_id": session.session_id, "message": message})
        _log(self.logger, "info", "hook_stop", continue_=result.continue_, decision=result.decision, reason=result.reason)
        return result

    def _run_pre_compact_hook(self, session: Session, reason: str) -> HookResult:
        if self.hooks is None or not self.hooks.has_hooks():
            return HookResult()
        result = self.hooks.run("PreCompact", {"session_id": session.session_id, "reason": reason, "model": self.model})
        _log(self.logger, "info", "hook_pre_compact", continue_=result.continue_, decision=result.decision, reason=result.reason)
        return result

    def _run_post_compact_hook(self, session: Session, reason: str, tokens_before: int, tokens_after: int) -> HookResult:
        if self.hooks is None or not self.hooks.has_hooks():
            return HookResult()
        result = self.hooks.run("PostCompact", {"session_id": session.session_id, "reason": reason, "model": self.model, "tokens_before": tokens_before, "tokens_after": tokens_after})
        _log(self.logger, "info", "hook_post_compact", continue_=result.continue_, decision=result.decision, reason=result.reason)
        return result

    def _skill_context_for_task(self, user_task: str, on_event: Callable[[str | ProgressEvent], None] | None) -> str | None:
        if self.skill_library is None:
            return None
        parts = []
        available = self.skill_library.render_available()
        if available:
            parts.append(available)
        selected = self.skill_library.select_for_task(user_task)
        if selected:
            for skill in selected:
                _emit(on_event, f"Loaded skill: {skill.name}")
            parts.append(self.skill_library.render_for_task(user_task))
        return "\n\n".join(parts) or None

    def _complete_with_progress(self, request: LLMRequest, on_event: Callable[[str | ProgressEvent], None] | None) -> LLMResponse:
        result_queue: queue.Queue[tuple[str, LLMResponse | BaseException]] = queue.Queue(maxsize=1)

        def run_complete() -> None:
            try:
                result_queue.put(("response", self.provider.complete(request)))
            except BaseException as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=run_complete, daemon=True)
        thread.start()
        started_at = time.monotonic()
        while True:
            elapsed = time.monotonic() - started_at
            remaining = self.model_timeout_seconds - elapsed
            if remaining <= 0:
                raise ModelResponseTimeout(f"model response timed out after {self.model_timeout_seconds:g} seconds")
            try:
                kind, payload = result_queue.get(timeout=min(self.progress_interval_seconds, remaining))
            except queue.Empty:
                _emit(on_event, "Still waiting for model response...")
                continue
            if kind == "error":
                raise payload  # type: ignore[misc]
            return payload  # type: ignore[return-value]

    def _stream_with_progress(self, request: LLMRequest, on_event: Callable[[str | ProgressEvent], None] | None) -> LLMResponse:
        event_queue: queue.Queue[object] = queue.Queue()
        sentinel = object()

        def consume() -> None:
            try:
                for event in self.provider.stream(request):  # type: ignore[attr-defined]
                    event_queue.put(event)
            except BaseException as exc:
                event_queue.put(exc)
            finally:
                event_queue.put(sentinel)

        threading.Thread(target=consume, daemon=True).start()
        accumulator = StreamAccumulator(source=type(self.provider).__name__)
        started_at = time.monotonic()
        last_event_at = started_at
        finish_reason: str | None = None
        last_event_kind = "started"
        while True:
            now = time.monotonic()
            if now - started_at >= self.model_timeout_seconds:
                _log(self.logger, "error", "model_stream_timeout", reason="total", last_event_kind=last_event_kind, text_chars=len(accumulator.text))
                raise ModelResponseTimeout(f"model response timed out after {self.model_timeout_seconds:g} seconds")
            try:
                event = event_queue.get(timeout=min(self.stream_idle_timeout_seconds, max(self.model_timeout_seconds - (now - started_at), 0.001)))
            except queue.Empty:
                _log(
                    self.logger,
                    "error",
                    "model_stream_timeout",
                    reason="idle",
                    last_event_kind=last_event_kind,
                    last_event_age=time.monotonic() - last_event_at,
                    text_chars=len(accumulator.text),
                    tool_arg_chars={index: len(buffer.arguments) for index, buffer in accumulator.tool_buffers.items()},
                )
                raise ModelResponseTimeout(f"model stream idle timed out after {self.stream_idle_timeout_seconds:g} seconds")
            if event is sentinel:
                return accumulator.to_response(finish_reason=finish_reason)
            if isinstance(event, BaseException):
                raise event
            last_event_at = time.monotonic()
            if isinstance(event, StreamTextDelta):
                last_event_kind = "text_delta"
                accumulator.add_text(event.text)
                _log(self.logger, "info", "model_stream_text_delta", chars=len(event.text))
                _emit(on_event, ProgressEvent("text_delta", event.text, end=""))
                continue
            if isinstance(event, StreamToolCallDelta):
                last_event_kind = "tool_delta"
                accumulator.add_tool_delta(event)
                _log(self.logger, "info", "model_stream_tool_delta", index=event.index, name=event.name, args_chars=len(event.arguments_delta))
                continue
            if isinstance(event, StreamDone):
                last_event_kind = "done"
                finish_reason = event.finish_reason
                _log(self.logger, "info", "model_stream_done", finish_reason=finish_reason, text_chars=len(accumulator.text), tool_calls=len(accumulator.tool_buffers))
                continue
            if isinstance(event, StreamUsage):
                last_event_kind = "usage"
                accumulator.set_usage(event.usage)
                _log(self.logger, "info", "model_stream_usage", usage=event.usage.to_dict())
                continue
            if isinstance(event, StreamError):
                raise ModelResponseTimeout(event.message)
        raise AssertionError("unreachable")


def _emit(on_event: Callable[[str | ProgressEvent], None] | None, message: str | ProgressEvent) -> None:
    if on_event is not None:
        on_event(message)


def _pending_worker_questions_text(provider: Callable[..., str | None] | None, user_task: str) -> str | None:
    if provider is None:
        return None
    try:
        return provider(user_task)
    except TypeError:
        return provider()


def _tool_result_signature(call: ToolCall, result: ToolResult) -> tuple[str, str, str, str, str]:
    try:
        args = json.dumps(call.args, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        args = repr(call.args)
    return (call.name, args, result.status, result.output, result.error)


def _repeated_tool_result_message(tool_name: str, formatted_tool_result: str) -> str:
    return f"连续多次调用 {tool_name} 得到的结果没有变化，我先停止本轮，避免继续重复查询。\n\n最新结果：\n{formatted_tool_result}"


def _background_status_repeated_message(snapshot: str | None) -> str:
    if snapshot:
        return f"本轮已经查询过 background_tasks_status，我先停止重复轮询。\n\n上一次状态快照：\n{snapshot}"
    return "本轮已经查询过 background_tasks_status，我先停止重复轮询。请基于已有状态快照回复用户。"


def _session_context_items(session: Session) -> list[dict]:
    from xiaoming.prompting.runtime import _bootstrap_context_input_items, _loaded_skill_input_items

    return _bootstrap_context_input_items(session, dynamic=False) + _loaded_skill_input_items(session) + _bootstrap_context_input_items(session, dynamic=True)


def _history_items_for_prompt(session: Session) -> list[dict]:
    return ContextManager(session.input_items).for_prompt()


def _runtime_context_item(content: str) -> dict:
    return {
        "role": "developer",
        "content": f"<runtime_context>\n{content}\n</runtime_context>",
        "xiaoming": {"kind": "runtime_context", "durable": False},
    }


def _loaded_skill_from_library(library: SkillLibrary | None, requested_name: str) -> LoadedSkill | None:
    if library is None or not requested_name:
        return None
    skill = library.load(requested_name)
    if skill is None:
        return None
    return LoadedSkill.create(
        name=skill.name,
        description=skill.description,
        content=skill.content,
        path=str(skill.path) if skill.path is not None else "",
    )


def _log(logger: XiaomingLogger | None, level: str, event: str, **fields) -> None:
    if logger is None:
        return
    if level == "error":
        logger.error(event, **fields)
    else:
        logger.info(event, **fields)


def _record(recorder: object | None, session_id: str | None, event_type: str, payload: dict) -> None:
    if recorder is None:
        return
    append = getattr(recorder, "append", None)
    if append is None:
        return
    append(session_id, event_type, payload)


def _record_interrupted_tool_call(recorder: object | None, session: Session, call: ToolCall, registry: ToolRegistry) -> None:
    output_type = call.output_type
    output = f"Tool: {call.name}\nStatus: interrupted\nError:\nTool execution interrupted by user before completion."
    item = {"type": output_type, "call_id": call.id, "output": output}
    session.input_items.append(item)
    _record(
        recorder,
        session.session_id,
        "tool_result",
        {
            "call_id": call.id,
            "tool": call.name,
            "status": "interrupted",
            "output": "",
            "error": "Tool execution interrupted by user before completion.",
        },
    )
    _record(recorder, session.session_id, "tool_output", {"item": item})


def _record_turn_aborted(recorder: object | None, session: Session, turn_id: str, reason: str, message: str) -> None:
    session.input_items.append(
        {
            "role": "user",
            "content": f"<turn_aborted>\n{message}\n</turn_aborted>",
            "xiaoming": {"kind": "turn_aborted", "durable": True},
        }
    )
    _record(
        recorder,
        session.session_id,
        "turn_aborted",
        {
            "turn_id": turn_id,
            "reason": reason,
            "message": message,
        },
    )


def _record_turn_failed(recorder: object | None, session: Session, turn_id: str, kind: str, message: str) -> None:
    _record(
        recorder,
        session.session_id,
        "turn_failed",
        {
            "turn_id": turn_id,
            "kind": kind,
            "message": message,
        },
    )


def _retry_delay(base_seconds: float, attempt: int) -> float:
    if base_seconds <= 0:
        return 0
    jitter = random.uniform(0.9, 1.1)
    return base_seconds * (2 ** max(attempt - 1, 0)) * jitter


def _short_error(message: str, max_chars: int = 180) -> str:
    message = " ".join(message.split())
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 15] + "...[truncated]"


def _has_stream(provider: LLMProvider) -> bool:
    return callable(getattr(provider, "stream", None))


def _status_event(message: str) -> ProgressEvent:
    return ProgressEvent("status", message)


def _should_fallback_to_non_stream(response: LLMResponse) -> bool:
    if response.fatal_error is None:
        return False
    return "streamed tool arguments were incomplete or invalid" in response.fatal_error.message
