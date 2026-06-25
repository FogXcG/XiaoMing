from __future__ import annotations

from typing import Any
import json

from openai import OpenAI

from xiaoming.agent_errors import FatalTurnError, ProviderCallError, RecoverableToolError
from xiaoming.llm.errors import provider_call_error
from xiaoming.llm.openai_tools import to_openai_tools
from xiaoming.llm.types import LLMRequest, LLMResponse, ToolCall, ToolSpec
from xiaoming.llm.usage import extract_token_usage


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            raw = self.client.responses.create(
                model=request.model,
                instructions=request.instructions,
                input=[_to_openai_input_item(item) for item in request.input_items],
                tools=to_openai_tools(request.tools),
                temperature=request.temperature,
                max_output_tokens=request.max_output_tokens,
                parallel_tool_calls=True,
            )
        except ProviderCallError:
            raise
        except Exception as exc:
            raise provider_call_error(exc, source="openai") from exc
        return extract_response_with_tools(raw, request.tools)


def extract_response(raw: Any) -> LLMResponse:
    return extract_response_with_tools(raw, [])


def extract_response_with_tools(raw: Any, tools: list[ToolSpec]) -> LLMResponse:
    output_items = [_to_plain_dict(item) for item in getattr(raw, "output", [])]
    tool_calls: list[ToolCall] = []
    recoverable_errors: list[RecoverableToolError] = []
    freeform_args = {tool.name: tool.freeform_arg for tool in tools if tool.input_mode == "freeform" and tool.freeform_arg}
    for item in getattr(raw, "output", []):
        item_type = _get(item, "type")
        if item_type == "custom_tool_call":
            call_id = _get(item, "call_id")
            name = _get(item, "name")
            if not call_id or not name:
                return LLMResponse(
                    message=None,
                    tool_calls=[],
                    output_items=output_items,
                    raw=raw,
                    fatal_error=FatalTurnError(
                        source="openai",
                        message="model returned an invalid custom tool call without a call id or name",
                        hint="Retry the request; if it repeats, switch models or split the task.",
                    ),
                    usage=extract_token_usage(raw),
                )
            arg_name = freeform_args.get(name, "input")
            tool_calls.append(ToolCall(id=call_id, name=name, args={arg_name: _get(item, "input") or ""}, output_type="custom_tool_call_output"))
            continue
        if item_type != "function_call":
            continue
        arguments = _get(item, "arguments") or "{}"
        call_id = _get(item, "call_id")
        name = _get(item, "name")
        if not call_id or not name:
            return LLMResponse(
                message=None,
                tool_calls=[],
                output_items=output_items,
                raw=raw,
                fatal_error=FatalTurnError(
                    source="openai",
                    message="model returned an invalid function call without a call id or name",
                    hint="Retry the request; if it repeats, switch models or split the task.",
                ),
                usage=extract_token_usage(raw),
            )
        try:
            args = json.loads(arguments)
        except json.JSONDecodeError as exc:
            recoverable_errors.append(
                RecoverableToolError(
                    call_id=call_id,
                    tool_name=name,
                    message=f"failed to parse tool arguments: {exc}",
                    retry_hint="Retry with a smaller tool call or split the change.",
                )
            )
            continue
        if not isinstance(args, dict):
            recoverable_errors.append(
                RecoverableToolError(
                    call_id=call_id,
                    tool_name=name,
                    message="tool arguments must decode to an object",
                    retry_hint="Retry using an object for tool arguments.",
                )
            )
            continue
        tool_calls.append(ToolCall(id=call_id, name=name, args=args))
    text = getattr(raw, "output_text", None) or None
    if recoverable_errors:
        text = None
    return LLMResponse(message=text, tool_calls=tool_calls, output_items=output_items, raw=raw, usage=extract_token_usage(raw), recoverable_errors=recoverable_errors)


def _get(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name)


def _to_plain_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    raise TypeError(f"cannot convert OpenAI output item to dict: {type(item)!r}")


def _to_openai_input_item(item: dict[str, Any]) -> dict[str, Any]:
    if item.get("role") == "developer":
        return {"role": "developer", "content": item.get("content") or ""}
    return {key: value for key, value in item.items() if key != "xiaoming"}
