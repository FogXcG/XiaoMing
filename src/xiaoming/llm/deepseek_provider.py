from __future__ import annotations

from typing import Any
import json
import os
import re

from openai import OpenAI

from xiaoming.agent_errors import FatalTurnError, ProviderCallError, RecoverableToolError
from xiaoming.llm.errors import provider_call_error
from xiaoming.llm.streaming import ModelStreamEvent, StreamDone, StreamTextDelta, StreamToolCallDelta, StreamUsage
from xiaoming.llm.types import LLMRequest, LLMResponse, ToolCall, ToolSpec
from xiaoming.llm.usage import extract_token_usage


DEEPSEEK_BASE_URL = "https://api.deepseek.com/beta"


class DeepSeekProvider:
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        kwargs: dict[str, Any] = {"base_url": base_url or DEEPSEEK_BASE_URL}
        resolved_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if resolved_api_key:
            kwargs["api_key"] = resolved_api_key
        self.client = OpenAI(**kwargs)

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            raw = self.client.chat.completions.create(
                model=request.model,
                messages=input_items_to_messages(request.input_items, request.instructions),
                tools=to_deepseek_tools(request.tools) or None,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
        except ProviderCallError:
            raise
        except Exception as exc:
            raise provider_call_error(exc, source="deepseek") from exc
        return extract_response(raw)

    def stream(self, request: LLMRequest):
        try:
            chunks = self.client.chat.completions.create(
                model=request.model,
                messages=input_items_to_messages(request.input_items, request.instructions),
                tools=to_deepseek_tools(request.tools) or None,
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"thinking": {"type": "disabled"}},
            )
            for chunk in chunks:
                yield from extract_stream_events(chunk)
        except ProviderCallError:
            raise
        except Exception as exc:
            raise provider_call_error(exc, source="deepseek") from exc


def to_deepseek_tool(spec: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "strict": True,
            "description": spec.description,
            "parameters": _deepseek_schema(spec.input_schema),
        },
    }


def to_deepseek_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    return [to_deepseek_tool(spec) for spec in specs]


def input_items_to_messages(input_items: list[dict[str, Any]], instructions: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    for item in _drop_dangling_tool_calls(input_items):
        if item.get("type") == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item["call_id"],
                    "content": item["output"],
                }
            )
            continue
        if item.get("role") == "developer":
            messages.append({"role": "latest_reminder", "content": item.get("content") or ""})
            continue
        if item.get("role") in {"user", "assistant", "system", "tool"}:
            messages.append({key: value for key, value in item.items() if key != "xiaoming"})
    return messages


def _drop_dangling_tool_calls(input_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    index = 0
    while index < len(input_items):
        item = input_items[index]
        tool_call_ids = _assistant_tool_call_ids(item)
        if not tool_call_ids:
            cleaned.append(item)
            index += 1
            continue
        next_index = index + 1
        outputs: list[dict[str, Any]] = []
        output_ids: set[str] = set()
        while next_index < len(input_items) and input_items[next_index].get("type") == "function_call_output":
            output = input_items[next_index]
            output_ids.add(str(output.get("call_id") or ""))
            outputs.append(output)
            next_index += 1
        if tool_call_ids.issubset(output_ids):
            cleaned.append(item)
            cleaned.extend(outputs)
        index = next_index
    return cleaned


def _assistant_tool_call_ids(item: dict[str, Any]) -> set[str]:
    if item.get("role") != "assistant":
        return set()
    tool_calls = item.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return set()
    ids = {str(call.get("id") or "") for call in tool_calls if isinstance(call, dict)}
    return {call_id for call_id in ids if call_id}


def extract_response(raw: Any) -> LLMResponse:
    choice = raw.choices[0]
    message = choice.message
    finish_reason = getattr(choice, "finish_reason", None)
    output_item = _message_to_plain_dict(message)
    output_item.setdefault("role", "assistant")
    tool_calls: list[ToolCall] = []
    recoverable_errors: list[RecoverableToolError] = []
    fatal_error: FatalTurnError | None = None
    for call in getattr(message, "tool_calls", None) or []:
        function = _get(call, "function")
        arguments = _get(function, "arguments") or "{}"
        call_id = _get(call, "id")
        name = _get(function, "name")
        if not call_id or not name:
            fatal_error = FatalTurnError(
                source="deepseek",
                message="model returned an invalid tool call without a call id or function name",
                hint="Retry the request; if it repeats, switch models or split the task.",
            )
            continue
        try:
            args = _parse_tool_arguments(arguments)
        except ValueError as exc:
            if finish_reason == "length":
                message_text = "model output was truncated while producing tool arguments"
            else:
                message_text = f"failed to parse tool arguments: {exc}"
            recoverable_errors.append(
                RecoverableToolError(
                    call_id=call_id,
                    tool_name=name,
                    message=message_text,
                    retry_hint="Retry with a smaller tool call, split the change, or use write_file/append_file for large new files.",
                )
            )
            continue
        tool_calls.append(ToolCall(id=call_id, name=name, args=args))
    if finish_reason == "length" and not tool_calls and not recoverable_errors:
        fatal_error = FatalTurnError(
            source="deepseek",
            message="model output was truncated before a complete tool call could be recovered",
            hint="Try again with a smaller request or split the change into smaller chunks.",
        )
    content = getattr(message, "content", None)
    if recoverable_errors:
        content = None
    return LLMResponse(
        message=content,
        tool_calls=tool_calls,
        output_items=[output_item],
        raw=raw,
        finish_reason=finish_reason,
        usage=extract_token_usage(raw),
        recoverable_errors=recoverable_errors,
        fatal_error=fatal_error,
    )


def extract_stream_events(chunk: Any) -> list[ModelStreamEvent]:
    events: list[ModelStreamEvent] = []
    usage = extract_token_usage(chunk)
    if usage is not None:
        events.append(StreamUsage(usage))
    choices = _get(chunk, "choices") or []
    if not choices:
        return events
    choice = choices[0]
    delta = _get(choice, "delta")
    content = _get(delta, "content") if delta is not None else None
    if content:
        events.append(StreamTextDelta(content))
    for call in (_get(delta, "tool_calls") if delta is not None else None) or []:
        function = _get(call, "function")
        events.append(
            StreamToolCallDelta(
                index=int(_get(call, "index") or 0),
                call_id=_get(call, "id"),
                name=_get(function, "name") if function is not None else None,
                arguments_delta=(_get(function, "arguments") if function is not None else None) or "",
            )
        )
    finish_reason = _get(choice, "finish_reason")
    if finish_reason:
        events.append(StreamDone(finish_reason))
    return events


def _get(item: Any, name: str) -> Any:
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name)


def _parse_tool_arguments(arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_quote_unquoted_object_values(arguments))
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must decode to an object")
    return parsed


def _quote_unquoted_object_values(arguments: str) -> str:
    def replace(match: re.Match[str]) -> str:
        prefix, token, suffix = match.groups()
        if token in {"true", "false", "null"}:
            return match.group(0)
        if re.fullmatch(r"-?\d+(\.\d+)?", token):
            return match.group(0)
        return f'{prefix}"{token}"{suffix}'

    return re.sub(r'(:\s*)([A-Za-z0-9_./*+-]+)(\s*[,}])', replace, arguments)


def _message_to_plain_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if hasattr(message, "dict"):
        return message.dict(exclude_none=True)
    raise TypeError(f"cannot convert DeepSeek message to dict: {type(message)!r}")


def _deepseek_schema(schema: dict[str, Any]) -> dict[str, Any]:
    converted = dict(schema)
    schema_type = converted.get("type")
    if isinstance(schema_type, list):
        converted["type"] = next((item for item in schema_type if item != "null"), "string")
    if converted.get("type") == "array":
        description = converted.get("description") or ""
        suffix = " Provide this value as a JSON array string or comma-separated list."
        return {"type": "string", "description": (description + suffix).strip()}
    if converted.get("type") == "object":
        properties = converted.get("properties")
        if isinstance(properties, dict):
            converted["properties"] = {name: _deepseek_schema(value) if isinstance(value, dict) else value for name, value in properties.items()}
            converted["required"] = list(converted["properties"].keys())
    return converted
