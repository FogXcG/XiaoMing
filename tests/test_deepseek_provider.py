from types import SimpleNamespace

from xiaoming.llm.deepseek_provider import DEEPSEEK_BASE_URL, DeepSeekProvider, extract_response, extract_stream_events, input_items_to_messages, to_deepseek_tool
from xiaoming.llm.streaming import StreamDone, StreamTextDelta, StreamToolCallDelta, StreamUsage
from xiaoming.llm.types import LLMRequest, ToolSpec


def test_to_deepseek_tool_uses_strict_chat_completions_function_shape():
    spec = ToolSpec(
        name="read_file",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    tool = to_deepseek_tool(spec)

    assert tool["function"]["parameters"] == spec.input_schema


def test_to_deepseek_tool_converts_array_schema_to_string():
    spec = ToolSpec(
        name="schedule_background_task",
        description="Schedule.",
        input_schema={
            "type": "object",
            "properties": {"success_criteria": {"type": "array", "items": {"type": "string"}}},
            "required": ["success_criteria"],
            "additionalProperties": False,
        },
    )

    tool = to_deepseek_tool(spec)

    assert tool["function"]["parameters"]["properties"]["success_criteria"]["type"] == "string"


def test_input_items_to_messages_converts_function_outputs_to_tool_messages():
    messages = input_items_to_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function"}]},
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ]
    )

    assert messages[-1] == {"role": "tool", "tool_call_id": "call_1", "content": "ok"}


def test_input_items_to_messages_drops_dangling_tool_call_history():
    messages = input_items_to_messages(
        [
            {"role": "user", "content": "clone repo"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function"}]},
            {"role": "user", "content": "用 ssh git clone"},
        ]
    )

    assert messages == [
        {"role": "user", "content": "clone repo"},
        {"role": "user", "content": "用 ssh git clone"},
    ]


def test_input_items_to_messages_maps_developer_to_latest_reminder():
    messages = input_items_to_messages(
        [
            {"role": "developer", "content": "<developer_context>stable instructions</developer_context>"},
            {"role": "user", "content": "hi"},
        ],
        instructions="base instructions",
    )

    assert messages == [
        {"role": "system", "content": "base instructions"},
        {"role": "latest_reminder", "content": "<developer_context>stable instructions</developer_context>"},
        {"role": "user", "content": "hi"},
    ]


def test_extract_response_reads_chat_tool_call():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="read_file", arguments='{"path":"app.py"}'),
            )
        ],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"app.py"}'},
                }
            ],
        },
    )
    raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])

    response = extract_response(raw)

    assert response.message is None
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].args == {"path": "app.py"}
    assert response.output_items[0]["role"] == "assistant"
    assert response.finish_reason == "tool_calls"
    assert response.recoverable_errors == []
    assert response.fatal_error is None


def test_extract_response_preserves_content_before_tool_call():
    message = SimpleNamespace(
        content="I will inspect the repository with list_files.",
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="list_files", arguments='{"path":"."}'),
            )
        ],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "content": "I will inspect the repository with list_files.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": '{"path":"."}'},
                }
            ],
        },
    )
    raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])

    response = extract_response(raw)

    assert response.message == "I will inspect the repository with list_files."
    assert response.tool_calls[0].name == "list_files"


def test_extract_response_repairs_unquoted_deepseek_tool_arguments():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="list_files", arguments='{"path": ., "pattern": *}'),
            )
        ],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": '{"path": ., "pattern": *}'},
                }
            ],
        },
    )
    raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])

    response = extract_response(raw)

    assert response.tool_calls[0].args == {"path": ".", "pattern": "*"}


def test_extract_response_reports_truncated_tool_arguments_as_recoverable_error():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="apply_patch", arguments='{"patch": "unterminated'),
            )
        ],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "apply_patch", "arguments": '{"patch": "unterminated'},
                }
            ],
        },
    )
    raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="length")])

    response = extract_response(raw)

    assert response.tool_calls == []
    assert response.fatal_error is None
    assert len(response.recoverable_errors) == 1
    error = response.recoverable_errors[0]
    assert error.call_id == "call_1"
    assert error.tool_name == "apply_patch"
    assert "truncated" in error.message
    assert "smaller" in error.retry_hint


def test_extract_response_reports_malformed_tool_arguments_as_recoverable_error():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id="call_1",
                type="function",
                function=SimpleNamespace(name="read_file", arguments='{"path": "app.py",}'),
            )
        ],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "app.py",}'},
                }
            ],
        },
    )
    raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")])

    response = extract_response(raw)

    assert response.tool_calls == []
    assert len(response.recoverable_errors) == 1
    assert response.recoverable_errors[0].call_id == "call_1"
    assert "failed to parse tool arguments" in response.recoverable_errors[0].message


def test_extract_response_reports_length_without_call_id_as_fatal_turn_error():
    message = SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=None,
                type="function",
                function=SimpleNamespace(name="apply_patch", arguments='{"patch": "unterminated'),
            )
        ],
        model_dump=lambda exclude_none=True: {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "apply_patch", "arguments": '{"patch": "unterminated'},
                }
            ],
        },
    )
    raw = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="length")])

    response = extract_response(raw)

    assert response.tool_calls == []
    assert response.recoverable_errors == []
    assert response.fatal_error is not None
    assert "truncated" in response.fatal_error.message


def test_extract_response_reads_chat_final_message():
    message = SimpleNamespace(content="Done.", tool_calls=None, model_dump=lambda exclude_none=True: {"role": "assistant", "content": "Done."})
    raw = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17, prompt_cache_hit_tokens=7, prompt_cache_miss_tokens=5),
    )

    response = extract_response(raw)

    assert response.message == "Done."
    assert response.tool_calls == []
    assert response.finish_reason == "stop"
    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 5
    assert response.usage.total_tokens == 17
    assert response.usage.cache_hit_tokens == 7
    assert response.usage.cache_miss_tokens == 5


def test_extract_stream_events_maps_content_delta():
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel", tool_calls=None), finish_reason=None)])

    assert extract_stream_events(chunk) == [StreamTextDelta("Hel")]


def test_extract_stream_events_maps_usage_chunk_without_choices():
    chunk = SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5, total_tokens=17, prompt_cache_hit_tokens=7))

    events = extract_stream_events(chunk)

    assert len(events) == 1
    assert isinstance(events[0], StreamUsage)
    assert events[0].usage.input_tokens == 12
    assert events[0].usage.output_tokens == 5
    assert events[0].usage.total_tokens == 17
    assert events[0].usage.cache_hit_tokens == 7


def test_extract_stream_events_maps_tool_call_delta_and_done():
    chunk = SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[
                        SimpleNamespace(index=0, id="call_1", function=SimpleNamespace(name="write_file", arguments='{"path"')),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )

    assert extract_stream_events(chunk) == [
        StreamToolCallDelta(index=0, call_id="call_1", name="write_file", arguments_delta='{"path"'),
        StreamDone("tool_calls"),
    ]


def test_deepseek_provider_stream_requests_streaming_chunks(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi", tool_calls=None), finish_reason="stop")])]

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("xiaoming.llm.deepseek_provider.OpenAI", FakeClient)

    provider = DeepSeekProvider()
    events = list(
        provider.stream(
            LLMRequest(
                instructions="rules",
                input_items=[{"role": "user", "content": "hi"}],
                tools=[],
                model="deepseek-v4-flash",
                temperature=0.2,
                max_output_tokens=128,
            )
        )
    )

    assert captured["stream"] is True
    assert captured["stream_options"] == {"include_usage": True}
    assert events == [StreamTextDelta("Hi"), StreamDone("stop")]


def test_provider_reads_deepseek_api_key_from_environment(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("xiaoming.llm.deepseek_provider.OpenAI", FakeClient)

    DeepSeekProvider()

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == DEEPSEEK_BASE_URL


def test_provider_disables_thinking_by_default(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = SimpleNamespace(content="Done.", tool_calls=None, model_dump=lambda exclude_none=True: {"role": "assistant", "content": "Done."})
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr("xiaoming.llm.deepseek_provider.OpenAI", FakeClient)

    provider = DeepSeekProvider()
    provider.complete(
        LLMRequest(
            instructions="rules",
            input_items=[{"role": "user", "content": "hi"}],
            tools=[],
            model="deepseek-v4-flash",
            temperature=0.2,
            max_output_tokens=128,
        )
    )

    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
