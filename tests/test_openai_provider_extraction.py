from types import SimpleNamespace

from xiaoming.llm.openai_provider import OpenAIProvider, extract_response, extract_response_with_tools
from xiaoming.llm.types import LLMRequest, ToolSpec


def test_extract_response_reads_function_call_and_plain_output_items():
    raw = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="function_call",
                call_id="call_1",
                name="read_file",
                arguments='{"path":"app.py","start_line":null,"limit":null}',
                model_dump=lambda: {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"app.py","start_line":null,"limit":null}',
                },
            )
        ],
    )

    response = extract_response(raw)

    assert response.message is None
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].args == {"path": "app.py", "start_line": None, "limit": None}
    assert isinstance(response.output_items[0], dict)


def test_extract_response_reads_final_message():
    raw = SimpleNamespace(
        output_text="Done.",
        output=[],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=4,
            total_tokens=14,
            input_tokens_details=SimpleNamespace(cached_tokens=6),
            output_tokens_details=SimpleNamespace(reasoning_tokens=2),
        ),
    )

    response = extract_response(raw)

    assert response.message == "Done."
    assert response.tool_calls == []
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 4
    assert response.usage.total_tokens == 14
    assert response.usage.cached_tokens == 6
    assert response.usage.reasoning_tokens == 2


def test_extract_response_reads_custom_tool_call_as_freeform_argument():
    raw = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(
                type="custom_tool_call",
                call_id="call_1",
                name="apply_patch",
                input="*** Begin Patch\n*** End Patch",
                model_dump=lambda: {
                    "type": "custom_tool_call",
                    "call_id": "call_1",
                    "name": "apply_patch",
                    "input": "*** Begin Patch\n*** End Patch",
                },
            )
        ],
    )
    spec = ToolSpec(
        name="apply_patch",
        description="Apply a patch.",
        input_schema={"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"], "additionalProperties": False},
        input_mode="freeform",
        freeform_arg="patch",
    )

    response = extract_response_with_tools(raw, [spec])

    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "apply_patch"
    assert response.tool_calls[0].args == {"patch": "*** Begin Patch\n*** End Patch"}
    assert response.tool_calls[0].output_type == "custom_tool_call_output"


def test_openai_provider_enables_parallel_tool_calls(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text="Done.", output=[])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            pass

        responses = FakeResponses()

    monkeypatch.setattr("xiaoming.llm.openai_provider.OpenAI", FakeOpenAI)

    provider = OpenAIProvider(api_key="test")
    response = provider.complete(
        LLMRequest(
            instructions="rules",
            input_items=[{"role": "user", "content": "search twice"}],
            tools=[
                ToolSpec(
                    name="web_search",
                    description="Search web.",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"], "additionalProperties": False},
                )
            ],
            model="gpt-5",
            temperature=0.2,
            max_output_tokens=128,
        )
    )

    assert response.message == "Done."
    assert captured["parallel_tool_calls"] is True
