from types import SimpleNamespace

from xiaoming.llm.openai_provider import extract_response, extract_response_with_tools
from xiaoming.llm.types import ToolSpec


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
