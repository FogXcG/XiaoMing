from xiaoming.llm.openai_tools import to_openai_tool
from xiaoming.llm.types import ToolSpec


def test_to_openai_tool_enables_strict_function_schema():
    spec = ToolSpec(
        name="read_file",
        description="Read a file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": ["integer", "null"]},
            },
            "required": ["path", "limit"],
            "additionalProperties": False,
        },
    )

    tool = to_openai_tool(spec)

    assert tool["type"] == "function"
    assert tool["name"] == "read_file"
    assert tool["description"] == "Read a file."
    assert tool["strict"] is True
    assert tool["parameters"]["additionalProperties"] is False
    assert tool["parameters"]["required"] == ["path", "limit"]


def test_to_openai_tool_rejects_non_object_schema():
    spec = ToolSpec(name="bad", description="Bad.", input_schema={"type": "string"})

    try:
        to_openai_tool(spec)
    except ValueError as exc:
        assert "object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_to_openai_tool_rejects_missing_additional_properties_false():
    spec = ToolSpec(
        name="bad",
        description="Bad.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )

    try:
        to_openai_tool(spec)
    except ValueError as exc:
        assert "additionalProperties" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_to_openai_tool_uses_custom_tool_for_freeform_specs():
    spec = ToolSpec(
        name="apply_patch",
        description="Apply a patch.",
        input_schema={
            "type": "object",
            "properties": {"patch": {"type": "string"}},
            "required": ["patch"],
            "additionalProperties": False,
        },
        input_mode="freeform",
        freeform_arg="patch",
    )

    tool = to_openai_tool(spec)

    assert tool == {
        "type": "custom",
        "name": "apply_patch",
        "description": "Apply a patch.",
        "format": {"type": "text"},
    }
