import os

import pytest

from xiaoming.llm.deepseek_provider import DeepSeekProvider
from xiaoming.llm.types import LLMRequest, ToolSpec


@pytest.mark.deepseek
def test_deepseek_v4_flash_tool_call_smoke():
    if os.environ.get("XIAOMING_RUN_DEEPSEEK_SMOKE") != "1":
        pytest.skip("set XIAOMING_RUN_DEEPSEEK_SMOKE=1 to run DeepSeek smoke test")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required")

    provider = DeepSeekProvider()
    tools = [
        ToolSpec(
            name="echo",
            description="Echo text back to the caller.",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )
    ]
    first = provider.complete(
        LLMRequest(
            instructions="You must call the echo tool when the user asks you to echo text.",
            input_items=[{"role": "user", "content": "Echo the text xiaoming-deepseek-smoke using the tool."}],
            tools=tools,
            model="deepseek-v4-flash",
            temperature=0.2,
            max_output_tokens=4096,
        )
    )

    assert first.tool_calls
    assert first.tool_calls[0].name == "echo"

    second = provider.complete(
        LLMRequest(
            instructions="You must call the echo tool when the user asks you to echo text.",
            input_items=[
                {"role": "user", "content": "Echo the text xiaoming-deepseek-smoke using the tool."},
                *first.output_items,
                {
                    "type": "function_call_output",
                    "call_id": first.tool_calls[0].id,
                    "output": "xiaoming-deepseek-smoke",
                },
            ],
            tools=tools,
            model="deepseek-v4-flash",
            temperature=0.2,
            max_output_tokens=4096,
        )
    )

    assert "xiaoming-deepseek-smoke" in (second.message or "")
