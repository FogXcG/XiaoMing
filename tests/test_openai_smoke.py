import os

import pytest

from xiaoming.llm.openai_provider import OpenAIProvider
from xiaoming.llm.types import LLMRequest


@pytest.mark.openai
def test_openai_provider_smoke():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required")

    provider = OpenAIProvider()
    response = provider.complete(
        LLMRequest(
            instructions="Return a short greeting.",
            input_items=[{"role": "user", "content": "Say hello."}],
            tools=[],
            model=os.environ.get("XIAOMING_OPENAI_MODEL", "gpt-5"),
            temperature=0.2,
            max_output_tokens=128,
        )
    )

    assert response.message or response.tool_calls == []
