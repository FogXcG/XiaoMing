from __future__ import annotations

from typing import Protocol

from collections.abc import Iterator

from xiaoming.llm.types import LLMRequest, LLMResponse
from xiaoming.llm.streaming import ModelStreamEvent


class LLMProvider(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...

    def stream(self, request: LLMRequest) -> Iterator[ModelStreamEvent]:
        ...
