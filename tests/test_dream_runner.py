from xiaoming.llm.types import LLMResponse, ToolCall
from xiaoming.memory.dream_runner import DreamRunner
from xiaoming.session import Session


class DreamProvider:
    def __init__(self):
        self.requests = []
        self.calls = 0

    def complete(self, request):
        self.requests.append(request)
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                message=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="write_diary_draft",
                        args={
                            "scope": "day",
                            "start_time": "2026-06-04T00:00:00+08:00",
                            "end_time": "2026-06-05T00:00:00+08:00",
                            "body": "I wrote a diary.",
                            "source_ids": ["fragment:item-1"],
                        },
                    )
                ],
                output_items=[],
                raw=None,
            )
        return LLMResponse(
            message=None,
            tool_calls=[ToolCall(id="call-2", name="accept_dream", args={"reason": "candidate view is coherent"})],
            output_items=[],
            raw=None,
        )


def test_dream_runner_commits_draft_after_accept():
    session = Session(session_id="session-1")
    session.input_items.append(
        {
            "role": "user",
            "content": "old context",
            "xiaoming": {
                "id": "item-1",
                "kind": "user_message",
                "created_at": "2026-06-04T10:00:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        }
    )
    provider = DreamProvider()
    runner = DreamRunner(provider=provider, model="deepseek-v4-flash", max_output_tokens=4096)

    result = runner.run(session)

    assert result.accepted is True
    assert len(session.memory_diaries) == 1
    assert next(iter(session.memory_diaries.values())).status == "active"
    assert provider.requests[0].tools[0].name == "list_memory_packets"


def test_dream_runner_rejects_when_model_never_accepts():
    class NoAcceptProvider:
        def complete(self, request):
            return LLMResponse(message="still thinking", tool_calls=[], output_items=[{"role": "assistant", "content": "still thinking"}], raw=None)

    session = Session(session_id="session-1")
    session.input_items.append(
        {
            "role": "user",
            "content": "old context",
            "xiaoming": {
                "id": "item-1",
                "kind": "user_message",
                "created_at": "2026-06-04T10:00:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        }
    )

    result = DreamRunner(provider=NoAcceptProvider(), model="deepseek-v4-flash", max_output_tokens=4096, max_turns=2).run(session)

    assert result.accepted is False
    assert result.reason == "dream exceeded max turns"
    assert session.memory_diaries == {}
