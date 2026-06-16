from xiaoming.memory.fragments import fragments_from_items


def test_fragments_from_items_uses_time_metadata_and_content():
    items = [
        {
            "role": "user",
            "content": "hello",
            "xiaoming": {
                "id": "item-1",
                "kind": "user_message",
                "created_at": "2026-06-05T10:42:00+08:00",
                "timezone": "Asia/Shanghai",
            },
        }
    ]

    fragments = fragments_from_items(items)

    assert len(fragments) == 1
    assert fragments[0].id == "fragment:item-1"
    assert fragments[0].source_event_id == "item-1"
    assert fragments[0].role_or_type == "user_message"
    assert fragments[0].created_at == "2026-06-05T10:42:00+08:00"
    assert fragments[0].timezone == "Asia/Shanghai"
    assert fragments[0].content == "hello"


def test_fragments_skip_non_text_items_without_call_output():
    assert fragments_from_items([{"role": "assistant", "content": None}]) == []


def test_fragments_include_tool_outputs():
    fragments = fragments_from_items(
        [
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "Tool: read_file\nStatus: success\nOutput:\nhello",
                "xiaoming": {
                    "created_at": "2026-06-05T10:43:00+08:00",
                    "timezone": "Asia/Shanghai",
                },
            }
        ]
    )

    assert fragments[0].role_or_type == "function_call_output"
    assert fragments[0].source_event_id == "call-1"
    assert "Tool: read_file" in fragments[0].content
