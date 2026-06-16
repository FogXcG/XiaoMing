from xiaoming.session import Session


def test_session_starts_empty_and_can_clear():
    session = Session()
    session.input_items.append({"role": "user", "content": "hi"})

    assert session.item_count == 1

    session.clear()

    assert session.item_count == 0
