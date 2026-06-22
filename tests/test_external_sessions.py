from pathlib import Path

from xiaoming.async_runtime.external_sessions import CodexRemoteControlSession


def test_codex_remote_control_starts_thread_and_turn(monkeypatch, tmp_path: Path):
    session = CodexRemoteControlSession(tmp_path)
    calls = []
    events = iter(
        [
            {"method": "item/agentMessage/delta", "params": {"threadId": "thread-1", "turnId": "turn-1", "delta": "OK"}},
            {"method": "thread/status/changed", "params": {"threadId": "thread-1", "status": {"type": "idle"}}},
        ]
    )

    monkeypatch.setattr(session, "_ensure_started", lambda: None)

    def fake_request(method, params=None):
        calls.append((method, params))
        if method == "thread/start":
            return {"result": {"thread": {"id": "thread-1"}}}
        if method == "turn/start":
            return {"result": {"turn": {"id": "turn-1"}}}
        raise AssertionError(method)

    monkeypatch.setattr(session, "_request", fake_request)
    monkeypatch.setattr(session, "_read_json", lambda deadline: next(events))

    answer, thread_id = session.send("hello")

    assert answer == "OK"
    assert thread_id == "thread-1"
    assert calls[0][0] == "thread/start"
    assert calls[0][1]["approvalPolicy"] == "never"
    assert calls[0][1]["sandbox"] == "danger-full-access"
    assert calls[1][0] == "turn/start"
    assert calls[1][1]["input"] == [{"type": "text", "text": "hello", "text_elements": []}]
    assert calls[1][1]["sandboxPolicy"] == {"type": "dangerFullAccess"}


def test_codex_remote_control_resumes_existing_thread(monkeypatch, tmp_path: Path):
    session = CodexRemoteControlSession(tmp_path, session_id="thread-1")
    calls = []
    events = iter(
        [
            {
                "method": "item/completed",
                "params": {
                    "threadId": "thread-1",
                    "turnId": "turn-2",
                    "item": {"type": "agentMessage", "text": "Still here"},
                },
            },
            {"method": "thread/status/changed", "params": {"threadId": "thread-1", "status": {"type": "idle"}}},
        ]
    )

    monkeypatch.setattr(session, "_ensure_started", lambda: None)

    def fake_request(method, params=None):
        calls.append((method, params))
        if method == "thread/resume":
            return {"result": {"thread": {"id": "thread-1"}}}
        if method == "turn/start":
            return {"result": {"turn": {"id": "turn-2"}}}
        raise AssertionError(method)

    monkeypatch.setattr(session, "_request", fake_request)
    monkeypatch.setattr(session, "_read_json", lambda deadline: next(events))

    answer, thread_id = session.send("continue")

    assert answer == "Still here"
    assert thread_id == "thread-1"
    assert calls[0][0] == "thread/resume"
    assert calls[0][1]["threadId"] == "thread-1"
    assert calls[0][1]["excludeTurns"] is True


def test_codex_remote_control_timeout_reports_progress_and_keeps_waiting(monkeypatch, tmp_path: Path):
    session = CodexRemoteControlSession(tmp_path, timeout_seconds=1)
    progress = []
    events = iter(
        [
            TimeoutError("Codex app-server response timed out"),
            {"method": "item/agentMessage/delta", "params": {"threadId": "thread-1", "turnId": "turn-1", "delta": "done"}},
            {"method": "thread/status/changed", "params": {"threadId": "thread-1", "status": {"type": "idle"}}},
        ]
    )

    monkeypatch.setattr(session, "_ensure_started", lambda: None)

    def fake_request(method, params=None):
        if method == "thread/start":
            return {"result": {"thread": {"id": "thread-1"}}}
        if method == "turn/start":
            return {"result": {"turn": {"id": "turn-1"}}}
        raise AssertionError(method)

    def fake_read_json(deadline):
        item = next(events)
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(session, "_request", fake_request)
    monkeypatch.setattr(session, "_read_json", fake_read_json)

    answer, thread_id = session.send("hello", on_progress=progress.append)

    assert answer == "done"
    assert thread_id == "thread-1"
    assert progress == ["Codex is still working; waiting for the next event."]
