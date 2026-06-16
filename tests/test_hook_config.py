import json
import sys

from xiaoming.hook_config import load_workspace_hooks


def test_load_workspace_hooks_returns_none_without_config(tmp_path):
    assert load_workspace_hooks(tmp_path) is None


def test_workspace_hook_command_can_return_hook_result(tmp_path):
    hook_script = tmp_path / "hook.py"
    hook_script.write_text(
        "import json, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "assert payload['event'] == 'UserPromptSubmit'\n"
        "print(json.dumps({'updated_input': payload['payload']['user_input'] + ' via hook'}))\n"
    )
    (tmp_path / ".xiaoming").mkdir()
    (tmp_path / ".xiaoming" / "hooks.json").write_text(
        json.dumps({"UserPromptSubmit": [{"command": f"{sys.executable} {hook_script}"}]})
    )

    hooks = load_workspace_hooks(tmp_path)
    result = hooks.run("UserPromptSubmit", {"user_input": "hello"})

    assert result.updated_input == "hello via hook"


def test_workspace_hook_command_failure_stops_event(tmp_path):
    hook_script = tmp_path / "hook.py"
    hook_script.write_text("import sys\nsys.exit(3)\n")
    (tmp_path / ".xiaoming").mkdir()
    (tmp_path / ".xiaoming" / "hooks.json").write_text(
        json.dumps({"UserPromptSubmit": [{"command": f"{sys.executable} {hook_script}"}]})
    )

    hooks = load_workspace_hooks(tmp_path)
    result = hooks.run("UserPromptSubmit", {"user_input": "hello"})

    assert result.continue_ is False
    assert "exited with status 3" in result.reason


def test_invalid_workspace_hook_config_is_ignored_and_logged(tmp_path):
    events = []

    class Logger:
        def error(self, event, **fields):
            events.append((event, fields))

    (tmp_path / ".xiaoming").mkdir()
    (tmp_path / ".xiaoming" / "hooks.json").write_text("{bad")

    assert load_workspace_hooks(tmp_path, logger=Logger()) is None
    assert events[0][0] == "hook_config_invalid"


def test_workspace_hook_command_logs_timing(tmp_path):
    events = []
    hook_script = tmp_path / "hook.py"
    hook_script.write_text("print('{}')\n")

    class Logger:
        def info(self, event, **fields):
            events.append((event, fields))

        def error(self, event, **fields):
            events.append((event, fields))

    (tmp_path / ".xiaoming").mkdir()
    (tmp_path / ".xiaoming" / "hooks.json").write_text(
        json.dumps({"Stop": [{"command": f"{sys.executable} {hook_script}"}]})
    )

    hooks = load_workspace_hooks(tmp_path, logger=Logger())
    hooks.run("Stop", {"message": "done"})

    finished = [fields for event, fields in events if event == "hook_command_finished"]
    assert finished
    assert finished[0]["event_name"] == "Stop"
    assert finished[0]["elapsed_ms"] >= 0
