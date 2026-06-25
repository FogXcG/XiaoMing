from argparse import Namespace
import json
import threading
import time

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.document import Document
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.widgets import TextArea

from xiaoming.cli import ChatRuntime, SlashCommandCompleter, TuiApprovalController, TuiOutput, _append_completed_tui_output, _apply_selected_completion, _format_tui_approval_action, _help_text, _make_output_window, _make_root_container, _print_progress, approve_action, build_instructions, build_loop, build_registry, build_universal_runtime_tools, discard_pending_terminal_input, enable_line_editing, parse_args, run_chat, run_initialization_wizard, run_loop_with_progress, should_run_initialization, tool_capability_hook
from xiaoming.hooks import HookManager
from xiaoming.progress import ProgressEvent


def test_parse_args_reads_task_and_options():
    args = parse_args(
        [
            "fix tests",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-flash",
            "--approval-mode",
            "auto_edit",
            "--permission-mode",
            "accept_edits",
            "--max-turns",
            "5",
            "--model-timeout",
            "60",
            "--stream",
        ]
    )

    assert args.task == "fix tests"
    assert args.provider == "deepseek"
    assert args.model == "deepseek-v4-flash"
    assert args.approval_mode == "auto_edit"
    assert args.permission_mode == "accept_edits"
    assert args.max_turns == 5
    assert args.model_timeout_seconds == 60
    assert args.stream is True


def test_parse_args_supports_chat_mode():
    args = parse_args(["chat", "--provider", "deepseek"])

    assert args.task == "chat"
    assert args.provider == "deepseek"


def test_parse_args_supports_disabling_stream():
    args = parse_args(["--no-stream"])

    assert args.stream is False


def test_parse_args_supports_init():
    args = parse_args(["--init"])

    assert args.init is True


def test_parse_args_supports_session_resume_options():
    args = parse_args(["--continue"])
    resumed = parse_args(["--resume", "session-123"])

    assert args.continue_session is True
    assert resumed.resume_session_id == "session-123"


def test_parse_args_defaults_to_chat_when_no_task():
    args = parse_args([])

    assert args.task is None
    assert args.stream is None


def test_help_lists_dream_command():
    assert "/dream" in _help_text()


def test_slash_command_completer_lists_commands_for_slash():
    completions = list(SlashCommandCompleter().get_completions(Document("/"), None))

    assert any(completion.text == "/help" for completion in completions)
    assert any(completion.text == "/model" for completion in completions)


def test_slash_command_completer_filters_by_prefix():
    completions = list(SlashCommandCompleter().get_completions(Document("/mo"), None))
    texts = {completion.text for completion in completions}

    assert "/model" in texts
    assert "/model-timeout" in texts
    assert "/help" not in texts


def test_slash_command_completer_ignores_non_command_input():
    completions = list(SlashCommandCompleter().get_completions(Document("hello /mo"), None))

    assert completions == []


def test_slash_command_completer_lists_permission_modes():
    completions = list(SlashCommandCompleter().get_completions(Document("/permission "), None))
    texts = {completion.text for completion in completions}

    assert "/permission-mode default" in texts
    assert "/permission-mode auto" in texts
    assert "/permission-mode bypass" in texts


def test_slash_command_completer_filters_permission_modes():
    completions = list(SlashCommandCompleter().get_completions(Document("/permission a"), None))
    texts = {completion.text for completion in completions}

    assert texts == {"/permission-mode accept_edits", "/permission-mode auto"}


def test_apply_selected_completion_applies_current_choice():
    buffer = Buffer()
    buffer.text = "/he"
    buffer.cursor_position = len("/he")
    buffer.complete_state = CompletionState(
        buffer.document,
        [Completion("/help", start_position=-3)],
        complete_index=0,
    )

    assert _apply_selected_completion(buffer) is True
    assert buffer.text == "/help"
    assert buffer.complete_state is None


def test_apply_selected_completion_ignores_unselected_menu():
    buffer = Buffer()
    buffer.text = "/"
    buffer.cursor_position = 1
    buffer.complete_state = CompletionState(
        buffer.document,
        [Completion("/help", start_position=-1)],
        complete_index=None,
    )

    assert _apply_selected_completion(buffer) is False
    assert buffer.text == "/"


def test_help_text_is_generated_from_slash_command_catalog():
    help_text = _help_text()

    assert "/talk <peer-id> <message>" in help_text
    assert "/skill install <github-tree-url>" in help_text


def test_enable_line_editing_imports_readline():
    imported = []

    class FakeReadline:
        def parse_and_bind(self, value):
            imported.append(("bind", value))

    def fake_import(name):
        imported.append(("import", name))
        return FakeReadline()

    assert enable_line_editing(importer=fake_import) is True
    assert ("import", "readline") in imported
    assert ("bind", "set enable-bracketed-paste on") in imported


def test_discard_pending_terminal_input_ignores_non_tty(monkeypatch):
    class FakeStdin:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", FakeStdin())

    assert discard_pending_terminal_input() is False


def test_print_progress_prints_stream_delta_once(capsys):
    _print_progress(ProgressEvent("text_delta", "Hi", end=""))

    assert capsys.readouterr().out == "Hi"


def test_approve_action_denies_on_eof(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt: (_ for _ in ()).throw(EOFError()))

    assert approve_action("touch file") is False


def test_approve_action_denies_when_stdin_is_not_tty(monkeypatch):
    class FakeStdin:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stdin", FakeStdin())

    assert approve_action("touch file") is False


def test_tui_approval_controller_waits_for_input_area_answer():
    output = TuiOutput()
    invalidations = []
    controller = TuiApprovalController(output, lambda: invalidations.append(True))
    result = []

    thread = threading.Thread(target=lambda: result.append(controller.request("Tool: write_file")))
    thread.start()
    deadline = time.time() + 1
    while time.time() < deadline and not controller.has_pending():
        time.sleep(0.01)

    text = output.flush()
    assert "Tool: write_file" in text
    assert "Approve? [y/N]" in text
    assert invalidations

    assert controller.consume_answer("y") is True
    thread.join(timeout=1)
    assert result == [True]


def test_tui_approval_controller_omits_large_write_file_preview():
    action = (
        "Tool: write_file\n"
        "File: index.html\n"
        "Bytes: 5000\n"
        "Lines: 120\n"
        "Content preview:\n"
        "<!DOCTYPE html>\n"
        + "x" * 3000
    )
    output = TuiOutput()
    controller = TuiApprovalController(output, lambda: None)
    result = []

    thread = threading.Thread(target=lambda: result.append(controller.request(action)))
    thread.start()
    deadline = time.time() + 1
    while time.time() < deadline and not controller.has_pending():
        time.sleep(0.01)

    text = output.flush()
    assert "Tool: write_file" in text
    assert "File: index.html" in text
    assert "Bytes: 5000" in text
    assert "Lines: 120" in text
    assert "Content preview: [omitted in TUI" in text
    assert "<!DOCTYPE html>" not in text
    assert "Approve? [y/N]" in text

    assert controller.consume_answer("n") is False
    thread.join(timeout=1)
    assert result == [False]


def test_format_tui_approval_action_limits_generic_long_actions():
    action = "\n".join(f"line {index}" for index in range(100))

    formatted = _format_tui_approval_action(action, max_chars=10_000, max_lines=5)

    assert "line 0" in formatted
    assert "line 4" in formatted
    assert "line 5" not in formatted
    assert "omitted 95 lines" in formatted


def test_output_window_wraps_long_lines():
    window = _make_output_window(Buffer())

    assert window.wrap_lines() is True


def test_root_container_includes_completion_menu():
    root = _make_root_container(_make_output_window(Buffer()), TextArea())

    assert any(isinstance(float_.content, CompletionsMenu) for float_ in root.floats)


def test_chat_runtime_rebuild_passes_custom_approval_callback(tmp_path, monkeypatch):
    monkeypatch.setenv("XIAOMING_HOME", str(tmp_path / "home"))
    args = Namespace(
        task=None,
        provider=None,
        model=None,
        approval_mode="suggest",
        permission_mode=None,
        max_turns=None,
        model_timeout_seconds=None,
        stream=False,
    )
    runtime = ChatRuntime(workspace=tmp_path, args=args)
    approvals = []
    runtime.direct_approve_callback = lambda action: approvals.append(action) or True
    runtime.loop = runtime._build_loop()

    result = runtime.loop.registry.run("write_file", {"path": "note.txt", "content": "hello"})

    assert result.status == "success"
    assert approvals
    assert (tmp_path / "note.txt").read_text() == "hello"


def test_build_instructions_includes_agent_rules(tmp_path):
    (tmp_path / "AGENTS.md").write_text("Project rule.\n")

    instructions = build_instructions(tmp_path)

    assert "Safety policy" in instructions
    assert "Project rule." in instructions
    assert "local coding agent working inside a user's repository" in instructions
    assert "Installing remote skills" in instructions
    assert "pass only message and, when useful, a short task_name" in instructions
    assert "do not take over the same file-changing work in the foreground" in instructions
    assert "the worker will inspect the workspace" in instructions


def test_build_instructions_encourages_parallel_read_only_tool_calls(tmp_path):
    worker = build_instructions(tmp_path, role="worker")
    orchestrator = build_instructions(tmp_path, role="orchestrator")

    for instructions in (worker, orchestrator):
        assert "When you need multiple independent read-only tool calls" in instructions
        assert "web_search, web_fetch, search_code, read_file, and list_files" in instructions
        assert "Do not batch write, shell mutation, patch, approval, or background task management tools" in instructions


def test_build_instructions_does_not_hardcode_identity(tmp_path):
    worker = build_instructions(tmp_path, role="worker")
    orchestrator = build_instructions(tmp_path, role="orchestrator")

    assert "You are Xiaoming" not in worker
    assert "You are Xiaoming" not in orchestrator
    assert "You are the user's primary conversation partner" not in orchestrator
    assert "act as" in worker
    assert "act as" in orchestrator
    assert "For simple greetings, reply naturally without introducing a fixed identity" in orchestrator


def test_build_instructions_includes_initial_personality_layers(tmp_path):
    instructions = build_instructions(tmp_path)

    assert "# Objective Reality" in instructions
    assert "LLM-driven agent system" in instructions
    assert "# Who am I\n\n\n# Core Philosophy" in instructions
    assert "Do not convert product labels" in instructions
    assert "仁、义、礼、智、信" in instructions
    assert "法、术、势" in instructions
    assert "道法自然，无为而无不为" in instructions
    assert "Conflict Resolution" in instructions


def test_build_registry_permission_request_hook_can_allow_approval(tmp_path):
    approvals = []
    registry = build_registry(
        tmp_path,
        "suggest",
        approve=lambda action: approvals.append(action) or False,
        include_workspace_tools=False,
        include_skill_install_tool=False,
        include_load_skill_tool=False,
        include_shell_tool=False,
        hooks=HookManager({"PermissionRequest": [lambda payload: {"decision": "allow"}]}),
    )

    result = registry.run("write_file", {"path": "note.txt", "content": "hello"})

    assert result.status == "success"
    assert (tmp_path / "note.txt").read_text() == "hello"
    assert approvals == []


def test_build_registry_permission_request_hook_can_deny_approval(tmp_path):
    approvals = []
    registry = build_registry(
        tmp_path,
        "suggest",
        approve=lambda action: approvals.append(action) or True,
        include_workspace_tools=False,
        include_skill_install_tool=False,
        include_load_skill_tool=False,
        include_shell_tool=False,
        hooks=HookManager({"PermissionRequest": [lambda payload: {"decision": "deny"}]}),
    )

    result = registry.run("write_file", {"path": "note.txt", "content": "hello"})

    assert result.status == "denied"
    assert not (tmp_path / "note.txt").exists()
    assert approvals == []


def test_orchestrator_and_worker_have_identical_tool_schema(tmp_path):
    args = Namespace(provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None, model_timeout_seconds=None, stream=False)
    main_loop = build_loop(
        tmp_path,
        args,
        extra_tools=build_universal_runtime_tools(coordinator_getter=lambda: None, talk_callback=lambda purpose, message, context, options: "unavailable"),
        role="orchestrator",
        capability_profile="orchestrator",
    )
    worker_loop = build_loop(
        tmp_path,
        args,
        extra_tools=build_universal_runtime_tools(coordinator_getter=lambda: None, talk_callback=lambda purpose, message, context, options: "ok"),
        role="worker",
        capability_profile="full",
    )

    main_specs = [(spec.name, spec.description, spec.input_schema, spec.input_mode, spec.freeform_arg) for spec in main_loop.registry.specs()]
    worker_specs = [(spec.name, spec.description, spec.input_schema, spec.input_mode, spec.freeform_arg) for spec in worker_loop.registry.specs()]
    assert main_specs == worker_specs


def test_tool_capability_hook_allows_orchestrator_all_tools():
    """Orchestrator now has full tool access — the model decides what to delegate."""
    hook = tool_capability_hook("orchestrator")

    result = hook({"tool": "write_file", "arguments": {"path": "note.txt", "content": "hello"}})

    assert result is None  # orchestrator allows all tools


def test_tool_capability_hook_allows_foreground_workspace_and_scheduler_tools():
    hook = tool_capability_hook("foreground")

    assert hook({"tool": "write_file", "arguments": {"path": "note.txt", "content": "hello"}}) is None
    assert hook({"tool": "schedule_background_task", "arguments": {"message": "继续处理"}}) is None


def test_tool_capability_hook_denies_read_only_mutation_tools():
    hook = tool_capability_hook("read_only")

    shell = hook({"tool": "shell", "arguments": {"cmd": "pytest"}})
    read_file = hook({"tool": "read_file", "arguments": {"path": "README.md"}})

    assert shell["decision"] == "deny"
    assert read_file is None


def test_tool_capability_hook_allows_skill_installer_tool_but_denies_shell():
    hook = tool_capability_hook("skill_install")

    install = hook({"tool": "install_skill", "arguments": {"url": "https://github.com/obra/superpowers"}})
    shell = hook({"tool": "shell", "arguments": {"cmd": "git clone https://github.com/obra/superpowers"}})

    assert install is None
    assert shell["decision"] == "deny"
    assert "install_skill" in shell["reason"]


def test_build_loop_loads_workspace_hooks(tmp_path):
    (tmp_path / ".xiaoming").mkdir()
    (tmp_path / ".xiaoming" / "hooks.json").write_text(json.dumps({"Stop": [{"command": "true"}]}))
    args = Namespace(provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None, model_timeout_seconds=None, stream=False)

    loop = build_loop(tmp_path, args)

    assert loop.hooks is not None


import pytest
@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_reuses_one_session(monkeypatch, capsys):
    calls = []

    class FakeLoop:
        def run(self, task, session=None):
            session.input_items.append({"role": "user", "content": task})
            calls.append((task, session.item_count))
            return "ok"

    inputs = iter(["first", "second", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(FakeLoop())

    assert result == 0
    assert [call[0] for call in calls] == ["first", "second"]
    assert calls[1][1] > calls[0][1]


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_supports_status_and_clear(monkeypatch, capsys):
    class FakeLoop:
        def run(self, task, session=None):
            session.input_items.append({"role": "user", "content": task})
            session.input_items.append({"role": "assistant", "content": "ok"})
            return "ok"

    inputs = iter(["first", "/status", "/clear", "/status", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(FakeLoop())
    output = capsys.readouterr().out

    assert result == 0
    assert "Session items: 2" in output
    assert "Context cleared." in output
    assert "Session items: 0" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_shows_commands_for_slash(monkeypatch, capsys):
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    inputs = iter(["/", "quit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(FakeLoop())
    output = capsys.readouterr().out

    assert result == 0
    assert "Type '/' for commands." in output
    assert "Commands:" in output
    assert "/help" in output
    assert "/status" in output
    assert "/exit" in output


def test_chat_runtime_continue_restores_latest_session(monkeypatch, tmp_path):
    from xiaoming.sessions.store import SessionStore

    store = SessionStore(tmp_path)
    record = store.create(title="hello", provider="deepseek", model="deepseek-v4-flash")
    store.append(record.id, "user_message", {"content": "first"})
    store.append(record.id, "assistant_message", {"content": "ok"})

    class FakeLoop:
        def run(self, task, session=None):
            session.input_items.append({"role": "user", "content": task})
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None, continue_session=True, resume_session_id=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )

    assert runtime.session_record.id == record.id
    assert [(item["role"], item["content"]) for item in runtime.session.input_items] == [("user", "first"), ("assistant", "ok")]
    assert runtime.session.input_items[0]["xiaoming"]["time"]


def test_initialization_wizard_writes_config_and_secret(tmp_path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("XIAOMING_HOME", str(home))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    prompts = iter(["", "", "sk-test"])
    output = []

    changed = run_initialization_wizard(workspace, input_fn=lambda prompt: next(prompts), output_fn=output.append, force=True)

    assert changed is True
    assert (home / "config.toml").read_text().splitlines()[:3] == ["[model]", 'provider = "deepseek"', 'model = "deepseek-v4-flash"']
    assert (home / "secrets.env").read_text().count("DEEPSEEK_API_KEY=sk-test") == 1
    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "sk-test"


def test_should_run_initialization_when_provider_key_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("XIAOMING_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert should_run_initialization(tmp_path) is True


def test_chat_runtime_dream_context_delegates_to_loop(tmp_path):
    class FakeLoop:
        def dream_context(self, session):
            return "Dream accepted: 1 diary draft(s). Reason: ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )

    assert runtime.dream_context() == "Dream accepted: 1 diary draft(s). Reason: ok"


def test_chat_runtime_defaults_to_latest_session(monkeypatch, tmp_path):
    from xiaoming.sessions.store import SessionStore

    store = SessionStore(tmp_path)
    record = store.create(title="latest", provider="deepseek", model="deepseek-v4-flash")
    store.append(record.id, "user_message", {"content": "persisted"})
    store.append(record.id, "assistant_message", {"content": "ok"})

    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )

    assert runtime.session_record.id == record.id
    assert [(item["role"], item["content"]) for item in runtime.session.input_items] == [("user", "persisted"), ("assistant", "ok")]
    assert runtime.session.input_items[0]["xiaoming"]["time"]


def test_chat_runtime_skips_empty_and_interrupted_sessions_by_default(monkeypatch, tmp_path):
    from xiaoming.sessions.store import SessionStore

    store = SessionStore(tmp_path)
    completed = store.create(title="completed", provider="deepseek", model="deepseek-v4-flash")
    store.append(completed.id, "user_message", {"content": "real task"})
    store.append(completed.id, "assistant_message", {"content": "done"})
    interrupted = store.create(title="interrupted", provider="deepseek", model="deepseek-v4-flash")
    store.append(interrupted.id, "user_message", {"content": "session"})
    store.create(title="empty", provider="deepseek", model="deepseek-v4-flash")

    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )

    assert runtime.session_record.id == completed.id


def test_chat_runtime_new_option_starts_fresh_session(monkeypatch, tmp_path):
    from xiaoming.sessions.store import SessionStore

    store = SessionStore(tmp_path)
    record = store.create(title="latest", provider="deepseek", model="deepseek-v4-flash")
    store.append(record.id, "user_message", {"content": "persisted"})
    store.append(record.id, "assistant_message", {"content": "ok"})

    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None, new_session=True),
        loop_factory=lambda workspace, args: FakeLoop(),
    )

    assert runtime.session_record.id != record.id
    assert runtime.session.input_items == []


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_prints_resumed_session_notice(monkeypatch, capsys, tmp_path):
    from xiaoming.sessions.store import SessionStore

    store = SessionStore(tmp_path)
    record = store.create(title="latest", provider="deepseek", model="deepseek-v4-flash")
    store.append(record.id, "user_message", {"content": "persisted"})
    store.append(record.id, "assistant_message", {"content": "ok"})

    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert f"Resumed session: {record.id}" in output
    assert "Title: latest" in output
    assert "Session items: 2" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_prints_new_session_notice(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "exit")

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert f"Started new session: {runtime.session_record.id}" in output
    assert "Session items: 0" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_supports_session_commands(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def run(self, task, session=None):
            session.input_items.append({"role": "user", "content": task})
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    initial_id = runtime.session_record.id
    inputs = iter(["/session", "/sessions", "/new", "/session", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert f"Session: {initial_id}" in output
    assert "Started new session." in output
    assert runtime.session_record.id != initial_id


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_supports_model_and_approval_commands(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    inputs = iter(["/status", "/model openai gpt-5", "/approval full_auto", "/model-timeout 60", "/stream on", "/status", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "Provider: deepseek" in output
    assert "Model: deepseek-v4-flash" in output
    assert "Model switched to openai gpt-5. Context cleared." in output
    assert "Approval mode set to full_auto. Context cleared." in output
    assert "Model timeout set to 60s. Context cleared." in output
    assert "Stream enabled. Context cleared." in output
    assert "Provider: openai" in output
    assert "Model: gpt-5" in output
    assert "Approval: full_auto" in output
    assert "Permission mode: auto" in output
    assert "Model timeout: 60s" in output
    assert "Stream: on" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_reports_model_switch_error_without_exiting(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    def failing_factory(workspace, args):
        if args.provider == "openai":
            raise RuntimeError("missing openai key")
        return FakeLoop()

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=failing_factory,
    )
    inputs = iter(["/model openai gpt-5", "/status", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "Error switching model: missing openai key" in output
    assert "Provider: deepseek" in output
    assert "Model: deepseek-v4-flash" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_lists_runtime_skills(monkeypatch, capsys, tmp_path):
    skill_dir = tmp_path / ".xiaoming" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: frontend\ndescription: Build UI.\n---\nBody\n")

    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    inputs = iter(["/skills", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "frontend - Build UI." in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_reloads_skills_without_clearing_session(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def __init__(self, skill_text):
            self.skill_library = None
            self.skill_text = skill_text

        def run(self, task, session=None):
            session.input_items.append({"role": "user", "content": task})
            return "ok"

    skill_texts = iter(["No skills found.", "frontend - Build UI."])

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(next(skill_texts)),
    )

    def fake_skills_text():
        return runtime.loop.skill_text

    monkeypatch.setattr(runtime, "skills_text", fake_skills_text)
    inputs = iter(["hello", "/skill reload", "/status", "/skills", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "Skills reloaded." in output
    assert "Session items: 1" in output
    assert "frontend - Build UI." in output


def test_run_loop_with_progress_prints_agent_events():
    events = []

    class FakeLoop:
        def run(self, task, session=None, on_event=None):
            if on_event:
                on_event("Thinking about the next step...")
            return "ok"

    result = run_loop_with_progress(FakeLoop(), "task", session=None, on_event=events.append)

    assert result == "ok"
    assert "Thinking about the next step..." in events


def test_run_loop_with_progress_prints_text_delta_without_prefix(capsys):
    from xiaoming.progress import ProgressEvent

    class FakeLoop:
        def run(self, task, session=None, on_event=None):
            if on_event:
                on_event(ProgressEvent("text_delta", "Hel", end=""))
                on_event(ProgressEvent("text_delta", "lo", end=""))
            return ""

    output_parts = []
    def collect(msg):
        if isinstance(msg, ProgressEvent):
            output_parts.append(msg.message)
        else:
            output_parts.append(str(msg))

    result = run_loop_with_progress(FakeLoop(), "task", session=None, on_event=collect)

    assert result == ""
    assert "".join(output_parts) == "Hello"


def test_tui_output_can_complete_streaming_text_once():
    output = TuiOutput()

    output.write("好的，我来查一下", end="")
    output.write("西安天气。", end="")
    output.complete_streaming()

    assert output.flush() == "好的，我来查一下西安天气。"
    assert output.peek_streaming() == ""
    assert output.flush() == ""


def test_tui_refresh_replaces_visible_streaming_text_when_completed():
    text, streaming_pos = _append_completed_tui_output(
        "header\n好的，我来查一下西安天气。",
        "好的，我来查一下西安天气。",
        len("header\n"),
    )

    assert text == "header\n好的，我来查一下西安天气。"
    assert streaming_pos == -1


def test_run_loop_with_progress_supports_legacy_loop_without_event_callback():
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    assert run_loop_with_progress(FakeLoop(), "task", session=None) == "ok"


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_does_not_print_blank_line_for_empty_stream_result(monkeypatch, capsys):
    class FakeLoop:
        def run(self, task, session=None):
            return ""

    inputs = iter(["task", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(FakeLoop())
    output = capsys.readouterr().out

    assert result == 0
    assert "\n\nxiaoming>" not in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_supports_logs_command(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    inputs = iter(["/logs", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert str(tmp_path / ".xiaoming" / "logs" / "xiaoming.log") in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_status_shows_stream_on_by_default(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, max_turns=None, model_timeout_seconds=None, stream=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    inputs = iter(["/status", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "Stream: on" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_supports_permission_commands(monkeypatch, capsys, tmp_path):
    class FakeLoop:
        def run(self, task, session=None):
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(
            task=None,
            provider=None,
            model=None,
            approval_mode=None,
            permission_mode=None,
            max_turns=None,
            model_timeout_seconds=None,
            stream=None,
        ),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    inputs = iter(["/permission-mode auto", "/allow Bash(pytest *)", "/permissions", "/status", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "Permission mode set to auto. Context cleared." in output
    assert "Added project rule: allow Bash(pytest *)" in output
    assert "allow Bash(pytest *) [project]" in output
    assert "Permission mode: auto" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_logs_unhandled_turn_errors(monkeypatch, capsys, tmp_path):
    class FailingLoop:
        def run(self, task, session=None):
            raise RuntimeError("boom")

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FailingLoop(),
    )
    inputs = iter(["fail", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out
    log_text = runtime.logger.path.read_text()

    assert result == 0
    assert "Error: boom" in output
    assert '"event": "cli_turn_exception"' in log_text
    assert "RuntimeError: boom" in log_text


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_ctrl_c_interrupts_current_turn_without_exiting(monkeypatch, capsys, tmp_path):
    class InterruptingLoop:
        def __init__(self):
            self.calls = 0

        def run(self, task, session=None):
            self.calls += 1
            if self.calls == 1:
                raise KeyboardInterrupt()
            session.input_items.append({"role": "user", "content": task})
            return "ok"

    loop = InterruptingLoop()
    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: loop,
    )
    inputs = iter(["hang", "next", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    discarded = []
    monkeypatch.setattr("xiaoming.cli.discard_pending_terminal_input", lambda: discarded.append(True) or True)

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert discarded == [True]
    assert "Interrupted current operation." in output
    assert "Run /rewind to restore changes from this turn." in output
    assert "ok" in output


@pytest.mark.skip(reason="prompt_toolkit requires TTY; pending UI test migration")
def test_run_chat_supports_checkpoint_commands(monkeypatch, capsys, tmp_path):
    path = tmp_path / "app.py"
    path.write_text("old\n")

    class FakeLoop:
        def run(self, task, session=None):
            runtime.checkpoint_store.snapshot_paths(runtime.active_checkpoint_id, ["app.py"])
            path.write_text("new\n")
            session.input_items.append({"role": "user", "content": task})
            return "ok"

    runtime = ChatRuntime(
        workspace=tmp_path,
        args=Namespace(task=None, provider=None, model=None, approval_mode=None, permission_mode=None, max_turns=None),
        loop_factory=lambda workspace, args: FakeLoop(),
    )
    inputs = iter(["change", "/checkpoints", "/rewind", "exit"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

    result = run_chat(runtime)
    output = capsys.readouterr().out

    assert result == 0
    assert "ok" in output
    assert "change" in output
    assert "Restored checkpoint:" in output
    assert path.read_text() == "old\n"
