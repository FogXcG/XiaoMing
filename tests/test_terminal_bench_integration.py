from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("evals/integrations/terminal_bench/xiaoming_agent.py")
    spec = importlib.util.spec_from_file_location("xiaoming_terminal_bench_agent", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_bench_command_quotes_task_description():
    module = _load_module()

    command = module.build_xiaoming_command("fix the bug; echo unsafe")

    assert "--permission-mode bypass" in command
    assert "--approval-mode full_auto" in command
    assert "'fix the bug; echo unsafe'" in command


def test_terminal_bench_env_passthrough_is_allowlisted(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")
    monkeypatch.setenv("XIAOMING_PIP_SPEC", "git+https://example.test/repo.git@branch")
    monkeypatch.setenv("XIAOMING_PIP_NO_INDEX", "1")
    monkeypatch.setenv("XIAOMING_PIP_FIND_LINKS", "/wheels")
    monkeypatch.setenv("UNRELATED_SECRET", "secret")

    env = module.xiaoming_env()

    assert env["DEEPSEEK_API_KEY"] == "secret"
    assert env["XIAOMING_PIP_SPEC"] == "git+https://example.test/repo.git@branch"
    assert env["XIAOMING_PIP_NO_INDEX"] == "1"
    assert env["XIAOMING_PIP_FIND_LINKS"] == "/wheels"
    assert "UNRELATED_SECRET" not in env


def test_terminal_bench_agent_exposes_private_env_property(monkeypatch):
    module = _load_module()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret")

    agent = module.XiaomingTerminalBenchAgent()

    assert agent._env["DEEPSEEK_API_KEY"] == "secret"
