from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any


ENV_PASSTHROUGH = (
    "DEEPSEEK_API_KEY",
    "OPENAI_API_KEY",
    "MOONSHOT_API_KEY",
    "KIMI_API_KEY",
    "BRAVE_SEARCH_API_KEY",
    "XIAOMING_DEEPSEEK_WEB_SEARCH",
    "XIAOMING_DEEPSEEK_WEB_SEARCH_MODEL",
    "DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS",
    "DEEPSEEK_WEB_SEARCH_MAX_TOKENS",
    "XIAOMING_PROVIDER",
    "XIAOMING_MODEL",
    "XIAOMING_PIP_SPEC",
    "XIAOMING_PIP_NO_INDEX",
    "XIAOMING_PIP_FIND_LINKS",
    "XIAOMING_PIP_TRUSTED_HOST",
)


def xiaoming_env() -> dict[str, str]:
    return {name: value for name in ENV_PASSTHROUGH if (value := os.environ.get(name))}


def build_xiaoming_command(task_description: str) -> str:
    provider = os.environ.get("XIAOMING_PROVIDER") or "deepseek"
    model = os.environ.get("XIAOMING_MODEL") or "deepseek-v4-flash"
    return " ".join(
        [
            "xiaoming-cli",
            "--new",
            "--no-stream",
            "--provider",
            shlex.quote(provider),
            "--model",
            shlex.quote(model),
            "--approval-mode full_auto",
            "--permission-mode bypass",
            "--max-turns 999",
            "--model-timeout 600",
            shlex.quote(task_description),
        ]
    )


def _terminal_bench_classes() -> tuple[type[Any], type[Any]]:
    try:
        from terminal_bench.agents.installed_agents.abstract_installed_agent import AbstractInstalledAgent
        from terminal_bench.terminal.models import TerminalCommand
    except Exception as exc:  # pragma: no cover - exercised only when terminal-bench is installed
        raise RuntimeError("terminal-bench is required to use XiaomingTerminalBenchAgent") from exc
    return AbstractInstalledAgent, TerminalCommand


try:
    AbstractInstalledAgent, TerminalCommand = _terminal_bench_classes()
except RuntimeError:
    class AbstractInstalledAgent:  # type: ignore[no-redef]
        pass

    class TerminalCommand:  # type: ignore[no-redef]
        def __init__(self, command: str, min_timeout_sec: int, max_timeout_sec: int, block: bool, append_enter: bool) -> None:
            self.command = command
            self.min_timeout_sec = min_timeout_sec
            self.max_timeout_sec = max_timeout_sec
            self.block = block
            self.append_enter = append_enter


class XiaomingTerminalBenchAgent(AbstractInstalledAgent):
    @staticmethod
    def name() -> str:
        return "xiaoming-cli"

    @property
    def _install_agent_script_path(self) -> Path:
        return Path(__file__).with_name("setup-xiaoming.sh")

    def _run_agent_commands(self, task_description: str) -> list[Any]:
        return [
            TerminalCommand(
                command=build_xiaoming_command(task_description),
                min_timeout_sec=0,
                max_timeout_sec=600,
                block=True,
                append_enter=True,
            )
        ]

    @property
    def _env(self) -> dict[str, str]:
        return xiaoming_env()
