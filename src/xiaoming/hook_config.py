from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any

from xiaoming.hooks import HookEvent, HookManager, HookResult


DEFAULT_HOOK_TIMEOUT_SECONDS = 10.0


def load_workspace_hooks(workspace: Path, timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS, logger: Any = None) -> HookManager | None:
    specs = _load_hook_specs(workspace, logger=logger)
    if not specs:
        return None
    hooks: dict[HookEvent, list] = {}
    for event, entries in specs.items():
        if event not in _VALID_EVENTS or not isinstance(entries, list):
            continue
        callbacks = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("command"), str):
                continue
            callbacks.append(_command_hook(workspace, event, entry["command"], timeout_seconds, logger=logger))
        if callbacks:
            hooks[event] = callbacks
    return HookManager(hooks) if hooks else None


def _load_hook_specs(workspace: Path, logger: Any = None) -> dict[str, Any]:
    for path in (workspace / ".xiaoming" / "hooks.json", workspace / ".agents" / "hooks.json"):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            _log_error(logger, "hook_config_invalid", path=str(path), error=str(exc))
            return {}
        if isinstance(data, dict) and isinstance(data.get("hooks"), dict):
            return data["hooks"]
        if isinstance(data, dict):
            return data
        _log_error(logger, "hook_config_invalid", path=str(path), error="root must be a JSON object")
        return {}
    return {}


def _command_hook(workspace: Path, event: str, command: str, timeout_seconds: float, logger: Any = None):
    def run(payload: dict[str, Any]) -> HookResult:
        request = {"event": event, "payload": payload, "workspace": str(workspace)}
        started = time.monotonic()
        _log_info(logger, "hook_command_started", event_name=event, command=command)
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                shell=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _log_error(logger, "hook_command_timeout", event_name=event, command=command, timeout_seconds=timeout_seconds)
            return HookResult(continue_=False, reason=f"hook command timed out after {timeout_seconds:g}s: {command}")
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _log_info(
            logger,
            "hook_command_finished",
            event_name=event,
            command=command,
            status=completed.returncode,
            elapsed_ms=elapsed_ms,
            stdout_chars=len(completed.stdout or ""),
            stderr_chars=len(completed.stderr or ""),
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            reason = f"hook command exited with status {completed.returncode}: {command}"
            if detail:
                reason += f"\n{detail}"
            return HookResult(continue_=False, reason=reason)
        output = completed.stdout.strip()
        if not output:
            return HookResult()
        try:
            data = json.loads(output)
        except json.JSONDecodeError as exc:
            return HookResult(continue_=False, reason=f"hook command returned invalid JSON: {exc}")
        if not isinstance(data, dict):
            return HookResult(continue_=False, reason="hook command returned non-object JSON")
        return HookResult.from_value(data)

    return run


def _log_info(logger: Any, event: str, **fields: Any) -> None:
    if logger is not None and hasattr(logger, "info"):
        logger.info(event, **fields)


def _log_error(logger: Any, event: str, **fields: Any) -> None:
    if logger is not None and hasattr(logger, "error"):
        logger.error(event, **fields)


_VALID_EVENTS = {
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "PermissionRequest",
    "PreCompact",
    "PostCompact",
    "Stop",
}
