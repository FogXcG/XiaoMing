from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
import tomllib
from typing import Any


VALID_APPROVAL_MODES = {"suggest", "auto_edit", "full_auto"}
VALID_PERMISSION_MODES = {"default", "plan", "accept_edits", "auto", "bypass"}
VALID_PROVIDERS = {"openai", "deepseek"}
DEFAULT_MAX_OUTPUT_TOKENS = 64_000
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_OPENAI_MODEL = "gpt-5"


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "deepseek"
    model: str = DEFAULT_DEEPSEEK_MODEL
    temperature: float = 0.2
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        if self.provider not in VALID_PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(VALID_PROVIDERS)}")


@dataclass(frozen=True)
class AgentConfig:
    approval_mode: str = "suggest"
    permission_mode: str = "default"
    max_turns: int = 999
    model_timeout_seconds: float = 180
    stream: bool = True
    stream_idle_timeout_seconds: float = 60

    def __post_init__(self) -> None:
        if self.approval_mode not in VALID_APPROVAL_MODES:
            raise ValueError(f"approval_mode must be one of {sorted(VALID_APPROVAL_MODES)}")
        if self.permission_mode not in VALID_PERMISSION_MODES:
            raise ValueError(f"permission_mode must be one of {sorted(VALID_PERMISSION_MODES)}")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.model_timeout_seconds <= 0:
            raise ValueError("model_timeout_seconds must be > 0")
        if self.stream_idle_timeout_seconds <= 0:
            raise ValueError("stream_idle_timeout_seconds must be > 0")


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path


@dataclass(frozen=True)
class Config:
    model: ModelConfig
    agent: AgentConfig
    workspace: WorkspaceConfig


def load_config(workspace: Path, cli_args: dict[str, Any]) -> Config:
    load_secrets_env()
    root = workspace.resolve()
    merged = _merged_file_config(root)
    model_config = merged.get("model", {})
    agent_config = merged.get("agent", {})
    provider = cli_args.get("provider") or model_config.get("provider") or "deepseek"
    default_model = DEFAULT_DEEPSEEK_MODEL if provider == "deepseek" else DEFAULT_OPENAI_MODEL
    model = ModelConfig(
        provider=provider,
        model=cli_args.get("model") or model_config.get("model") or default_model,
    )
    stream_value = cli_args.get("stream")
    if stream_value is None:
        stream_value = agent_config.get("stream")
    approval_mode = cli_args.get("approval_mode") or agent_config.get("approval_mode") or "suggest"
    agent = AgentConfig(
        approval_mode=approval_mode,
        permission_mode=cli_args.get("permission_mode") or agent_config.get("permission_mode") or _approval_to_permission_mode(approval_mode),
        max_turns=int(cli_args.get("max_turns") or agent_config.get("max_turns") or 999),
        model_timeout_seconds=float(cli_args.get("model_timeout_seconds") or agent_config.get("model_timeout_seconds") or 180),
        stream=True if stream_value is None else bool(stream_value),
        stream_idle_timeout_seconds=float(cli_args.get("stream_idle_timeout_seconds") or agent_config.get("stream_idle_timeout_seconds") or 60),
    )
    return Config(model=model, agent=agent, workspace=WorkspaceConfig(root=root))


def _approval_to_permission_mode(approval_mode: str) -> str:
    if approval_mode == "auto_edit":
        return "accept_edits"
    if approval_mode == "full_auto":
        return "auto"
    return "default"


def xiaoming_home() -> Path:
    return Path(os.environ.get("XIAOMING_HOME") or Path.home() / ".xiaoming").expanduser()


def global_config_path() -> Path:
    return xiaoming_home() / "config.toml"


def workspace_config_path(workspace: Path) -> Path:
    return workspace.resolve() / ".xiaoming" / "config.toml"


def secrets_env_path() -> Path:
    return xiaoming_home() / "secrets.env"


def api_key_env_name(provider: str) -> str:
    return "OPENAI_API_KEY" if provider == "openai" else "DEEPSEEK_API_KEY"


def api_key_present(provider: str) -> bool:
    load_secrets_env()
    return bool(os.environ.get(api_key_env_name(provider)))


def load_secrets_env(path: Path | None = None) -> dict[str, str]:
    env_path = path or secrets_env_path()
    if not env_path.exists():
        return {}
    loaded: dict[str, str] = {}
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def write_secrets_env(provider: str, api_key: str) -> Path:
    path = secrets_env_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    key_name = api_key_env_name(provider)
    existing = load_secrets_env(path)
    existing[key_name] = api_key
    lines = ["# Xiaoming local API keys. Keep this file private."]
    lines.extend(f"{key}={value}" for key, value in sorted(existing.items()))
    path.write_text("\n".join(lines) + "\n")
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass
    os.environ[key_name] = api_key
    return path


def save_global_config(
    *,
    provider: str,
    model: str | None = None,
    approval_mode: str = "suggest",
    permission_mode: str = "default",
    stream: bool = True,
) -> Path:
    path = global_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model = model or (DEFAULT_DEEPSEEK_MODEL if provider == "deepseek" else DEFAULT_OPENAI_MODEL)
    path.write_text(
        "\n".join(
            [
                "[model]",
                f'provider = "{provider}"',
                f'model = "{resolved_model}"',
                "",
                "[agent]",
                f'approval_mode = "{approval_mode}"',
                f'permission_mode = "{permission_mode}"',
                f"stream = {str(stream).lower()}",
                "",
            ]
        )
    )
    return path


def _merged_file_config(workspace: Path) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in (global_config_path(), workspace_config_path(workspace)):
        data = _read_config_file(path)
        for section, values in data.items():
            if isinstance(values, dict):
                merged.setdefault(section, {}).update(values)
    return merged


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
