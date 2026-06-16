from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


VALID_APPROVAL_MODES = {"suggest", "auto_edit", "full_auto"}
VALID_PERMISSION_MODES = {"default", "plan", "accept_edits", "auto", "bypass"}
VALID_PROVIDERS = {"openai", "deepseek"}
DEFAULT_MAX_OUTPUT_TOKENS = 64_000


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
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
    root = workspace.resolve()
    provider = cli_args.get("provider") or "deepseek"
    default_model = "deepseek-v4-flash" if provider == "deepseek" else "gpt-5"
    model = ModelConfig(
        provider=provider,
        model=cli_args.get("model") or default_model,
    )
    stream_value = cli_args.get("stream")
    agent = AgentConfig(
        approval_mode=cli_args.get("approval_mode") or "suggest",
        permission_mode=cli_args.get("permission_mode") or _approval_to_permission_mode(cli_args.get("approval_mode") or "suggest"),
        max_turns=int(cli_args.get("max_turns") or 999),
        model_timeout_seconds=float(cli_args.get("model_timeout_seconds") or 180),
        stream=True if stream_value is None else bool(stream_value),
        stream_idle_timeout_seconds=float(cli_args.get("stream_idle_timeout_seconds") or 60),
    )
    return Config(model=model, agent=agent, workspace=WorkspaceConfig(root=root))


def _approval_to_permission_mode(approval_mode: str) -> str:
    if approval_mode == "auto_edit":
        return "accept_edits"
    if approval_mode == "full_auto":
        return "auto"
    return "default"
