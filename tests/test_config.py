from pathlib import Path

from xiaoming.config import DEFAULT_MAX_OUTPUT_TOKENS, AgentConfig, load_config, load_secrets_env, save_global_config, write_secrets_env


def test_default_config_uses_suggest_mode(tmp_path: Path):
    config = load_config(workspace=tmp_path, cli_args={})

    assert config.model.provider == "deepseek"
    assert config.model.model == "deepseek-v4-flash"
    assert config.model.max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS
    assert config.model.max_output_tokens == 64_000
    assert config.agent.approval_mode == "suggest"
    assert config.agent.permission_mode == "default"
    assert config.agent.max_turns == 999
    assert config.agent.model_timeout_seconds == 180
    assert config.agent.stream is True
    assert config.agent.stream_idle_timeout_seconds == 60
    assert config.workspace.root == tmp_path.resolve()


def test_cli_args_override_defaults(tmp_path: Path):
    config = load_config(
        workspace=tmp_path,
        cli_args={
            "model": "gpt-5",
            "approval_mode": "auto_edit",
            "max_turns": 7,
            "model_timeout_seconds": 60,
            "stream": True,
            "stream_idle_timeout_seconds": 30,
        },
    )

    assert config.model.model == "gpt-5"
    assert config.agent.approval_mode == "auto_edit"
    assert config.agent.max_turns == 7
    assert config.agent.model_timeout_seconds == 60
    assert config.agent.stream is True
    assert config.agent.stream_idle_timeout_seconds == 30


def test_approval_mode_maps_to_permission_mode_when_unspecified(tmp_path: Path):
    config = load_config(workspace=tmp_path, cli_args={"approval_mode": "full_auto"})

    assert config.agent.permission_mode == "auto"


def test_permission_mode_overrides_approval_mapping(tmp_path: Path):
    config = load_config(workspace=tmp_path, cli_args={"approval_mode": "full_auto", "permission_mode": "plan"})

    assert config.agent.approval_mode == "full_auto"
    assert config.agent.permission_mode == "plan"


def test_deepseek_provider_defaults_to_v4_flash(tmp_path: Path):
    config = load_config(
        workspace=tmp_path,
        cli_args={
            "provider": "deepseek",
            "model": None,
        },
    )

    assert config.model.provider == "deepseek"
    assert config.model.model == "deepseek-v4-flash"


def test_load_config_reads_global_and_project_config(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("XIAOMING_HOME", str(home))
    save_global_config(provider="openai", model="gpt-5")
    project_config = workspace / ".xiaoming" / "config.toml"
    project_config.parent.mkdir()
    project_config.write_text("[model]\nprovider = \"deepseek\"\nmodel = \"deepseek-v4-pro\"\n")

    config = load_config(workspace=workspace, cli_args={})

    assert config.model.provider == "deepseek"
    assert config.model.model == "deepseek-v4-pro"


def test_load_secrets_env_sets_missing_keys(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("XIAOMING_HOME", str(home))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    write_secrets_env(provider="deepseek", api_key="sk-test")

    loaded = load_secrets_env()

    assert loaded == {"DEEPSEEK_API_KEY": "sk-test"}
    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "sk-test"


def test_agent_config_rejects_unknown_approval_mode():
    try:
        AgentConfig(approval_mode="danger", max_turns=20)
    except ValueError as exc:
        assert "approval_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_agent_config_rejects_unknown_permission_mode():
    try:
        AgentConfig(permission_mode="danger")
    except ValueError as exc:
        assert "permission_mode" in str(exc)
    else:
        raise AssertionError("expected ValueError")
