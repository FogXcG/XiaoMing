from pathlib import Path

from xiaoming.logging import XiaomingLogger, redact_secrets


def test_logger_writes_json_lines(tmp_path: Path):
    logger = XiaomingLogger.create(tmp_path)

    logger.info("test_event", value="ok")

    log_text = logger.path.read_text()
    assert '"event": "test_event"' in log_text
    assert '"value": "ok"' in log_text
    assert logger.path == tmp_path / ".xiaoming" / "logs" / "xiaoming.log"


def test_logger_redacts_common_secrets(tmp_path: Path):
    logger = XiaomingLogger.create(tmp_path)

    logger.info("secret_event", api_key="sk-1234567890abcdef", header="Authorization: Bearer token-value")

    log_text = logger.path.read_text()
    assert "sk-1234567890abcdef" not in log_text
    assert "Bearer token-value" not in log_text
    assert "[REDACTED]" in log_text


def test_redact_secrets_handles_nested_values():
    redacted = redact_secrets({"headers": {"Authorization": "Bearer abc"}, "items": ["sk-1234567890abcdef"]})

    assert redacted == {"headers": {"Authorization": "[REDACTED]"}, "items": ["[REDACTED]"]}


def test_redact_secrets_keeps_token_usage_counters():
    redacted = redact_secrets(
        {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
                "prompt_cache_hit_tokens": 8,
            },
            "api_token": "sk-1234567890abcdef",
        }
    )

    assert redacted["usage"]["input_tokens"] == 10
    assert redacted["usage"]["output_tokens"] == 2
    assert redacted["usage"]["total_tokens"] == 12
    assert redacted["usage"]["prompt_cache_hit_tokens"] == 8
    assert redacted["api_token"] == "[REDACTED]"
