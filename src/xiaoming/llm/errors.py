from __future__ import annotations

from typing import Any

from xiaoming.agent_errors import AgentErrorInfo, ProviderCallError


def provider_call_error(exc: BaseException, *, source: str) -> ProviderCallError:
    status_code = _status_code(exc)
    message = str(exc) or type(exc).__name__
    details: dict[str, Any] = {"source": source, "exception": type(exc).__name__}
    if status_code is not None:
        details["http_status_code"] = status_code

    if status_code == 401 or status_code == 403:
        return ProviderCallError(AgentErrorInfo(kind="unauthorized", message=message, retryable=False, details=details))
    if status_code == 400:
        return ProviderCallError(AgentErrorInfo(kind="bad_request", message=message, retryable=False, details=details))
    if status_code == 429:
        return ProviderCallError(AgentErrorInfo(kind="rate_limited", message=message, retryable=True, details=details))
    if status_code is not None and status_code >= 500:
        return ProviderCallError(AgentErrorInfo(kind="server_overloaded", message=message, retryable=True, details=details))

    name = type(exc).__name__.lower()
    if any(token in name for token in ("timeout", "connection", "connect", "network")):
        return ProviderCallError(AgentErrorInfo(kind="http_connection_failed", message=message, retryable=True, details=details))

    return ProviderCallError(AgentErrorInfo(kind="unknown", message=message, retryable=False, details=details))


def _status_code(exc: BaseException) -> int | None:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None
