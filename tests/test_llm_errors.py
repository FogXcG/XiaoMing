from types import SimpleNamespace

from xiaoming.llm.errors import provider_call_error


class FakeStatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FakeResponseError(Exception):
    def __init__(self, status_code):
        super().__init__(f"response {status_code}")
        self.response = SimpleNamespace(status_code=status_code)


class FakeTimeoutError(Exception):
    pass


def test_provider_call_error_marks_429_as_retryable_rate_limit():
    error = provider_call_error(FakeStatusError(429), source="openai")

    assert error.error_info.kind == "rate_limited"
    assert error.error_info.retryable is True
    assert error.error_info.details["http_status_code"] == 429


def test_provider_call_error_marks_400_as_non_retryable_bad_request():
    error = provider_call_error(FakeResponseError(400), source="deepseek")

    assert error.error_info.kind == "bad_request"
    assert error.error_info.retryable is False


def test_provider_call_error_marks_timeout_class_as_retryable_connection_failure():
    error = provider_call_error(FakeTimeoutError("timed out"), source="deepseek")

    assert error.error_info.kind == "http_connection_failed"
    assert error.error_info.retryable is True
