from xiaoming.cli import build_registry
from xiaoming.tools.web import WebFetchTool, WebSearchTool, _HttpResponse, html_to_text


def test_build_registry_includes_web_tools(tmp_path):
    names = {spec.name for spec in build_registry(tmp_path, approval_mode="full_auto").specs()}

    assert "web_search" in names
    assert "web_fetch" in names


def test_web_fetch_rejects_localhost():
    result = WebFetchTool().run({"url": "http://localhost:8000", "max_chars": None})

    assert result.status == "error"
    assert "public hostnames" in result.error


def test_web_fetch_upgrades_http_and_extracts_html(monkeypatch):
    captured = {}

    def fake_http_get(url):
        captured["url"] = url
        return _HttpResponse("<html><body><h1>Hello</h1><p>World</p><script>bad()</script></body></html>", url, "text/html", 80)

    monkeypatch.setattr("xiaoming.tools.web._http_get", fake_http_get)

    result = WebFetchTool().run({"url": "http://example.com/page", "max_chars": 2000})

    assert result.status == "success"
    assert captured["url"] == "https://example.com/page"
    assert "Hello" in result.output
    assert "World" in result.output
    assert "bad()" not in result.output


def test_web_search_filters_domains(monkeypatch):
    monkeypatch.setattr(
        "xiaoming.tools.web._search_with_fallback",
        lambda query, max_results: (
            "test",
            [
                {"title": "Allowed", "url": "https://docs.example.com/a", "snippet": "A"},
                {"title": "Blocked", "url": "https://bad.test/b", "snippet": "B"},
            ],
            ["test"],
        ),
    )

    result = WebSearchTool().run({"query": "docs", "max_results": 5, "allowed_domains": "example.com", "blocked_domains": None})

    assert result.status == "success"
    assert "Allowed" in result.output
    assert "Blocked" not in result.output


def test_web_search_news_query_uses_china_accessible_backend_first(monkeypatch):
    called = []
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def fake_backend(backend, query, max_results):
        called.append(backend)
        if backend == "bing_cn":
            return [{"title": "News", "url": "https://news.example.com/a", "snippet": "Today"}]
        return []

    monkeypatch.setattr("xiaoming.tools.web._search_backend", fake_backend)

    result = WebSearchTool().run({"query": "today hot news", "max_results": 5, "allowed_domains": None, "blocked_domains": None})

    assert result.status == "success"
    assert called == ["bing_cn"]
    assert "News" in result.output


def test_web_search_chinese_news_prefers_sogou_news(monkeypatch):
    called = []
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    def fake_backend(backend, query, max_results):
        called.append(backend)
        if backend == "sogou_news":
            return [{"title": "热点新闻", "url": "https://news.example.com/a", "snippet": "Today"}]
        return []

    monkeypatch.setattr("xiaoming.tools.web._search_backend", fake_backend)

    result = WebSearchTool().run({"query": "今天有什么热点新闻", "max_results": 5, "allowed_domains": None, "blocked_domains": None})

    assert result.status == "success"
    assert called == ["sogou_news"]
    assert "热点新闻" in result.output


def test_web_search_prefers_deepseek_when_deepseek_and_kimi_are_configured(monkeypatch):
    called = []

    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_backend(backend, query, max_results):
        called.append(backend)
        if backend == "deepseek_anthropic":
            return [{"title": "DeepSeek", "url": "https://news.example.com/a", "snippet": "Today"}]
        return []

    monkeypatch.setattr("xiaoming.tools.web._search_backend", fake_backend)

    result = WebSearchTool().run({"query": "今天有什么热点新闻", "max_results": 5, "allowed_domains": None, "blocked_domains": None})

    assert result.status == "success"
    assert called == ["deepseek_anthropic"]
    assert "Search backend: deepseek_anthropic" in result.output
    assert "DeepSeek" in result.output


def test_web_search_uses_deepseek_anthropic_after_kimi(monkeypatch):
    called = []

    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")

    def fake_backend(backend, query, max_results):
        called.append(backend)
        if backend == "deepseek_anthropic":
            return [{"title": "DeepSeek", "url": "https://news.example.com/a", "snippet": "Today"}]
        return []

    monkeypatch.setattr("xiaoming.tools.web._search_backend", fake_backend)

    result = WebSearchTool().run({"query": "今天有什么热点新闻", "max_results": 5, "allowed_domains": None, "blocked_domains": None})

    assert result.status == "success"
    assert called == ["deepseek_anthropic"]
    assert "Search backend: deepseek_anthropic" in result.output
    assert "DeepSeek" in result.output


def test_deepseek_anthropic_search_can_be_disabled(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("XIAOMING_DEEPSEEK_WEB_SEARCH", "0")

    assert "deepseek_anthropic" not in __import__("xiaoming.tools.web", fromlist=["_candidate_backends"])._candidate_backends("今天有什么热点新闻")


def test_deepseek_anthropic_search_parses_server_tool_results(monkeypatch):
    from xiaoming.tools import web

    captured = {}
    payload = {
        "content": [
            {"type": "thinking", "thinking": "searching"},
            {
                "type": "server_tool_use",
                "id": "call_1",
                "name": "web_search",
                "input": {"query": "AI news"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "call_1",
                "content": [
                    {"type": "web_search_result", "title": "AI News A", "url": "https://news.example.com/a", "encrypted_content": "..."},
                    {"type": "web_search_result", "title": "AI News B", "url": "https://news.example.com/b", "encrypted_content": "..."},
                ],
            },
            {"type": "text", "text": "Summary with https://news.example.com/a"},
        ],
        "usage": {"server_tool_use": {"web_search_requests": 1}},
    }

    def fake_post(url, body, headers, timeout_seconds):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout_seconds"] = timeout_seconds
        return payload

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(web, "_deepseek_anthropic_post", fake_post)

    results = web._deepseek_anthropic_search("AI news", 2)

    assert captured["url"] == "https://api.deepseek.com/anthropic/v1/messages"
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["body"]["tools"] == [{"type": "web_search_20250305", "name": "web_search", "max_uses": 2}]
    assert results == [
        {"title": "AI News A", "url": "https://news.example.com/a", "snippet": "Summary with https://news.example.com/a\nDeepSeek web search requests: 1"},
        {"title": "AI News B", "url": "https://news.example.com/b", "snippet": "Summary with https://news.example.com/a\nDeepSeek web search requests: 1"},
    ]


def test_kimi_search_uses_moonshot_web_search_tool(monkeypatch):
    from xiaoming.tools import web

    captured = {}

    class FakeToolFunction:
        name = "$web_search"
        arguments = '{"query":"西安天气"}'

    class FakeToolCall:
        id = "call_1"
        type = "function"
        function = FakeToolFunction()

    class FakeMessage:
        def __init__(self, content=None, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls or []

    class FakeChoice:
        def __init__(self, message):
            self.message = message

    class FakeResponse:
        def __init__(self, message):
            self.choices = [FakeChoice(message)]

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            captured.setdefault("requests", []).append(kwargs)
            if self.calls == 1:
                return FakeResponse(FakeMessage(tool_calls=[FakeToolCall()]))
            return FakeResponse(FakeMessage(content="西安天气晴。来源：https://weather.example.com/xian"))

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)

    results = web._kimi_search("西安天气", 3)

    assert captured["client"]["api_key"] == "test-key"
    assert captured["client"]["base_url"] == web.MOONSHOT_BASE_URL
    assert captured["requests"][0]["tools"] == [{"type": "builtin_function", "function": {"name": "$web_search"}}]
    assert captured["requests"][0]["temperature"] == 0.6
    assert captured["requests"][0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert captured["requests"][1]["messages"][-1]["role"] == "tool"
    assert results == [
        {
            "title": "Kimi web search citation",
            "url": "https://weather.example.com/xian",
            "snippet": "西安天气晴。来源：https://weather.example.com/xian",
        }
    ]


def test_kimi_content_prefers_source_urls_over_inline_api_urls():
    from xiaoming.tools import web

    results = web._kimi_content_to_results(
        "Use `https://api.moonshot.cn/v1`. Sources:\n1. https://platform.moonshot.cn/docs/guide/use-web-search`",
        2,
    )

    assert results[0]["url"] == "https://platform.moonshot.cn/docs/guide/use-web-search"


def test_web_search_rejects_conflicting_domain_filters():
    result = WebSearchTool().run({"query": "docs", "max_results": 5, "allowed_domains": "example.com", "blocked_domains": "bad.test"})

    assert result.status == "error"
    assert "cannot both be set" in result.error


def test_web_search_error_tells_model_not_to_repeat_backend(monkeypatch):
    monkeypatch.setattr("xiaoming.tools.web._search_backend", lambda *_: (_ for _ in ()).throw(TimeoutError("timeout")))

    result = WebSearchTool().run({"query": "today hot news", "max_results": 5, "allowed_domains": None, "blocked_domains": None})

    assert result.status == "error"
    assert "Do not retry the same search backend repeatedly" in result.error


def test_web_search_schema_is_deepseek_compatible():
    schema = WebSearchTool().spec.input_schema

    assert schema["properties"]["allowed_domains"]["type"] == ["string", "null"]
    assert "items" not in schema["properties"]["allowed_domains"]
    assert schema["properties"]["blocked_domains"]["type"] == ["string", "null"]
    assert "items" not in schema["properties"]["blocked_domains"]


def test_html_to_text_strips_script_and_collapses_text():
    text = html_to_text("<h1>Title</h1><p>Hello <b>world</b></p><script>alert(1)</script>")

    assert "Title" in text
    assert "Hello world" in text
    assert "alert" not in text


def test_web_fetch_curl_does_not_inject_xiaoming_web_proxy(monkeypatch):
    calls = {}

    monkeypatch.setenv("XIAOMING_WEB_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/curl")

    def fake_run(command, **kwargs):
        calls["command"] = command
        marker = b"\n__XIAOMING_CURL_META__"
        return type("Completed", (), {"returncode": 0, "stdout": b"ok" + marker + b"https://example.com\ttext/plain", "stderr": b""})()

    monkeypatch.setattr("subprocess.run", fake_run)

    result = WebFetchTool().run({"url": "https://example.com", "max_chars": 1000})

    assert result.status == "success"
    assert "--proxy" not in calls["command"]
