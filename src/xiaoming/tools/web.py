from __future__ import annotations

from html.parser import HTMLParser
import json
import os
import re
import shutil
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from xiaoming.context.truncation import truncate_middle
from xiaoming.llm.types import ToolSpec
from xiaoming.tools.base import ToolResult


DEFAULT_TIMEOUT_SECONDS = 6
DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS = 60
MAX_FETCH_BYTES = 1_000_000
MAX_OUTPUT_CHARS = 30_000
KIMI_WEB_SEARCH_MODEL = "kimi-k2.6"
MOONSHOT_BASE_URL = "https://api.moonshot.cn/v1"
DEEPSEEK_ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
DEEPSEEK_WEB_SEARCH_MODEL = "deepseek-v4-flash"


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web for current information. Returns titles, URLs, and snippets. "
        "Use this for recent information or facts outside the repository. Include source URLs in the final answer. "
        "For broad research, make multiple parallel web_search calls with different queries and angles simultaneously."
    )
    supports_parallel_tool_calls = True
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": ["integer", "null"]},
            "allowed_domains": {
                "type": ["string", "null"],
                "description": "Optional comma-separated domains to include, for example: example.com,docs.example.com",
            },
            "blocked_domains": {
                "type": ["string", "null"],
                "description": "Optional comma-separated domains to exclude, for example: example.com,spam.test",
            },
        },
        "required": ["query", "max_results", "allowed_domains", "blocked_domains"],
        "additionalProperties": False,
    }

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        query = str(args.get("query") or "").strip()
        if len(query) < 2:
            return ToolResult(self.name, "error", error="query must be at least 2 characters")
        allowed = _domain_list(args.get("allowed_domains"))
        blocked = _domain_list(args.get("blocked_domains"))
        if allowed and blocked:
            return ToolResult(self.name, "error", error="allowed_domains and blocked_domains cannot both be set")
        max_results = _bounded_int(args.get("max_results"), default=5, minimum=1, maximum=10)
        try:
            backend, results, attempted = _search_with_fallback(query, max_results)
            results = _filter_results(results, allowed, blocked)
            if not results:
                return ToolResult(self.name, "success", output=f"No search results found for: {query}\nAttempted backends: {', '.join(attempted)}")
            lines = [
                f"Search backend: {backend}",
                f"Search URL: {_search_url_for_backend(backend, query, max_results)}",
                f"Attempted backends: {', '.join(attempted)}",
                f"Search results for: {query}",
                "",
            ]
            for index, result in enumerate(results[:max_results], start=1):
                lines.append(f"{index}. {result['title']}")
                lines.append(f"   URL: {result['url']}")
                if result.get("snippet"):
                    lines.append(f"   Snippet: {result['snippet']}")
            return ToolResult(self.name, "success", output=truncate_middle("\n".join(lines), MAX_OUTPUT_CHARS))
        except Exception as exc:
            return ToolResult(
                self.name,
                "error",
                error=f"{exc}. Do not retry the same search backend repeatedly; try a more specific query or use web_fetch on a known news/source URL.",
            )


class WebFetchTool:
    name = "web_fetch"
    description = (
        "Fetch a public HTTP or HTTPS URL and return readable text. This is read-only. "
        "Use it to inspect documentation or web pages returned by web_search."
    )
    supports_parallel_tool_calls = True
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "max_chars": {"type": ["integer", "null"]},
        },
        "required": ["url", "max_chars"],
        "additionalProperties": False,
    }

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        url = str(args.get("url") or "").strip()
        max_chars = _bounded_int(args.get("max_chars"), default=12000, minimum=1000, maximum=MAX_OUTPUT_CHARS)
        try:
            normalized = _validate_public_url(url)
            response = _http_get(normalized)
            data = response.text.encode(_charset(response.content_type), errors="replace")
            if response.raw_size > MAX_FETCH_BYTES:
                return ToolResult(self.name, "error", error=f"response exceeds {MAX_FETCH_BYTES} bytes")
            if _looks_binary(data, response.content_type):
                return ToolResult(self.name, "error", error=f"binary or unsupported content type: {response.content_type or 'unknown'}")
            text = response.text
            if "html" in response.content_type.lower() or _looks_like_html(text):
                text = html_to_text(text)
            header = f"Fetched URL: {response.final_url}\nContent-Type: {response.content_type or 'unknown'}\n\n"
            return ToolResult(self.name, "success", output=header + truncate_middle(_normalize_text(text), max_chars))
        except Exception as exc:
            return ToolResult(self.name, "error", error=str(exc))


def _brave_search(query: str, max_results: int) -> list[dict[str, str]]:
    url = _search_url_for_backend("brave", query, max_results)
    payload = json.loads(_http_get(url, headers={"Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"]}).text)
    return [
        {"title": item.get("title") or "", "url": item.get("url") or "", "snippet": item.get("description") or ""}
        for item in (payload.get("web", {}).get("results") or [])
        if item.get("url")
    ]


def _kimi_search(query: str, max_results: int) -> list[dict[str, str]]:
    from openai import OpenAI

    api_key = _kimi_api_key()
    if not api_key:
        return []
    client = OpenAI(api_key=api_key, base_url=os.environ.get("MOONSHOT_BASE_URL") or MOONSHOT_BASE_URL)
    model = os.environ.get("XIAOMING_KIMI_WEB_SEARCH_MODEL") or KIMI_WEB_SEARCH_MODEL
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a web search backend for a coding agent. Search the web and return a concise answer "
                "with source URLs. Prefer authoritative and recent sources. Do not mention that you are an AI model."
            ),
        },
        {
            "role": "user",
            "content": f"Search query: {query}\nReturn up to {max_results} useful results or citations.",
        },
    ]
    tools = [{"type": "builtin_function", "function": {"name": "$web_search"}}]
    for _ in range(3):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.6,
            extra_body={"thinking": {"type": "disabled"}},
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        if not tool_calls:
            content = getattr(message, "content", None) or ""
            return _kimi_content_to_results(content, max_results)
        messages.append(_kimi_assistant_message(message))
        for tool_call in tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "content": tool_call.function.arguments or "{}",
                }
            )
    return []


def _deepseek_anthropic_search(query: str, max_results: int) -> list[dict[str, str]]:
    api_key = _deepseek_api_key()
    if not api_key:
        return []
    base_url = (os.environ.get("DEEPSEEK_ANTHROPIC_BASE_URL") or DEEPSEEK_ANTHROPIC_BASE_URL).rstrip("/")
    url = f"{base_url}/v1/messages"
    max_uses = max(1, min(max_results, 5))
    body = {
        "model": os.environ.get("XIAOMING_DEEPSEEK_WEB_SEARCH_MODEL") or DEEPSEEK_WEB_SEARCH_MODEL,
        "max_tokens": _deepseek_web_search_max_tokens(),
        "messages": [
            {
                "role": "user",
                "content": f"Search query: {query}\nReturn up to {max_results} useful results with source URLs.",
            }
        ],
        "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": max_uses}],
    }
    payload = _deepseek_anthropic_post(
        url,
        body,
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        timeout_seconds=_deepseek_web_search_timeout_seconds(),
    )
    return _deepseek_anthropic_content_to_results(payload, max_results)


def _deepseek_anthropic_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    if not shutil.which("curl"):
        raise RuntimeError("curl is required for DeepSeek Anthropic web search")
    marker = "\n__XIAOMING_CURL_STATUS__"
    command = ["curl", "-sS", "--max-time", str(timeout_seconds), "-X", "POST"]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])
    command.extend(["-w", marker + "%{http_code}", "--data-binary", "@-", url])
    completed = subprocess.run(command, input=json.dumps(body).encode(), text=False, capture_output=True, timeout=timeout_seconds + 5, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"curl failed with exit code {completed.returncode}")
    raw_body, sep, status_bytes = completed.stdout.rpartition(marker.encode())
    if not sep:
        raise RuntimeError("DeepSeek Anthropic response missing HTTP status")
    status_text = status_bytes.decode("utf-8", errors="replace").strip()
    status = int(status_text or "0")
    text = raw_body.decode("utf-8", errors="replace")
    if status >= 400:
        raise RuntimeError(f"DeepSeek Anthropic web search HTTP {status}: {truncate_middle(text, 1000)}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"DeepSeek Anthropic web search returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("DeepSeek Anthropic web search returned a non-object response")
    return data


def _deepseek_anthropic_content_to_results(payload: dict[str, Any], max_results: int) -> list[dict[str, str]]:
    content = payload.get("content")
    if not isinstance(content, list):
        return []
    text_summary = _normalize_text("\n".join(str(block.get("text") or "") for block in content if isinstance(block, dict) and block.get("type") == "text"))
    usage_note = _deepseek_usage_note(payload)
    snippet = "\n".join(part for part in [text_summary, usage_note] if part)
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "web_search_tool_result":
            continue
        for item in block.get("content") or []:
            if not isinstance(item, dict) or item.get("type") != "web_search_result":
                continue
            url = str(item.get("url") or "").strip()
            title = _normalize_text(str(item.get("title") or "DeepSeek web search result"))
            if not url or url in seen:
                continue
            seen.add(url)
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                return results
    if text_summary:
        urls = _extract_source_urls(text_summary) or _extract_urls(text_summary)
        for url in urls[:max_results]:
            if url in seen:
                continue
            seen.add(url)
            results.append({"title": "DeepSeek web search citation", "url": url, "snippet": snippet})
        if results:
            return results
        return [{"title": "DeepSeek web search answer", "url": "https://api-docs.deepseek.com/", "snippet": snippet}]
    return []


def _duckduckgo_search(query: str, max_results: int) -> list[dict[str, str]]:
    url = _search_url_for_backend("duckduckgo", query, max_results)
    html = _http_get(url).text
    return _parse_duckduckgo_results(html)[:max_results]


def _bing_cn_search(query: str, max_results: int) -> list[dict[str, str]]:
    html = _http_get(_search_url_for_backend("bing_cn", query, max_results)).text
    return _parse_bing_results(html)[:max_results]


def _sogou_news_search(query: str, max_results: int) -> list[dict[str, str]]:
    html = _http_get(_search_url_for_backend("sogou_news", query, max_results)).text
    return _parse_generic_search_results(html, preferred_hosts=["sogou.com"])[:max_results]


def _so_search(query: str, max_results: int) -> list[dict[str, str]]:
    html = _http_get(_search_url_for_backend("so_search", query, max_results)).text
    return _parse_generic_search_results(html, preferred_hosts=["so.com"])[:max_results]


def _google_news_search(query: str, max_results: int) -> list[dict[str, str]]:
    url = _search_url_for_backend("google_news", query, max_results)
    xml_text = _http_get(url).text
    root = ET.fromstring(xml_text)
    results: list[dict[str, str]] = []
    for item in root.findall(".//item"):
        title = item.findtext("title") or ""
        link = item.findtext("link") or ""
        source = item.find("source")
        source_name = source.text if source is not None and source.text else ""
        pub_date = item.findtext("pubDate") or ""
        snippet = " - ".join(part for part in [source_name, pub_date] if part)
        if title and link:
            results.append({"title": _normalize_text(title), "url": link, "snippet": snippet})
        if len(results) >= max_results:
            break
    return results


def _search_with_fallback(query: str, max_results: int) -> tuple[str, list[dict[str, str]], list[str]]:
    attempted: list[str] = []
    errors: list[str] = []
    for backend in _candidate_backends(query):
        attempted.append(backend)
        try:
            results = _search_backend(backend, query, max_results)
        except Exception as exc:
            errors.append(f"{backend}: {exc}")
            continue
        if results:
            return backend, results, attempted
    detail = "; ".join(errors) if errors else "all backends returned no results"
    raise RuntimeError(f"all search backends failed or returned no results; attempted={', '.join(attempted)}; {detail}")


def _candidate_backends(query: str) -> list[str]:
    candidates: list[str] = []
    if _deepseek_web_search_enabled() and _deepseek_api_key():
        candidates.append("deepseek_anthropic")
    if _kimi_api_key():
        candidates.append("kimi")
    if os.environ.get("BRAVE_SEARCH_API_KEY"):
        candidates.append("brave")
    if _contains_cjk(query):
        if _looks_like_news_query(query):
            candidates.extend(["sogou_news", "bing_cn", "so_search"])
            candidates.append("google_news")
        else:
            candidates.extend(["bing_cn", "sogou_news", "so_search"])
    elif _looks_like_news_query(query):
        candidates.extend(["bing_cn", "google_news", "duckduckgo"])
    else:
        candidates.extend(["bing_cn", "duckduckgo", "so_search"])
    return _dedupe(candidates)


def _search_backend(backend: str, query: str, max_results: int) -> list[dict[str, str]]:
    if backend == "kimi":
        return _kimi_search(query, max_results)
    if backend == "deepseek_anthropic":
        return _deepseek_anthropic_search(query, max_results)
    if backend == "brave":
        return _brave_search(query, max_results)
    if backend == "google_news":
        return _google_news_search(query, max_results)
    if backend == "bing_cn":
        return _bing_cn_search(query, max_results)
    if backend == "sogou_news":
        return _sogou_news_search(query, max_results)
    if backend == "so_search":
        return _so_search(query, max_results)
    return _duckduckgo_search(query, max_results)


def _parse_duckduckgo_results(html: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    pattern = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.I | re.S)
    for match in pattern.finditer(html):
        url = urllib.parse.unquote(match.group(1))
        if url.startswith("//duckduckgo.com/l/?"):
            parsed = urllib.parse.urlparse("https:" + url)
            params = urllib.parse.parse_qs(parsed.query)
            url = params.get("uddg", [url])[0]
        parsed_url = urllib.parse.urlparse(url)
        if parsed_url.netloc.endswith("duckduckgo.com") and parsed_url.path.endswith("/y.js"):
            continue
        title = _normalize_text(re.sub(r"<[^>]+>", "", match.group(2)))
        if url and title:
            results.append({"title": title, "url": url, "snippet": ""})
    return results


def _parse_bing_results(html: str) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    pattern = re.compile(r'<li[^>]+class="b_algo"[^>]*>.*?<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?(?:<p[^>]*>(.*?)</p>)?', re.I | re.S)
    for match in pattern.finditer(html):
        url = match.group(1)
        title = _clean_html(match.group(2))
        snippet = _clean_html(match.group(3) or "")
        if url and title:
            results.append({"title": title, "url": url, "snippet": snippet})
    if results:
        return results
    return _parse_generic_search_results(html, preferred_hosts=["bing.com", "microsoft.com"])


def _parse_generic_search_results(html: str, preferred_hosts: list[str] | None = None) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    pattern = re.compile(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
    for match in pattern.finditer(html):
        raw_url = match.group(1)
        title = _clean_html(match.group(2))
        if not title or len(title) < 4:
            continue
        url = _normalize_result_url(raw_url)
        if not url or url in seen:
            continue
        host = urllib.parse.urlparse(url).hostname or ""
        if preferred_hosts and any(_domain_matches(host, domain) for domain in preferred_hosts):
            continue
        if host and any(_domain_matches(host, blocked) for blocked in ["baidu.com", "sogou.com", "so.com", "bing.com"]):
            continue
        seen.add(url)
        results.append({"title": title, "url": url, "snippet": ""})
        if len(results) >= 10:
            break
    return results


def _normalize_result_url(raw_url: str) -> str:
    url = raw_url.replace("&amp;", "&")
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/"):
        return ""
    if not url.startswith(("http://", "https://")):
        return ""
    return urllib.parse.unquote(url)


def _clean_html(text: str) -> str:
    return _normalize_text(re.sub(r"<[^>]+>", "", text).replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"'))


def _looks_like_news_query(query: str) -> bool:
    lowered = query.lower()
    return any(word in lowered for word in ["news", "热点", "新闻", "latest", "today", "今日", "头条"])


def _contains_cjk(query: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", query))


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _search_url_for_backend(backend: str, query: str, max_results: int) -> str:
    if backend == "kimi":
        return "https://platform.kimi.com/?q=" + urllib.parse.quote(query)
    if backend == "deepseek_anthropic":
        return "https://api.deepseek.com/anthropic/v1/messages?q=" + urllib.parse.quote(query)
    if backend == "brave":
        return "https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": query, "count": max_results})
    if backend == "google_news":
        return "https://news.google.com/rss/search?" + urllib.parse.urlencode({"q": query, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"})
    if backend == "bing_cn":
        return "https://cn.bing.com/search?" + urllib.parse.urlencode({"q": query})
    if backend == "sogou_news":
        return "https://news.sogou.com/news?" + urllib.parse.urlencode({"query": query})
    if backend == "so_search":
        return "https://www.so.com/s?" + urllib.parse.urlencode({"q": query})
    return "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})


def _filter_results(results: list[dict[str, str]], allowed: list[str], blocked: list[str]) -> list[dict[str, str]]:
    filtered = []
    for result in results:
        host = urllib.parse.urlparse(result.get("url") or "").hostname or ""
        if allowed and not any(_domain_matches(host, domain) for domain in allowed):
            continue
        if blocked and any(_domain_matches(host, domain) for domain in blocked):
            continue
        filtered.append(result)
    return filtered


def _kimi_api_key() -> str | None:
    return os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")


def _deepseek_api_key() -> str | None:
    return os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")


def _deepseek_web_search_enabled() -> bool:
    return str(os.environ.get("XIAOMING_DEEPSEEK_WEB_SEARCH") or "1").strip().lower() not in {"0", "false", "no", "off"}


def _deepseek_web_search_timeout_seconds() -> int:
    return _bounded_int(os.environ.get("DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS"), default=DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS, minimum=5, maximum=180)


def _deepseek_web_search_max_tokens() -> int:
    return _bounded_int(os.environ.get("DEEPSEEK_WEB_SEARCH_MAX_TOKENS"), default=1200, minimum=200, maximum=4000)


def _deepseek_usage_note(payload: dict[str, Any]) -> str:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return ""
    server_tool_use = usage.get("server_tool_use")
    if not isinstance(server_tool_use, dict):
        return ""
    requests = server_tool_use.get("web_search_requests")
    if requests is None:
        return ""
    return f"DeepSeek web search requests: {requests}"


def _kimi_assistant_message(message: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"role": "assistant"}
    content = getattr(message, "content", None)
    if content:
        data["content"] = content
    tool_calls = getattr(message, "tool_calls", None) or []
    if tool_calls:
        data["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": getattr(tool_call, "type", "function"),
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments or "{}",
                },
            }
            for tool_call in tool_calls
        ]
    return data


def _kimi_content_to_results(content: str, max_results: int) -> list[dict[str, str]]:
    text = _normalize_text(content)
    if not text:
        return []
    urls = _extract_source_urls(text) or _extract_urls(text)
    if not urls:
        return [{"title": "Kimi web search answer", "url": "https://platform.kimi.com/", "snippet": text}]
    results: list[dict[str, str]] = []
    for url in urls[:max_results]:
        results.append({"title": "Kimi web search citation", "url": url, "snippet": text})
    return results


def _extract_source_urls(text: str) -> list[str]:
    match = re.search(r"(?i)(sources?|references?)\s*:|来源\s*[:：]|参考\s*[:：]", text)
    if not match:
        return []
    return _extract_urls(text[match.end() :])


def _extract_urls(text: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://[^\s<>)\"']+", text):
        url = match.group(0).rstrip(".,;:。；，`")
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _validate_public_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "http":
        parsed = parsed._replace(scheme="https")
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("url must be a valid http or https URL")
    if parsed.username or parsed.password:
        raise ValueError("urls with embedded credentials are not allowed")
    host = parsed.hostname or ""
    if "." not in host or host in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("only public hostnames are allowed")
    return urllib.parse.urlunparse(parsed)


class _HttpResponse:
    def __init__(self, text: str, final_url: str, content_type: str, raw_size: int):
        self.text = text
        self.final_url = final_url
        self.content_type = content_type
        self.raw_size = raw_size


def _http_get(url: str, headers: dict[str, str] | None = None) -> _HttpResponse:
    if shutil.which("curl"):
        return _curl_get(url, headers or {})
    return _urllib_get(url, headers or {})


def _curl_get(url: str, headers: dict[str, str]) -> _HttpResponse:
    marker = "\n__XIAOMING_CURL_META__"
    command = [
        "curl",
        "-sSL",
        "--max-time",
        str(DEFAULT_TIMEOUT_SECONDS),
        "--max-filesize",
        str(MAX_FETCH_BYTES),
        "-A",
        "xiaoming-cli/0.1",
    ]
    for key, value in headers.items():
        command.extend(["-H", f"{key}: {value}"])
    command.extend(["-w", marker + "%{url_effective}\t%{content_type}", url])
    completed = subprocess.run(command, text=False, capture_output=True, timeout=DEFAULT_TIMEOUT_SECONDS + 5, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(stderr or f"curl failed with exit code {completed.returncode}")
    if len(completed.stdout) > MAX_FETCH_BYTES + 4096:
        raise RuntimeError(f"response exceeds {MAX_FETCH_BYTES} bytes")
    marker_bytes = marker.encode()
    body, sep, meta = completed.stdout.rpartition(marker_bytes)
    if not sep:
        body = completed.stdout
        meta_text = ""
    else:
        meta_text = meta.decode("utf-8", errors="replace")
    final_url, _, content_type = meta_text.partition("\t")
    return _HttpResponse(body.decode(_charset(content_type), errors="replace"), final_url or url, content_type.strip(), len(body))


def _urllib_get(url: str, headers: dict[str, str]) -> _HttpResponse:
    request = urllib.request.Request(url, headers={"User-Agent": "xiaoming-cli/0.1", **headers})
    with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
        data = response.read(MAX_FETCH_BYTES + 1)
        if len(data) > MAX_FETCH_BYTES:
            raise RuntimeError(f"response exceeds {MAX_FETCH_BYTES} bytes")
        content_type = response.headers.get("content-type", "")
        return _HttpResponse(data.decode(_charset(content_type), errors="replace"), response.geturl(), content_type, len(data))


def _domain_list(value: object) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).lower().strip() for item in value if str(item).strip()]
    return [item.strip().lower() for item in str(value).split(",") if item.strip()]


def _domain_matches(host: str, domain: str) -> bool:
    host = host.lower()
    domain = domain.lower().removeprefix("*.").removeprefix("domain:")
    return host == domain or host.endswith("." + domain)


def _bounded_int(value: object, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    return max(minimum, min(maximum, int(value)))


def _charset(content_type: str) -> str:
    match = re.search(r"charset=([^;\s]+)", content_type, re.I)
    return match.group(1) if match else "utf-8"


def _looks_binary(data: bytes, content_type: str) -> bool:
    lowered = content_type.lower()
    if lowered and not any(kind in lowered for kind in ["text", "html", "json", "xml", "markdown", "javascript"]):
        return True
    return b"\x00" in data[:1024]


def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"<(html|body|main|article|p|div|h1|title)\b", text, re.I))


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
        if tag in {"p", "br", "div", "section", "article", "h1", "h2", "h3", "li"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str):
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "section", "article", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str):
        if not self.skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return _normalize_text(" ".join(parser.parts))


def _normalize_text(text: str) -> str:
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n\s*", "\n\n", text)
    return text.strip()
