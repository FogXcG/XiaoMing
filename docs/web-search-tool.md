# DeepSeek Web Search Tool

This document defines a minimal `web_search` tool backed only by DeepSeek's Anthropic-compatible hosted web search.

The LLM decides when to call `web_search`. The host application executes the DeepSeek request and returns normalized search results with source URLs.

## Tool Definition

OpenAI-compatible function tool:

```json
{
  "type": "function",
  "function": {
    "name": "web_search",
    "description": "Search the web for current information using DeepSeek hosted web search. Returns titles, URLs, and snippets. Use this for recent information or facts outside local context. Include source URLs in the final answer.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "Search query."
        },
        "max_results": {
          "type": "integer",
          "description": "Maximum number of search results to return. Default 5. Clamp to 1-10."
        }
      },
      "required": ["query"],
      "additionalProperties": false
    }
  }
}
```

## Runtime Contract

Input validation:

- Reject `query` shorter than 2 characters.
- Default `max_results` to `5`.
- Clamp `max_results` to `1..10`.
- Set DeepSeek `max_uses` to `min(max_results, 5)`.

Output format returned to the LLM:

```text
Search backend: deepseek_anthropic
Search URL: https://api.deepseek.com/anthropic/v1/messages?q=...
Search results for: query text

1. Result title
   URL: https://example.com/article
   Snippet: Short relevant excerpt or DeepSeek summary.
```

Recommended output limit:

- Cap tool output to roughly `30,000` characters.
- Preserve source URLs when truncating long content.

## DeepSeek Request

DeepSeek hosted web search is called through the Anthropic-compatible messages API.

Endpoint:

```text
https://api.deepseek.com/anthropic/v1/messages
```

Headers:

```text
content-type: application/json
x-api-key: $DEEPSEEK_API_KEY
anthropic-version: 2023-06-01
```

Request body:

```json
{
  "model": "deepseek-v4-flash",
  "max_tokens": 1200,
  "messages": [
    {
      "role": "user",
      "content": "Search query: today's AI news\nReturn up to 5 useful results with source URLs."
    }
  ],
  "tools": [
    {
      "type": "web_search_20250305",
      "name": "web_search",
      "max_uses": 5
    }
  ]
}
```

## Environment Variables

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export DEEPSEEK_WEB_SEARCH_MODEL=deepseek-v4-flash
export DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS=60
export DEEPSEEK_WEB_SEARCH_MAX_TOKENS=1200
```

`DEEPSEEK_ANTHROPIC_BASE_URL`, `DEEPSEEK_WEB_SEARCH_MODEL`, `DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS`, and `DEEPSEEK_WEB_SEARCH_MAX_TOKENS` can all have defaults in the host application.

## Reference Pseudocode

```python
def web_search(args):
    query = str(args.get("query") or "").strip()
    if len(query) < 2:
        return error("query must be at least 2 characters")

    max_results = clamp(args.get("max_results") or 5, 1, 10)
    max_uses = min(max_results, 5)

    body = {
        "model": env("DEEPSEEK_WEB_SEARCH_MODEL", "deepseek-v4-flash"),
        "max_tokens": int(env("DEEPSEEK_WEB_SEARCH_MAX_TOKENS", "1200")),
        "messages": [
            {
                "role": "user",
                "content": f"Search query: {query}\nReturn up to {max_results} useful results with source URLs.",
            }
        ],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": max_uses,
            }
        ],
    }

    payload = post_json(
        url=env("DEEPSEEK_ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic").rstrip("/") + "/v1/messages",
        headers={
            "content-type": "application/json",
            "x-api-key": env("DEEPSEEK_API_KEY"),
            "anthropic-version": "2023-06-01",
        },
        body=body,
        timeout_seconds=int(env("DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS", "60")),
    )

    results = parse_deepseek_web_search_results(payload)
    return format_search_results("deepseek_anthropic", query, results[:max_results])
```

## Response Parsing

Preferred parsing:

- Read `content[]`.
- Find blocks with `type == "web_search_tool_result"`.
- Inside those blocks, read items with `type == "web_search_result"`.
- Extract `title`, `url`, and optional summary/snippet.

Fallback parsing:

- If DeepSeek returns text with citations but no structured result blocks, extract URLs from the text.
- Return the text as the snippet so the caller still has useful source context.

Minimal normalized result:

```json
{
  "title": "Result title",
  "url": "https://example.com/source",
  "snippet": "Short summary or citation context"
}
```
