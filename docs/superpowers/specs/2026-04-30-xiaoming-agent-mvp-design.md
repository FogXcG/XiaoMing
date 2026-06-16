# Xiaoming Agent MVP Design

## Goal

Build a local CLI coding agent MVP that can work inside a repository, inspect files, make focused code changes, run verification commands, iterate on tool results, and report the final outcome.

The MVP uses the OpenAI Responses API with native function tool calling. It does not train a model or implement a custom model runtime.

## Non-Goals

The first version will not include multi-agent delegation, long-term memory, a web UI, an IDE plugin, MCP integration, full multi-provider support, automatic git commits, vector retrieval, cloud sandboxes, background jobs, streaming output, or model-generated context compaction.

The core question for the MVP is whether a model can complete a small real repository task through a controlled local tool loop.

## Architecture

```text
CLI
  -> AgentLoop
  -> OpenAIProvider
  -> OpenAI Responses API
  -> Normalized ToolCall
  -> ToolRegistry
  -> Local Tools
  -> ToolResult
  -> AgentLoop
```

The CLI accepts the user task, loads configuration, and handles approval prompts. The agent loop owns turn management, tool execution, context items, and termination. The LLM layer hides OpenAI SDK details behind internal request and response types. Tools provide local capabilities such as file reads, search, patch application, and shell execution. Policy code handles workspace path checks, command policy, and approval decisions.

The agent loop must not depend on OpenAI raw response objects. The provider converts OpenAI SDK objects into plain internal types and plain dictionaries that can be sent back to the Responses API.

## Project Layout

```text
xiaoming/
  pyproject.toml
  README.md
  src/
    xiaoming/
      __init__.py
      cli.py
      config.py
      agent_loop.py
      prompts/
        system.md
      llm/
        __init__.py
        types.py
        provider.py
        openai_provider.py
        openai_tools.py
      tools/
        __init__.py
        base.py
        registry.py
        list_files.py
        read_file.py
        search_code.py
        apply_patch.py
        shell.py
        git_status.py
      policy/
        __init__.py
        paths.py
        approvals.py
        shell_policy.py
      context/
        __init__.py
        truncation.py
        formatting.py
  tests/
    test_agent_loop.py
    test_openai_tools.py
    test_openai_provider_extraction.py
    test_read_file.py
    test_search_code.py
    test_shell_policy.py
    test_apply_patch.py
```

## CLI

The primary command is:

```bash
xiaoming "修复当前测试失败"
```

Useful options:

```bash
xiaoming "任务" --model gpt-5 --approval-mode auto-edit --max-turns 20
```

Configuration precedence is CLI arguments, then project config, then environment variables, then defaults. OpenAI credentials come from `OPENAI_API_KEY`, with optional `OPENAI_BASE_URL`.

When a tool requires approval, the CLI presents the exact action:

```text
Run command?
npm test

Approve? [y/N]
```

If the user denies approval, the denial is returned to the model as a tool result.

## Configuration

Example:

```toml
[model]
provider = "openai"
model = "gpt-5"
temperature = 0.2
max_output_tokens = 4096

[agent]
max_turns = 20
approval_mode = "suggest"

[workspace]
root = "."
```

Approval modes:

```text
suggest     All write operations and shell commands require confirmation.
auto_edit   apply_patch may run automatically; shell still needs approval or whitelist.
full_auto   apply patches and whitelisted shell commands automatically.
```

The MVP default is `suggest`.

## Instruction Priority

Instructions are assembled with this priority:

```text
safety policy > current user request > AGENTS.md/project rules > default agent prompt
```

Safety policy cannot be overridden by project rules or a user task. The current user request can override project habits. Project rules supplement the default prompt.

## LLM Layer

The MVP uses OpenAI Responses API because it is suited to tool-using agent workflows and supports native function tools.

Provider interface:

```python
class LLMProvider:
    def complete(self, request: LLMRequest) -> LLMResponse:
        ...
```

Internal types:

```python
@dataclass
class LLMRequest:
    instructions: str
    input_items: list[dict]
    tools: list[ToolSpec]
    model: str
    temperature: float
    max_output_tokens: int

@dataclass
class ToolCall:
    id: str
    name: str
    args: dict

@dataclass
class LLMResponse:
    message: str | None
    tool_calls: list[ToolCall]
    output_items: list[dict]
    raw: object
```

`output_items` must contain plain dictionaries that can be passed back as Responses API `input`. OpenAI SDK objects must not leak into `AgentLoop`.

`message` is extracted by `OpenAIProvider`, preferably from the SDK's text convenience field and otherwise by traversing output text blocks. `AgentLoop` does not inspect OpenAI fields directly.

## Tool Calling

Tools expose one internal schema:

```python
@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict
```

Each local tool implements:

```python
class Tool:
    name: str
    description: str
    input_schema: dict

    def run(self, args: dict) -> ToolResult:
        ...
```

`ToolSpec` is converted to an OpenAI function tool:

```python
{
  "type": "function",
  "name": "read_file",
  "description": "Read a UTF-8 text file from the workspace.",
  "parameters": {
    "type": "object",
    "properties": {
      "path": {"type": "string"},
      "start_line": {"type": ["integer", "null"]},
      "limit": {"type": ["integer", "null"]}
    },
    "required": ["path", "start_line", "limit"],
    "additionalProperties": false
  },
  "strict": true
}
```

OpenAI tool schemas use `strict: true`. Every object schema uses `additionalProperties: false`. All properties are listed in `required`; nullable optional values use a union with `null`.

The MVP sets `parallel_tool_calls=False` because local coding operations usually have sequential dependencies and because serial execution simplifies permissions and context.

## Agent Loop

The loop keeps a list of Responses API input items:

```python
input_items = [
    {"role": "user", "content": user_task}
]

for turn in range(max_turns):
    response = provider.complete(
        LLMRequest(
            instructions=instructions,
            input_items=input_items,
            tools=tool_registry.specs(),
            model=config.model,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        )
    )

    input_items.extend(response.output_items)

    if not response.tool_calls:
        return response.message

    for tool_call in response.tool_calls:
        result = tool_registry.run(tool_call.name, tool_call.args)
        input_items.append({
            "type": "function_call_output",
            "call_id": tool_call.id,
            "output": result.to_text(),
        })

raise MaxTurnsExceeded
```

Tool output is always a string. Structured results are serialized before they are returned to the model. Tool failures are also returned as `function_call_output` so the model can correct its next action.

The loop stops when the model returns a final message with no tool calls, when max turns is reached, when the user rejects a required action, or when the provider reports an unrecoverable error.

## System Prompt

`src/xiaoming/prompts/system.md`:

```text
You are Xiaoming, a local coding agent working inside a user's repository.

Rules:
- Use tools to inspect, modify, and verify the repository.
- Do not assume file contents. Use search_code or read_file before editing.
- Modify existing files only through apply_patch.
- Prefer small, focused changes.
- Do not refactor unrelated code.
- After editing, run the smallest relevant verification command.
- If a command requires approval, request it through the tool result flow.
- If information is insufficient, inspect more context or ask the user.
- When done, summarize changed files and verification results.
```

If `AGENTS.md` exists in the workspace, startup reads it and merges it into instructions under the priority rules above.

## Tools

### list_files

Lists workspace files.

Parameters:

```json
{
  "path": "string | null",
  "pattern": "string | null"
}
```

It does not return `.git`, and it ignores common generated directories such as `node_modules`, `.venv`, `dist`, and `build`. Long output is truncated.

### read_file

Reads a text file slice.

Parameters:

```json
{
  "path": "string",
  "start_line": "integer | null",
  "limit": "integer | null"
}
```

The resolved path must stay inside the workspace. The default limit is 200 lines. Binary files are rejected.

### search_code

Searches code using `rg`.

Parameters:

```json
{
  "query": "string",
  "path": "string | null"
}
```

The search path must stay inside the workspace. Result count and output length are capped.

### apply_patch

Applies Codex-style patches only.

Supported format:

```text
*** Begin Patch
*** Update File: path/to/file.py
@@
-old line
+new line
*** End Patch
```

Parameters:

```json
{
  "patch": "string"
}
```

The patch can only affect files inside the workspace. The tool does not execute shell patch commands and does not accept multiple patch dialects. Approval depends on `approval_mode`. Failures are returned to the model.

### shell

Runs verification commands.

Parameters:

```json
{
  "command": "string"
}
```

The MVP does not attempt complete shell parsing. Automatic execution is allowed only for exact whitelist commands or explicitly safe prefixes. Any command with shell control syntax requires approval or is rejected.

Auto-approved commands:

```text
git status
git diff
pytest
python -m pytest
npm test
npm run test
npm run lint
```

Commands such as `pytest tests/test_foo.py`, `npm install`, and `curl example.com` require approval. Commands such as `rm -rf .`, `git reset --hard`, `git checkout --`, and `sudo ...` are rejected by default.

The policy treats `;`, `&&`, `||`, `|`, `>`, `<`, and command substitution as control syntax.

### git_status

Returns current git status. It has no parameters.

## Safety

All file paths are resolved and must stay inside the workspace root. Sensitive files such as `.env` and locations such as `~/.ssh` are denied unless explicitly allowed. Symlinks may not be used to escape the workspace.

Shell commands pass through policy before execution. Write operations require approval by default. Network commands require approval. Destructive commands are rejected by default.

The MVP is not a complete sandbox, but it blocks obvious dangerous operations.

## Context Management

The first version only truncates context; it does not ask a model to summarize old context.

The agent keeps instructions, the original task, recent model outputs, and recent tool results. Tool outputs are capped. File reads are paginated by default. Shell output keeps the beginning and end when truncated. If the request still exceeds context limits after truncation, the agent stops and asks the user to narrow the task.

Tool results use a stable text shape:

```text
Tool: read_file
Status: success
Output:
...
```

Errors are returned the same way:

```text
Tool: shell
Status: error
Error:
Command rejected by policy: rm -rf .
```

## Error Handling

Provider errors:

- Missing API key fails immediately.
- Timeout retries once.
- Rate limit backs off once, then fails.
- Context length errors trigger truncation and one retry.
- Unexpected SDK response format reports a concise raw summary.

Tool errors:

- Unknown tool returns a tool error.
- Invalid arguments return a schema error.
- Execution failures return stderr or an error message.
- Approval denial returns a denial tool result.

Agent errors:

- Max turns stops execution and reports current state.
- Patch failure allows the model to reread files and retry.
- Test failure allows further iteration until max turns.

## Testing

Default tests do not call the OpenAI API.

Unit tests cover:

- `ToolSpec` conversion to OpenAI function tools.
- Strict schema generation.
- OpenAI response item conversion to plain dictionaries.
- OpenAI function call extraction.
- Final message extraction.
- `function_call_output` construction.
- Workspace path restrictions.
- Shell whitelist, approval, and rejection policy.
- `read_file` pagination and truncation.
- `apply_patch` success and failure cases.

Integration tests use fake providers to cover:

- A `read_file` tool call.
- An `apply_patch` tool call.
- A `shell` tool call.
- Multi-turn agent loop execution.
- Final answer termination.
- Recovery after a tool failure.

Real OpenAI smoke tests are skipped by default and require an explicit marker and `OPENAI_API_KEY`.

## Acceptance Criteria

The MVP is complete when:

- The CLI starts an agent task.
- The OpenAI Responses API provider works with native function tool calling.
- At least `list_files`, `read_file`, `search_code`, `apply_patch`, and `shell` are available.
- A small real bugfix flow can be completed end to end.
- Final output summarizes changed files and verification.
- Dangerous commands are not automatically executed.
- Default tests do not require network or API credentials.
- Core logic has unit coverage.

## Later Extensions

Future phases may add Anthropic provider support, MCP tool adapters, richer approval UI, model-generated context compaction, git diff review, session persistence, streaming output, multi-agent task splitting, project memory, and GitHub issue or PR integration.
