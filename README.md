# Xiaoming

Xiaoming is a personal assistant agent.

The long-term goal is not only a coding assistant. Xiaoming is meant to stay close to the user, keep a continuous conversation when allowed, handle multiple tasks in the background, and represent the user when coordinating with other people or agents.

Today, Xiaoming is implemented as a local CLI agent. The CLI is the current development surface for the larger personal assistant vision, with asynchronous workers, native tool calling, persistent sessions, skills, web search, logs, checkpoints, hooks, and context management.

## Vision

Xiaoming should eventually work like a personal Jarvis:

- Stay close to the user, on the device that makes the most sense: desktop, phone, wearable, home device, or future hardware.
- Keep a continuous conversation with the user without blocking on long-running work.
- Listen through voice when authorized, respond through voice or text, and remain available throughout the day.
- Handle multiple tasks at once through asynchronous workers.
- Ask the user only when confirmation, preference, permission, or sensitive information is needed.
- Evolve its own capabilities because one static codebase cannot cover every future need.
- Communicate with other users' Xiaoming agents through ordinary natural-language messaging.
- Coordinate with home Xiaoming agents running on home devices, so a personal Xiaoming can control or delegate to the user's local home assistant.

## Core Ideas

### Always Conversational

The user should be able to keep talking to Xiaoming while work is happening. Long tasks should not freeze the conversation. Xiaoming should acknowledge work, schedule it, report progress, and return to the user when it needs input.

### Asynchronous by Default

Background workers are core to Xiaoming's design. They allow Xiaoming to:

- keep chatting with the user,
- run multiple tasks at the same time,
- wait for confirmations without blocking the main conversation,
- report task progress and results back to the main assistant.

### Main Xiaoming Is the Conversation Layer

In interactive chat, the main Xiaoming is the user's real-time conversation partner and coordinator. Its first responsibility is keeping the conversation responsive and clean.

Main Xiaoming can answer simple questions directly and may handle small, bounded foreground work when that is the most natural path. Larger, slower, multi-file, setup-heavy, skill-installing, repository-cloning, or otherwise blocking work is scheduled through the background task system.

If the user sends a new message while a foreground task is still running, the runtime gives the new user message priority and moves the unfinished foreground task to a background worker with forked conversation context.

### Foreground To Background

Foreground execution is an ergonomics feature, not the main execution model. It is meant for quick work that benefits from immediate interaction.

When foreground work risks blocking the user:

- the current foreground turn can be interrupted,
- the unfinished task is scheduled as a background task,
- the worker receives the inherited context and a directive to continue rather than restart,
- main Xiaoming immediately returns to handling the user's latest message.

### Workers Do the Work

Workers are independent coding agents started as subprocesses. A worker has access to repository tools, shell, write tools, skills, web search/fetch, and `talk`.

When a worker needs a decision or clarification, it calls `talk`. Xiaoming relays that request to the user or answers from delegated task intent when it is safe and clear. Permission approval uses the parent approval path.

Workers finish by producing a final natural-language result. Xiaoming records that result as the worker submission and verifies it before telling the user the task is complete.

### Voice First, Text Compatible

The ideal Xiaoming can listen and speak for long periods, but text remains the simplest and most inspectable interface. The CLI is the current text interface; voice can be layered on top of the same session, task, permission, and worker model.

### Self-Evolving

Xiaoming should be able to improve Xiaoming. This must be done safely:

- self-evolution work should happen in isolated candidate worktrees or versions,
- the running stable Xiaoming should not directly overwrite itself,
- candidates must pass tests and smoke checks before activation,
- a previous stable version must always be available,
- failed candidates should preserve logs and worktrees for diagnosis.

See [docs/TODO.md](docs/TODO.md) for the current self-evolution discussion item.

### Natural-Language Agent-to-Agent Communication

For personal assistant use cases, Xiaoming-to-Xiaoming communication should be as simple as messaging between users. The subject changes from a human typing to an assistant representing that human.

Example:

```text
Alice: Help me ask Zhang San if he wants to watch a movie this weekend.

Alice's Xiaoming -> Zhang San's Xiaoming:
Hi, I am Alice's Xiaoming. Alice would like to watch a movie with Zhang San this weekend.
Is Zhang San available on Saturday evening or Sunday afternoon? Please only share available time windows,
not private calendar details.
```

This does not require a complex protocol at first. The transport can be an existing messaging system. The critical parts are identity, consent, privacy boundaries, auditability, and user confirmation for commitments.

## Current Capabilities

- Stateful multi-turn CLI chat.
- Default resume of the latest valid session for the current workspace.
- Native tool calling with OpenAI-compatible providers.
- DeepSeek support, defaulting to `deepseek-v4-flash`.
- OpenAI support.
- Streaming output by default.
- Model timeout and idle progress messages.
- Asynchronous background tasks with worker subprocesses.
- Multiple task scheduling with conflict-aware queueing.
- Worker-to-user communication through main Xiaoming.
- Background task cancellation from chat and slash commands.
- LLM-based background task verification with read-only inspection tools.
- Persistent sessions and resume.
- Context tracking and compaction with model-specific context windows.
- Checkpoints and rewind for Xiaoming file-editing tools.
- Project skills under `.agents/skills/`.
- Built-in `skill-installer` and `find-skills` skills.
- Native skill installation into the current workspace.
- Web search with DeepSeek hosted search, Kimi/Moonshot, Brave, and built-in fallback search backends.
- Web fetch for public HTTP/HTTPS pages.
- Logs with secret redaction.
- Workspace hooks.
- Lightweight local eval runner.
- Terminal-Bench adapter for external benchmarking.

## Architecture

At a high level:

```text
User
  -> xiaoming-cli
  -> main Xiaoming orchestrator
  -> foreground work when quick and interactive
  -> AsyncCoordinator
  -> LLMScheduler
  -> WorkerProcess
  -> worker AgentLoop
  -> tools / shell / skills / talk
  -> verifier worker
  -> main Xiaoming notice back to user
```

### CLI Runtime

`xiaoming-cli` and `xiaoming` are both entry points to `xiaoming.cli:main`.

Interactive chat uses `ChatRuntime`. It:

- creates or resumes a session,
- injects bootstrap context such as `AGENTS.md`,
- starts the async coordinator,
- builds the main orchestrator loop,
- creates per-turn checkpoints,
- handles slash commands,
- buffers background notices so they do not interrupt active user input.

### Main Orchestrator

The main orchestrator normally runs with a restricted coordination tool surface:

- `web_search`
- `schedule_background_task`
- `background_tasks_status`
- `follow_background_task`
- `cancel_background_task`
- `answer_worker_question`

For small foreground work, the runtime can temporarily use a foreground capability profile so the main loop can complete the quick task directly. If the user sends another message before the foreground task completes, the task is moved to a background worker and the main loop resumes normal conversation handling.

This keeps the conversation layer responsive while avoiding the overhead of background workers for every tiny task.

### Background Coordinator

`AsyncCoordinator` owns background task state. It persists tasks under `.xiaoming/tasks/` and manages:

- task creation,
- conflict-aware queueing,
- worker subprocess lifecycle,
- worker progress,
- pending worker questions,
- user answers routed back to workers,
- task cancellation,
- task verification,
- completion or failure notices.

The scheduler is LLM-based. It chooses whether a new user message should start a new task, attach to an existing task, wait for conflicting work, cancel and restart work, or simply remain chat.

### Worker Runtime

Each background task starts `python -m xiaoming.worker_main` as a subprocess.

Worker stdout is reserved for the JSON-line worker protocol. Normal output is redirected to stderr and worker logs are written under:

```text
.xiaoming/logs/workers/<task-id>.log
.xiaoming/logs/workers/<task-id>.stderr.log
```

Workers receive:

- the task contract,
- bootstrap context,
- worker protocol instructions,
- normal workspace tools,
- write tools,
- shell,
- skill install/load tools,
- web search/fetch,
- `talk`.

Workers complete by returning a final answer. The coordinator treats that final answer as a worker submission, records it on the task, and starts verification.

### Verification

Completed worker submissions are verified before being accepted. Verification runs in a separate verifier worker with read-only tools:

- `list_files`
- `read_file`
- `search_code`
- `git_status`
- `web_fetch`
- `web_search`

The verifier judges whether the worker result satisfies the user goal and task contract. If accepted, the task is marked complete and main Xiaoming notifies the user. If revision is needed, the verifier feedback is routed back to the same worker. After the configured maximum revision attempts, the task moves to `needs_user_decision` so main Xiaoming can explain the current state and ask the user how to proceed.

### Prompt and Context Runtime

Each model request is built from:

- base instructions,
- bootstrap context,
- loaded skills,
- hook-injected context,
- compacted or normal history,
- durable turn context diffs,
- ephemeral turn context,
- the current user message.

Turn context includes workspace path, date, provider, model, stream mode, permission mode, session id, checkpoint id, pending worker questions, and background task summary.

Base instructions include Xiaoming's initial personality layer from `src/xiaoming/prompts/personality.md`:

- `Objective Reality`: factual runtime constraints such as LLM-driven agent architecture, CLI entrypoint, tools, workers, skills, sessions, and context management.
- `Who am I`: intentionally empty by default. Xiaoming should not invent a fixed identity when this section is empty.
- `Core Philosophy`: 儒、法、道、墨、兵 as abstract philosophy, plus conflict resolution order.

The prompt runtime refreshes base instructions on each turn. If the packaged personality prompt, role prompt, or project rules change, the next turn updates the session base instructions; compaction and resume do not remove this layer.

Loaded skills are durable session context. Bootstrap context is injected as developer-role context. Loaded skill instructions are reintroduced on later turns so the session can continue after resume or compaction.

## Installation

Install in editable mode for local development:

```bash
python -m pip install -e '.[dev]'
```

The package exposes:

```bash
xiaoming
xiaoming-cli
xiaoming-eval
```

## Usage

DeepSeek is the default provider:

```bash
export DEEPSEEK_API_KEY=...
xiaoming-cli
```

Run one task:

```bash
xiaoming "查看当前项目结构"
xiaoming "修复当前测试失败" --provider deepseek --model deepseek-v4-flash
```

OpenAI:

```bash
export OPENAI_API_KEY=...
xiaoming "修复当前测试失败" --provider openai --model gpt-5
```

Interactive chat:

```bash
xiaoming-cli
xiaoming chat --provider deepseek --model deepseek-v4-flash
```

Persistent sessions:

```bash
xiaoming-cli
xiaoming-cli --resume <session-id>
xiaoming-cli --new
```

By default, `xiaoming-cli` resumes the latest resumable session for the current workspace. Sessions are stored under `.xiaoming/sessions/` as append-only JSONL event logs.

## Chat Commands

Inside chat, type `/` or `/help` to list commands.

```text
xiaoming> /help
xiaoming> /status
xiaoming> /context
xiaoming> /compact
xiaoming> /tasks
xiaoming> /cancel
xiaoming> /cancel all
xiaoming> /quiet
xiaoming> /verbose
xiaoming> /skills
xiaoming> /logs
xiaoming> /session
xiaoming> /sessions
xiaoming> /checkpoints
xiaoming> /new
xiaoming> /rewind
xiaoming> /rewind <checkpoint-id>
xiaoming> /resume <session-id>
xiaoming> /skill reload
xiaoming> /skill install https://github.com/<owner>/<repo>/tree/<ref>/<path-to-skill>
xiaoming> /model
xiaoming> /model <openai|deepseek> <model>
xiaoming> /provider <openai|deepseek>
xiaoming> /approval <suggest|auto_edit|full_auto>
xiaoming> /permission-mode <default|plan|accept_edits|auto|bypass>
xiaoming> /permissions
xiaoming> /allow Tool(pattern)
xiaoming> /deny Tool(pattern)
xiaoming> /ask Tool(pattern)
xiaoming> /model-timeout <seconds>
xiaoming> /stream on
xiaoming> /stream off
xiaoming> /clear
xiaoming> /exit
```

## Background Tasks

Interactive `xiaoming-cli` schedules workspace-changing work as background tasks. Task state is stored under:

```text
.xiaoming/tasks/
```

Useful commands:

```text
xiaoming> /tasks
xiaoming> /cancel
xiaoming> /cancel all
xiaoming> /quiet
xiaoming> /verbose
```

`/quiet` reduces background progress notices. `/verbose` restores progress notices. Terminal task notices are de-duplicated.

On CLI restart, tasks that were actively running or waiting for user input are marked failed because their worker process no longer exists. Queued tasks that had not started remain waiting and are started again when their conflicts are gone.

## Web Search

`web_search` is available to both the main orchestrator and workers.

Search backend priority is:

1. DeepSeek Anthropic-compatible hosted web search, when `DEEPSEEK_API_KEY` is set and `XIAOMING_DEEPSEEK_WEB_SEARCH` is not disabled.
2. Kimi/Moonshot hosted web search, when `MOONSHOT_API_KEY` or `KIMI_API_KEY` is set.
3. Brave Search, when `BRAVE_SEARCH_API_KEY` is set.
4. Built-in fallback search backends, including China-accessible backends and RSS-style news search.

DeepSeek hosted web search:

```bash
export DEEPSEEK_API_KEY=...
# Optional: disable DeepSeek hosted web search
export XIAOMING_DEEPSEEK_WEB_SEARCH=0
# Optional: override model or timeout
export XIAOMING_DEEPSEEK_WEB_SEARCH_MODEL=deepseek-v4-flash
export DEEPSEEK_WEB_SEARCH_TIMEOUT_SECONDS=60
export DEEPSEEK_WEB_SEARCH_MAX_TOKENS=1200
```

Kimi/Moonshot hosted web search:

```bash
export MOONSHOT_API_KEY=...
# Optional: override the default kimi-k2.6 search model
export XIAOMING_KIMI_WEB_SEARCH_MODEL=kimi-k2.6
# Optional: override the default https://api.moonshot.cn/v1 endpoint
export MOONSHOT_BASE_URL=https://api.moonshot.cn/v1
```

`web_fetch` can fetch public HTTP/HTTPS pages and returns readable text. It rejects private hostnames, local addresses, embedded credentials, large binary responses, and unsupported content.

## Skills

Skills live under:

```text
src/xiaoming/builtin_skills/<name>/SKILL.md
.agents/skills/<name>/SKILL.md
.agents/skills/<bundle>/skills/<name>/SKILL.md
.xiaoming/skills/<name>/SKILL.md
```

The legacy `.xiaoming/skills/<name>/SKILL.md` path is still discovered.

Example:

```markdown
---
name: frontend
description: Build frontend UI.
---

Use semantic HTML and accessible CSS.
```

List discovered skills:

```text
xiaoming> /skills
```

When skills exist, Xiaoming exposes their names, descriptions, paths, and usage rules to the model. The model can call the native `load_skill` tool to load the full `SKILL.md` when a skill is relevant.

Use a skill explicitly by mentioning `$<name>` in the task:

```text
xiaoming> 用 $frontend 写一个简单网页
```

Built-in skills:

- `skill-installer`: installs GitHub-hosted skills into this workspace.
- `find-skills`: helps discover installable skills instead of guessing sources.

### Installing Skills

Install explicitly:

```text
xiaoming> /skill install https://github.com/<owner>/<repo>/tree/<ref>/<path-to-skill>
xiaoming> /skill reload
```

Workers can also install skills through the native `install_skill` tool. It supports:

- GitHub tree URLs.
- `repo` plus `paths`, for example repo `owner/repo` and paths `["skills/example"]`.

The default destination is:

```text
.agents/skills/<skill-name>/
```

The installer refuses to overwrite existing skill directories, rejects unsafe paths and symlinks, enforces file and byte limits, downloads GitHub archives when possible, and falls back to sparse git checkout when needed.

Xiaoming intentionally does not use `npx skills add -g` for installation because project-local skills are preferred.

## Runtime Options

Useful options:

```bash
xiaoming "任务" --provider deepseek --model deepseek-v4-flash --approval-mode suggest --max-turns 999 --model-timeout 180
xiaoming "任务" --provider openai --model gpt-5 --approval-mode suggest --max-turns 999 --model-timeout 180
xiaoming "任务" --no-stream
```

Defaults:

- Provider: `deepseek`
- Model: `deepseek-v4-flash`
- Temperature: `0.2`
- Max output tokens: `64000`
- Approval mode: `suggest`
- Permission mode: derived from approval mode
- Max turns: `999`
- Model timeout: `180` seconds
- Streaming: on

During a slow model call, Xiaoming prints `Still waiting for model response...`; after the timeout it returns an error instead of waiting indefinitely.

Streaming is on by default. Use `--no-stream` or `/stream off` to disable it. DeepSeek streams text deltas while the model is generating. Tool calls are still executed only after their full arguments arrive. OpenAI currently falls back to non-streaming mode.

## Permissions

Approval modes:

- `suggest`: ask before writes and unknown shell commands.
- `auto_edit`: allow workspace edits, still ask for shell operations that are not clearly safe.
- `full_auto`: maps to automatic permission mode for normal CLI use.

Permission modes:

- `default`: ask for writes and unknown shell commands.
- `plan`: blocks file writes.
- `accept_edits`: allows workspace file edits.
- `auto`: allows workspace edits and known safe development commands.
- `bypass`: allows more actions after path and dangerous-command checks.

File tools are restricted to the current workspace. Sensitive paths such as `.env`, `.ssh`, `.git`, `.xiaoming`, and skill directories receive stricter handling.

Shell policy allows common read-only commands, rejects dangerous commands such as `sudo`, `git reset --hard`, and download-piped-to-shell patterns, and asks on unknown or write-capable shell commands. Shell redirection is treated as write-capable and asks for approval before execution.

Project permission rules can be managed in chat:

```text
xiaoming> /allow Bash(pytest*)
xiaoming> /deny Bash(rm*)
xiaoming> /ask WriteFile(.env)
xiaoming> /permissions
```

Rules are stored under the workspace `.xiaoming` configuration.

## Checkpoints

```text
xiaoming> /checkpoints
xiaoming> /rewind
xiaoming> /rewind <checkpoint-id>
```

Xiaoming creates a checkpoint before each user turn. File-writing tools snapshot affected files before modifying them. `/rewind` restores the latest checkpoint; `/rewind <checkpoint-id>` restores a specific checkpoint.

This tracks Xiaoming's file edit tools, not arbitrary file changes made by shell commands.

## Context

Xiaoming tracks estimated context usage and automatically compacts long sessions before they exceed the active model's context window.

```text
xiaoming> /context
xiaoming> /compact
```

`/context` shows estimated tokens, model window, compact threshold, session item count, compaction count, and the last model usage when available. `/compact` manually summarizes older history while keeping recent user messages.

Model-specific context windows are configured in code. For example:

- DeepSeek V4 and V3.2 family: `1000000`
- `deepseek-chat` and `deepseek-reasoner`: `1000000`
- `gpt-5`: `400000`
- `gpt-4.1`: `1047576`
- fallback default: `128000`

The default compaction threshold is 90% of the active model window.

## Sessions

Sessions are stored under:

```text
.xiaoming/sessions/
```

The session store keeps an index plus append-only JSONL event logs. Resume reconstructs the session from events and repairs dangling tool calls from interrupted turns by adding synthetic interrupted tool outputs.

`xiaoming-cli` resumes the latest resumable session for the current workspace unless `--new` is used.

## Logs

Xiaoming writes runtime logs to:

```text
.xiaoming/logs/xiaoming.log
.xiaoming/logs/workers/<task-id>.log
.xiaoming/logs/workers/<task-id>.stderr.log
```

Use `/logs` in chat to print the active main log path.

Logs include session id, turn starts, model calls, usage, tool calls, tool errors, worker lifecycle events, available worker tools and skills, fatal model errors, and CLI exceptions. API keys, authorization headers, tokens, passwords, common `sk-...` secrets, and similar fields are redacted before writing.

## Hooks

Workspace hooks can be configured in `.xiaoming/hooks.json` or `.agents/hooks.json`. Hook commands receive JSON on stdin:

```json
{
  "event": "UserPromptSubmit",
  "payload": {"user_input": "hello"},
  "workspace": "/path/to/workspace"
}
```

They may print a JSON hook result on stdout, for example:

```json
{"updated_input": "hello with extra context"}
```

Supported events are:

- `SessionStart`
- `UserPromptSubmit`
- `PreToolUse`
- `PostToolUse`
- `PermissionRequest`
- `PreCompact`
- `PostCompact`
- `Stop`

Hook results can update input, add context, allow or deny permission requests, ask for approval, stop execution, or suppress output.

## Evaluations

Xiaoming includes a lightweight local eval runner:

```bash
xiaoming-eval evals/cases/local_smoke/slash_help.json
xiaoming-eval evals/cases/local_smoke
xiaoming-eval evals/cases
```

Each case runs in an isolated temporary workspace, feeds configured input turns, appends `exit`, and writes a JSON report under:

```text
evals/reports/
```

Case groups include:

- `local_smoke`
- `config`
- `sessions`
- `skills`
- `permissions`
- `async`
- `web_search`
- `chat`

There is also a Terminal-Bench adapter under:

```text
evals/integrations/terminal_bench/
```

The adapter is external to Xiaoming's normal CLI path. It installs Xiaoming in a Terminal-Bench task container and runs a non-interactive command with `--new --no-stream --approval-mode full_auto --permission-mode bypass`.

## Safety Boundaries

The current implementation is a local CLI agent, not a hardened sandbox. It has policy checks, path restrictions, approval flows, checkpoints, hooks, and logs, but it does not provide OS-level isolation.

For risky work:

- use a clean repository or worktree,
- keep approval mode at `suggest`,
- inspect `/tasks` before approving worker requests,
- keep logs and session files,
- use `/rewind` for Xiaoming file-tool edits when needed.

Self-evolution and cross-agent communication require stricter safety boundaries than local coding tasks. Those systems should keep stable versions, logs, user consent, privacy limits, and rollback paths.

## Tests

Default tests do not call OpenAI:

```bash
pytest -v
```

The optional OpenAI smoke test calls OpenAI:

```bash
OPENAI_API_KEY=... pytest -m openai -v
```

The optional DeepSeek smoke test verifies `deepseek-v4-flash` and tool calling:

```bash
DEEPSEEK_API_KEY=... pytest -m deepseek -v
```
