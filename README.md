# Xiaoming

A personal AI agent with hierarchical memory, async task coordination, and tool orchestration — built as a local CLI agent, designed to evolve into a JARVIS-like personal assistant.

## Roadmap

- [x] Core agent loop with LLM multi-provider (DeepSeek, OpenAI)
- [x] Tool system (15+ tools) with permissions and safety policy
- [x] Streaming inference with context compaction
- [x] Persistent sessions and file checkpoints
- [x] Skills system (discovery, loading, installation)
- [x] Web search and web fetch
- [x] Hooks extension system
- [x] Async worker coordination with LLM scheduler
- [x] Worker verification and auto-revision loop
- [x] Dream hierarchical memory (fragments, diaries, dream runner)
- [x] Eval harness and terminal-bench adapter
- [ ] Voice interaction — combine with XiaoBai's real-time voice capabilities
- [ ] Self-evolution — candidate worktrees, smoke tests, safe activation
- [ ] Agent-to-Agent communication — natural-language messaging between Xiaoming instances
- [ ] Multi-platform — desktop, phone, wearable, home device
- [ ] JARVIS — voice × async agent = personal AI assistant

## Features

### Agent Runtime
- **Agent Loop** — multi-turn reasoning engine with streaming, tool calling, error recovery, loop detection, and automatic context compaction.
- **LLM Multi-Provider** — unified provider protocol supporting DeepSeek and OpenAI, with streaming/non-streaming, schema conversion, and retry logic.
- **Context Management** — automatic compaction at 90% context threshold, durable/ephemeral turn context, model-specific context window configs.

### Hierarchical Dream Memory
- **Fragments** — raw session events are derived into time-gap-based memory fragments.
- **Packetizer** — fragments are grouped into packets by time gaps for LLM processing.
- **Dream Runner** — an LLM-powered "dream" cycle that reads packets, writes first-person diary drafts (day/week/month/year), self-checks, and atomically accepts or rejects them.
- **Memory View** — hierarchical prompt injection: year → month → week → day diaries layered into the model context.

### Async Task Coordination
- **LLM Scheduler** — decides whether to start, queue, attach, or cancel tasks based on conflict detection across files, modules, and domains.
- **Worker Subprocesses** — isolated worker processes with configurable tool profiles (full, read-only, verify, skill-install).
- **Write Leases** — prevents concurrent writes to the same file across workers.
- **Verification Loop** — completed tasks are verified by an LLM with read-only tools; up to 3 auto-revision attempts before escalating to the user.
- **Worker-User Communication** — workers can ask questions through the coordinator; LLM-based decider routes answers.

### Tool System
- 15+ built-in tools: file operations, shell, git, web search, web fetch, background tasks, skills, talk, and more.
- Unified tool registry with OpenAI-compatible schema generation.
- Parallel execution for read-only tools (up to 4 concurrent calls).
- Loop detection — repeated identical tool results trigger limits.
- Sub-agent tool set trimming (Full, ReadOnly, Verify, SkillInstall).

### Skills
- **SKILL.md spec** — markdown-based skill definition with YAML frontmatter.
- Multi-path discovery under `.agents/skills/`, `.xiaoming/skills/`, and built-in skills.
- Progressive loading and dynamic prompt injection.
- GitHub skill installation with safety checks (path validation, file limits, symlink rejection).
- Built-in `skill-installer` and `find-skills` skills.

### Safety & Extensibility
- **Permission Engine** — path-based and command-based decisions with configurable modes (default, plan, accept_edits, auto, bypass).
- **Shell Policy** — read-only allowlist, dangerous command detection, download-piped-to-shell rejection.
- **Hooks System** — event-driven extension (SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, PermissionRequest, PreCompact, PostCompact, Stop).
- **Checkpoints** — file snapshots before each turn with `/rewind` support.
- **Logging** — structured logs with automatic API key and secret redaction.

### Eval Framework
- Local eval runner with JSON case definitions and assertion-based reporting.
- Isolated temporary workspaces per case.
- Terminal-Bench adapter for external benchmarking.

## Quick Start

### Prerequisites
- Python 3.11+
- A DeepSeek API key (default provider) or OpenAI API key

### Install

```bash
git clone https://github.com/FogXcG/XiaoMing.git
cd Xiaoming
python -m pip install -e '.[dev]'
```

### Run

```bash
export DEEPSEEK_API_KEY=your-key

# Interactive chat (default: resumes latest session)
xiaoming-cli

# One-shot task
xiaoming "查看当前项目结构"

# Or with OpenAI
export OPENAI_API_KEY=your-key
xiaoming "fix failing tests" --provider openai --model gpt-5
```

## Usage

### Chat Commands

```text
/help              List all commands
/status            Show runtime status
/context           Show context usage and token estimates
/compact           Manually compact conversation history
/tasks             List background tasks
/cancel [id|all]   Cancel background tasks
/skills            List discovered skills
/skill install <url> Install a skill from GitHub
/skill reload      Reload skills
/checkpoints       List file checkpoints
/rewind [id]       Restore to a checkpoint
/sessions          List saved sessions
/resume <id>       Resume a specific session
/new               Start a new session
/model <p> <m>     Switch model
/provider <name>   Switch provider (openai|deepseek)
/approval <mode>   Set approval mode (suggest|auto_edit|full_auto)
/permissions       List permission rules
/allow|deny|ask    Manage permission rules
/stream on|off     Toggle streaming
/logs              Show log path
/clear             Clear terminal
/exit              Quit
```

### Background Tasks

Long-running or workspace-changing work is automatically scheduled as background tasks:

```text
xiaoming> /tasks          # View all tasks and their statuses
xiaoming> /cancel task-1  # Cancel a specific task
xiaoming> /quiet          # Reduce progress notices
xiaoming> /verbose        # Restore progress notices
```

### Skills

Skills are reusable instruction bundles in markdown:

```markdown
---
name: frontend
description: Build frontend UI.
---

Use semantic HTML and accessible CSS.
```

```text
xiaoming> /skills              # List available skills
xiaoming> 用 $frontend 写网页   # Use a skill explicitly
```

Install community skills:

```text
xiaoming> /skill install https://github.com/<owner>/<repo>/tree/<ref>/<path>
```

### Web Search

Available to both the main agent and workers. Backend priority:

1. DeepSeek hosted web search (default, when `DEEPSEEK_API_KEY` is set)
2. Kimi/Moonshot (`MOONSHOT_API_KEY` or `KIMI_API_KEY`)
3. Brave Search (`BRAVE_SEARCH_API_KEY`)
4. Built-in fallback (DuckDuckGo, Bing, Google News, Sogou)

## Architecture

```
User → CLI → Main Agent Loop (Orchestrator)
                ├── Foreground: quick, interactive tasks
                └── Background: AsyncCoordinator
                      ├── LLMScheduler (conflict-aware)
                      ├── WorkerProcess (subprocess)
                      │     └── AgentLoop → tools / shell / skills
                      └── Verifier (read-only inspection, auto-revision)
```

- **Main Agent** — restricted tool surface for conversation and coordination.
- **Async Coordinator** — owns task lifecycle: creation, scheduling, execution, verification.
- **Workers** — independent subprocesses with full tool access via JSONL protocol.
- **Verifier** — read-only tools, judges result quality, requests revisions or escalates.

## Configuration

### Key Environment Variables

| Variable | Purpose |
|----------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key (default provider) |
| `OPENAI_API_KEY` | OpenAI API key |
| `MOONSHOT_API_KEY` / `KIMI_API_KEY` | Kimi/Moonshot web search |
| `BRAVE_SEARCH_API_KEY` | Brave Search API key |
| `XIAOMING_DEEPSEEK_WEB_SEARCH` | Set to `0` to disable DeepSeek hosted search |

### Defaults

| Setting | Value |
|---------|-------|
| Provider | `deepseek` |
| Model | `deepseek-v4-flash` |
| Temperature | `0.2` |
| Max output tokens | `64000` |
| Approval mode | `suggest` |
| Streaming | on |
| Model timeout | `180s` |

## Development

```bash
# Run tests (no API calls)
pytest -v

# Run with DeepSeek
DEEPSEEK_API_KEY=... pytest -m deepseek -v

# Run eval cases
xiaoming-eval evals/cases/local_smoke
xiaoming-eval evals/cases
```

## Safety

Xiaoming is a local CLI agent — it does not provide OS-level sandboxing. For risky work, use a clean worktree, keep approval mode at `suggest`, and leverage checkpoints and logs.

## License

MIT
