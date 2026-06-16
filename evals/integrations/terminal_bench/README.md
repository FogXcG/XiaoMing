# Terminal-Bench Integration

This directory contains a thin Terminal-Bench adapter for Xiaoming.

The adapter does not add `terminal-bench` as a Xiaoming runtime dependency. Install Terminal-Bench in a separate evaluation environment, then point it at:

```text
evals/integrations/terminal_bench/xiaoming_agent.py:XiaomingTerminalBenchAgent
```

## Agent behavior

For each task, the adapter installs `xiaoming-cli` in the task container and runs:

```bash
xiaoming-cli --new --no-stream --provider deepseek --model deepseek-v4-flash \
  --approval-mode full_auto --permission-mode bypass --max-turns 999 \
  --model-timeout 600 '<task description>'
```

The adapter passes through only the Xiaoming/model environment variables it needs, such as `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, and `KIMI_API_KEY`.

## Installing Xiaoming in task containers

By default `setup-xiaoming.sh` installs from:

```bash
git+https://github.com/FogXcG/XiaoMing.git
```

To evaluate an unmerged branch or local fork, set:

```bash
export XIAOMING_PIP_SPEC='git+https://github.com/FogXcG/XiaoMing.git@feature/xiaoming-hardening'
```

## Notes

- This is an external benchmark adapter, not part of Xiaoming's normal CLI path.
- Terminal-Bench tasks are evaluated in isolated task environments; Xiaoming's async chat mode is not used here.
- For deterministic comparisons, prefer `--no-stream` and record Terminal-Bench's official result files alongside Xiaoming's `.xiaoming/logs` and `.xiaoming/sessions`.

