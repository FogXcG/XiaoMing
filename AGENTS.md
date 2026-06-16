# AGENTS.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Regression Testing

**Use fast local tests first, then real CLI evals for agent behavior.**

For focused code changes, run the smallest relevant pytest slice. Recent Terminal-Bench
integration fixes are covered by:
```
pytest -q tests/test_permissions.py tests/test_terminal_bench_integration.py tests/test_prompting.py
```

For real CLI regression checks, use Terminal-Bench through the Xiaoming installed-agent
adapter. The reliable local workflow is:
```
rm -rf /tmp/xiaoming-tbench/wheels
mkdir -p /tmp/xiaoming-tbench/wheels
python3 -m pip wheel . -w /tmp/xiaoming-tbench/wheels
python3 -m http.server 9876 --bind 0.0.0.0 --directory /tmp/xiaoming-tbench/wheels
```

Then run individual tasks with a lowercase run id:
```
XIAOMING_PIP_SPEC='http://172.27.0.1:9876/xiaoming_agent-0.1.0-py3-none-any.whl' \
XIAOMING_PIP_FIND_LINKS='http://172.27.0.1:9876' \
XIAOMING_PIP_TRUSTED_HOST='172.27.0.1' \
PYTHONPATH=/home/cicada/data/xiaoming \
uv tool run --from terminal-bench --with 'requests[socks]' --with 'httpx[socks]' \
  tb runs create \
  --dataset-path /tmp/xiaoming-tbench/terminal-bench-core-0.1.1 \
  --task-id fix-permissions \
  --agent-import-path evals.integrations.terminal_bench.xiaoming_agent:XiaomingTerminalBenchAgent \
  --output-path /tmp/xiaoming-tbench/runs \
  --run-id xiaoming-local-YYYYMMDD-hhmmss \
  --n-concurrent 1 \
  --global-agent-timeout-sec 900 \
  --no-upload-results \
  --log-level info
```

Known passing smoke cases:
- `fix-permissions`
- `csv-to-parquet`
- `heterogeneous-dates`

Results are written under `/tmp/xiaoming-tbench/runs/<run-id>/results.json`.
Stop the temporary wheel HTTP server after the run. Avoid `XIAOMING_PIP_NO_INDEX`
unless the local wheelhouse contains platform-compatible wheels for the task
container's Python version.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
