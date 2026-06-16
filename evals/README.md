# Xiaoming Evals

Lightweight regression cases for Xiaoming's CLI behavior.

Run one case:

```bash
xiaoming-eval evals/cases/local_smoke/slash_help.json
```

Run a directory of cases:

```bash
xiaoming-eval evals/cases/local_smoke
```

Run all bundled cases recursively:

```bash
xiaoming-eval evals/cases
```

By default the runner starts Xiaoming with:

```bash
python -m xiaoming.cli --new
```

Use `--command` to test another executable:

```bash
xiaoming-eval evals/cases/local_smoke --command xiaoming-cli --new
```

Each case runs in an isolated temporary workspace, feeds the configured input turns, appends `exit`, and writes a JSON report under `evals/reports/`.

Case groups:

- `local_smoke`: stable CLI commands that do not call the model.
- `config`: runtime configuration and invalid command handling.
- `sessions`: session lifecycle commands.
- `skills`: skill list and reload commands.
- `permissions`: project permission rule persistence.
- `async`: background task controls and scheduling behavior.
- `web_search`: web search backend behavior.
- `chat`: simple model-facing conversation checks.
