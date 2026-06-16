from __future__ import annotations

import json
import sys
from pathlib import Path

from xiaoming.evals.runner import EvalCase, load_cases, run_case, write_report


def test_run_case_passes_with_stdout_file_log_and_session_assertions(tmp_path: Path):
    script = tmp_path / "fake_cli.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "import sys",
                "workspace = Path.cwd()",
                "text = sys.stdin.read()",
                "print('ready')",
                "print(text)",
                "(workspace / 'created.txt').write_text('ok')",
                "(workspace / '.xiaoming' / 'logs').mkdir(parents=True)",
                "(workspace / '.xiaoming' / 'logs' / 'xiaoming.log').write_text('tool web_search success')",
                "(workspace / '.xiaoming' / 'sessions').mkdir(parents=True)",
                "(workspace / '.xiaoming' / 'sessions' / 'case.jsonl').write_text('Search backend: deepseek_anthropic')",
            ]
        )
    )
    case = EvalCase.from_dict(
        {
            "id": "smoke",
            "inputs": ["hello"],
            "assertions": {
                "stdout_contains": ["ready", "hello"],
                "files_exist": ["created.txt"],
                "log_contains": ["web_search"],
                "session_contains": ["deepseek_anthropic"],
            },
        }
    )

    result = run_case(case, command=[sys.executable, str(script)], work_root=tmp_path / "work")

    assert result.passed is True
    assert result.exit_code == 0
    assert result.failures == []
    assert result.workspace.exists()


def test_run_case_reports_assertion_failures(tmp_path: Path):
    script = tmp_path / "fake_cli.py"
    script.write_text("print('no match')\n")
    case = EvalCase.from_dict({"id": "fail", "assertions": {"stdout_contains": ["expected"]}})

    result = run_case(case, command=[sys.executable, str(script)], work_root=tmp_path / "work")

    assert result.passed is False
    assert any("stdout missing" in failure for failure in result.failures)


def test_run_case_supports_negative_log_and_session_assertions(tmp_path: Path):
    script = tmp_path / "fake_cli.py"
    script.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "workspace = Path.cwd()",
                "(workspace / '.xiaoming' / 'logs').mkdir(parents=True)",
                "(workspace / '.xiaoming' / 'logs' / 'xiaoming.log').write_text('safe log')",
                "(workspace / '.xiaoming' / 'sessions').mkdir(parents=True)",
                "(workspace / '.xiaoming' / 'sessions' / 'case.jsonl').write_text('safe session')",
            ]
        )
    )
    case = EvalCase.from_dict(
        {
            "id": "negative",
            "assertions": {
                "log_not_contains": ["web_search"],
                "session_not_contains": ["Search backend:"],
            },
        }
    )

    assert case.assertions.log_not_contains == ["web_search"]
    assert case.assertions.session_not_contains == ["Search backend:"]
    result = run_case(case, command=[sys.executable, str(script)], work_root=tmp_path / "work")

    assert result.passed is True


def test_run_case_can_assert_duration_upper_bound(tmp_path: Path):
    script = tmp_path / "fake_cli.py"
    script.write_text("import time\ntime.sleep(0.05)\nprint('slow')\n")
    case = EvalCase.from_dict(
        {
            "id": "slow",
            "assertions": {
                "duration_less_than_seconds": 0.001,
            },
        }
    )

    result = run_case(case, command=[sys.executable, str(script)], work_root=tmp_path / "work")

    assert result.passed is False
    assert any("duration" in failure for failure in result.failures)


def test_write_report_outputs_json(tmp_path: Path):
    script = tmp_path / "fake_cli.py"
    script.write_text("print('ok')\n")
    case = EvalCase.from_dict({"id": "report", "assertions": {"stdout_contains": ["ok"]}})
    result = run_case(case, command=[sys.executable, str(script)], work_root=tmp_path / "work")

    report_path = write_report([result], tmp_path / "reports")

    payload = json.loads(report_path.read_text())
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["failed"] == 0
    assert payload["results"][0]["case_id"] == "report"


def test_load_cases_recurses_directories(tmp_path: Path):
    nested = tmp_path / "cases" / "local_smoke"
    nested.mkdir(parents=True)
    (nested / "one.json").write_text(json.dumps({"id": "one"}))
    (tmp_path / "cases" / "two.json").write_text(json.dumps({"id": "two"}))

    cases = load_cases([tmp_path / "cases"])

    assert [case.id for case in cases] == ["one", "two"]
