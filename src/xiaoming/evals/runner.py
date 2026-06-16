from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalAssertions:
    stdout_contains: list[str] = field(default_factory=list)
    stdout_not_contains: list[str] = field(default_factory=list)
    stderr_contains: list[str] = field(default_factory=list)
    stderr_not_contains: list[str] = field(default_factory=list)
    files_exist: list[str] = field(default_factory=list)
    files_not_exist: list[str] = field(default_factory=list)
    log_contains: list[str] = field(default_factory=list)
    log_not_contains: list[str] = field(default_factory=list)
    session_contains: list[str] = field(default_factory=list)
    session_not_contains: list[str] = field(default_factory=list)
    duration_less_than_seconds: float | None = None
    exit_code: int | None = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "EvalAssertions":
        data = data or {}
        return cls(
            stdout_contains=_string_list(data.get("stdout_contains")),
            stdout_not_contains=_string_list(data.get("stdout_not_contains")),
            stderr_contains=_string_list(data.get("stderr_contains")),
            stderr_not_contains=_string_list(data.get("stderr_not_contains")),
            files_exist=_string_list(data.get("files_exist")),
            files_not_exist=_string_list(data.get("files_not_exist")),
            log_contains=_string_list(data.get("log_contains")),
            log_not_contains=_string_list(data.get("log_not_contains")),
            session_contains=_string_list(data.get("session_contains")),
            session_not_contains=_string_list(data.get("session_not_contains")),
            duration_less_than_seconds=_optional_float(data.get("duration_less_than_seconds")),
            exit_code=_optional_int(data.get("exit_code"), default=0),
        )


@dataclass(frozen=True)
class EvalCase:
    id: str
    inputs: list[str] = field(default_factory=list)
    fixture: str | None = None
    timeout_seconds: float = 120
    env: dict[str, str] = field(default_factory=dict)
    assertions: EvalAssertions = field(default_factory=EvalAssertions)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        case_id = str(data.get("id") or "").strip()
        if not case_id:
            raise ValueError("eval case requires a non-empty id")
        return cls(
            id=case_id,
            inputs=_string_list(data.get("inputs")),
            fixture=str(data["fixture"]) if data.get("fixture") else None,
            timeout_seconds=float(data.get("timeout_seconds") or 120),
            env={str(key): str(value) for key, value in dict(data.get("env") or {}).items()},
            assertions=EvalAssertions.from_dict(data.get("assertions")),
        )


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    passed: bool
    exit_code: int | None
    duration_seconds: float
    workspace: Path
    stdout: str
    stderr: str
    failures: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 3),
            "workspace": str(self.workspace),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "failures": self.failures,
        }


def run_case(case: EvalCase, *, command: list[str] | None = None, work_root: Path | None = None, fixtures_root: Path | None = None) -> EvalResult:
    command = command or _default_command()
    work_root = work_root or Path(tempfile.mkdtemp(prefix="xiaoming-evals-"))
    workspace = _prepare_workspace(case, work_root, fixtures_root)
    started = time.monotonic()
    failures: list[str] = []
    env = os.environ.copy()
    env.update(case.env)
    stdin_text = "\n".join([*case.inputs, "exit", ""])
    try:
        completed = subprocess.run(command, input=stdin_text, text=True, cwd=workspace, env=env, capture_output=True, timeout=case.timeout_seconds, check=False)
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        exit_code = None
        stdout = _decode_timeout_output(exc.stdout)
        stderr = _decode_timeout_output(exc.stderr)
        failures.append(f"command timed out after {case.timeout_seconds}s")
    duration = time.monotonic() - started
    failures.extend(_check_assertions(case.assertions, workspace, stdout, stderr, exit_code, duration))
    return EvalResult(
        case_id=case.id,
        passed=not failures,
        exit_code=exit_code,
        duration_seconds=duration,
        workspace=workspace,
        stdout=stdout,
        stderr=stderr,
        failures=failures,
    )


def load_cases(paths: list[Path]) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for path in paths:
        if path.is_dir():
            for child in sorted(path.rglob("*.json")):
                cases.append(_load_case_file(child))
        else:
            cases.append(_load_case_file(path))
    return cases


def write_report(results: list[EvalResult], reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = reports_dir / f"report-{stamp}.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": len(results),
            "passed": sum(1 for result in results if result.passed),
            "failed": sum(1 for result in results if not result.passed),
        },
        "results": [result.to_dict() for result in results],
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xiaoming-eval")
    parser.add_argument("cases", nargs="+", type=Path, help="JSON case files or directories containing JSON cases")
    parser.add_argument("--command", nargs="+", help="Command used to start Xiaoming. Defaults to current Python module entrypoint.")
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--fixtures-root", type=Path, default=Path("evals/fixtures"))
    parser.add_argument("--reports-dir", type=Path, default=Path("evals/reports"))
    args = parser.parse_args(argv)

    cases = load_cases(args.cases)
    results = [run_case(case, command=args.command, work_root=args.work_root, fixtures_root=args.fixtures_root) for case in cases]
    report_path = write_report(results, args.reports_dir)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status} {result.case_id} ({result.duration_seconds:.1f}s) workspace={result.workspace}")
        for failure in result.failures:
            print(f"  - {failure}")
    print(f"Report: {report_path}")
    return 0 if all(result.passed for result in results) else 1


def _load_case_file(path: Path) -> EvalCase:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return EvalCase.from_dict(data)


def _default_command() -> list[str]:
    return [sys.executable, "-m", "xiaoming.cli", "--new"]


def _prepare_workspace(case: EvalCase, work_root: Path, fixtures_root: Path | None) -> Path:
    work_root.mkdir(parents=True, exist_ok=True)
    workspace = work_root / f"{case.id}-{uuid.uuid4().hex[:8]}"
    if case.fixture:
        if fixtures_root is None:
            raise ValueError("fixtures_root is required when case.fixture is set")
        fixture_path = fixtures_root / case.fixture
        if not fixture_path.exists():
            raise FileNotFoundError(f"fixture not found: {fixture_path}")
        shutil.copytree(fixture_path, workspace)
    else:
        workspace.mkdir(parents=True)
    return workspace


def _check_assertions(assertions: EvalAssertions, workspace: Path, stdout: str, stderr: str, exit_code: int | None, duration_seconds: float) -> list[str]:
    failures: list[str] = []
    if assertions.exit_code is not None and exit_code != assertions.exit_code:
        failures.append(f"exit code {exit_code!r} != expected {assertions.exit_code!r}")
    if assertions.duration_less_than_seconds is not None and duration_seconds >= assertions.duration_less_than_seconds:
        failures.append(f"duration {duration_seconds:.3f}s >= expected {assertions.duration_less_than_seconds:.3f}s")
    failures.extend(_contains_failures("stdout", stdout, assertions.stdout_contains, should_contain=True))
    failures.extend(_contains_failures("stdout", stdout, assertions.stdout_not_contains, should_contain=False))
    failures.extend(_contains_failures("stderr", stderr, assertions.stderr_contains, should_contain=True))
    failures.extend(_contains_failures("stderr", stderr, assertions.stderr_not_contains, should_contain=False))
    failures.extend(_path_failures(workspace, assertions.files_exist, should_exist=True))
    failures.extend(_path_failures(workspace, assertions.files_not_exist, should_exist=False))
    log_text = _read_tree_text(workspace / ".xiaoming" / "logs")
    session_text = _read_tree_text(workspace / ".xiaoming" / "sessions")
    failures.extend(_contains_failures("log", log_text, assertions.log_contains, should_contain=True))
    failures.extend(_contains_failures("log", log_text, assertions.log_not_contains, should_contain=False))
    failures.extend(_contains_failures("session", session_text, assertions.session_contains, should_contain=True))
    failures.extend(_contains_failures("session", session_text, assertions.session_not_contains, should_contain=False))
    return failures


def _contains_failures(label: str, text: str, needles: list[str], *, should_contain: bool) -> list[str]:
    failures: list[str] = []
    for needle in needles:
        found = needle in text
        if should_contain and not found:
            failures.append(f"{label} missing {needle!r}")
        if not should_contain and found:
            failures.append(f"{label} unexpectedly contains {needle!r}")
    return failures


def _path_failures(workspace: Path, paths: list[str], *, should_exist: bool) -> list[str]:
    failures: list[str] = []
    for path in paths:
        exists = (workspace / path).exists()
        if should_exist and not exists:
            failures.append(f"missing file {path!r}")
        if not should_exist and exists:
            failures.append(f"unexpected file {path!r}")
    return failures


def _read_tree_text(path: Path) -> str:
    if not path.exists():
        return ""
    chunks: list[str] = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        try:
            chunks.append(child.read_text(errors="replace"))
        except OSError:
            continue
    return "\n".join(chunks)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _optional_int(value: Any, *, default: int | None) -> int | None:
    if value is None:
        return default
    if value == "any":
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _decode_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
