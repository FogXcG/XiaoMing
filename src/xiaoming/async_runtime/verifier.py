from __future__ import annotations

from fnmatch import fnmatch
import json
from pathlib import Path
import queue
import re
import threading
from typing import Protocol

from xiaoming.async_runtime.tasks import TaskResultReport, TaskSpec, VerificationResult
from xiaoming.llm.provider import LLMProvider
from xiaoming.llm.types import LLMRequest
from xiaoming.tools.list_files import ListFilesTool
from xiaoming.tools.read_file import ReadFileTool
from xiaoming.tools.registry import ToolRegistry


class Verifier(Protocol):
    def verify(self, spec: TaskSpec, report: TaskResultReport) -> VerificationResult:
        ...


class TaskVerifier:
    def __init__(self, workspace: Path):
        self.workspace = workspace

    def verify(self, spec: TaskSpec, report: TaskResultReport) -> VerificationResult:
        reasons: list[str] = []
        if report.status != "completed":
            reasons.append(f"report status is {report.status}")
        if not report.summary.strip():
            reasons.append("report summary is empty")
        reported_artifacts = [_normalize_path(self.workspace, artifact) for artifact in report.artifacts]
        for artifact in spec.expected_artifacts:
            artifact_paths = _artifact_paths(self.workspace, artifact, reported_artifacts)
            if not artifact_paths:
                continue
            if not any(_artifact_reported(path, reported_artifacts) for path in artifact_paths):
                reasons.append(f"expected artifact not reported: {artifact}")
            if not any((self.workspace / path).exists() for path in artifact_paths):
                reasons.append(f"expected artifact missing: {artifact}")
        changed_paths = report.changed_files + report.created_files
        if spec.allowed_write_paths:
            for path in changed_paths:
                if not _allowed(self.workspace, path, spec.allowed_write_paths):
                    reasons.append(f"changed path outside allowed scope: {path}")
        for command in spec.verification_commands:
            matches = [record for record in report.verification if record.command == command]
            if not matches:
                continue
            if not any(record.status == "passed" for record in matches):
                reasons.append(f"verification command did not pass: {command}")
        for record in report.verification:
            if record.status == "failed":
                reasons.append(f"verification failed: {record.command} {record.reason}".strip())
        return VerificationResult(accepted=not reasons, reasons=reasons)


class LLMTaskVerifier:
    def __init__(self, workspace: Path, provider: LLMProvider, model: str, timeout_seconds: float = 30, max_attempts: int = 3, max_tool_turns: int = 6):
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.max_tool_turns = max_tool_turns
        self.registry = ToolRegistry([ListFilesTool(workspace), ReadFileTool(workspace)])

    def verify(self, spec: TaskSpec, report: TaskResultReport) -> VerificationResult:
        payload = {
            "workspace": str(self.workspace),
            "task_spec": spec.to_dict(),
            "worker_report": report.to_dict(),
        }
        errors: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._verify_once(payload)
            except BaseException as exc:
                errors.append(f"attempt {attempt}: {exc}")
        return VerificationResult(accepted=False, reasons=["LLM verifier failed after 3 attempts: " + "; ".join(errors)])

    def _verify_once(self, payload: dict) -> VerificationResult:
        result_queue: queue.Queue[VerificationResult | BaseException] = queue.Queue(maxsize=1)

        def run() -> None:
            try:
                result_queue.put(self._verify_with_tools(payload))
            except BaseException as exc:
                result_queue.put(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            result = result_queue.get(timeout=self.timeout_seconds)
        except queue.Empty:
            raise TimeoutError(f"verifier model timed out after {self.timeout_seconds:g} seconds")
        if isinstance(result, BaseException):
            raise result
        return result

    def _verify_with_tools(self, payload: dict) -> VerificationResult:
        input_items = [{"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)}]
        tools = self.registry.specs()
        for _ in range(self.max_tool_turns + 1):
            response = self.provider.complete(
                LLMRequest(
                    model=self.model,
                    instructions=_LLM_VERIFIER_INSTRUCTIONS,
                    input_items=input_items,
                    tools=tools,
                    temperature=0,
                    max_output_tokens=1600,
                )
            )
            input_items.extend(response.output_items)
            if response.fatal_error is not None:
                raise ValueError(response.fatal_error.message)
            if response.recoverable_errors:
                raise ValueError("; ".join(error.message for error in response.recoverable_errors))
            if not response.tool_calls:
                text = (response.message or "").strip()
                if not text:
                    raise ValueError("empty verifier response")
                return _verification_from_json(text)
            for call in response.tool_calls:
                result = self.registry.run(call.name, call.args)
                input_items.append({"type": call.output_type, "call_id": call.id, "output": self.registry.format_result(result)})
        raise ValueError(f"verifier exceeded max_tool_turns={self.max_tool_turns}")


_LLM_VERIFIER_INSTRUCTIONS = """In this runtime, act as the background task verifier.
Return only one JSON object. Do not use markdown.
Judge whether the worker report satisfies the task contract using semantic reasoning.
You may inspect the workspace using the available read-only tools before deciding.
Use list_files to check whether expected files/directories exist. Use read_file for small relevant files when content matters.
Do not assume reported paths exist; inspect them when existence or content is important.
Be practical: accept completed work when the report and evidence show the user's requested outcome was achieved, even if paths are relative vs absolute or artifact descriptions are phrased differently.
Reject when the report says failed/blocked, the summary is empty, required deliverables are clearly missing, reported evidence contradicts completion, or the task materially failed its success criteria.
Required JSON shape:
{"accepted": boolean, "reasons": ["short reason", "..."]}
If accepted is true, reasons may be empty or contain brief positive evidence. If accepted is false, reasons must explain the blocker clearly.
"""


def _verification_from_json(text: str) -> VerificationResult:
    parsed = json.loads(_extract_json_object(text))
    accepted = parsed.get("accepted")
    if not isinstance(accepted, bool):
        raise ValueError("verifier response missing boolean accepted")
    reasons = parsed.get("reasons")
    if reasons is None:
        reasons = []
    if not isinstance(reasons, list):
        raise ValueError("verifier response reasons must be an array")
    return VerificationResult(accepted=accepted, reasons=[str(reason) for reason in reasons if str(reason).strip()])


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("verifier response did not contain a JSON object")
    return stripped[start : end + 1]


def _allowed(workspace: Path, path: str, patterns: list[str]) -> bool:
    normalized = _normalize_path(workspace, path)
    for pattern in patterns:
        pattern = _normalize_path(workspace, pattern)
        if fnmatch(normalized, pattern):
            return True
        if not any(char in pattern for char in "*?[]") and normalized.startswith(pattern.rstrip("/") + "/"):
            return True
        if pattern.endswith("/**") and normalized.startswith(pattern[:-3].rstrip("/") + "/"):
            return True
    return False


_PATH_RE = re.compile(r"(?:\.{1,2}/|/)?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*(?:/)?")


def _artifact_paths(workspace: Path, description: str, reported_artifacts: list[str]) -> list[str]:
    paths: list[str] = []
    for match in _PATH_RE.finditer(description):
        path = _normalize_path(workspace, match.group(0))
        if _looks_like_path(path) and path not in paths:
            paths.append(path)
    for artifact in reported_artifacts:
        if artifact and (artifact in description or Path(artifact).name in description) and artifact not in paths:
            paths.append(artifact)
    if not paths:
        candidate = _normalize_path(workspace, description)
        if _looks_like_path(candidate):
            paths.append(candidate)
    return paths


def _artifact_reported(expected_path: str, reported_artifacts: list[str]) -> bool:
    for artifact in reported_artifacts:
        reported_path = artifact
        if reported_path == expected_path:
            return True
        if reported_path.startswith(expected_path.rstrip("/") + "/"):
            return True
        if expected_path in artifact:
            return True
    return False


def _normalize_path(workspace: Path, path: str) -> str:
    text = path.strip().strip("`\"'")
    if not text:
        return ""
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return str(candidate.resolve().relative_to(workspace.resolve()))
        except ValueError:
            return str(candidate).strip("/")
    if text.startswith("./"):
        return text[2:].strip("/")
    return text.strip("/")


def _looks_like_path(path: str) -> bool:
    if not path:
        return False
    if "/" in path or "." in Path(path).name:
        return True
    return bool(re.fullmatch(r"[A-Za-z0-9._-]+", path))
