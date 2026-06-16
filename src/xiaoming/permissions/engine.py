from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex

from xiaoming.permissions.rules import match_rules
from xiaoming.permissions.types import PermissionBehavior, PermissionDecision, PermissionMode, PermissionRule


READ_COMMANDS = {
    "cat",
    "du",
    "file",
    "find",
    "grep",
    "head",
    "jq",
    "ls",
    "pwd",
    "rg",
    "sort",
    "stat",
    "tail",
    "tree",
    "uniq",
    "wc",
    "which",
}
READ_GIT_SUBCOMMANDS = {"diff", "log", "show", "status"}
NEUTRAL_COMMANDS = {"echo", "printf", "true", "false", ":"}
AUTO_COMMANDS = {
    ("npm", "test"),
    ("npm", "run", "build"),
    ("npm", "run", "lint"),
    ("npm", "run", "test"),
    ("pnpm", "build"),
    ("pnpm", "lint"),
    ("pnpm", "test"),
    ("python", "-m", "pytest"),
    ("pytest",),
}
SHELL_OPERATORS = {"&&", "||", ";", "|"}
REDIRECT_OPERATORS = {">", ">>", "<", "2>", "2>>"}
SENSITIVE_PARTS = {".env", ".ssh"}
SENSITIVE_WRITE_PARTS = {".env", ".git", ".ssh", ".xiaoming", "skills"}


@dataclass
class PermissionEngine:
    workspace: Path
    mode: PermissionMode = PermissionMode.DEFAULT
    rules: list[PermissionRule] = field(default_factory=list)

    def decide_shell(self, command: str) -> PermissionDecision:
        normalized = " ".join(command.strip().split())
        explicit = match_rules(self.rules, "Bash", normalized)
        if explicit is not None:
            return explicit
        try:
            tokens = _shell_tokens(command.strip())
        except ValueError as exc:
            return PermissionDecision(PermissionBehavior.ASK, reason=f"could not parse shell command: {exc}")
        if _has_download_piped_to_shell(tokens):
            return PermissionDecision(PermissionBehavior.DENY, reason="download piped to shell")
        segments = _split_shell_segments_from_tokens(tokens)
        if not segments:
            return PermissionDecision(PermissionBehavior.ASK, reason="empty shell command")
        if self.mode == PermissionMode.BYPASS:
            return PermissionDecision(PermissionBehavior.ALLOW, reason="bypass mode")
        decisions = [self._decide_shell_segment(segment) for segment in segments]
        return _strictest(decisions)

    def _decide_shell_segment(self, segment: list[str]) -> PermissionDecision:
        if not segment:
            return PermissionDecision(PermissionBehavior.ALLOW, reason="empty shell segment")
        if any(token in REDIRECT_OPERATORS for token in segment):
            return PermissionDecision(PermissionBehavior.ASK, reason="shell redirection can write files")
        command = segment[0]
        lowered = [token.lower() for token in segment]
        if _is_dangerous_shell(lowered):
            return PermissionDecision(PermissionBehavior.DENY, reason="dangerous shell command")
        if _mentions_sensitive_path(segment):
            return PermissionDecision(PermissionBehavior.ASK, reason="command mentions sensitive path")
        if _is_read_only_segment(lowered):
            return PermissionDecision(PermissionBehavior.ALLOW, reason="read-only shell command")
        if self.mode == PermissionMode.AUTO and _is_auto_safe_segment(lowered):
            return PermissionDecision(PermissionBehavior.ALLOW, reason="auto-safe development command")
        return PermissionDecision(PermissionBehavior.ASK, reason=f"unknown or write-capable shell command: {command}")

    def decide_file(self, tool: str, requested_path: str) -> PermissionDecision:
        resolved = (self.workspace.resolve() / requested_path).resolve()
        root = self.workspace.resolve()
        if resolved != root and root not in resolved.parents:
            return PermissionDecision(PermissionBehavior.DENY, reason="path is outside workspace")
        rel = "." if resolved == root else str(resolved.relative_to(root))
        explicit = match_rules(self.rules, tool, rel)
        if explicit is not None:
            return explicit
        if self.mode == PermissionMode.BYPASS:
            return PermissionDecision(PermissionBehavior.ALLOW, reason="bypass mode")
        if tool in {"Read", "List", "Search", "GitStatus"}:
            if _is_sensitive_path(resolved):
                return PermissionDecision(PermissionBehavior.ASK, reason="path is sensitive")
            return PermissionDecision(PermissionBehavior.ALLOW, reason="read-only file tool")
        if self.mode == PermissionMode.PLAN:
            return PermissionDecision(PermissionBehavior.DENY, reason="plan mode blocks writes")
        if _is_sensitive_write_path(resolved):
            return PermissionDecision(PermissionBehavior.ASK, reason="path is sensitive")
        if self.mode in {PermissionMode.ACCEPT_EDITS, PermissionMode.AUTO}:
            return PermissionDecision(PermissionBehavior.ALLOW, reason=f"{self.mode.value} allows workspace edits")
        return PermissionDecision(PermissionBehavior.ASK, reason="file write requires approval")


def _split_shell_segments(command: str) -> list[list[str]]:
    return _split_shell_segments_from_tokens(_shell_tokens(command))


def _shell_tokens(command: str) -> list[str]:
    command = "\n".join(line for line in command.splitlines() if not line.lstrip().startswith("#"))
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>")
    lexer.whitespace_split = True
    raw_tokens = list(lexer)
    return _coalesce_operators(raw_tokens)


def _split_shell_segments_from_tokens(tokens: list[str]) -> list[list[str]]:
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in SHELL_OPERATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _coalesce_operators(tokens: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"&", "|", ">", "<"} and index + 1 < len(tokens) and tokens[index + 1] == token:
            result.append(token + token)
            index += 2
            continue
        if token == "2" and index + 1 < len(tokens) and tokens[index + 1] in {">", ">>"}:
            result.append("2" + tokens[index + 1])
            index += 2
            continue
        result.append(token)
        index += 1
    return result


def _has_download_piped_to_shell(tokens: list[str]) -> bool:
    saw_download = False
    after_pipe = False
    for token in tokens:
        lowered = token.lower()
        if lowered in {"curl", "wget"}:
            saw_download = True
            after_pipe = False
            continue
        if lowered == "|":
            after_pipe = saw_download
            continue
        if after_pipe and lowered in {"sh", "bash"}:
            return True
        if lowered in {"&&", "||", ";"}:
            saw_download = False
            after_pipe = False
    return False


def _is_dangerous_shell(tokens: list[str]) -> bool:
    if not tokens:
        return False
    if tokens[0] == "sudo":
        return True
    if tokens[:3] == ["git", "reset", "--hard"]:
        return True
    if tokens[:3] == ["git", "checkout", "--"]:
        return True
    if tokens[0] == "rm" and "-rf" in tokens and any(target in {".", "/", "*"} for target in tokens[1:]):
        return True
    if tokens[0] in {"curl", "wget"} and any(token in {"sh", "bash"} for token in tokens[1:]):
        return True
    return False


def _mentions_sensitive_path(tokens: list[str]) -> bool:
    return any(any(part == sensitive or part.startswith(sensitive + "/") for sensitive in SENSITIVE_PARTS) for part in tokens)


def _is_sensitive_path(path: Path) -> bool:
    return any(part in SENSITIVE_PARTS for part in path.parts)


def _is_sensitive_write_path(path: Path) -> bool:
    return any(part in SENSITIVE_WRITE_PARTS for part in path.parts)


def _is_read_only_segment(tokens: list[str]) -> bool:
    if not tokens:
        return True
    if tokens[0] in READ_COMMANDS:
        return True
    if tokens[0] == "git" and len(tokens) > 1 and tokens[1] in READ_GIT_SUBCOMMANDS:
        return True
    if tokens[0] in NEUTRAL_COMMANDS:
        return True
    return False


def _is_auto_safe_segment(tokens: list[str]) -> bool:
    return any(tuple(tokens[: len(pattern)]) == pattern for pattern in AUTO_COMMANDS)


def _strictest(decisions: list[PermissionDecision]) -> PermissionDecision:
    severity = {
        PermissionBehavior.ALLOW: 0,
        PermissionBehavior.ASK: 1,
        PermissionBehavior.DENY: 2,
    }
    return max(decisions, key=lambda decision: severity[decision.behavior])
