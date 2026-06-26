from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from xiaoming.llm.types import ToolSpec
from xiaoming.skill_installer import (
    Fetch,
    SkillInstallError,
    SkillInstallResult,
    install_skills_from_github,
    parse_github_tree_url,
)
from xiaoming.skills import Skill, SkillLibrary
from xiaoming.tools.base import ToolResult


@dataclass(frozen=True)
class _InstallTarget:
    name: str
    destination: Path
    source_path: str


class SkillTool:
    name = "skill"
    description = (
        "Manage Xiaoming skills. Use action=load to load installed skill instructions, "
        "action=install to install remote skills into this workspace, and action=list "
        "to list currently available skills. Installing only makes a skill available; "
        "call action=load before following a skill's instructions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["load", "install", "list"]},
            "name": {"type": "string", "description": "Skill name for action=load."},
            "url": {"type": "string", "description": "GitHub tree URL for action=install."},
            "repo": {"type": "string", "description": "GitHub repo in owner/repo form for action=install."},
            "paths": {"type": "array", "items": {"type": "string"}, "description": "Skill directory paths for action=install with repo."},
            "ref": {"type": "string", "description": "Git ref for repo/path installs. Defaults to main."},
            "dest": {"type": "string", "description": "Optional destination root. Relative paths are resolved under the workspace."},
        },
        "required": ["action"],
        "additionalProperties": False,
    }
    supports_parallel_tool_calls = False

    def __init__(
        self,
        workspace: Path,
        library: SkillLibrary,
        approval_mode: str,
        approve: Callable[[str], bool],
        fetch: Fetch | None = None,
        allow_install: bool = True,
        allow_load: bool = True,
    ):
        self.workspace = workspace
        self.library = library
        self.approval_mode = approval_mode
        self.approve = approve
        self.fetch = fetch
        self.allow_install = allow_install
        self.allow_load = allow_load

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        action = str(args.get("action") or "").strip().lower()
        if action == "load":
            return self._load(args)
        if action == "install":
            return self._install(args)
        if action == "list":
            return self._list()
        return ToolResult(self.name, "error", error="action must be one of: load, install, list")

    def _load(self, args: dict[str, Any]) -> ToolResult:
        if not self.allow_load:
            return ToolResult(self.name, "error", error="skill loading is not available in this context")
        skill_name = str(args.get("name") or "").strip()
        if not skill_name:
            return ToolResult(self.name, "error", error="name is required for action=load")
        skill = self.library.load(skill_name)
        if skill is None:
            self.library.refresh(self.workspace)
            skill = self.library.load(skill_name)
        if skill is None:
            return ToolResult(self.name, "error", error=f"unknown skill: {skill_name}")
        return ToolResult(self.name, "success", output=_render_skill(skill))

    def _list(self) -> ToolResult:
        self.library.refresh(self.workspace)
        if not self.library.skills:
            return ToolResult(self.name, "success", output="<skills>\n</skills>")
        return ToolResult(
            self.name,
            "success",
            output="\n".join(["<skills>"] + [_render_skill_summary(skill) for skill in self.library.skills] + ["</skills>"]),
        )

    def _install(self, args: dict[str, Any]) -> ToolResult:
        if not self.allow_install:
            return ToolResult(self.name, "error", error="skill installation is not available in this context")
        try:
            targets = _install_targets(args, self.workspace)
        except SkillInstallError as exc:
            return ToolResult(self.name, "error", error=str(exc))
        if self.approval_mode == "suggest" and not self.approve(_approval_text(args, targets)):
            return ToolResult(self.name, "denied", error="skill install denied by user")
        installed: list[SkillInstallResult] = []
        already_installed: list[_InstallTarget] = []
        missing_paths: list[str] = []
        try:
            for target in targets:
                if target.destination.exists():
                    if (target.destination / "SKILL.md").exists():
                        already_installed.append(target)
                        continue
                    raise SkillInstallError(f"destination already exists but is not a skill directory: {target.destination}")
                missing_paths.append(target.source_path)
            if missing_paths:
                installed = self._install_missing(args, missing_paths)
        except SkillInstallError as exc:
            return ToolResult(self.name, "error", error=str(exc))
        self.library.refresh(self.workspace)
        return ToolResult(self.name, "success", output=_format_install_result(installed, already_installed))

    def _install_missing(self, args: dict[str, Any], missing_paths: list[str]) -> list[SkillInstallResult]:
        url = str(args.get("url") or "").strip()
        if url:
            source = parse_github_tree_url(url)
            return install_skills_from_github(
                repo=f"{source.owner}/{source.repo}",
                paths=[source.path],
                workspace=self.workspace,
                ref=source.ref,
                dest=_dest_root(args, self.workspace),
                fetch=self.fetch,
            )
        repo = str(args.get("repo") or "").strip()
        if not repo:
            raise SkillInstallError("provide either url or repo with paths")
        ref = str(args.get("ref") or "main")
        dest_path = _dest_root(args, self.workspace)
        return install_skills_from_github(
            repo=repo,
            paths=missing_paths,
            workspace=self.workspace,
            ref=ref,
            dest=dest_path,
            fetch=self.fetch,
        )


def _install_targets(args: dict[str, Any], workspace: Path) -> list[_InstallTarget]:
    url = str(args.get("url") or "").strip()
    repo = str(args.get("repo") or "").strip()
    paths = args.get("paths") or []
    dest_root = _dest_root(args, workspace) or workspace / ".agents" / "skills"
    if isinstance(paths, str):
        raise SkillInstallError("paths must be an array of skill directory paths, not a string")
    if url:
        source = parse_github_tree_url(url)
        return [_target_from_source_path(source.path, dest_root)]
    if not repo or not paths:
        raise SkillInstallError("provide either url or repo with paths")
    return [_target_from_source_path(str(path).strip("/"), dest_root) for path in paths]


def _target_from_source_path(source_path: str, dest_root: Path) -> _InstallTarget:
    if not source_path:
        raise SkillInstallError("skill path must not be empty")
    name = Path(source_path).name
    return _InstallTarget(name=name, destination=dest_root / name, source_path=source_path)


def _dest_root(args: dict[str, Any], workspace: Path) -> Path | None:
    dest = args.get("dest")
    if not dest:
        return None
    dest_path = Path(str(dest)).expanduser()
    if not dest_path.is_absolute():
        dest_path = workspace / dest_path
    return dest_path


def _render_skill(skill: Skill) -> str:
    path = str(skill.path) if skill.path is not None else ""
    parts = [
        "<skill>",
        f"<name>{skill.name}</name>",
    ]
    if skill.description:
        parts.append(f"<description>{skill.description}</description>")
    if path:
        parts.append(f"<path>{path}</path>")
    parts.extend(["<instructions>", skill.content, "</instructions>", "</skill>"])
    return "\n".join(parts)


def _render_skill_summary(skill: Skill) -> str:
    path = str(skill.path) if skill.path is not None else ""
    parts = ["<skill>", f"<name>{skill.name}</name>"]
    if skill.description:
        parts.append(f"<description>{skill.description}</description>")
    if path:
        parts.append(f"<path>{path}</path>")
    parts.append("</skill>")
    return "\n".join(parts)


def _format_install_result(installed: list[SkillInstallResult], already_installed: list[_InstallTarget]) -> str:
    status = "installed"
    if installed and already_installed:
        status = "partially_installed"
    elif already_installed and not installed:
        status = "already_installed"
    parts = ["<skill_install_result>", f"<status>{status}</status>", "<skills>"]
    for result in installed:
        parts.extend(
            [
                "<skill>",
                "<status>installed</status>",
                f"<name>{result.name}</name>",
                f"<destination>{result.destination}</destination>",
                f"<files>{result.files}</files>",
                f"<bytes>{result.bytes_written}</bytes>",
                "</skill>",
            ]
        )
    for target in already_installed:
        parts.extend(
            [
                "<skill>",
                "<status>already_installed</status>",
                f"<name>{target.name}</name>",
                f"<destination>{target.destination}</destination>",
                "</skill>",
            ]
        )
    parts.extend(
        [
            "</skills>",
            "<next_action>Call skill with action=load and the skill name before using installed skill instructions.</next_action>",
            "</skill_install_result>",
        ]
    )
    return "\n".join(parts)


def _approval_text(args: dict[str, Any], targets: list[_InstallTarget]) -> str:
    source = args.get("url") or f"{args.get('repo')} {args.get('paths')}"
    destinations = "\n".join(f"- {target.destination}" for target in targets)
    return (
        "Tool: skill\n"
        "Action: install\n"
        f"Source: {source}\n"
        f"Destinations:\n{destinations}\n"
        "This downloads remote skill files into the workspace. Existing valid skill directories are treated as already installed and are not overwritten."
    )
