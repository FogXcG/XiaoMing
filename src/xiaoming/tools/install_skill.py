from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from xiaoming.llm.types import ToolSpec
from xiaoming.skill_installer import (
    Fetch,
    SkillInstallError,
    install_skill_from_url,
    install_skills_from_github,
)
from xiaoming.skills import SkillLibrary
from xiaoming.tools.base import ToolResult


class InstallSkillTool:
    name = "install_skill"
    description = "Install skills from a GitHub tree URL or repo/path into this workspace's .agents/skills and refresh the current skill library."
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "repo": {"type": "string"},
            "paths": {"type": "array", "items": {"type": "string"}},
            "ref": {"type": "string"},
            "dest": {"type": "string"},
        },
        "additionalProperties": False,
    }

    def __init__(
        self,
        workspace: Path,
        library: SkillLibrary,
        approval_mode: str,
        approve: Callable[[str], bool],
        fetch: Fetch | None = None,
    ):
        self.workspace = workspace
        self.library = library
        self.approval_mode = approval_mode
        self.approve = approve
        self.fetch = fetch

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        if self.approval_mode == "suggest" and not self.approve(_approval_text(args)):
            return ToolResult(self.name, "denied", error="skill install denied by user")
        try:
            results = self._install(args)
        except SkillInstallError as exc:
            return ToolResult(self.name, "error", error=str(exc))
        self.library.refresh(self.workspace)
        return ToolResult(
            self.name,
            "success",
            output="\n".join(
                [_format_result(result) for result in results]
                + ["Skills were reloaded and are available immediately."]
            ),
        )

    def _install(self, args: dict[str, Any]):
        url = str(args.get("url") or "").strip()
        repo = str(args.get("repo") or "").strip()
        paths = args.get("paths") or []
        if isinstance(paths, str):
            raise SkillInstallError(
                "paths must be an array of skill directory paths, not a string"
            )
        if url:
            return [install_skill_from_url(url, self.workspace, fetch=self.fetch)]
        if not repo or not paths:
            raise SkillInstallError("provide either url or repo with paths")
        ref = str(args.get("ref") or "main")
        dest = args.get("dest")
        dest_path = Path(str(dest)).expanduser() if dest else None
        if dest_path is not None and not dest_path.is_absolute():
            dest_path = self.workspace / dest_path
        return install_skills_from_github(
            repo=repo,
            paths=[str(path) for path in paths],
            workspace=self.workspace,
            ref=ref,
            dest=dest_path,
            fetch=self.fetch,
        )


def _format_result(result) -> str:
    return (
        f"Installed skill: {result.name}\n"
        f"Destination: {result.destination}\n"
        f"Files: {result.files}\n"
        f"Bytes: {result.bytes_written}"
    )


def _approval_text(args: dict[str, Any]) -> str:
    source = args.get("url") or f"{args.get('repo')} {args.get('paths')}"
    destination = args.get("dest") or ".agents/skills/<skill-name>"
    return (
        "Tool: install_skill\n"
        f"Source: {source}\n"
        f"Destination: {destination}\n"
        "This downloads remote skill files into the workspace. Existing skill directories will not be overwritten."
    )
