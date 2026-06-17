from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.llm.types import ToolSpec
from xiaoming.skill_installer import (
    SkillInstallError,
    install_skill_from_url,
)
from xiaoming.skills import SkillLibrary
from xiaoming.tools.base import ToolResult


class FetchSkillTool:
    name = "fetch_skill"
    description = (
        "Download a skill from a GitHub URL into .agents/skills/ and return "
        "the loaded skill instructions. Use this when find-skills discovered "
        "a relevant skill that is not yet installed locally. "
        "If the skill is already in .agents/skills/, use load_skill instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "GitHub tree URL to the skill directory, e.g. https://github.com/owner/repo/tree/main/skills/my-skill"},
        },
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(self, workspace: Path, library: SkillLibrary):
        self.workspace = workspace
        self.library = library

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        url = str(args["url"]).strip()
        try:
            result = install_skill_from_url(
                url,
                self.workspace,
                max_files=100,
                max_total_bytes=2_000_000,
            )
        except SkillInstallError as exc:
            if "destination already exists" in str(exc):
                return ToolResult(
                    self.name,
                    "success",
                    output=(
                        f"Skill is already installed at {str(exc).split(': ')[-1] if ': ' in str(exc) else str(exc)}. "
                        "Use load_skill with the skill name to load it into this session."
                    ),
                )
            return ToolResult(self.name, "error", error=str(exc))

        self.library.refresh(self.workspace)
        return ToolResult(
            self.name,
            "success",
            output=_format_fetch_result(result),
        )


def _format_fetch_result(result) -> str:
    return (
        f"<fetch_skill_result>\n"
        f"<name>{result.name}</name>\n"
        f"<destination>{result.destination}</destination>\n"
        f"<files>{result.files}</files>\n"
        f"<bytes>{result.bytes_written}</bytes>\n"
        f"</fetch_skill_result>"
    )
