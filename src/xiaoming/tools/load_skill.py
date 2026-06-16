from __future__ import annotations

from pathlib import Path
from typing import Any

from xiaoming.llm.types import ToolSpec
from xiaoming.skills import Skill, SkillLibrary
from xiaoming.tools.base import ToolResult


class LoadSkillTool:
    name = "load_skill"
    description = "Load the full instructions for an available skill by name."
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
        },
        "required": ["name"],
        "additionalProperties": False,
    }

    def __init__(self, library: SkillLibrary, workspace: Path | None = None):
        self.library = library
        self.workspace = workspace

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)

    def run(self, args: dict[str, Any]) -> ToolResult:
        name = args["name"]
        skill = self.library.load(name)
        if skill is None and self.workspace is not None:
            self.library.refresh(self.workspace)
            skill = self.library.load(name)
        if skill is None:
            return ToolResult(self.name, "error", error=f"unknown skill: {name}")
        return ToolResult(self.name, "success", output=_render_skill(skill))


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
