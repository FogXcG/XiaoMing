from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


BUILTIN_SKILLS_ROOT = Path(__file__).parent / "builtin_skills"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    path: Path | None = None


class SkillLibrary:
    def __init__(self, skills: list[Skill] | None = None):
        self.skills = sorted(skills or [], key=lambda skill: skill.name)
        self._by_name = _build_skill_index(self.skills)

    @classmethod
    def discover(cls, workspace: Path) -> "SkillLibrary":
        return cls(_discover_skills(workspace))

    def refresh(self, workspace: Path) -> None:
        self.skills = sorted(_discover_skills(workspace), key=lambda skill: skill.name)
        self._by_name = _build_skill_index(self.skills)

    def load(self, name: str) -> Skill | None:
        return self._by_name.get(name)

    def select_for_task(self, task: str) -> list[Skill]:
        selected: list[Skill] = []
        for match in re.finditer(r"(?<!\w)\$([A-Za-z0-9_.-]+)", task):
            skill = self._by_name.get(match.group(1))
            if skill is not None and skill not in selected:
                selected.append(skill)
        for alias, skill in self._by_name.items():
            if skill in selected:
                continue
            if _contains_plain_skill_name(task, alias):
                selected.append(skill)
        return selected

    def render_for_task(self, task: str) -> str:
        selected = self.select_for_task(task)
        if not selected:
            return ""
        blocks = ["Active skills:"]
        for skill in selected:
            blocks.append(f"Skill: {skill.name}")
            if skill.description:
                blocks.append(f"Description: {skill.description}")
            blocks.append(skill.content)
        return "\n\n".join(blocks)

    def explicit_skills_for_text(self, text: str) -> list[Skill]:
        return self.select_for_task(text)

    def render_available(self) -> str:
        if not self.skills:
            return ""
        lines = [
            "Available skills:",
            "How to use skills:",
            "- Discovery: the list below is available in this session (name, description, and file path).",
            "- Trigger rules: if the user names a skill with $name or plain text, or the task clearly matches a skill description, use that skill for this turn.",
            "- Before inspecting files, writing files, running commands, or otherwise acting on a matching task, call load_skill with the skill name to load full instructions.",
            "- Multiple matching skills: use the minimal set that covers the request, with process skills before implementation skills.",
            "- If you skip an obvious skill, briefly explain why before continuing.",
            "- Loaded skill instructions remain available for the session and should be followed when relevant.",
            "- Progressive disclosure: after loading SKILL.md, load referenced files only when needed; prefer scripts/assets from the skill when they exist.",
        ]
        for skill in self.skills:
            description = f" - {skill.description}" if skill.description else ""
            path = f" (file: {_display_path(skill.path)})" if skill.path is not None else ""
            lines.append(f"- {skill.name}{description}{path}")
        return "\n".join(lines)

    def list_text(self) -> str:
        if not self.skills:
            return "No skills found. Add skills under .agents/skills/<name>/SKILL.md."
        lines = []
        for skill in self.skills:
            if skill.description:
                lines.append(f"{skill.name} - {skill.description}")
            else:
                lines.append(skill.name)
        return "\n".join(lines)


def _discover_skills(workspace: Path) -> list[Skill]:
    skills: list[Skill] = []
    seen_names: set[str] = set()
    for root in [BUILTIN_SKILLS_ROOT, workspace / ".agents" / "skills", workspace / ".xiaoming" / "skills"]:
        if not root.exists():
            continue
        for skill_file in _skill_files(root):
            skill = _read_skill(skill_file)
            if skill.name in seen_names:
                continue
            skills.append(skill)
            seen_names.add(skill.name)
    return skills


def _skill_files(root: Path) -> list[Path]:
    files = set(root.glob("*/SKILL.md"))
    files.update(root.glob("*/skills/*/SKILL.md"))
    return sorted(files)


def _build_skill_index(skills: list[Skill]) -> dict[str, Skill]:
    index = {skill.name: skill for skill in skills}
    for skill in skills:
        if skill.path is not None:
            index.setdefault(skill.path.parent.name, skill)
        if ":" in skill.name:
            index.setdefault(skill.name.rsplit(":", 1)[-1], skill)
    return index


def _contains_plain_skill_name(task: str, name: str) -> bool:
    if not name:
        return False
    escaped = re.escape(name)
    return re.search(rf"(?<![A-Za-z0-9_.:-]){escaped}(?![A-Za-z0-9_.:-])", task, flags=re.IGNORECASE) is not None


def _read_skill(path: Path) -> Skill:
    text = path.read_text()
    metadata, content = _split_frontmatter(text)
    name = metadata.get("name") or path.parent.name
    description = metadata.get("description") or ""
    return Skill(name=name, description=description, content=content.strip(), path=path)


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    parts = path.parts
    for marker in (".agents", ".xiaoming"):
        if marker in parts:
            return str(Path(*parts[parts.index(marker) :]))
    return str(path)


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    metadata: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return metadata, "\n".join(lines[index + 1 :])
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip().strip('"').strip("'")
    return {}, text
