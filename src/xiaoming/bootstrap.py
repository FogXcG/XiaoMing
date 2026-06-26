from __future__ import annotations

from pathlib import Path

from xiaoming.session import BootstrapContext


def discover_bootstrap_contexts(workspace: Path) -> list[BootstrapContext]:
    contexts: list[BootstrapContext] = []
    for root in [workspace / ".agents" / "skills", workspace / ".xiaoming" / "skills"]:
        if not root.exists():
            continue
        for plugin_root in sorted(path for path in root.iterdir() if path.is_dir()):
            context = _superpowers_context(plugin_root)
            if context is not None:
                contexts.append(context)
    return contexts


def _superpowers_context(plugin_root: Path) -> BootstrapContext | None:
    skill_path = plugin_root / "skills" / "using-superpowers" / "SKILL.md"
    if not skill_path.exists():
        return None
    content = skill_path.read_text()
    wrapped = (
        "<EXTREMELY_IMPORTANT>\n"
        "You have superpowers.\n\n"
        "**Below is the full content of your 'superpowers:using-superpowers' skill - your introduction to using skills. "
        "For all other skills, use the 'skill' tool with action='load':**\n\n"
        f"{content}\n"
        "</EXTREMELY_IMPORTANT>"
    )
    return BootstrapContext.create(
        plugin_name=plugin_root.name,
        source=f"{plugin_root.name}:using-superpowers",
        content=wrapped,
        path=str(skill_path),
    )
