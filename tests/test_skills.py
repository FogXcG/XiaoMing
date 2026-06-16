from pathlib import Path

from xiaoming.skills import SkillLibrary


def test_skill_library_discovers_workspace_skills(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: frontend
description: Build frontend UI.
---

Use accessible HTML and CSS.
"""
    )

    library = SkillLibrary.discover(tmp_path)

    skill = library.load("frontend")
    assert skill is not None
    assert skill.description == "Build frontend UI."
    assert "Use accessible HTML" in skill.content
    assert skill.path == skill_dir / "SKILL.md"


def test_skill_library_discovers_plugin_nested_skills(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "superpowers" / "skills" / "using-superpowers"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: using-superpowers
description: Use superpowers skills.
---

Load relevant skills before work.
"""
    )

    library = SkillLibrary.discover(tmp_path)

    skill = library.load("using-superpowers")
    assert skill is not None
    assert skill.description == "Use superpowers skills."
    assert "Load relevant skills" in skill.content
    assert skill.path == skill_dir / "SKILL.md"


def test_skill_library_supports_legacy_xiaoming_skill_directory(tmp_path: Path):
    skill_dir = tmp_path / ".xiaoming" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Frontend\nUse semantic markup.\n")

    library = SkillLibrary.discover(tmp_path)

    assert library.load("frontend") is not None


def test_skill_library_selects_explicit_dollar_mentions(tmp_path: Path):
    skill_dir = tmp_path / ".xiaoming" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Frontend\nUse semantic markup.\n")

    library = SkillLibrary.discover(tmp_path)

    selected = library.select_for_task("用 $frontend 写一个页面")

    assert [skill.name for skill in selected] == ["frontend"]


def test_skill_library_selects_plain_skill_name_mentions(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "brainstorming"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: brainstorming\ndescription: Use before creative work.\n---\nBody\n")

    library = SkillLibrary.discover(tmp_path)

    selected = library.select_for_task("用 brainstorming 帮我设计一个页面")

    assert [skill.name for skill in selected] == ["brainstorming"]


def test_skill_library_selects_namespaced_plain_skill_mentions(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "writing-plans"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: superpowers:writing-plans\ndescription: Write plans.\n---\nBody\n")

    library = SkillLibrary.discover(tmp_path)

    selected = library.select_for_task("请使用 superpowers:writing-plans 写计划")

    assert [skill.name for skill in selected] == ["superpowers:writing-plans"]


def test_skill_library_does_not_semantically_guess_plain_requests(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "brainstorming"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: brainstorming\ndescription: Use before creative work.\n---\nBody\n")

    library = SkillLibrary.discover(tmp_path)

    assert library.select_for_task("帮我开发一个简单网页") == []


def test_skill_library_renders_selected_skills_for_prompt(tmp_path: Path):
    skill_dir = tmp_path / ".xiaoming" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Frontend\nUse semantic markup.\n")

    library = SkillLibrary.discover(tmp_path)

    rendered = library.render_for_task("用 $frontend 写一个页面")

    assert "Active skills:" in rendered
    assert "Skill: frontend" in rendered
    assert "Use semantic markup." in rendered


def test_skill_library_lists_skills(tmp_path: Path):
    skill_dir = tmp_path / ".xiaoming" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: frontend\ndescription: Build UI.\n---\nBody\n")

    library = SkillLibrary.discover(tmp_path)

    listing = library.list_text()
    assert "frontend - Build UI." in listing
    assert "skill-installer - Install skills" in listing


def test_skill_library_renders_available_skill_metadata_without_content(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: frontend\ndescription: Build UI.\n---\nUse semantic markup.\n")

    library = SkillLibrary.discover(tmp_path)

    rendered = library.render_available()

    assert "Available skills:" in rendered
    assert "frontend - Build UI." in rendered
    assert ".agents/skills/frontend/SKILL.md" in rendered
    assert "load_skill" in rendered
    assert "Trigger rules:" in rendered
    assert "plain text" in rendered
    assert "skip an obvious skill" in rendered
    assert "Use semantic markup." not in rendered


def test_skill_library_loads_skill_by_name(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "frontend"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: frontend\ndescription: Build UI.\n---\nUse semantic markup.\n")

    library = SkillLibrary.discover(tmp_path)

    assert library.load("frontend").content == "Use semantic markup."
    assert library.load("missing") is None


def test_skill_library_loads_skill_by_directory_alias_when_frontmatter_name_is_namespaced(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "writing-plans"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: superpowers:writing-plans\ndescription: Write plans.\n---\nBody\n")

    library = SkillLibrary.discover(tmp_path)

    assert library.load("superpowers:writing-plans").content == "Body"
    assert library.load("writing-plans").content == "Body"


def test_skill_library_includes_builtin_skill_installer(tmp_path: Path):
    library = SkillLibrary.discover(tmp_path)

    skill = library.load("skill-installer")

    assert skill is not None
    assert "Install skills" in skill.description
    assert "install_skill" in skill.content
    assert "openai/skills" not in skill.content
    assert "load `find-skills`" in skill.content
    assert "Only use OpenAI skills" in skill.content


def test_skill_library_includes_builtin_find_skills(tmp_path: Path):
    library = SkillLibrary.discover(tmp_path)

    skill = library.load("find-skills")

    assert skill is not None
    assert "discover and install agent skills" in skill.description
    assert "npx skills find" in skill.content
    assert "npx skills add" not in skill.content
    assert "-g -y" not in skill.content
