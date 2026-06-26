from io import BytesIO
from pathlib import Path
import zipfile

from xiaoming.skills import SkillLibrary
from xiaoming.tools.skill import SkillTool


def _repo_zip(files: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(f"acme-skills-main/{path}", content)
    return buf.getvalue()


def test_skill_tool_installs_refreshes_and_loads_skill(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip(
            {
                "skills/frontend/SKILL.md": b"---\nname: frontend\ndescription: Build UI.\n---\nUse semantic HTML.",
                "skills/frontend/guide.md": b"Guide content.",
            }
        ),
    }
    library = SkillLibrary([])
    tool = SkillTool(tmp_path, library, approval_mode="auto_edit", approve=lambda action: True, fetch=lambda url: responses[url])

    install = tool.run({"action": "install", "url": "https://github.com/acme/skills/tree/main/skills/frontend"})
    load = tool.run({"action": "load", "name": "frontend"})

    assert install.status == "success"
    assert "<status>installed</status>" in install.output
    assert (tmp_path / ".agents" / "skills" / "frontend" / "SKILL.md").exists()
    assert library.load("frontend") is not None
    assert load.status == "success"
    assert "<name>frontend</name>" in load.output
    assert "Use semantic HTML." in load.output


def test_skill_tool_rejects_invalid_install_url(tmp_path: Path):
    library = SkillLibrary([])
    tool = SkillTool(tmp_path, library, approval_mode="auto_edit", approve=lambda action: True)

    result = tool.run({"action": "install", "url": "https://google.com"})

    assert result.status == "error"
    assert "GitHub tree URL" in result.error or "expected" in result.error.lower()


def test_skill_tool_treats_existing_skill_as_already_installed(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip(
            {"skills/frontend/SKILL.md": b"---\nname: frontend\n---\nContent."}
        ),
    }
    library = SkillLibrary([])
    tool = SkillTool(tmp_path, library, approval_mode="auto_edit", approve=lambda action: True, fetch=lambda url: responses[url])

    first = tool.run({"action": "install", "url": "https://github.com/acme/skills/tree/main/skills/frontend"})
    second = tool.run({"action": "install", "url": "https://github.com/acme/skills/tree/main/skills/frontend"})

    assert first.status == "success"
    assert second.status == "success"
    assert "<status>already_installed</status>" in second.output
