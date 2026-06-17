from pathlib import Path
from io import BytesIO
import zipfile

from xiaoming.skills import Skill, SkillLibrary
from xiaoming.tools.fetch_skill import FetchSkillTool


def _repo_zip(files: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            archived_path = f"acme-skills-main/{path}"
            zf.writestr(archived_path, content)
    return buf.getvalue()


def make_skill(name: str, description: str = "", content: str = "") -> Skill:
    return Skill(name=name, description=description, content=content)


def test_fetch_skill_downloads_and_refreshes_library(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip({
            "skills/frontend/SKILL.md": b"---\nname: frontend\ndescription: Build UI.\n---\nUse semantic HTML.",
            "skills/frontend/guide.md": b"Guide content.",
        }),
    }

    skills_dir = tmp_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True)

    library = SkillLibrary([])
    tool = FetchSkillTool(workspace=tmp_path, library=library)
    # Patch the installer fetch
    import xiaoming.tools.fetch_skill as fetch_skill_module
    orig = fetch_skill_module.install_skill_from_url
    def patched_install(url, workspace, max_files=100, max_total_bytes=2_000_000, fetch=None):
        return orig(url, workspace, max_files=max_files, max_total_bytes=max_total_bytes, fetch=lambda u: responses[u])

    fetch_skill_module.install_skill_from_url = patched_install
    try:
        result = tool.run({"url": "https://github.com/acme/skills/tree/main/skills/frontend"})
    finally:
        fetch_skill_module.install_skill_from_url = orig

    assert result.status == "success"
    assert "frontend" in result.output
    # Verify skill was installed to disk
    assert (skills_dir / "frontend" / "SKILL.md").exists()
    # Verify library was refreshed
    loaded = library.load("frontend")
    assert loaded is not None
    assert loaded.name == "frontend"


def test_fetch_skill_rejects_invalid_url(tmp_path: Path):
    library = SkillLibrary([])
    tool = FetchSkillTool(workspace=tmp_path, library=library)

    result = tool.run({"url": "https://google.com"})

    assert result.status == "error"
    assert "GitHub tree URL" in result.error or "expected" in result.error.lower()


def test_fetch_skill_handles_duplicate(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip({
            "skills/frontend/SKILL.md": b"---\nname: frontend\n---\nContent.",
        }),
    }

    library = SkillLibrary([])
    tool = FetchSkillTool(workspace=tmp_path, library=library)

    import xiaoming.tools.fetch_skill as fetch_skill_module
    orig = fetch_skill_module.install_skill_from_url
    def patched_install(url, workspace, max_files=100, max_total_bytes=2_000_000, fetch=None):
        return orig(url, workspace, max_files=max_files, max_total_bytes=max_total_bytes, fetch=lambda u: responses[u])

    fetch_skill_module.install_skill_from_url = patched_install
    try:
        # First install
        result1 = tool.run({"url": "https://github.com/acme/skills/tree/main/skills/frontend"})
        assert result1.status == "success"

        # Second install (duplicate)
        result2 = tool.run({"url": "https://github.com/acme/skills/tree/main/skills/frontend"})
        assert result2.status == "success"
        assert "already installed" in result2.output.lower()
    finally:
        fetch_skill_module.install_skill_from_url = orig
