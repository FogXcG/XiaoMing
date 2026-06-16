from pathlib import Path
from io import BytesIO
import zipfile

from xiaoming.skill_installer import GithubSkillSource, SkillInstallError, install_skill_from_url, install_skills_from_github, parse_github_tree_url


def test_parse_github_tree_url():
    source = parse_github_tree_url("https://github.com/openai/skills/tree/main/skills/.experimental/create-plan")

    assert source == GithubSkillSource(owner="openai", repo="skills", ref="main", path="skills/.experimental/create-plan")


def test_install_skill_from_github_url_downloads_skill_directory(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip(
            {
                "skills/frontend/SKILL.md": b"---\nname: frontend\ndescription: Build UI.\n---\nBody\n",
                "skills/frontend/references/tokens.md": b"Tokens\n",
                "skills/backend/SKILL.md": b"---\nname: backend\n---\nBackend\n",
            }
        ),
    }

    result = install_skill_from_url(
        "https://github.com/acme/skills/tree/main/skills/frontend",
        tmp_path,
        fetch=lambda url: responses[url],
    )

    assert result.name == "frontend"
    assert result.files == 2
    assert (tmp_path / ".agents" / "skills" / "frontend" / "SKILL.md").read_text() == "---\nname: frontend\ndescription: Build UI.\n---\nBody\n"
    assert (tmp_path / ".agents" / "skills" / "frontend" / "references" / "tokens.md").read_text() == "Tokens\n"


def test_install_skills_from_repo_paths_downloads_multiple_skills(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip(
            {
                "skills/frontend/SKILL.md": b"---\nname: frontend\n---\nFrontend\n",
                "skills/frontend/references/tokens.md": b"Tokens\n",
                "skills/backend/SKILL.md": b"---\nname: backend\n---\nBackend\n",
            }
        ),
    }

    results = install_skills_from_github(
        repo="acme/skills",
        paths=["skills/frontend", "skills/backend"],
        workspace=tmp_path,
        fetch=lambda url: responses[url],
    )

    assert [result.name for result in results] == ["frontend", "backend"]
    assert (tmp_path / ".agents" / "skills" / "frontend" / "SKILL.md").read_text() == "---\nname: frontend\n---\nFrontend\n"
    assert (tmp_path / ".agents" / "skills" / "backend" / "SKILL.md").read_text() == "---\nname: backend\n---\nBackend\n"


def test_install_skills_from_repo_path_supports_custom_destination(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip(
            {"skills/frontend/SKILL.md": b"---\nname: frontend\n---\nFrontend\n"}
        ),
    }

    results = install_skills_from_github(
        repo="acme/skills",
        paths=["skills/frontend"],
        workspace=tmp_path,
        dest=tmp_path / ".xiaoming" / "skills",
        fetch=lambda url: responses[url],
    )

    assert results[0].destination == tmp_path / ".xiaoming" / "skills" / "frontend"
    assert (tmp_path / ".xiaoming" / "skills" / "frontend" / "SKILL.md").exists()


def test_install_skill_refuses_existing_destination(tmp_path: Path):
    destination = tmp_path / ".agents" / "skills" / "frontend"
    destination.mkdir(parents=True)
    (destination / "SKILL.md").write_text("existing\n")

    try:
        install_skill_from_url("https://github.com/acme/skills/tree/main/skills/frontend", tmp_path, fetch=lambda url: b"{}")
    except SkillInstallError as exc:
        assert "already exists" in str(exc)
    else:
        raise AssertionError("expected install error")


def test_install_skill_requires_skill_md(tmp_path: Path):
    responses = {
        "https://codeload.github.com/acme/skills/zip/main": _repo_zip(
            {"skills/frontend/README.md": b"Readme\n"}
        ),
    }

    try:
        install_skill_from_url(
            "https://github.com/acme/skills/tree/main/skills/frontend",
            tmp_path,
            fetch=lambda url: responses[url],
        )
    except SkillInstallError as exc:
        assert "SKILL.md" in str(exc)
    else:
        raise AssertionError("expected install error")


def test_install_skill_reports_download_failures(tmp_path: Path):
    def fetch(url):
        raise OSError("network down")

    try:
        install_skill_from_url("https://github.com/acme/skills/tree/main/skills/frontend", tmp_path, fetch=fetch)
    except SkillInstallError as exc:
        assert "failed to download GitHub archive" in str(exc)
        assert "network down" in str(exc)
    else:
        raise AssertionError("expected install error")


def _repo_zip(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path, content in files.items():
            archive.writestr(f"skills-main/{path}", content)
    return buffer.getvalue()
