from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen
import zipfile


Fetch = Callable[[str], bytes]


@dataclass(frozen=True)
class GithubSkillSource:
    owner: str
    repo: str
    ref: str
    path: str


@dataclass(frozen=True)
class SkillInstallResult:
    name: str
    destination: Path
    files: int
    bytes_written: int


class SkillInstallError(RuntimeError):
    pass


def parse_github_tree_url(url: str) -> GithubSkillSource:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != "github.com":
        raise SkillInstallError("only GitHub tree URLs are supported")
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 5 or parts[2] != "tree":
        raise SkillInstallError("expected GitHub URL like https://github.com/<owner>/<repo>/tree/<ref>/<path>")
    owner, repo, _tree, ref = parts[:4]
    skill_path = "/".join(parts[4:]).strip("/")
    if not skill_path:
        raise SkillInstallError("GitHub tree URL must point to a skill directory")
    return GithubSkillSource(owner=owner, repo=repo, ref=ref, path=skill_path)


def install_skill_from_url(
    url: str,
    workspace: Path,
    fetch: Fetch | None = None,
    max_files: int = 100,
    max_total_bytes: int = 2_000_000,
) -> SkillInstallResult:
    source = parse_github_tree_url(url)
    return install_skills_from_github(
        repo=f"{source.owner}/{source.repo}",
        paths=[source.path],
        workspace=workspace,
        ref=source.ref,
        fetch=fetch,
        max_files=max_files,
        max_total_bytes=max_total_bytes,
    )[0]


def install_skills_from_github(
    repo: str,
    paths: list[str],
    workspace: Path,
    ref: str = "main",
    dest: Path | None = None,
    fetch: Fetch | None = None,
    max_files: int = 100,
    max_total_bytes: int = 2_000_000,
) -> list[SkillInstallResult]:
    owner, repo_name = _parse_repo(repo)
    if not paths:
        raise SkillInstallError("at least one skill path is required")
    fetcher = fetch or _fetch_url
    source = GithubSkillSource(owner=owner, repo=repo_name, ref=ref, path="")
    dest_root = dest or workspace / ".agents" / "skills"
    normalized_paths = []
    for skill_path in paths:
        normalized = str(skill_path).strip("/")
        _validate_relative_path(normalized)
        destination = dest_root / Path(normalized).name
        if destination.exists():
            raise SkillInstallError(f"destination already exists: {destination}")
        normalized_paths.append(normalized)

    with tempfile.TemporaryDirectory(prefix="xiaoming-skill-install-") as tmp:
        tmp_path = Path(tmp)
        repo_root = _prepare_repo(
            source,
            normalized_paths,
            tmp_path,
            fetcher,
            allow_git_fallback=fetch is None,
        )
        plans = _build_copy_plans(repo_root, normalized_paths, dest_root, max_files, max_total_bytes)
        for source_dir, destination, _files, _bytes in plans:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, destination)
        return [
            SkillInstallResult(
                name=destination.name,
                destination=destination,
                files=files,
                bytes_written=bytes_written,
            )
            for _source_dir, destination, files, bytes_written in plans
        ]


def _parse_repo(repo: str) -> tuple[str, str]:
    parts = [part for part in repo.strip().split("/") if part]
    if len(parts) != 2:
        raise SkillInstallError("repo must be in owner/repo format")
    return parts[0], parts[1]


def _prepare_repo(
    source: GithubSkillSource,
    paths: list[str],
    tmp_path: Path,
    fetch: Fetch,
    allow_git_fallback: bool,
) -> Path:
    try:
        return _download_repo_zip(source, tmp_path, fetch)
    except SkillInstallError:
        if allow_git_fallback:
            return _git_sparse_checkout(source, paths, tmp_path)
        raise


def _download_repo_zip(source: GithubSkillSource, tmp_path: Path, fetch: Fetch) -> Path:
    url = f"https://codeload.github.com/{source.owner}/{source.repo}/zip/{quote(source.ref, safe='')}"
    try:
        archive = fetch(url)
    except Exception as exc:
        raise SkillInstallError(f"failed to download GitHub archive: {exc}") from exc
    zip_path = tmp_path / "repo.zip"
    zip_path.write_bytes(archive)
    try:
        with zipfile.ZipFile(zip_path) as zip_file:
            _safe_extract_zip(zip_file, tmp_path)
            top_levels = {name.split("/")[0] for name in zip_file.namelist() if name}
    except zipfile.BadZipFile as exc:
        raise SkillInstallError("downloaded GitHub archive was not a valid zip file") from exc
    if len(top_levels) != 1:
        raise SkillInstallError("unexpected GitHub archive layout")
    return tmp_path / next(iter(top_levels))


def _safe_extract_zip(zip_file: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for info in zip_file.infolist():
        target = (destination / info.filename).resolve()
        if target != root and root not in target.parents:
            raise SkillInstallError("archive contains files outside the destination")
    zip_file.extractall(destination)


def _git_sparse_checkout(source: GithubSkillSource, paths: list[str], tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    errors = []
    for repo_url in [
        f"https://github.com/{source.owner}/{source.repo}.git",
        f"git@github.com:{source.owner}/{source.repo}.git",
    ]:
        try:
            _run_git(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--depth",
                    "1",
                    "--sparse",
                    "--single-branch",
                    "--branch",
                    source.ref,
                    repo_url,
                    str(repo_root),
                ]
            )
            _run_git(["git", "-C", str(repo_root), "sparse-checkout", "set", *paths])
            return repo_root
        except SkillInstallError as exc:
            errors.append(str(exc))
            if repo_root.exists():
                shutil.rmtree(repo_root, ignore_errors=True)
    raise SkillInstallError("failed to clone GitHub repo: " + " | ".join(errors))


def _run_git(args: list[str]) -> None:
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise SkillInstallError(result.stderr.strip() or "git command failed")


def _build_copy_plans(
    repo_root: Path,
    paths: list[str],
    dest_root: Path,
    max_files: int,
    max_total_bytes: int,
) -> list[tuple[Path, Path, int, int]]:
    plans: list[tuple[Path, Path, int, int]] = []
    total_files = 0
    total_bytes = 0
    for skill_path in paths:
        source_dir = repo_root / skill_path
        _validate_skill_directory(source_dir, skill_path)
        files, bytes_written = _measure_skill(source_dir)
        total_files += files
        total_bytes += bytes_written
        if total_files > max_files:
            raise SkillInstallError(f"skill has too many files ({total_files} > {max_files})")
        if total_bytes > max_total_bytes:
            raise SkillInstallError(f"skill is too large ({total_bytes} bytes > {max_total_bytes})")
        plans.append((source_dir, dest_root / Path(skill_path).name, files, bytes_written))
    return plans


def _validate_skill_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SkillInstallError(f"skill path not found: {label}")
    if not (path / "SKILL.md").is_file():
        raise SkillInstallError(f"skill directory must contain SKILL.md: {label}")
    for child in path.rglob("*"):
        if child.is_symlink():
            raise SkillInstallError(f"refusing symlink in skill: {child.relative_to(path)}")


def _measure_skill(path: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if child.is_file():
            relative = str(child.relative_to(path))
            _validate_relative_path(relative)
            files += 1
            total_bytes += child.stat().st_size
    return files, total_bytes


def _validate_relative_path(path: str) -> None:
    parts = Path(path).parts
    if not path or Path(path).is_absolute() or ".." in parts:
        raise SkillInstallError(f"unsafe path in skill: {path}")


def _fetch_url(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "xiaoming-cli"})
    with urlopen(request, timeout=30) as response:
        return response.read()
