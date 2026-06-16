from pathlib import Path

from xiaoming.permissions.engine import PermissionEngine
from xiaoming.permissions.types import PermissionMode
from xiaoming.tools.git_status import GitStatusTool
from xiaoming.tools.list_files import ListFilesTool
from xiaoming.tools.read_file import ReadFileTool
from xiaoming.tools.search_code import SearchCodeTool


def test_list_files_ignores_git_and_node_modules(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("ignored\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "pkg.js").write_text("ignored\n")

    result = ListFilesTool(tmp_path).run({"path": None, "pattern": None})

    assert result.status == "success"
    assert "src/app.py" in result.output
    assert ".git/config" not in result.output
    assert "node_modules/pkg.js" not in result.output


def test_read_file_returns_numbered_slice(tmp_path: Path):
    path = tmp_path / "app.py"
    path.write_text("a\nb\nc\n")

    result = ReadFileTool(tmp_path).run({"path": "app.py", "start_line": 2, "limit": 1})

    assert result.status == "success"
    assert result.output == "2: b"


def test_read_file_reports_sensitive_path_requires_approval(tmp_path: Path):
    (tmp_path / ".env").write_text("SECRET=1\n")
    engine = PermissionEngine(tmp_path, mode=PermissionMode.DEFAULT)

    result = ReadFileTool(tmp_path, permission_engine=engine).run({"path": ".env", "start_line": 1, "limit": 1})

    assert result.status == "error"
    assert "requires approval" in result.error


def test_list_files_denies_outside_workspace_with_permission_engine(tmp_path: Path):
    engine = PermissionEngine(tmp_path, mode=PermissionMode.BYPASS)

    result = ListFilesTool(tmp_path, permission_engine=engine).run({"path": "..", "pattern": None})

    assert result.status == "error"
    assert "rejected" in result.error


def test_search_code_finds_text(tmp_path: Path):
    path = tmp_path / "app.py"
    path.write_text("needle = 1\n")

    result = SearchCodeTool(tmp_path).run({"query": "needle", "path": None})

    assert result.status == "success"
    assert "app.py" in result.output
    assert "needle = 1" in result.output


def test_git_status_returns_error_outside_git_repo(tmp_path: Path):
    result = GitStatusTool(tmp_path).run({})

    assert result.status == "error"
    assert result.error
