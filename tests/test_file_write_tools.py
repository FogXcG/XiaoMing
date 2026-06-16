from pathlib import Path

from xiaoming.checkpoints.store import CheckpointStore
from xiaoming.tools.append_file import AppendFileTool
from xiaoming.tools.edit_file import EditFileTool
from xiaoming.tools.write_file import WriteFileTool, summarize_write_for_approval


def test_write_file_creates_new_file(tmp_path: Path):
    result = WriteFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run(
        {"path": "index.html", "content": "<!DOCTYPE html>\n"}
    )

    assert result.status == "success"
    assert (tmp_path / "index.html").read_text() == "<!DOCTYPE html>\n"
    assert "Wrote file: index.html" in result.output


def test_write_file_requires_approval_in_suggest_mode(tmp_path: Path):
    approvals = []

    result = WriteFileTool(tmp_path, approval_mode="suggest", approve=lambda action: approvals.append(action) or False).run(
        {"path": "index.html", "content": "<h1>Hello</h1>\n"}
    )

    assert result.status == "denied"
    assert not (tmp_path / "index.html").exists()
    assert "Tool: write_file" in approvals[0]
    assert "File: index.html" in approvals[0]
    assert "<h1>Hello</h1>" in approvals[0]


def test_write_file_refuses_to_overwrite_existing_file(tmp_path: Path):
    (tmp_path / "index.html").write_text("old\n")

    result = WriteFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run(
        {"path": "index.html", "content": "new\n"}
    )

    assert result.status == "error"
    assert "file already exists" in result.error
    assert (tmp_path / "index.html").read_text() == "old\n"


def test_write_tools_snapshot_files_before_modifying(tmp_path: Path):
    path = tmp_path / "app.py"
    path.write_text("old\n")
    checkpoints = CheckpointStore(tmp_path)
    checkpoint = checkpoints.create(session_id="session-1", prompt="edit")
    tool = EditFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True, checkpoint_store=checkpoints, checkpoint_id=checkpoint.id)

    result = tool.run({"path": "app.py", "old_text": "old\n", "new_text": "new\n"})
    checkpoints.restore(checkpoint.id)

    assert result.status == "success"
    assert path.read_text() == "old\n"


def test_write_file_rejects_oversized_content(tmp_path: Path):
    result = WriteFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True, max_content_chars=10).run(
        {"path": "large.txt", "content": "x" * 11}
    )

    assert result.status == "error"
    assert "content is too large" in result.error


def test_append_file_appends_to_existing_file(tmp_path: Path):
    (tmp_path / "index.html").write_text("<!DOCTYPE html>\n")

    result = AppendFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run(
        {"path": "index.html", "content": "<h1>Hello</h1>\n"}
    )

    assert result.status == "success"
    assert (tmp_path / "index.html").read_text() == "<!DOCTYPE html>\n<h1>Hello</h1>\n"


def test_append_file_requires_existing_file(tmp_path: Path):
    result = AppendFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run(
        {"path": "missing.txt", "content": "hello\n"}
    )

    assert result.status == "error"
    assert "file does not exist" in result.error


def test_edit_file_replaces_unique_text(tmp_path: Path):
    (tmp_path / "app.py").write_text("print('old')\n")

    result = EditFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run(
        {"path": "app.py", "old_text": "old", "new_text": "new"}
    )

    assert result.status == "success"
    assert (tmp_path / "app.py").read_text() == "print('new')\n"


def test_edit_file_reports_multiple_matches(tmp_path: Path):
    (tmp_path / "app.py").write_text("value = 1\nvalue = 1\n")

    result = EditFileTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run(
        {"path": "app.py", "old_text": "value = 1", "new_text": "value = 2"}
    )

    assert result.status == "error"
    assert "matched 2 times" in result.error
    assert "longer unique old_text" in result.error


def test_summarize_write_for_approval_truncates_preview():
    summary = summarize_write_for_approval("large.txt", "\n".join(str(i) for i in range(100)), preview_lines=3)

    assert "Tool: write_file" in summary
    assert "File: large.txt" in summary
    assert "... truncated ..." in summary
