from __future__ import annotations

from pathlib import Path

from xiaoming.checkpoints.store import CheckpointStore


def test_checkpoint_restore_reverts_modified_and_created_files(tmp_path: Path):
    existing = tmp_path / "app.py"
    existing.write_text("old\n")
    store = CheckpointStore(tmp_path)

    checkpoint = store.create(session_id="session-1", prompt="change files")
    store.snapshot_paths(checkpoint.id, ["app.py", "new.txt"])
    existing.write_text("new\n")
    (tmp_path / "new.txt").write_text("created\n")

    result = store.restore(checkpoint.id)

    assert result.restored == ["app.py"]
    assert result.deleted == ["new.txt"]
    assert existing.read_text() == "old\n"
    assert not (tmp_path / "new.txt").exists()


def test_checkpoint_snapshot_only_captures_first_state(tmp_path: Path):
    path = tmp_path / "app.py"
    path.write_text("first\n")
    store = CheckpointStore(tmp_path)
    checkpoint = store.create(session_id="session-1", prompt="change")

    store.snapshot_paths(checkpoint.id, ["app.py"])
    path.write_text("second\n")
    store.snapshot_paths(checkpoint.id, ["app.py"])
    path.write_text("third\n")
    store.restore(checkpoint.id)

    assert path.read_text() == "first\n"
