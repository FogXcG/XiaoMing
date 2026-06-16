from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from xiaoming.policy.paths import resolve_workspace_path


@dataclass(frozen=True)
class CheckpointRecord:
    id: str
    session_id: str | None
    prompt: str
    created_at: str
    path: Path


@dataclass(frozen=True)
class CheckpointRestoreResult:
    restored: list[str]
    deleted: list[str]


class CheckpointStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.root = self.workspace / ".xiaoming" / "checkpoints"
        self.index_path = self.root / "index.json"

    def create(self, session_id: str | None, prompt: str) -> CheckpointRecord:
        self.root.mkdir(parents=True, exist_ok=True)
        checkpoint_id = str(uuid4())
        path = self.root / checkpoint_id
        path.mkdir(parents=True, exist_ok=True)
        record = CheckpointRecord(id=checkpoint_id, session_id=session_id, prompt=prompt, created_at=_now(), path=path)
        (path / "files").mkdir()
        (path / "metadata.json").write_text(json.dumps(_record_to_json(record), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        (path / "manifest.json").write_text(json.dumps({"files": {}}, indent=2, sort_keys=True) + "\n")
        self._upsert(record)
        return record

    def latest(self) -> CheckpointRecord | None:
        records = self.list()
        return records[0] if records else None

    def get(self, checkpoint_id: str) -> CheckpointRecord | None:
        for record in self.list():
            if record.id == checkpoint_id:
                return record
        return None

    def list(self) -> list[CheckpointRecord]:
        if not self.index_path.exists():
            return []
        try:
            data = json.loads(self.index_path.read_text())
        except json.JSONDecodeError:
            return []
        records: list[CheckpointRecord] = []
        for item in data.get("checkpoints", []):
            try:
                path = self.workspace / item["path"] if not Path(str(item["path"])).is_absolute() else Path(str(item["path"]))
                records.append(
                    CheckpointRecord(
                        id=str(item["id"]),
                        session_id=item.get("session_id"),
                        prompt=str(item.get("prompt") or ""),
                        created_at=str(item["created_at"]),
                        path=path,
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return records

    def snapshot_paths(self, checkpoint_id: str | None, paths: list[str]) -> None:
        if not checkpoint_id:
            return
        record = self.get(checkpoint_id)
        if record is None:
            return
        manifest = self._read_manifest(record)
        for path_text in paths:
            resolved = resolve_workspace_path(self.workspace, path_text, allow_sensitive=True)
            rel = str(resolved.relative_to(self.workspace))
            if rel in manifest["files"]:
                continue
            snapshot_rel = f"files/{len(manifest['files']):06d}"
            manifest["files"][rel] = {
                "existed": resolved.exists(),
                "snapshot": snapshot_rel,
            }
            if resolved.exists():
                snapshot = record.path / snapshot_rel
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(resolved, snapshot)
        self._write_manifest(record, manifest)

    def restore(self, checkpoint_id: str) -> CheckpointRestoreResult:
        record = self.get(checkpoint_id)
        if record is None:
            raise ValueError(f"unknown checkpoint: {checkpoint_id}")
        manifest = self._read_manifest(record)
        restored: list[str] = []
        deleted: list[str] = []
        for rel, entry in manifest["files"].items():
            target = resolve_workspace_path(self.workspace, rel, allow_sensitive=True)
            if entry.get("existed"):
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record.path / entry["snapshot"], target)
                restored.append(rel)
            elif target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                deleted.append(rel)
        return CheckpointRestoreResult(restored=sorted(restored), deleted=sorted(deleted))

    def _read_manifest(self, record: CheckpointRecord) -> dict[str, Any]:
        try:
            data = json.loads((record.path / "manifest.json").read_text())
        except json.JSONDecodeError:
            return {"files": {}}
        if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
            return {"files": {}}
        return data

    def _write_manifest(self, record: CheckpointRecord, manifest: dict[str, Any]) -> None:
        (record.path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def _upsert(self, record: CheckpointRecord) -> None:
        data = {"checkpoints": []}
        if self.index_path.exists():
            try:
                loaded = json.loads(self.index_path.read_text())
                if isinstance(loaded, dict) and isinstance(loaded.get("checkpoints"), list):
                    data = loaded
            except json.JSONDecodeError:
                pass
        item = _record_to_json(record)
        checkpoints = [entry for entry in data["checkpoints"] if entry.get("id") != record.id]
        checkpoints.insert(0, item)
        data["checkpoints"] = checkpoints[:50]
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _record_to_json(record: CheckpointRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "session_id": record.session_id,
        "prompt": record.prompt,
        "created_at": record.created_at,
        "path": str(record.path.relative_to(record.path.parents[2])) if ".xiaoming" in record.path.parts else str(record.path),
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
