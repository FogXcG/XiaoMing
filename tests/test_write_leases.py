from xiaoming.async_runtime.leases import FileWriteLeaseClient, WriteLeaseServer
from xiaoming.tools.write_file import WriteFileTool


def test_write_lease_server_denies_same_file_to_different_task():
    server = WriteLeaseServer()

    first = server.acquire("task-1", ["README.md"])
    second = server.acquire("task-2", ["README.md"])

    assert first.granted is True
    assert second.granted is False


def test_write_lease_server_releases_task_files():
    server = WriteLeaseServer()
    server.acquire("task-1", ["README.md"])

    server.release("task-1")

    assert server.acquire("task-2", ["README.md"]).granted is True


def test_write_file_tool_respects_lease_callback(tmp_path):
    tool = WriteFileTool(
        tmp_path,
        approval_mode="full_auto",
        approve=lambda action: True,
        lease_callback=lambda tool_name, paths: False,
    )

    result = tool.run({"path": "README.md", "content": "hello\n"})

    assert result.status == "error"
    assert "lease denied" in result.error
    assert not (tmp_path / "README.md").exists()


def test_file_write_lease_client_coordinates_across_instances(tmp_path):
    first = FileWriteLeaseClient(tmp_path / "leases", "task-1")
    second = FileWriteLeaseClient(tmp_path / "leases", "task-2")

    assert first.acquire("write_file", ["README.md"]) is True
    assert second.acquire("write_file", ["README.md"]) is False

    first.release_all()
    assert second.acquire("write_file", ["README.md"]) is True
