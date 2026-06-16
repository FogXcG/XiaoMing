from pathlib import Path

from xiaoming.logging import XiaomingLogger
from xiaoming.tools.apply_patch import ApplyPatchTool, summarize_patch_for_approval
from xiaoming.tools.base import ToolResult
from xiaoming.tools.registry import ToolRegistry


def test_apply_patch_updates_existing_file(tmp_path: Path):
    path = tmp_path / "app.py"
    path.write_text("print('old')\n")
    patch = """*** Begin Patch
*** Update File: app.py
@@
-print('old')
+print('new')
*** End Patch
"""

    result = ApplyPatchTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run({"patch": patch})

    assert result.status == "success"
    assert path.read_text() == "print('new')\n"


def test_apply_patch_tool_spec_marks_patch_as_freeform(tmp_path: Path):
    spec = ApplyPatchTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).spec

    assert spec.input_mode == "freeform"
    assert spec.freeform_arg == "patch"


def test_apply_patch_requires_approval_in_suggest_mode(tmp_path: Path):
    path = tmp_path / "app.py"
    path.write_text("old\n")
    patch = """*** Begin Patch
*** Update File: app.py
@@
-old
+new
*** End Patch
"""

    approvals = []

    result = ApplyPatchTool(tmp_path, approval_mode="suggest", approve=lambda action: approvals.append(action) or False).run({"patch": patch})

    assert result.status == "denied"
    assert path.read_text() == "old\n"
    assert "Tool: apply_patch" in approvals[0]
    assert "Files: app.py" in approvals[0]
    assert "-old" in approvals[0]
    assert "+new" in approvals[0]


def test_summarize_patch_for_approval_lists_unified_diff_files():
    patch = """--- /dev/null
+++ b/index.html
@@ -0,0 +1,2 @@
+<h1>Hello</h1>
+<p>World</p>
"""

    summary = summarize_patch_for_approval(patch)

    assert "Tool: apply_patch" in summary
    assert "Files: index.html" in summary
    assert "+<h1>Hello</h1>" in summary


def test_apply_patch_adds_file_from_unified_diff(tmp_path: Path):
    patch = """--- /dev/null
+++ b/index.html
@@ -0,0 +1,3 @@
+<!DOCTYPE html>
+<title>Demo</title>
+<h1>Hello</h1>
"""

    result = ApplyPatchTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run({"patch": patch})

    assert result.status == "success"
    assert (tmp_path / "index.html").read_text() == "<!DOCTYPE html>\n<title>Demo</title>\n<h1>Hello</h1>\n"


def test_apply_patch_adds_file_from_wrapped_unified_diff(tmp_path: Path):
    patch = """*** Begin Patch
--- /dev/null
+++ b/style.css
@@ -0,0 +1,2 @@
+body {
+  color: red;
+}
*** End Patch
"""

    result = ApplyPatchTool(tmp_path, approval_mode="auto_edit", approve=lambda action: True).run({"patch": patch})

    assert result.status == "success"
    assert (tmp_path / "style.css").read_text() == "body {\n  color: red;\n}\n"


def test_registry_returns_unknown_tool_error():
    registry = ToolRegistry([])

    result = registry.run("missing", {})

    assert result.status == "error"
    assert "unknown tool" in result.error


def test_registry_converts_tool_crash_to_error_result():
    class CrashingTool:
        name = "crash"
        description = "Crash."
        input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

        @property
        def spec(self):
            from xiaoming.llm.types import ToolSpec

            return ToolSpec(self.name, self.description, self.input_schema)

        def run(self, args):
            raise RuntimeError("boom")

    registry = ToolRegistry([CrashingTool()])

    result = registry.run("crash", {})

    assert result.status == "error"
    assert "tool crashed: RuntimeError: boom" == result.error


def test_registry_logs_tool_crashes(tmp_path: Path):
    class CrashingTool:
        name = "crash"
        description = "Crash."
        input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

        @property
        def spec(self):
            from xiaoming.llm.types import ToolSpec

            return ToolSpec(self.name, self.description, self.input_schema)

        def run(self, args):
            raise RuntimeError("boom")

    logger = XiaomingLogger.create(tmp_path)
    registry = ToolRegistry([CrashingTool()], logger=logger)

    registry.run("crash", {})

    log_text = logger.path.read_text()
    assert '"event": "tool_crashed"' in log_text
    assert '"tool": "crash"' in log_text
    assert "RuntimeError: boom" in log_text


def test_registry_exposes_specs():
    class DummyTool:
        name = "dummy"
        description = "Dummy."
        input_schema = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

        @property
        def spec(self):
            from xiaoming.llm.types import ToolSpec

            return ToolSpec(self.name, self.description, self.input_schema)

        def run(self, args):
            return ToolResult(self.name, "success", output="ok")

    registry = ToolRegistry([DummyTool()])

    assert registry.specs()[0].name == "dummy"
    assert registry.run("dummy", {}).output == "ok"


def test_registry_describes_common_tool_calls():
    registry = ToolRegistry([])

    assert registry.describe_call("write_file", {"path": "index.html"}) == "create file index.html"
    assert registry.describe_call("append_file", {"path": "index.html"}) == "append to index.html"
    assert registry.describe_call("edit_file", {"path": "app.py"}) == "edit app.py"
    assert registry.describe_call("shell", {"cmd": "pytest -v"}) == "run command: pytest -v"
    assert registry.describe_call("read_file", {"path": "README.md"}) == "read README.md"
    assert registry.describe_call("unknown", {}) == ""
