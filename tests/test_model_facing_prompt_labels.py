from pathlib import Path

from xiaoming.async_runtime import question_decider, responder, scheduler, verifier
from xiaoming.cli import build_instructions
from xiaoming.tools.background_task import BackgroundTasksStatusTool, CancelBackgroundTaskTool, ReplyMailboxMessageTool, ScheduleBackgroundTaskTool
from xiaoming.tools.talk import TalkTool
from xiaoming import worker_main


def test_model_facing_instructions_do_not_include_product_name(tmp_path: Path):
    texts = [
        build_instructions(tmp_path, role="orchestrator"),
        build_instructions(tmp_path, role="worker"),
        scheduler._SCHEDULER_INSTRUCTIONS,
        responder._RESPONDER_INSTRUCTIONS,
        question_decider._DECIDER_INSTRUCTIONS,
        verifier._LLM_VERIFIER_INSTRUCTIONS,
        worker_main._worker_protocol_context_item()["content"],
        worker_main._worker_protocol_repair_prompt(),
    ]

    for text in texts:
        assert "Xiaoming" not in text
        assert "xiaoming" not in text


def test_model_facing_tool_descriptions_do_not_include_product_name():
    tools = [
        ScheduleBackgroundTaskTool(lambda: None),
        BackgroundTasksStatusTool(lambda: None),
        CancelBackgroundTaskTool(lambda: None),
        ReplyMailboxMessageTool(lambda: None),
        TalkTool(lambda purpose, message, context, options: ""),
    ]

    for tool in tools:
        assert "Xiaoming" not in tool.spec.description
        assert "xiaoming" not in tool.spec.description
        rendered_schema = str(tool.spec.input_schema)
        assert "Xiaoming" not in rendered_schema
        assert "xiaoming" not in rendered_schema


def test_builtin_skill_prompts_do_not_include_product_name():
    root = Path(__file__).resolve().parents[1] / "src" / "xiaoming" / "builtin_skills"

    for path in root.glob("*/SKILL.md"):
        text = path.read_text()
        assert "Xiaoming" not in text
        assert "xiaoming" not in text
