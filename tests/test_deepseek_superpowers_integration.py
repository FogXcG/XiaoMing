import os

import pytest

from xiaoming.agent_loop import AgentLoop
from xiaoming.llm.deepseek_provider import DeepSeekProvider
from xiaoming.session import Session
from xiaoming.tools.background_task import BackgroundTasksStatusTool, ScheduleBackgroundTaskTool
from xiaoming.tools.base import ToolResult
from xiaoming.tools.registry import ToolRegistry
from xiaoming.tools.skill import SkillTool


class ScriptedCoordinator:
    def __init__(self):
        self.scheduled = []
        self.status_checks = 0

    def schedule_background_task(self, task_spec):
        self.scheduled.append(task_spec)
        return ToolResult("schedule_background_task", "success", output=f"scheduled: {task_spec.title}")

    def tasks_text(self):
        self.status_checks += 1
        return "安装 superpowers skill  failed  GitHub repository root URL is not a single skill directory; retry using git clone."


@pytest.mark.deepseek
def test_deepseek_superpowers_skill_install_retry_with_git_clone(tmp_path):
    if os.environ.get("XIAOMING_RUN_DEEPSEEK_SUPERPOWERS_CASE") != "1":
        pytest.skip("set XIAOMING_RUN_DEEPSEEK_SUPERPOWERS_CASE=1 to run the real DeepSeek superpowers case")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is required")

    coordinator = ScriptedCoordinator()
    registry = ToolRegistry(
        [
            SkillTool(tmp_path, _SkillLibrary(), approval_mode="auto_edit", approve=lambda action: True),
            ScheduleBackgroundTaskTool(lambda: coordinator),
            BackgroundTasksStatusTool(lambda: coordinator),
        ]
    )
    loop = AgentLoop(
        provider=DeepSeekProvider(),
        registry=registry,
        instructions=_INSTRUCTIONS,
        model="deepseek-v4-flash",
        temperature=0,
        max_output_tokens=4096,
        max_turns=8,
        stream=False,
        model_timeout_seconds=120,
    )
    session = Session()

    first = loop.run("你了解 superpowers 这个 skill 吗？", session=session)
    assert "superpowers" in first.lower()
    assert coordinator.scheduled == []

    loop.run("帮我安装 superpowers 这个 skill：https://github.com/obra/superpowers", session=session)
    assert len(coordinator.scheduled) == 1
    assert "https://github.com/obra/superpowers" in coordinator.scheduled[0].goal

    third = loop.run("安装失败的话，用 git clone 安装。", session=session)
    assert len(coordinator.scheduled) >= 2, third
    retry = coordinator.scheduled[-1]
    assert "git clone" in (retry.goal + " " + retry.notes).lower()
    assert retry.success_criteria
    assert "已完成" not in third


class _SkillLibrary:
    skills = {"skill-installer": object()}

    def load(self, name):
        if name != "skill-installer":
            return None
        return _Skill()

    def refresh(self, workspace):
        return None


class _Skill:
    name = "skill-installer"
    description = "Install skills from GitHub URLs into this workspace."
    path = None
    content = (
        "Use this skill when the user wants to install an Agent Skill from a GitHub directory URL.\n"
        "Interactive chat workflow: explain that remote skill installation changes the local workspace, "
        "then call schedule_background_task with a complete structured task contract. If installation fails, "
        "the user may ask you to retry using git clone; schedule that retry in the background too."
    )


_INSTRUCTIONS = """
In this runtime, act as a local coding agent.
Answer simple questions directly.
When the user asks to install a skill, first briefly explain that installation changes the workspace, then call skill with action=load for skill-installer, then call schedule_background_task with a complete structured task contract.
When the user asks about install progress or references a failed install, call background_tasks_status before deciding next steps.
If a background install failed and the user asks to use git clone, schedule a new background task whose goal explicitly uses git clone.
Do not claim a background task is completed unless the status says it was accepted or completed successfully.
""".strip()
