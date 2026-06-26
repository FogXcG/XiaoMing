from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


ToolProfile = Literal["full", "read_only", "skill_install", "verify"]


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    description: str
    when_to_use: str
    system_prompt: str
    tool_profile: ToolProfile
    default_skills: list[str] = field(default_factory=list)
    default_context_policy: str = "briefed"
    background_default: bool = True
    model: str | None = None
    max_turns: int | None = None


class AgentRegistry:
    def __init__(self, agents: list[AgentDefinition]):
        self._agents = {agent.name: agent for agent in agents}

    def get(self, name: str) -> AgentDefinition:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise KeyError(f"unknown agent type: {name}") from exc

    def list(self) -> list[AgentDefinition]:
        return list(self._agents.values())


WORKER_PROMPT = (
    "You are an independent coding agent. Your user is the coordinator, who represents the human user. "
    "Complete the assigned task using available tools and skills. Decide your own approach from the task goal and context. "
    "Ask the coordinator through talk when you need human intent, design approval, or a decision. "
    "For skill installation, use the native skill tool with action=install instead of recreating installer behavior with shell, git clone, curl, mkdir, cp, or write_file."
)

VERIFIER_WORKER_PROMPT = (
    "You are an independent verification agent. Use only read-only tools to decide whether the worker result satisfies the user goal. "
    "Do not trust reported paths without inspecting them when existence or content matters."
)

def builtin_agent_registry() -> AgentRegistry:
    return AgentRegistry(
        [
            AgentDefinition(
                name="worker",
                description="The single worker role for all concrete background work.",
                when_to_use="Use for any concrete task that may inspect, modify, install, research, or verify work.",
                system_prompt=WORKER_PROMPT,
                tool_profile="full",
                default_context_policy="forked",
            ),
            AgentDefinition(
                name="verifier",
                description="Read-only worker that verifies whether reported work satisfies the task.",
                when_to_use="Use internally after a worker reports completion.",
                system_prompt=VERIFIER_WORKER_PROMPT,
                tool_profile="verify",
                default_context_policy="forked",
            ),
        ]
    )
