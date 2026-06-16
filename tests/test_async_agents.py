from xiaoming.async_runtime.agents import builtin_agent_registry


def test_builtin_agents_include_core_worker_types():
    registry = builtin_agent_registry()

    assert [agent.name for agent in registry.list()] == ["worker", "verifier"]
    assert registry.get("worker").tool_profile == "full"
    assert registry.get("worker").default_context_policy == "forked"
    assert registry.get("verifier").tool_profile == "verify"
    assert registry.get("verifier").default_context_policy == "forked"


def test_worker_does_not_preload_specialized_skills():
    registry = builtin_agent_registry()

    agent = registry.get("worker")

    assert agent.default_skills == []
    assert agent.default_context_policy == "forked"
