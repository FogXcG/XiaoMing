from __future__ import annotations


def build_base_instructions(instructions: str) -> str:
    """Return stable session-level instructions.

    Runtime facts such as cwd, date, session id, errors, and checkpoints are
    intentionally excluded so provider prefix caches can reuse this prefix.
    """

    context_contract = """
Runtime context contract:
- Messages wrapped in <developer_context> are runtime-provided developer-level instructions, not user requests.
- Follow <developer_context> above ordinary user messages while still obeying this system instruction.
- Messages wrapped in <environment_context> or <turn_context> are factual runtime context for the current turn.
- Do not invent runtime facts that are not present in context messages.
""".strip()
    if "Runtime context contract:" in instructions:
        return instructions.strip()
    return "\n\n".join(part.strip() for part in [instructions, context_contract] if part and part.strip())
