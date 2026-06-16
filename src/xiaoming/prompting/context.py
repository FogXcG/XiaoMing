from __future__ import annotations

from xiaoming.prompting.items import PromptItem, TurnContext


class ContextBuilder:
    def build_initial_durable_context(self, ctx: TurnContext, turn_id: str) -> list[PromptItem]:
        return [
            PromptItem(role="developer", kind="developer_context", content=_developer_context(ctx), turn_id=turn_id, durable=True),
            PromptItem(role="user", kind="environment_context", content=_environment_context(ctx), turn_id=turn_id, durable=True),
        ]

    def build_durable_diff(self, previous: TurnContext, current: TurnContext, turn_id: str) -> list[PromptItem]:
        sections: list[str] = []
        prev = previous.durable_snapshot()
        curr = current.durable_snapshot()
        for key in sorted(curr):
            if prev.get(key) != curr.get(key):
                sections.append(f"<changed name=\"{key}\" previous=\"{_xml(prev.get(key))}\" current=\"{_xml(curr.get(key))}\" />")
        if current.project_instructions_hash and previous.project_instructions_hash != current.project_instructions_hash and current.project_instructions_text:
            sections.append(f"<project_instructions>{_xml(current.project_instructions_text)}</project_instructions>")
        if current.skills_summary_hash and previous.skills_summary_hash != current.skills_summary_hash and current.skills_summary_text:
            sections.append(f"<available_skills>{_xml(current.skills_summary_text)}</available_skills>")
        if current.plugins_summary_hash and previous.plugins_summary_hash != current.plugins_summary_hash and current.plugins_summary_text:
            sections.append(f"<available_plugins>{_xml(current.plugins_summary_text)}</available_plugins>")
        if not sections:
            return []
        content = "<developer_context type=\"durable_diff\">\n" + "\n".join(sections) + "\n</developer_context>"
        return [PromptItem(role="developer", kind="developer_context_diff", content=content, turn_id=turn_id, durable=True)]

    def build_ephemeral_context(self, ctx: TurnContext, turn_id: str) -> list[PromptItem]:
        sections = [f"<current_date>{_xml(ctx.current_date)}</current_date>"]
        if ctx.checkpoint_id:
            sections.append(f"<checkpoint_id>{_xml(ctx.checkpoint_id)}</checkpoint_id>")
        if ctx.last_error_id:
            sections.append(f"<last_error id=\"{_xml(ctx.last_error_id)}\" />")
        if ctx.interrupted_turn_id:
            sections.append(f"<interrupted_turn_id>{_xml(ctx.interrupted_turn_id)}</interrupted_turn_id>")
        if ctx.pending_worker_questions_text:
            sections.append(f"<pending_worker_questions>{_xml(ctx.pending_worker_questions_text)}</pending_worker_questions>")
        if ctx.background_tasks_text:
            sections.append(f"<background_tasks>{_xml(ctx.background_tasks_text)}</background_tasks>")
        content = "<turn_context>\n" + "\n".join(sections) + "\n</turn_context>"
        return [PromptItem(role="user", kind="ephemeral_context", content=content, turn_id=turn_id, durable=False, consumed=False)]


def _developer_context(ctx: TurnContext) -> str:
    parts = [
        "<developer_context type=\"initial\">",
        f"<provider>{_xml(ctx.provider)}</provider>",
        f"<model>{_xml(ctx.model)}</model>",
        f"<stream>{str(ctx.stream).lower()}</stream>",
        f"<permission_mode>{_xml(ctx.permission_mode)}</permission_mode>",
    ]
    if ctx.approval_policy:
        parts.append(f"<approval_policy>{_xml(ctx.approval_policy)}</approval_policy>")
    if ctx.project_instructions_text:
        parts.append(f"<project_instructions>{_xml(ctx.project_instructions_text)}</project_instructions>")
    if ctx.skills_summary_text:
        parts.append(f"<available_skills>{_xml(ctx.skills_summary_text)}</available_skills>")
    if ctx.plugins_summary_text:
        parts.append(f"<available_plugins>{_xml(ctx.plugins_summary_text)}</available_plugins>")
    parts.append("</developer_context>")
    return "\n".join(parts)


def _environment_context(ctx: TurnContext) -> str:
    parts = [
        "<environment_context>",
        f"<cwd>{_xml(ctx.cwd)}</cwd>",
        f"<session_id>{_xml(ctx.session_id or '')}</session_id>",
        f"<resumed>{str(ctx.resumed).lower()}</resumed>",
        "</environment_context>",
    ]
    return "\n".join(parts)


def _xml(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
