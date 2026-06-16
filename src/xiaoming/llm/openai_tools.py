from __future__ import annotations

from typing import Any

from xiaoming.llm.types import ToolSpec


def to_openai_tool(spec: ToolSpec) -> dict[str, Any]:
    if spec.input_mode == "freeform":
        return {
            "type": "custom",
            "name": spec.name,
            "description": spec.description,
            "format": {"type": "text"},
        }
    schema = spec.input_schema
    if schema.get("type") != "object":
        raise ValueError(f"tool {spec.name} schema must be an object")
    if schema.get("additionalProperties") is not False:
        raise ValueError(f"tool {spec.name} schema must set additionalProperties to false")
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict):
        raise ValueError(f"tool {spec.name} schema must define properties")
    if sorted(required or []) != sorted(properties.keys()):
        raise ValueError(f"tool {spec.name} strict schema must require every property")
    return {
        "type": "function",
        "name": spec.name,
        "description": spec.description,
        "parameters": schema,
        "strict": True,
    }


def to_openai_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    return [to_openai_tool(spec) for spec in specs]
