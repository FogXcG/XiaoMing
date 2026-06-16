from __future__ import annotations

import json
from typing import Any, TextIO


class ProtocolError(ValueError):
    pass


def encode_message(kind: str, **payload: Any) -> str:
    data = {"kind": kind, **payload}
    return json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n"


def decode_message(line: str) -> dict[str, Any]:
    try:
        data = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(str(exc)) from exc
    if not isinstance(data, dict):
        raise ProtocolError("protocol message must be a JSON object")
    if not isinstance(data.get("kind"), str):
        raise ProtocolError("protocol message must include string kind")
    return data


def write_message(output: TextIO, kind: str, **payload: Any) -> None:
    output.write(encode_message(kind, **payload))
    output.flush()
