from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEvent:
    kind: str
    message: str
    end: str = "\n"
