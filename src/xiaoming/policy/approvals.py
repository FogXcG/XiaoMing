from __future__ import annotations

from collections.abc import Callable


ApprovalCallback = Callable[[str], bool]
