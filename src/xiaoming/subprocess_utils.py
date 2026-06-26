from __future__ import annotations

import os
import subprocess
from typing import Any


def noninteractive_env(env: dict[str, str] | None = None) -> dict[str, str]:
    result = dict(os.environ if env is None else env)
    result["GIT_TERMINAL_PROMPT"] = "0"
    result["GIT_ASKPASS"] = "true"
    result["SSH_ASKPASS"] = "true"
    return result


def run_noninteractive(*popenargs: Any, **kwargs: Any) -> subprocess.CompletedProcess:
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    kwargs.setdefault("env", noninteractive_env(kwargs.get("env")))
    return subprocess.run(*popenargs, **kwargs)
