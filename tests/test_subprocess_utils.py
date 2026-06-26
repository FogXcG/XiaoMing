import subprocess

from xiaoming.subprocess_utils import noninteractive_env, run_noninteractive


def test_noninteractive_env_disables_git_terminal_prompts():
    env = noninteractive_env({"PATH": "/bin", "GIT_TERMINAL_PROMPT": "1"})

    assert env["PATH"] == "/bin"
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert env["GIT_ASKPASS"] == "true"
    assert env["SSH_ASKPASS"] == "true"


def test_run_noninteractive_closes_stdin_and_sets_env(monkeypatch):
    captured = {}

    def fake_run(*popenargs, **kwargs):
        captured["popenargs"] = popenargs
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(popenargs[0], 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_noninteractive(["git", "clone", "https://example.test/repo.git"])

    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert captured["kwargs"]["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert captured["kwargs"]["env"]["GIT_ASKPASS"] == "true"
    assert captured["kwargs"]["env"]["SSH_ASKPASS"] == "true"
