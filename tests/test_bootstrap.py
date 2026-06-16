from pathlib import Path

from xiaoming.bootstrap import discover_bootstrap_contexts


def test_discovers_superpowers_using_superpowers_bootstrap(tmp_path: Path):
    skill_dir = tmp_path / ".agents" / "skills" / "superpowers" / "skills" / "using-superpowers"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: using-superpowers\n---\nUse skills first.\n")

    contexts = discover_bootstrap_contexts(tmp_path)

    assert len(contexts) == 1
    assert contexts[0].plugin_name == "superpowers"
    assert contexts[0].source == "superpowers:using-superpowers"
    assert "You have superpowers." in contexts[0].content
    assert "Use skills first." in contexts[0].content
