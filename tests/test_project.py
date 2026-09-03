"""Per-project config in a `.reviewer/` folder."""

from __future__ import annotations

import pytest

from reviewer.project import find_dir, is_ignored, load_ignore, load_rules


@pytest.fixture
def project(tmp_path):
    d = tmp_path / ".reviewer"
    d.mkdir()
    (d / "rules.md").write_text("- Money is always integer cents.\n")
    (d / "ignore").write_text("# generated\nsrc/generated/*\n*.pb.ts\n")
    return tmp_path


def test_found_from_a_subdirectory(project):
    deep = project / "src" / "lib" / "deep"
    deep.mkdir(parents=True)
    assert find_dir(deep) == project / ".reviewer"


def test_absent_when_there_is_none(tmp_path):
    assert find_dir(tmp_path) is None
    assert load_rules(tmp_path) is None
    assert load_ignore(tmp_path) == []


def test_rules_are_read(project):
    assert "integer cents" in load_rules(project)


def test_empty_rules_file_is_treated_as_none(project):
    (project / ".reviewer" / "rules.md").write_text("   \n")
    assert load_rules(project) is None


def test_oversized_rules_are_truncated(project):
    from reviewer.project import MAX_RULES_CHARS

    (project / ".reviewer" / "rules.md").write_text("x" * (MAX_RULES_CHARS + 500))
    assert len(load_rules(project)) == MAX_RULES_CHARS


def test_ignore_skips_comments_and_blanks(project):
    assert load_ignore(project) == ["src/generated/*", "*.pb.ts"]


@pytest.mark.parametrize("path,expected", [
    ("src/generated/api.py", True),
    ("types/schema.pb.ts", True),
    ("src/app.py", False),
    ("src/generated.py", False),
])
def test_ignore_matching(path, expected, project):
    assert is_ignored(path, load_ignore(project)) is expected


def test_rules_are_framed_as_configuration_not_file_content(project):
    """rules.md is written by maintainers, but it still ends up in a prompt next
    to untrusted source. It must be labelled as the former."""
    from reviewer.prompts.specialists import role_message

    msg = role_message("security", load_rules(project))
    assert "integer cents" in msg
    assert "not content from the files under review" in msg


def test_rules_go_in_the_role_message_not_the_shared_prefix(project):
    """Putting per-project rules in the system prompt would give every reviewer a
    different cached prefix and defeat the cache warm."""
    from reviewer.prompts.specialists import SHARED_SYSTEM, role_message

    assert "integer cents" not in SHARED_SYSTEM
    assert "integer cents" in role_message("security", load_rules(project))
