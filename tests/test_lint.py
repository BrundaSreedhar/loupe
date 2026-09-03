"""The linter pre-pass.

Two things it must get right: never crash a review, and never run someone else's
tooling without being asked.
"""

from __future__ import annotations

import pytest

from loupe.lint import LintIssue, as_prompt_section


def _issue(**kw):
    base = {"tool": "ruff", "path": "a.py", "line": 12, "code": "F841",
            "message": "unused variable"}
    return LintIssue(**{**base, **kw})


def test_no_section_when_nothing_was_found():
    assert as_prompt_section([]) == ""


def test_section_names_the_issues_and_forbids_repeating_them():
    section = as_prompt_section([_issue()])
    assert "a.py:12" in section and "F841" in section
    assert "Do not report any of these" in section
    assert "different words" in section, "paraphrased duplicates must be excluded too"


@pytest.mark.parametrize("policy,source,expected", [
    ("auto", "local", True),
    ("auto", "github", False),   # someone else's branch: do not execute their config
    ("on", "github", True),
    ("off", "local", False),
])
def test_when_linting_runs(policy, source, expected, monkeypatch, isolated_config):
    monkeypatch.setenv("LOUPE_LINT", policy)
    assert isolated_config().lint_enabled(source) is expected


def test_a_broken_linter_does_not_stop_the_review(monkeypatch):
    """A linter that raises is a lost precision opportunity, not a failed review."""
    import loupe.lint as lint_mod

    def boom(paths, root):
        raise RuntimeError("eslint exploded")

    monkeypatch.setattr(lint_mod, "RUNNERS", (boom,))
    assert lint_mod.run(["a.py"], ".") == []


def test_output_is_capped(monkeypatch):
    """A repo with thousands of pre-existing violations must not push the diff out
    of the context window."""
    import loupe.lint as lint_mod

    monkeypatch.setattr(lint_mod, "MAX_ISSUES", 5)
    monkeypatch.setattr(lint_mod, "RUNNERS", (lambda p, r: [_issue() for _ in range(50)],))
    assert len(lint_mod.run(["a.py"], ".")) == 5


def test_node_skips_entirely_for_a_pull_request(monkeypatch, isolated_config):
    """The safety property: reviewing a PR must not execute that PR's tooling."""
    monkeypatch.setenv("LOUPE_LINT", "auto")
    isolated_config()

    import loupe.lint as lint_mod
    from loupe.nodes.lint import lint
    from loupe.schema import FileContext, ReviewRequest

    called = []
    monkeypatch.setattr(lint_mod, "run", lambda *a, **k: called.append(1) or [])

    ctx = {"a.py": FileContext(path="a.py", content="x", strategy="whole_file", tokens=1)}
    out = lint({"request": ReviewRequest(source="github", ref="r"), "contexts": ctx})
    assert out["lint_issues"] == []
    assert called == [], "a PR review must not execute the repo's linters"


def test_real_linter_finds_a_real_problem(tmp_path):
    """End to end against ruff, if it is installed."""
    from loupe.lint import available, run

    if "ruff" not in available(str(tmp_path)):
        pytest.skip("ruff not installed")
    (tmp_path / "bad.py").write_text("def f():\n    unused = 1\n    return 2\n")
    issues = run(["bad.py"], str(tmp_path))
    assert any(i.code == "F841" for i in issues), [i.render() for i in issues]
