"""Narration while the review runs.

Driven through the handler's own interface rather than through a graph run: these
are assertions about what a stage says, and a fake graph would only be testing
LangGraph's event names back at itself. The event names themselves are covered by
the end-to-end test, which runs the real graph.
"""

from __future__ import annotations

import io
from uuid import uuid4

from rich.console import Console

from loupe.index import Definition, Edge
from loupe.lint import LintIssue
from loupe.progress import Reporter, reporter
from loupe.safety import SafetyIssue
from loupe.schema import FileContext, Verdict


def run(node: str, inputs: dict, outputs: dict) -> str:
    """One stage, start to finish, and everything it printed."""
    console = Console(file=io.StringIO(), width=120, no_color=True)
    handler = Reporter(console)
    run_id = uuid4()
    handler.on_chain_start({}, inputs, run_id=run_id, name=node)
    handler.on_chain_end(outputs, run_id=run_id, name=node)
    return console.file.getvalue()


def test_each_reviewer_is_named_and_says_what_it_found():
    text = run("specialist", {"role": "security", "contexts": {"a.py": 1, "b.py": 2}},
               {"findings": [1, 2]})
    assert "security" in text
    assert "2 files" in text
    assert "2 findings" in text


def test_a_reviewer_that_found_nothing_says_so():
    """An empty list is the correct review of clean code, and has to look
    different from a reviewer that never ran."""
    assert "nothing" in run("specialist", {"role": "performance", "contexts": {"a.py": 1}},
                            {"findings": []})


def test_verification_names_the_file_and_the_outcome():
    verdicts = [Verdict(finding_id="a", status="CONFIRMED", reasoning="r"),
                Verdict(finding_id="b", status="REJECTED", reasoning="r")]
    text = run("verify", {"path": "src/app.py", "findings": [1, 2]}, {"verdicts": verdicts})

    assert "src/app.py" in text
    assert "2 claims" in text
    assert "1 stood up, 1 rejected" in text


def test_everything_rejected_reads_as_rejection_not_as_silence():
    text = run("verify", {"path": "src/app.py", "findings": [1]},
               {"verdicts": [Verdict(finding_id="a", status="REJECTED", reasoning="r")]})
    assert "nothing stood up" in text


def _ctx(path: str, definitions: list[str] | None = None) -> FileContext:
    return FileContext(
        path=path,
        content="",
        strategy="whole_file",
        tokens=1,
        definitions=definitions or [],
    )


def test_the_files_being_reviewed_are_named():
    text = run(
        "prepare", {}, {"contexts": {"src/a.py": _ctx("src/a.py"), "src/b.py": _ctx("src/b.py")},
                        "dropped": []}
    )
    assert "2 files" in text
    assert "src/a.py" in text and "src/b.py" in text


def test_a_file_dropped_for_budget_is_not_passed_over_in_silence():
    text = run("prepare", {}, {"contexts": {"src/a.py": _ctx("src/a.py")},
                               "dropped": ["src/huge.py"]})
    assert "over budget" in text


def test_the_definitions_a_change_touched_are_counted():
    text = run(
        "prepare",
        {},
        {
            "contexts": {
                "src/a.py": _ctx("src/a.py", ["save", "load"]),
                "src/b.py": _ctx("src/b.py", ["run"]),
            },
            "dropped": [],
        },
    )
    assert "touching 3 definitions" in text


def test_a_change_touching_no_definition_says_nothing_about_definitions():
    """A diff of imports and constants touches none. "touching 0 definitions" is
    noise; saying nothing is the honest rendering."""
    text = run("prepare", {}, {"contexts": {"src/a.py": _ctx("src/a.py")}, "dropped": []})
    assert "definition" not in text


def test_the_secret_scan_reports_both_outcomes():
    assert "nothing" in run("preflight", {}, {"safety": []})

    found = run("preflight", {}, {"safety": [
        SafetyIssue("secret", "AWS key", "a.py", 3, "..."),
        SafetyIssue("injection", "instruction override", "b.py", 9, "..."),
    ]})
    assert "credential" in found
    assert "written at the reviewer" in found


def test_following_calls_out_of_the_diff_is_reported():
    defn = Definition(name="validate", kind="function", path="app/validators.py",
                      start=1, end=3, module="app.validators")
    text = run("expand", {}, {
        "edges": [Edge(caller="app/handlers.py", name="validate", definition=defn, shown=True)],
        "references": [object()],
    })
    assert "1 call" in text
    assert "1 file" in text


def test_the_linter_pre_pass_names_its_tool():
    text = run("lint", {}, {"lint_issues": [LintIssue("ruff", "a.py", 1, "F401", "unused")]})
    assert "ruff" in text
    assert "1 issue" in text


def test_a_stage_with_no_news_announces_itself_but_claims_nothing():
    """Every stage says it is running — that is the point of the spinner, and in
    a pipe the same text is printed instead. What a stage must not do is invent a
    result: a linter that found nothing reports nothing."""
    text = run("lint", {}, {"lint_issues": []})
    assert text.strip(), "the stage never said it was running"
    assert "issue" not in text

    text = run("expand", {}, {"edges": [], "references": []})
    assert text.strip()
    assert "definition" not in text


def test_every_stage_names_a_spinner_rich_actually_has():
    """An unknown spinner name raises at the first frame, which would take the
    review down for the sake of decoration."""
    from rich.spinner import SPINNERS

    from loupe.progress import _SPINNERS

    assert set(_SPINNERS.values()) <= set(SPINNERS)


def test_each_reviewer_keeps_one_colour():
    """Four branches run at once. Telling them apart is the point of the colour,
    so the spinner and the line it leaves behind have to agree."""
    from loupe.progress import _ROLE_STYLE

    console = Console(file=io.StringIO(), width=120)
    handler = Reporter(console)
    line = handler._after_specialist({"role": "security"}, {"findings": []})

    assert _ROLE_STYLE["security"] in line
    assert set(_ROLE_STYLE) >= {"security", "correctness", "performance", "maintainability"}


def test_the_last_line_says_whether_anything_survived():
    assert "nothing survived" in run("finalize", {}, {"accepted": []})
    assert "2 findings" in run("finalize", {}, {"accepted": [1, 2]})


def test_nobody_watching_means_no_callbacks():
    """The eval harness runs this graph thousands of times and wants none of it,
    and `--output json` would be corrupted by narration on stdout."""
    from loupe.progress import callbacks

    assert reporter(None) is None
    assert callbacks(None) == []
    assert len(callbacks(reporter(Console()))) == 1
