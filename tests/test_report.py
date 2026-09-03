"""What the terminal shows.

Two of these exist because of the same complaint: a review printed "1 merged → 0
confirmed" and nothing else, so there was no way to find out what had been found
or why it was thrown away. The reasoning was in the result the whole time.
"""

from __future__ import annotations

import io

from rich.console import Console

from loupe.emit import render, render_changes, render_rejected
from loupe.index import Definition, Edge
from loupe.schema import Finding, ReviewRequest, ReviewResult, Verdict


def out(fn, result) -> str:
    console = Console(file=io.StringIO(), width=100, no_color=True)
    fn(result, console)
    return console.file.getvalue()


def finding(fid: str = "f1", line: int = 6) -> Finding:
    return Finding(
        file="app/handlers.py", line=line, category="correctness", severity="high",
        summary="validate() may return a non-numeric id",
        failure_scenario="A string id reaches to_cents and multiplies a str by 100.",
        confidence=0.85, evidence="    checked = validate(payload)",
        id=fid, produced_by="correctness",
    )


def rejected_result() -> ReviewResult:
    f = finding()
    return ReviewResult(
        request_ref="HEAD", mode="multi", verified=True,
        raw=[f], merged=[f], accepted=[],
        verdicts=[Verdict(
            finding_id="f1", status="REJECTED",
            reasoning="Nothing in this file shows a non-numeric id reaching to_cents.",
        )],
    )


def test_a_rejected_finding_says_what_it_was_and_why_it_went():
    """The complaint this was written for: 1 merged, 0 confirmed, and no way to
    see either the claim or the reasoning that killed it."""
    text = out(render_rejected, rejected_result())

    assert "gate rejected 1 finding" in text
    assert "app/handlers.py:6" in text
    assert "validate() may return a non-numeric id" in text
    assert "non-numeric id reaching to_cents" in text


def test_an_accepted_finding_is_not_also_listed_as_rejected():
    f = finding()
    result = ReviewResult(
        request_ref="HEAD", mode="multi", verified=True, raw=[f], merged=[f], accepted=[f],
        verdicts=[Verdict(finding_id="f1", status="CONFIRMED", reasoning="real")],
    )
    assert out(render_rejected, result) == ""


def test_an_unverified_run_has_no_rejections_to_show():
    """With the gate off nothing was rejected, so silence is correct rather than
    a rendering that quietly drops the section."""
    f = finding()
    result = ReviewResult(request_ref="HEAD", mode="multi", verified=False,
                          raw=[f], merged=[f], accepted=[f])
    assert out(render_rejected, result) == ""


def _edge(caller: str, name: str, path: str, shown: bool = True) -> Edge:
    return Edge(
        caller=caller, name=name, shown=shown,
        definition=Definition(name=name, kind="function", path=path, start=12, end=20,
                              module=path.replace("/", ".").removesuffix(".py")),
    )


def test_the_change_map_shows_what_each_changed_file_calls():
    result = ReviewResult(
        request_ref="HEAD", mode="multi", verified=True,
        edges=[_edge("app/handlers.py", "validate", "app/validators.py")],
    )
    text = out(render_changes, result)

    assert "What this change reaches" in text
    assert "app/handlers.py" in text
    assert "validate()" in text
    assert "app/validators.py:12" in text


def test_a_call_into_another_changed_file_is_called_out():
    """The case worth seeing: two files in one change, one calling the other. A
    defect spanning them is invisible in either file alone."""
    result = ReviewResult(
        request_ref="HEAD", mode="multi", verified=True,
        edges=[
            _edge("app/handlers.py", "to_cents", "app/money.py"),
            _edge("app/money.py", "round_half_up", "app/rounding.py"),
        ],
    )
    text = out(render_changes, result)

    # app/money.py is both a caller and a callee, so it is part of the change.
    assert "also changed here" in text


def test_no_resolved_calls_draws_no_map():
    """A change with nothing to draw prints nothing, rather than an empty box."""
    result = ReviewResult(request_ref="HEAD", mode="multi", verified=True)
    assert out(render_changes, result) == ""


def test_a_silent_review_still_explains_itself():
    """End to end: no findings reported, but the reader can see one was filed and
    read the reason it was dropped."""
    console = Console(file=io.StringIO(), width=100, no_color=True)
    render(rejected_result(), ReviewRequest(source="local", ref="HEAD"), console)
    text = console.file.getvalue()

    assert "No findings" in text
    assert "gate rejected" in text
    assert "non-numeric id reaching to_cents" in text
