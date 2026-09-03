"""Errors that are handled must still be visible.

Every failure in this pipeline is caught so one bad stage cannot kill a review.
That is right, and it was also how a review that lost most of its work came to
look exactly like a clean one.
"""

from __future__ import annotations

import logging

from rich.console import Console

from agentgate.emit import render
from agentgate.logs import setup_logging
from agentgate.schema import Finding, Problem, ReviewRequest, ReviewResult


def _result(**kw) -> ReviewResult:
    base = {"request_ref": "r", "mode": "multi", "verified": True}
    return ReviewResult(**{**base, **kw})


def _render(result) -> str:
    console = Console(width=100, no_color=True, force_terminal=False)
    with console.capture() as cap:
        render(result, ReviewRequest(source="local", ref="r"), console)
    return cap.get()


def test_clean_review_says_no_findings():
    out = _render(_result(usage={"raw_count": 3, "merged_count": 2}))
    assert "No findings." in out
    assert "did not complete cleanly" not in out


def test_a_review_that_failed_does_not_look_clean():
    """The bug this exists to prevent: every stage erroring produced an empty
    finding list, which rendered as a reassuring green 'No findings.'"""
    out = _render(_result(problems=[
        Problem(stage="verify", detail="could not verify 4 findings", severity="error")
    ]))
    assert "No findings." not in out
    assert "did not complete cleanly" in out
    assert "could not verify 4 findings" in out


def test_problems_are_shown_alongside_findings():
    finding = Finding(
        id="f1", produced_by="correctness", file="a.py", line=1,
        category="correctness", severity="high", summary="bug",
        failure_scenario="concrete", confidence=0.9,
    )
    out = _render(_result(
        accepted=[finding],
        problems=[Problem(stage="dedupe", detail="could not merge 2 findings")],
        usage={"raw_count": 2, "merged_count": 2, "accepted_count": 1},
    ))
    assert "bug" in out
    assert "Problems during this review" in out
    assert "could not merge 2 findings" in out


def test_problems_use_a_reducer_so_concurrent_failures_are_not_lost():
    """Several nodes fail at once. A plain assignment keeps only the last."""
    import operator
    from typing import get_args, get_type_hints

    from agentgate.state import ReviewState

    hints = get_type_hints(ReviewState, include_extras=True)
    assert operator.add in get_args(hints["problems"])


def test_setup_logging_quiets_http_chatter_by_default():
    """At INFO, httpx logs a line per request and buries everything about the
    review itself."""
    setup_logging(1)
    assert logging.getLogger("agentgate").level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING
    setup_logging(3)
    assert logging.getLogger("httpx").level == logging.DEBUG
    setup_logging(0)
