"""Errors that are handled must still be visible.

Every failure in this pipeline is caught so one bad stage cannot kill a review.
That is right, and it was also how a review that lost most of its work came to
look exactly like a clean one.
"""

from __future__ import annotations

import logging

from rich.console import Console

from loupe.emit import render
from loupe.logs import setup_logging
from loupe.schema import Finding, Problem, ReviewRequest, ReviewResult


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

    from loupe.state import ReviewState

    hints = get_type_hints(ReviewState, include_extras=True)
    assert operator.add in get_args(hints["problems"])


def test_setup_logging_quiets_http_chatter_by_default():
    """At INFO, httpx logs a line per request and buries everything about the
    review itself."""
    setup_logging(1)
    assert logging.getLogger("loupe").level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING
    setup_logging(3)
    assert logging.getLogger("httpx").level == logging.DEBUG
    setup_logging(0)


def test_an_unlisted_dependency_is_quiet_by_default():
    """The bug: naming noisy libraries one at a time only silences the known ones.

    The Anthropic SDK talks through `httpx2`, which was not on the list, so every
    model call printed an HTTP request line through a running spinner. Root sits at
    WARNING so anything not explicitly raised is quiet, whatever it is called.
    """
    import io

    console = Console(file=io.StringIO(), width=120, no_color=True)
    setup_logging(1, console=console)

    logging.getLogger("httpx2").info('HTTP Request: POST https://api.anthropic.com "200 OK"')
    logging.getLogger("some.brand.new.dependency").info("chatter nobody asked for")
    logging.getLogger("loupe.nodes.specialists").info("a line about your review")

    out = console.file.getvalue()
    assert "api.anthropic.com" not in out
    assert "chatter nobody asked for" not in out
    # ...and ours still gets through, which is the half that makes -v worth typing.
    assert "a line about your review" in out
    setup_logging(0)


def test_the_log_handler_shares_the_console_it_was_given():
    """Rich only coordinates a live region with its own Console. A second one writes
    straight to the terminal at wherever the cursor is, which is mid-spinner."""
    import io

    from rich.logging import RichHandler

    console = Console(file=io.StringIO(), width=120, no_color=True)
    setup_logging(1, console=console)
    handlers = [h for h in logging.getLogger().handlers if isinstance(h, RichHandler)]
    assert handlers, "no RichHandler installed"
    assert handlers[0].console is console
    setup_logging(0)


def test_dependency_chatter_is_dropped_but_real_warnings_survive():
    """google_genai warns on every call about how langchain calls it. The user
    cannot act on it and it fires constantly; a genuine SDK warning must still
    get through, so it is filtered by message, not by silencing the logger."""
    import io

    console = Console(file=io.StringIO(), width=100, no_color=True)
    setup_logging(1, console=console)
    log = logging.getLogger("google_genai.models")

    log.warning(
        "Direct use of automatic function calling (AFC) in Models.generate_content "
        "is not recommended."
    )
    log.warning("Your API key will expire in 3 days.")

    out = console.file.getvalue()
    assert "automatic function calling" not in out
    assert "API key will expire" in out
    setup_logging(0)


def test_everything_shows_at_maximum_verbosity():
    import io

    console = Console(file=io.StringIO(), width=100, no_color=True)
    setup_logging(3, console=console)
    logging.getLogger("google_genai.models").warning(
        "Direct use of automatic function calling (AFC) is not recommended."
    )
    assert "automatic function calling" in console.file.getvalue()
    setup_logging(0)
