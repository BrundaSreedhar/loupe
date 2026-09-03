"""The pre-flight check. Runs before any file content leaves the machine."""

from __future__ import annotations

import logging

from ..config import ON_SECRET
from ..safety import SafetyIssue, redact_text, scan_text
from ..state import ReviewState

log = logging.getLogger(__name__)


class SecretsFound(RuntimeError):
    """Raised when credentials are found and the policy is to refuse."""


def preflight(state: ReviewState) -> dict:
    request = state["request"]
    issues: list[SafetyIssue] = []

    for fd in request.reviewable:
        if fd.content_after:
            issues.extend(scan_text(fd.content_after, fd.path))
    if request.body:
        issues.extend(scan_text(request.body, "<pull request description>"))

    secrets = [i for i in issues if i.kind == "secret"]
    injections = [i for i in issues if i.kind == "injection"]

    if secrets:
        if ON_SECRET == "block":  # noqa: S105 — a policy name, not a credential
            raise SecretsFound(
                f"{len(secrets)} possible credential(s) in the diff; refusing to send. "
                f"First: {secrets[0].path}:{secrets[0].line} ({secrets[0].label})."
            )
        if ON_SECRET == "redact":  # noqa: S105 — as above
            for fd in request.reviewable:
                if fd.content_after:
                    fd.content_after = redact_text(fd.content_after)
            log.warning("redacted %d possible credential(s) before review", len(secrets))
        else:
            log.warning("sending %d possible credential(s) to the model", len(secrets))

    if injections:
        # Not filtered out: the reviewer is told to treat source as data and to
        # report tampering, so leaving the text in place is what lets it do that.
        log.warning(
            "%d line(s) look like instructions aimed at the reviewer: %s",
            len(injections),
            ", ".join(f"{i.path}:{i.line}" for i in injections[:5]),
        )

    return {"safety": issues}
