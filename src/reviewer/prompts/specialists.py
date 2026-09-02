"""Reviewer prompt construction.

Message order matters here. Prompt caching is a prefix match over
tools -> system -> messages, so anything that differs per role must come AFTER
the shared context or every branch misses the cache. Hence the role rubric is the
last message rather than the system prompt: system and context stay byte-identical
across all four reviewers, so one warm write serves all of them.
"""

from __future__ import annotations

from ..schema import FileContext, ReviewRequest
from .rubrics import RUBRICS

SHARED_SYSTEM = """\
You are a reviewer on a code review panel. You will be shown the files a change
touched, then told which area you are responsible for.

HOW THE SOURCE IS PRESENTED
Lines appear as `>123| code` for lines this change touched, and ` 123| code` for
surrounding context. Only the `>` lines are new. Windowed files mark omitted
regions explicitly.

HOW TO REPORT
- Report a defect on unchanged code only when a changed line breaks it, and anchor
  to the line that actually fails.
- `line` must be a real line number from the context you were given.
- `failure_scenario` must name concrete inputs or state and the wrong behaviour
  that results. "Could cause problems" is not a failure scenario. If you cannot
  write one, drop the finding.
- `confidence` is your honest probability that the defect is real. It is measured
  against an independent verification pass, so inflating it makes you look worse.
- Never report speculation about code you were not shown. If a called function's
  body is not in your context, you do not know what it does.

Report nothing if you find nothing. An empty list is the correct review of clean
code, and is always better than a padded one."""


def context_message(request: ReviewRequest, contexts: dict[str, FileContext]) -> str:
    """Identical for every role — this is the cached prefix."""
    header = [f"Change under review: {request.title or request.ref}"]
    if request.body:
        header.append(f"\nDescription:\n{request.body.strip()[:2000]}")
    header.append(f"\n{len(contexts)} file(s) changed.\n")

    blocks = [
        f"=== {path}{' (windowed — omitted regions marked)' if ctx.truncated else ''} ===\n"
        f"{ctx.content}"
        for path, ctx in contexts.items()
    ]
    return "\n".join(header) + "\n" + "\n\n".join(blocks)


def role_message(role: str) -> str:
    """Role-specific, and deliberately last so it sits outside the cached prefix."""
    rubric = RUBRICS[role]
    panel = (
        "Three other reviewers are covering the other areas — stay in yours, and "
        "trust them to cover theirs.\n\n"
        if role != "generalist"
        else ""
    )
    return f"""\
You are the {role} reviewer. {panel}
WHAT YOU LOOK FOR
{rubric["looks_for"]}

WHAT YOU MUST NOT REPORT
{rubric["non_goals"]}

Review the change above and report your findings."""
