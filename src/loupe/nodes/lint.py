"""Run the project's own linters over the files being reviewed."""

from __future__ import annotations

import logging

from ..config import lint_enabled
from ..lint import available, run
from ..schema import Problem
from ..state import ReviewState

log = logging.getLogger(__name__)


def lint(state: ReviewState) -> dict:
    request = state["request"]
    if not lint_enabled(request.source):
        log.info("lint pre-pass off for a %s review", request.source)
        return {"lint_issues": []}

    paths = list(state.get("contexts") or {})
    if not paths:
        return {"lint_issues": []}

    tools = available(request.repo_root)
    if not tools:
        log.info("no linters found for this repo; nothing to pre-check")
        return {"lint_issues": []}

    issues = run(paths, request.repo_root)
    log.info("%s found %d issue(s) the reviewers will be told to skip",
             "/".join(tools), len(issues))
    problems = []
    if issues:
        problems.append(Problem(
            stage="lint",
            detail=f"{len(issues)} issue(s) already found by {'/'.join(tools)} — "
                   "reviewers were told not to repeat them",
        ))
    return {"lint_issues": issues, "problems": problems}
