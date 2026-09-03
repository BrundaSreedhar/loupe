"""Apply verdicts, rank, truncate."""

from __future__ import annotations

import logging

from ..config import MAX_REPORTED
from ..memory import diff, load, save
from ..schema import Finding
from ..state import ReviewState

log = logging.getLogger(__name__)

_SEVERITY_WEIGHT = {"high": 3.0, "medium": 2.0, "low": 1.0}


def rank_key(f: Finding) -> float:
    return _SEVERITY_WEIGHT[f.severity] * max(f.confidence, 0.01)


def finalize(state: ReviewState) -> dict:
    merged = state.get("merged") or []

    if state.get("verify"):
        verdicts = {v.finding_id: v for v in (state.get("verdicts") or [])}
        # A re-judged finding replaces its first verdict outright.
        verdicts.update({v.finding_id: v for v in (state.get("consensus") or [])})
        accepted: list[Finding] = []
        for f in merged:
            v = verdicts.get(f.id)
            if v is None or v.status != "CONFIRMED":
                continue
            if v.corrected_line is not None:
                f = f.model_copy(update={"line": v.corrected_line})
            accepted.append(f)
    else:
        accepted = list(merged)

    accepted.sort(key=rank_key, reverse=True)
    accepted = accepted[:MAX_REPORTED]

    request = state.get("request")
    if request is None or not state.get("remember", True):
        # No repo identity means nothing to remember against — and memory is a
        # convenience, never a precondition for producing a review.
        return {"accepted": accepted}

    repo, branch = request.identity
    previous = load(repo, branch)
    delta = diff(previous, accepted, first_run=not previous)
    save(repo, branch, accepted)

    if delta.resolved:
        log.info("%d finding(s) from the last review are gone", len(delta.resolved))
    if delta.persisting:
        log.info("%d finding(s) were already reported and are still open",
                 len(delta.persisting))

    return {"accepted": accepted, "delta": delta}
