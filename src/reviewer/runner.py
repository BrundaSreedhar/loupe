"""Single entry point for running a review. The CLI and the eval harness both
go through here, so the thing being measured is the thing that ships."""

from __future__ import annotations

from typing import Literal

from .graph import build_graph
from .schema import Finding, Problem, ReviewRequest, ReviewResult, Verdict

_GRAPH = None


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_review(
    request: ReviewRequest,
    mode: Literal["single", "multi"] = "multi",
    verify: bool = True,
    run_name: str | None = None,
) -> ReviewResult:
    final = _graph().invoke(
        {
            "request": request,
            "mode": mode,
            "verify": verify,
            "findings": [],
            "verdicts": [],
            "problems": [],
        },
        config={
            "run_name": run_name or f"review:{request.ref}",
            "tags": [mode, "verified" if verify else "unverified"],
            "metadata": {
                "mode": mode,
                "verify": verify,
                "source": request.source,
                "ref": request.ref,
                "files": len(request.reviewable),
            },
            # N findings fan out in one superstep; the default limit is per-step,
            # but deep verify fan-outs on large diffs still want headroom.
            "recursion_limit": 100,
        },
    )

    raw: list[Finding] = final.get("findings") or []
    merged: list[Finding] = final.get("merged") or []
    verdicts: list[Verdict] = final.get("verdicts") or []
    accepted: list[Finding] = final.get("accepted") or []
    problems: list[Problem] = final.get("problems") or []

    confirmed = sum(1 for v in verdicts if v.status == "CONFIRMED")
    return ReviewResult(
        request_ref=request.ref,
        mode=mode,
        verified=verify,
        raw=raw,
        merged=merged,
        accepted=accepted,
        verdicts=verdicts,
        problems=problems,
        usage={
            "raw_count": len(raw),
            "merged_count": len(merged),
            "accepted_count": len(accepted),
            "merge_rate": 1 - (len(merged) / len(raw)) if raw else 0.0,
            "rejection_rate": 1 - (confirmed / len(verdicts)) if verdicts else 0.0,
            "dropped_files": len(final.get("dropped") or []),
            # How many files the reviewers were actually shown. Zero means the
            # review never happened, which must not be reported as "found nothing".
            "contexts": len(final.get("contexts") or {}),
        },
    )
