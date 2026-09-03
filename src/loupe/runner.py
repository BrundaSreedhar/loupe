"""Single entry point for running a review. The CLI and the eval harness both
go through here, so the thing being measured is the thing that ships."""

from __future__ import annotations

from typing import Literal

from .fallback import reset_usage, usage
from .graph import build_graph
from .quota import raise_if_terminal
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
    remember: bool = True,
    lint: bool | None = None,
) -> ReviewResult:
    reset_usage()
    try:
        final = _invoke(request, mode, verify, run_name, remember, lint)
    except Exception as exc:  # noqa: BLE001 — re-raised immediately; this only
        # decides which exception the caller sees.
        # A daily cap has to arrive as DailyQuotaExhausted whichever node hit it.
        # The nodes with broad excepts already translate it, but a reviewer branch
        # has none — it has no failure to swallow — so an exhausted quota came out
        # as the provider's own error, the eval harness read it as one bad case,
        # and ground through the rest of the corpus failing identically.
        raise_if_terminal(exc)
        raise
    return _assemble(request, mode, verify, final)


def _invoke(request, mode, verify, run_name, remember, lint):
    return _graph().invoke(
        {
            "request": request,
            "mode": mode,
            "verify": verify,
            "remember": remember,
            "lint": lint,
            "findings": [],
            "verdicts": [],
            "consensus": [],
            "problems": [],
        },
        config={
            "run_name": run_name or f"review:{request.ref}",
            "tags": [mode, "verified" if verify else "unverified"],
            "metadata": {
                "mode": mode,
                "verify": verify,
                "remember": remember,
                "lint": lint,
                "source": request.source,
                "ref": request.ref,
                "files": len(request.reviewable),
            },
            # N findings fan out in one superstep; the default limit is per-step,
            # but deep verify fan-outs on large diffs still want headroom.
            "recursion_limit": 100,
        },
    )

def _assemble(
    request: ReviewRequest, mode: str, verify: bool, final: dict
) -> ReviewResult:
    raw: list[Finding] = final.get("findings") or []
    merged: list[Finding] = final.get("merged") or []
    verdicts: list[Verdict] = final.get("verdicts") or []
    accepted: list[Finding] = final.get("accepted") or []
    delta = final.get("delta")
    problems: list[Problem] = list(final.get("problems") or [])
    fell_back = usage()
    if fell_back:
        problems.append(Problem(
            stage="fallback",
            detail=f"{len(fell_back)} stage(s) ran on the local model after the "
                   f"daily quota ran out ({', '.join(sorted(fell_back))}). These "
                   "findings come from a smaller model and are not comparable "
                   "with a normal run.",
            severity="error",
        ))

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
        delta=delta,
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
            "fallback_stages": len(fell_back),
            # Definitions pulled in from elsewhere in the repository. Zero means
            # the reviewers saw the diff and nothing else.
            "references": len(final.get("references") or []),
            "reconsidered": len(final.get("consensus") or []),
        },
    )
