"""Look up what the changed lines call.

Filesystem and parser work only — no model call — so this is cheap and its failure
mode is "the reviewers see what they saw before", never a lost review. It runs
before `warm_cache` because the definitions it finds go into the cached prefix,
and a prefix that changes after the warm is a prefix nobody hits.
"""

from __future__ import annotations

import logging

from ..config import (
    DEF_MAX_LINES,
    INDEX,
    INDEX_MAX_FILES,
    INDEX_TOKEN_BUDGET,
    ON_SECRET,
)
from ..context import estimate
from ..index import enabled, gather
from ..schema import Problem
from ..state import ReviewState
from ._common import valid_lines

log = logging.getLogger(__name__)


def expand(state: ReviewState) -> dict:
    request = state["request"]
    contexts = state.get("contexts") or {}
    if not contexts or not enabled(INDEX, request.source):
        return {}

    try:
        found = gather(
            request,
            contexts,
            max_files=INDEX_MAX_FILES,
            max_lines=DEF_MAX_LINES,
            token_budget=INDEX_TOKEN_BUDGET,
            estimate=estimate,
            visible_lines=valid_lines,
            on_secret=ON_SECRET,
        )
        references, stats, edges = found.references, found.stats, found.edges
    except Exception as exc:  # noqa: BLE001 — expansion is an improvement to the
        # context, never a precondition for reviewing. Losing it costs recall on
        # cross-file defects and nothing else, so it must not fail the review.
        log.warning("could not index the repository (%s); reviewing the diff alone", exc)
        return {"problems": [Problem(
            stage="expand",
            detail=f"could not read the repository to look up called definitions "
                   f"({type(exc).__name__}) — reviewers saw the changed files only",
        )]}

    if not stats.files_indexed:
        # No repository to read, but the changed files parsed fine — and what this
        # change edited does not depend on the index.
        return {"changed": found.changed}

    log.info(
        "index: %d file(s)%s · %d name(s) called from changed lines → "
        "%d shown, %d ambiguous, %d not in this repo",
        stats.files_indexed,
        " (capped)" if stats.partial_index else "",
        stats.names,
        stats.resolved,
        stats.ambiguous,
        stats.unresolved,
    )
    for ref in references:
        log.debug("showing %s (called %d×)", ref.definition.label, ref.callers)

    problems: list[Problem] = []
    if stats.secrets:
        # Worth saying out loud: these files are not part of the change, so this
        # is a credential the person running the review did not put in front of it.
        detail = {
            "block": "left out of the review entirely",
            "redact": "blanked before sending",
            "warn": "sent to the model as-is",
        }.get(ON_SECRET, "handled by policy")
        problems.append(Problem(
            stage="expand",
            detail=f"{stats.secrets} referenced definition(s) contained something "
                   f"that looks like a credential — {detail}",
        ))
    if stats.over_budget:
        problems.append(Problem(
            stage="expand",
            detail=f"{stats.over_budget} called definition(s) found but not shown "
                   f"(over LOUPE_INDEX_TOKEN_BUDGET)",
        ))

    return {
        "references": references,
        "edges": edges,
        "summaries": found.summaries,
        "changed": found.changed,
        "problems": problems,
    }
