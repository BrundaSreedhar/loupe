"""Merge findings that describe the same defect.

Runs before verification on purpose: verification is one model call per finding,
so every duplicate removed here is a call not made. Positional grouping is done
first, deterministically and for free, so the model is only asked about findings
that are actually near each other.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import MERGE_LINE_WINDOW, merger_llm
from ..prompts.merge import SYSTEM, user_prompt
from ..quota import raise_if_terminal
from ..schema import Finding, MergeResult, Problem
from ..state import ReviewState

log = logging.getLogger(__name__)


def group_by_locality(findings: list[Finding]) -> list[list[Finding]]:
    by_file: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        by_file[f.file].append(f)

    groups: list[list[Finding]] = []
    for _, fs in sorted(by_file.items()):
        fs.sort(key=lambda x: x.line)
        current = [fs[0]]
        for f in fs[1:]:
            if f.line - current[-1].line <= MERGE_LINE_WINDOW:
                current.append(f)
            else:
                groups.append(current)
                current = [f]
        groups.append(current)
    return groups


def dedupe(state: ReviewState) -> dict:
    findings = state.get("findings") or []
    if not findings:
        return {"merged": []}

    llm = merger_llm().with_structured_output(MergeResult, method="json_schema")
    merged: list[Finding] = []
    problems: list[Problem] = []

    for group in group_by_locality(findings):
        if len(group) == 1:
            merged.append(group[0])
            continue
        try:
            result: MergeResult = llm.invoke(
                [SystemMessage(SYSTEM), HumanMessage(user_prompt(group))],
                config={
                    "run_name": f"merge:{group[0].file}:{group[0].line}",
                    "tags": ["dedupe"],
                    "metadata": {"group_size": len(group)},
                },
            )
        except Exception as exc:  # noqa: BLE001 — a failed merge must not lose
            # findings; keep them all, unmerged. A quota failure still propagates.
            raise_if_terminal(exc)
            log.warning("merge failed at %s:%d (%s); keeping %d finding(s) unmerged",
                        group[0].file, group[0].line, type(exc).__name__, len(group))
            problems.append(Problem(
                stage="dedupe",
                detail=f"could not merge {len(group)} finding(s) at {group[0].file}:"
                       f"{group[0].line} — they may appear as duplicates",
            ))
            merged.extend(group)
            continue

        for m in result.findings:
            covered = [group[i] for i in m.covers if 0 <= i < len(group)] or group
            payload = m.model_dump(exclude={"covers"})
            merged.append(
                Finding(
                    id=covered[0].id,
                    produced_by=covered[0].produced_by,
                    merged_from=[c.id for c in covered],
                    **payload,
                )
            )

    log.info("dedupe: %d finding(s) -> %d", len(findings), len(merged))
    return {"merged": merged, "problems": problems}
