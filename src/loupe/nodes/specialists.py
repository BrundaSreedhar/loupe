"""One reviewer branch. Reached via Send(), once per role."""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import GROUNDING, GROUNDING_WINDOW, specialist_llm, structured
from ..fingerprint import compute as fingerprint
from ..grounding import apply as check_citations
from ..lint import as_prompt_section
from ..project import load_rules
from ..prompts.specialists import SHARED_SYSTEM, context_message, role_message
from ..schema import Finding, FindingBatch, Problem, validate_fix
from ..state import SpecialistTask
from ._common import cached_block, drop_ungrounded

log = logging.getLogger(__name__)


def specialist(task: SpecialistTask) -> dict:
    role = task["role"]
    contexts = task["contexts"]
    if not contexts:
        return {"findings": []}

    llm = structured(specialist_llm(), FindingBatch, f"specialist:{role}")
    batch: FindingBatch = llm.invoke(
        [
            SystemMessage(SHARED_SYSTEM),
            HumanMessage(content=cached_block(
                context_message(task["request"], contexts, task.get("references") or [])
            )),
            HumanMessage(
                role_message(
                    role,
                    load_rules(task["request"].repo_root),
                    as_prompt_section(task.get("lint_issues") or []),
                )
            ),
        ],
        config={
            "run_name": f"specialist:{role}",
            "tags": ["specialist", role],
            "metadata": {"role": role, "files": len(contexts)},
        },
    )

    findings = [
        Finding(id=f"{role[:4]}-{uuid4().hex[:8]}", produced_by=role, **raw.model_dump())
        for raw in batch.findings
    ]
    sources = {
        f.path: f.content_after for f in task["request"].files if f.content_after is not None
    }
    problems: list[Problem] = []

    # Order matters through the next three steps. Both checks can change or
    # discard a finding, and the fingerprint is taken over the code around its
    # line — so it has to be computed last, on the line the finding ends up at.
    findings, dropped = drop_ungrounded(findings, contexts)
    if dropped:
        log.warning("%s reviewer: dropped %d finding(s) pointing at code it was "
                    "not shown: %s", role, len(dropped), "; ".join(dropped[:3]))
        problems.append(Problem(
            stage=f"specialist:{role}",
            detail=f"{len(dropped)} finding(s) discarded — they referenced files or "
                   f"lines outside the review context",
        ))

    cited = None
    if GROUNDING:
        cited = check_citations(findings, sources, GROUNDING_WINDOW)
        findings = cited.kept
        if cited.skipped:
            # Every finding came back with no citation at all. That is the model
            # failing to fill a field, not a batch of inventions, and dropping
            # them would turn a provider-side failure into a clean review.
            log.warning("%s reviewer: no finding carried a quoted line; the "
                        "citation check did not run", role)
            problems.append(Problem(
                stage=f"specialist:{role}",
                detail=f"{cited.missing} finding(s) came back with no quoted source "
                       "line, so they could not be checked against the file — they "
                       "are reported unchecked",
                severity="error",
            ))
        elif cited.dropped:
            log.warning("%s reviewer: dropped %d finding(s) whose quoted line is "
                        "not in the file: %s", role, cited.dropped,
                        "; ".join(cited.reasons[:3]))
            problems.append(Problem(
                stage=f"specialist:{role}",
                detail=f"{cited.dropped} finding(s) discarded — the source they "
                       f"quoted is not in the file they blamed",
            ))
        if cited.moved:
            log.info("%s reviewer: re-anchored %d finding(s) to the line they quoted",
                     role, cited.moved)

    findings = [
        validate_fix(f, sources.get(f.file, "")).model_copy(update={
            "fingerprint": fingerprint(
                f.file, f.category, sources.get(f.file, ""), f.line
            ) if sources.get(f.file) else ""
        })
        for f in findings
    ]

    log.info("%s reviewer: %d finding(s)%s", role, len(findings),
             f", {len(dropped)} ungrounded" if dropped else "")
    return {"findings": findings, "problems": problems}
