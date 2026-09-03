"""One reviewer branch. Reached via Send(), once per role."""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import specialist_llm
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

    llm = specialist_llm().with_structured_output(FindingBatch, method="json_schema")
    batch: FindingBatch = llm.invoke(
        [
            SystemMessage(SHARED_SYSTEM),
            HumanMessage(content=cached_block(context_message(task["request"], contexts))),
            HumanMessage(role_message(role)),
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
    findings = [validate_fix(f, sources.get(f.file, "")) for f in findings]
    kept, dropped = drop_ungrounded(findings, contexts)
    log.info("%s reviewer: %d finding(s)%s", role, len(kept),
             f", {len(dropped)} dropped as ungrounded" if dropped else "")
    problems = []
    if dropped:
        log.warning("%s reviewer: dropped %d finding(s) pointing at code it was "
                    "not shown: %s", role, len(dropped), "; ".join(dropped[:3]))
        problems.append(Problem(
            stage=f"specialist:{role}",
            detail=f"{len(dropped)} finding(s) discarded — they referenced files or "
                   f"lines outside the review context",
        ))
    return {"findings": kept, "problems": problems}
