"""One reviewer branch. Reached via Send(), once per role."""

from __future__ import annotations

import logging
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import specialist_llm
from ..prompts.specialists import SHARED_SYSTEM, context_message, role_message
from ..schema import Finding, FindingBatch
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
    kept, dropped = drop_ungrounded(findings, contexts)
    if dropped:
        log.info("%s reviewer: dropped %d ungrounded finding(s): %s", role, len(dropped), dropped)
    return {"findings": kept}
