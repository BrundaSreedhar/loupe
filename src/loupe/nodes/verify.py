"""The verification gate. Reached via Send(), once per file or once per finding."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import structured, verifier_llm
from ..prompts.verify import SYSTEM, user_prompt
from ..quota import raise_if_terminal
from ..schema import Problem, Verdict, VerdictBatch
from ..state import VerifyTask

log = logging.getLogger(__name__)


def _reject_all(findings, reason: str) -> dict:
    """Fail closed: a finding that could not be verified is not a confirmed one."""
    return {
        "verdicts": [
            Verdict(finding_id=f.id, status="REJECTED", reasoning=reason) for f in findings
        ]
    }


def verify(task: VerifyTask) -> dict:
    findings = task["findings"]
    path = task["path"]
    if not findings:
        return {"verdicts": []}

    llm = structured(verifier_llm(), VerdictBatch, "verify")
    try:
        batch: VerdictBatch = llm.invoke(
            [
                SystemMessage(SYSTEM),
                HumanMessage(
                    user_prompt(findings, path, task["source"], task.get("references") or [])
                ),
            ],
            config={
                "run_name": f"verify:{path}",
                "tags": ["verify"],
                # finding_ids is what makes one bad verdict traceable back to the
                # claim that produced it; without it a fan-out is anonymous spans.
                "metadata": {
                    "path": path,
                    "batch_size": len(findings),
                    "finding_ids": [f.id for f in findings],
                    "categories": sorted({f.category for f in findings}),
                },
            },
        )
    except Exception as exc:  # noqa: BLE001 — an unverifiable finding is not a
        # confirmed one. Fail closed.
        # ...but a quota failure is not evidence about the finding. Rejecting
        # everything on an exhausted quota produces a clean-looking review that
        # simply never happened, so that one propagates instead.
        raise_if_terminal(exc)
        log.error("verification of %s failed (%s); rejecting %d finding(s) it could "
                  "not check", path, type(exc).__name__, len(findings))
        out = _reject_all(findings, f"Verification failed: {type(exc).__name__}: {exc}")
        out["problems"] = [Problem(
            stage="verify",
            detail=f"could not verify {len(findings)} finding(s) in {path} "
                   f"({type(exc).__name__}) — they were dropped, not cleared",
            severity="error",
        )]
        return out

    by_index = {v.index: v for v in batch.verdicts}
    verdicts: list[Verdict] = []
    for i, f in enumerate(findings):
        decision = by_index.get(i)
        if decision is None:
            # A finding the verifier silently skipped has not been confirmed.
            verdicts.append(
                Verdict(
                    finding_id=f.id,
                    status="REJECTED",
                    reasoning="The verifier returned no verdict for this finding.",
                )
            )
            continue
        verdicts.append(
            Verdict(
                finding_id=f.id,
                status=decision.status,
                reasoning=decision.reasoning,
                corrected_line=decision.corrected_line,
            )
        )
    return {"verdicts": verdicts}
