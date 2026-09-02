"""The precision gate. Reached via Send(), once per surviving finding."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import verifier_llm
from ..prompts.verify import SYSTEM, user_prompt
from ..schema import Verdict, VerdictDecision
from ..state import VerifyTask


def verify(task: VerifyTask) -> dict:
    finding = task["finding"]
    llm = verifier_llm().with_structured_output(VerdictDecision, method="json_schema")
    try:
        decision: VerdictDecision = llm.invoke(
            [SystemMessage(SYSTEM), HumanMessage(user_prompt(finding, task["source"]))],
            config={
                "run_name": f"verify:{finding.file}:{finding.line}",
                "tags": ["verify", finding.category],
                # finding_id is what makes one bad verdict traceable back to its
                # claim; without it the fan-out is N anonymous sibling spans.
                "metadata": {
                    "finding_id": finding.id,
                    "category": finding.category,
                    "produced_by": finding.produced_by,
                    "self_confidence": finding.confidence,
                },
            },
        )
    except Exception as exc:  # noqa: BLE001 — an unverifiable finding is not
        # a confirmed one. Fail closed.
        return {
            "verdicts": [
                Verdict(
                    finding_id=finding.id,
                    status="REJECTED",
                    reasoning=f"Verification failed: {type(exc).__name__}: {exc}",
                )
            ]
        }

    return {
        "verdicts": [
            Verdict(
                finding_id=finding.id,
                status=decision.status,
                reasoning=decision.reasoning,
                corrected_line=decision.corrected_line,
            )
        ]
    }
