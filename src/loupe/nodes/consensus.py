"""Re-verify a borderline finding and take the majority. Reached via Send()."""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from ..config import CONSENSUS_SAMPLES, structured, verifier_llm
from ..consensus import tally
from ..prompts.verify import SYSTEM, user_prompt
from ..quota import raise_if_terminal
from ..schema import Problem, Verdict, VerdictBatch
from ..state import ConsensusTask

log = logging.getLogger(__name__)


def reconsider(task: ConsensusTask) -> dict:
    """Run the remaining samples for one finding and replace its verdict.

    The first verdict already exists, so this runs SAMPLES - 1 more. Each is an
    independent call rather than one call asking for N opinions — asking a model
    to disagree with itself inside a single response is not a second sample.
    """
    finding = task["finding"]
    votes: list[Verdict] = [task["first"]]
    llm = structured(verifier_llm(), VerdictBatch, "consensus")

    for i in range(max(CONSENSUS_SAMPLES - 1, 0)):
        try:
            batch: VerdictBatch = llm.invoke(
                [
                    SystemMessage(SYSTEM),
                    HumanMessage(user_prompt(
                        [finding], finding.file, task["source"], task.get("references") or []
                    )),
                ],
                config={
                    "run_name": f"consensus:{finding.file}:{finding.line}",
                    "tags": ["consensus"],
                    "metadata": {"finding_id": finding.id, "sample": i + 2},
                },
            )
        except Exception as exc:  # noqa: BLE001 — a lost sample is not a verdict
            raise_if_terminal(exc)
            log.warning("consensus sample %d failed for %s: %s", i + 2, finding.id, exc)
            continue
        if batch.verdicts:
            d = batch.verdicts[0]
            votes.append(
                Verdict(
                    finding_id=finding.id,
                    status=d.status,
                    reasoning=d.reasoning,
                    corrected_line=d.corrected_line,
                )
            )

    status, agreeing, total = tally(votes)
    log.info("consensus on %s: %s (%d of %d)", finding.id, status, agreeing, total)

    problems = []
    if total < CONSENSUS_SAMPLES:
        problems.append(Problem(
            stage="consensus",
            detail=f"{finding.file}:{finding.line} was judged on {total} of "
                   f"{CONSENSUS_SAMPLES} samples — the rest failed",
        ))

    # Replaces the first verdict rather than adding to it; finalize keys on id.
    return {
        "consensus": [Verdict(
            finding_id=finding.id,
            status=status,
            reasoning=f"{agreeing} of {total} passes agreed. "
                      + (votes[-1].reasoning if votes else ""),
            corrected_line=next(
                (v.corrected_line for v in votes if v.corrected_line), None
            ),
        )],
        "problems": problems,
    }
