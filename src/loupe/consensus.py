"""Deciding which findings deserve a second look, and what to do with the votes.

Cursor reported moving developer action rate from about half to over two thirds by
majority-voting its reviewer's output, with no change to which bugs were found —
the entire gain came from filtering. That is the strongest single result in this
area, and it costs a call per extra sample.

Sampling every finding is unaffordable on a rate-limited tier and mostly wasted:
a verifier that confirms with clear reasoning does not change its mind. So only
findings where the two independent signals *disagree* are re-sampled.
"""

from __future__ import annotations

from collections import Counter

from .config import CONSENSUS_HIGH, CONSENSUS_LOW
from .schema import Finding, Verdict


def is_borderline(finding: Finding, verdict: Verdict) -> bool:
    """The reviewer's self-reported confidence and the gate's verdict disagree.

    Confirmed-and-confident, or rejected-and-unsure, are agreement — a second
    sample would almost certainly say the same thing and buy nothing.
    """
    if verdict.status == "CONFIRMED":
        return finding.confidence < CONSENSUS_LOW
    return finding.confidence > CONSENSUS_HIGH


def tally(verdicts: list[Verdict]) -> tuple[str, int, int]:
    """Majority over the samples. Returns (status, agreeing, total).

    Ties go to REJECTED: an even split is not confidence, and the cost of a false
    positive is what this whole stage exists to reduce.
    """
    if not verdicts:
        return "REJECTED", 0, 0
    counts = Counter(v.status for v in verdicts)
    confirmed = counts.get("CONFIRMED", 0)
    total = sum(counts.values())
    if confirmed * 2 > total:
        return "CONFIRMED", confirmed, total
    return "REJECTED", total - confirmed, total


def vote_note(agreeing: int, total: int) -> str:
    """What the reader is shown. `2 of 3` and `3 of 3` are different claims, and
    saying which saves the reader re-deriving it."""
    return f"{agreeing} of {total} verification passes agreed"
