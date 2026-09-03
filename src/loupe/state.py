"""Graph state.

The `Annotated[..., operator.add]` reducers are the only thing making the fan-ins
correct. Four specialists write `findings` concurrently and N verifiers write
`verdicts` concurrently; without a reducer, last-write-wins silently discards
everything but one branch's work.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired

from typing_extensions import TypedDict

from .lint import LintIssue
from .safety import SafetyIssue
from .schema import FileContext, Finding, Problem, ReviewRequest, Verdict


class ReviewState(TypedDict):
    request: ReviewRequest
    contexts: NotRequired[dict[str, FileContext]]
    dropped: NotRequired[list[str]]
    lint_issues: NotRequired[list[LintIssue]]
    safety: NotRequired[list[SafetyIssue]]

    mode: Literal["single", "multi"]
    verify: bool

    findings: Annotated[list[Finding], operator.add]
    merged: NotRequired[list[Finding]]
    verdicts: Annotated[list[Verdict], operator.add]
    # Re-judged borderline findings. Kept in its own channel so a consensus
    # verdict cannot be confused with the single-pass one it replaces.
    consensus: Annotated[list[Verdict], operator.add]
    # Reduced like the others: several nodes fail concurrently, and a plain
    # assignment would keep only whichever branch finished last.
    problems: Annotated[list[Problem], operator.add]
    accepted: NotRequired[list[Finding]]


class SpecialistTask(TypedDict):
    """Send() payload for one reviewer branch."""

    role: str
    request: ReviewRequest
    contexts: dict[str, FileContext]
    lint_issues: list[LintIssue]


class ConsensusTask(TypedDict):
    """Send() payload for re-judging one borderline finding."""

    finding: Finding
    first: Verdict
    source: str


class VerifyTask(TypedDict):
    """Send() payload for one verification call.

    Carries the *full* file source, not the windowed context the reviewer saw —
    the gate is only worth having if it reasons from different evidence. Holding a
    list rather than a single finding lets per-file and per-finding verification
    share one node: a batch of one is still a batch.
    """

    path: str
    findings: list[Finding]
    source: str
