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

from .index import Definition, Edge, Reference
from .lint import LintIssue
from .safety import SafetyIssue
from .schema import FileContext, Finding, Problem, ReviewRequest, Verdict


class ReviewState(TypedDict):
    request: ReviewRequest
    contexts: NotRequired[dict[str, FileContext]]
    dropped: NotRequired[list[str]]
    lint_issues: NotRequired[list[LintIssue]]
    safety: NotRequired[list[SafetyIssue]]
    # Definitions the changed lines call, resolved from the repository on disk.
    # Part of the shared prompt prefix, so every reviewer sees the same list.
    references: NotRequired[list[Reference]]
    # Every call from a changed line to a definition this repo owns — including
    # the ones whose source the prompt left out. Drawn for the reader, not sent.
    edges: NotRequired[list[Edge]]
    # path -> one line on what that file is for, for the diagram in the report.
    summaries: NotRequired[dict[str, str]]
    # The definitions this change edited, from the changed files' own trees. The
    # unit the reverse lookups — callers, tests, history — are all keyed by.
    changed: NotRequired[list[Definition]]

    mode: Literal["single", "multi"]
    verify: bool
    # None defers to LOUPE_LINT. False is for callers whose diff exists only in
    # memory: a linter reads the file on disk, which is not the file under review.
    lint: NotRequired[bool | None]

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
    delta: NotRequired[object]
    remember: NotRequired[bool]


class SpecialistTask(TypedDict):
    """Send() payload for one reviewer branch."""

    role: str
    request: ReviewRequest
    contexts: dict[str, FileContext]
    lint_issues: list[LintIssue]
    references: list[Reference]


class ConsensusTask(TypedDict):
    """Send() payload for re-judging one borderline finding."""

    finding: Finding
    first: Verdict
    source: str
    references: list[Reference]


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
    # The same definitions the reviewers were shown. Without them the gate must
    # treat every cross-file claim as an unverifiable assumption and reject it,
    # which would make the expansion above worthless.
    references: list[Reference]
