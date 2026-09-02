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

from .schema import FileContext, Finding, ReviewRequest, Verdict


class ReviewState(TypedDict):
    request: ReviewRequest
    contexts: NotRequired[dict[str, FileContext]]
    dropped: NotRequired[list[str]]

    mode: Literal["single", "multi"]
    verify: bool

    findings: Annotated[list[Finding], operator.add]
    merged: NotRequired[list[Finding]]
    verdicts: Annotated[list[Verdict], operator.add]
    accepted: NotRequired[list[Finding]]


class SpecialistTask(TypedDict):
    """Send() payload for one reviewer branch."""

    role: str
    request: ReviewRequest
    contexts: dict[str, FileContext]


class VerifyTask(TypedDict):
    """Send() payload for one finding's verification.

    Carries the *full* file source, not the windowed context the reviewer saw —
    the gate is only worth having if it reasons from different evidence.
    """

    finding: Finding
    source: str
