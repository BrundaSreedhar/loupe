"""Data contracts.

The split between `RawFinding` and `Finding` is deliberate: the first is what a
model is asked to produce, the second is what the system tracks about it. Keeping
provenance out of the model-facing schema means a specialist can't invent its own
id, and the LLM-facing JSON schema stays small enough to be reliably filled.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["security", "correctness", "performance", "maintainability"]
Severity = Literal["high", "medium", "low"]
Role = Literal["security", "correctness", "performance", "maintainability", "generalist"]


# ─── Input ──────────────────────────────────────────────────────────────────


class Hunk(BaseModel):
    """One @@ block. Line numbers are the point — a finding that can't be anchored
    to a real line can't become an inline PR comment."""

    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    content: str

    @property
    def new_end(self) -> int:
        return self.new_start + max(self.new_lines - 1, 0)


class FileDiff(BaseModel):
    path: str
    old_path: str | None = None
    change_type: Literal["added", "modified", "deleted", "renamed"] = "modified"
    hunks: list[Hunk] = Field(default_factory=list)
    content_after: str | None = None
    is_binary: bool = False

    @property
    def changed_lines(self) -> set[int]:
        lines: set[int] = set()
        for h in self.hunks:
            lines.update(range(h.new_start, h.new_end + 1))
        return lines


class ReviewRequest(BaseModel):
    """Normalized across adapters. Everything downstream is source-agnostic."""

    source: Literal["local", "github"]
    ref: str
    title: str = ""
    body: str = ""
    files: list[FileDiff] = Field(default_factory=list)
    repo_root: str = "."
    # Populated by the GitHub adapter so `emit` can post back.
    repo: str | None = None
    pr_number: int | None = None
    head_sha: str | None = None

    @property
    def reviewable(self) -> list[FileDiff]:
        return [f for f in self.files if not f.is_binary and f.change_type != "deleted"]


class FileContext(BaseModel):
    """The budgeted source window a reviewer actually sees."""

    path: str
    content: str
    strategy: Literal["whole_file", "windowed", "diff_only"]
    tokens: int
    truncated: bool = False


# ─── Output ─────────────────────────────────────────────────────────────────


class RawFinding(BaseModel):
    """What a reviewer must produce. `failure_scenario` is required on purpose:
    it is the field that makes a vague observation impossible to file."""

    file: str = Field(description="Repo-relative path, exactly as given in the diff.")
    line: int = Field(description="1-indexed line in the post-change file.")
    category: Category
    severity: Severity
    summary: str = Field(description="One sentence naming the defect. Not a tour of the code.")
    failure_scenario: str = Field(
        description="Concrete inputs or state that lead to a wrong result or crash. "
        "If you cannot write one, do not report the finding."
    )
    suggested_fix: str | None = None
    confidence: float = Field(
        ge=0.0, le=1.0, description="0-1. Calibration is measured, not trusted."
    )


class FindingBatch(BaseModel):
    """Structured-output container — models fill a list far more reliably inside
    a named object than as a bare top-level array."""

    findings: list[RawFinding] = Field(default_factory=list)


class Finding(RawFinding):
    id: str
    produced_by: Role
    merged_from: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    finding_id: str
    status: Literal["CONFIRMED", "REJECTED"]
    reasoning: str
    corrected_line: int | None = None


class VerdictDecision(BaseModel):
    """Model-facing half of a Verdict — no id, the system already knows it."""

    status: Literal["CONFIRMED", "REJECTED"]
    reasoning: str = Field(
        description="Why the defect is or is not real, citing the actual source you were given."
    )
    corrected_line: int | None = Field(
        default=None, description="Set only if the defect is real but reported at the wrong line."
    )


class ReviewResult(BaseModel):
    request_ref: str
    mode: Literal["single", "multi"]
    verified: bool
    raw: list[Finding] = Field(default_factory=list)
    merged: list[Finding] = Field(default_factory=list)
    accepted: list[Finding] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    usage: dict[str, float] = Field(default_factory=dict)


class MergedFinding(RawFinding):
    """A merge output. `covers` indexes back into the group the merger was given."""

    covers: list[int] = Field(default_factory=list)


class MergeResult(BaseModel):
    findings: list[MergedFinding] = Field(default_factory=list)
