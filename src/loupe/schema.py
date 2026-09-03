"""Data contracts.

The split between `RawFinding` and `Finding` is deliberate: the first is what a
model is asked to produce, the second is what the system tracks about it. Keeping
provenance out of the model-facing schema means a specialist can't invent its own
id, and the LLM-facing JSON schema stays small enough to be reliably filled.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .filters import is_reviewable_path

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
        return [
            f
            for f in self.files
            if not f.is_binary
            and f.change_type != "deleted"
            and is_reviewable_path(f.path)
        ]

    @property
    def identity(self) -> tuple[str, str]:
        """(repo, branch) — the key review memory is stored under."""
        if self.source == "github" and self.repo:
            return self.repo, f"pr-{self.pr_number}"
        return self.repo_root, self.ref

    @property
    def skipped(self) -> list[str]:
        """Paths present in the diff that no reviewer will see. Reported rather
        than silently dropped."""
        reviewed = {f.path for f in self.reviewable}
        return [f.path for f in self.files if f.path not in reviewed]


class FileContext(BaseModel):
    """The budgeted source window a reviewer actually sees."""

    path: str
    content: str
    strategy: Literal["whole_file", "windowed", "diff_only"]
    tokens: int
    truncated: bool = False


# ─── Output ─────────────────────────────────────────────────────────────────


class FixSuggestion(BaseModel):
    """A replacement, not a description of one.

    "Change <= to < in the loop bound" is prose a person still has to apply.
    Exact replacement text can be rendered as a GitHub suggestion block, which is
    one click to accept — and is checkable, because a replacement identical to
    what is already there is a no-op and can be dropped mechanically.
    """

    start_line: int = Field(description="First line being replaced, in the post-change file.")
    end_line: int = Field(description="Last line being replaced. Same as start_line for one line.")
    replacement: str = Field(
        description="The exact replacement source. Real indentation, no line-number "
        "prefixes, no ``` fences, no commentary. It is pasted in verbatim."
    )
    note: str | None = Field(default=None, description="One short sentence on why.")


class Claim(BaseModel):
    """The part of a finding every producer fills in — a reviewer filing one, or
    the merger rewriting two into one. `failure_scenario` is required on purpose:
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
    fix: FixSuggestion | None = Field(
        default=None,
        description="Only when you can write the corrected code exactly. Omit it "
        "rather than guess — a wrong fix is worse than none.",
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="0-1. Calibration is measured, not trusted."
    )


class RawFinding(Claim):
    """What a reviewer produces: a claim plus the line it reasoned from.

    The citation is checked against the real file before the finding goes any
    further, which costs nothing and catches a claim about code that is not there.
    It is also what the report shows next to the summary, so the evidence arrives
    with the argument instead of having to be reconstructed.
    """

    evidence: str = Field(
        default="",
        description="The exact source line your claim rests on, copied "
        "character-for-character from the numbered listing WITHOUT the `>123| ` "
        "prefix. Two or three consecutive lines if the defect needs them. This is "
        "matched against the real file: a quote that is not in it means the "
        "finding is discarded, so copy, do not retype.",
    )


class FindingBatch(BaseModel):
    """Structured-output container — models fill a list far more reliably inside
    a named object than as a bare top-level array."""

    findings: list[RawFinding] = Field(default_factory=list)


class Finding(RawFinding):
    id: str
    produced_by: Role
    merged_from: list[str] = Field(default_factory=list)
    # Stable across runs, unlike `id`. Empty when the source was unavailable to
    # compute one — such a finding is always treated as new.
    fingerprint: str = ""


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


class IndexedVerdict(BaseModel):
    """One decision inside a batch. `index` refers to the numbered finding list
    the verifier was shown."""

    index: int
    status: Literal["CONFIRMED", "REJECTED"]
    reasoning: str = Field(
        description="Why the defect is or is not real, citing the source you were given."
    )
    corrected_line: int | None = Field(
        default=None, description="Set only if the defect is real but reported at the wrong line."
    )


class VerdictBatch(BaseModel):
    verdicts: list[IndexedVerdict] = Field(default_factory=list)


class Problem(BaseModel):
    """Something that went wrong but did not stop the review.

    These used to be caught, logged into a void, and forgotten — so a review that
    silently lost half its findings looked exactly like a clean one. Collecting
    them means the run can say what it could not do."""

    stage: str
    detail: str
    severity: Literal["warning", "error"] = "warning"


class ReviewResult(BaseModel):
    request_ref: str
    mode: Literal["single", "multi"]
    verified: bool
    raw: list[Finding] = Field(default_factory=list)
    merged: list[Finding] = Field(default_factory=list)
    accepted: list[Finding] = Field(default_factory=list)
    verdicts: list[Verdict] = Field(default_factory=list)
    problems: list[Problem] = Field(default_factory=list)
    delta: object | None = None
    usage: dict[str, float] = Field(default_factory=dict)


class MergedFinding(Claim):
    """A merge output. `covers` indexes back into the group the merger was given.

    No `evidence` field: the merger is doing bookkeeping, not reading source, and
    a citation it invented would be checked against a file it was never shown. The
    merged finding inherits the citation of the finding it kept.
    """

    covers: list[int] = Field(default_factory=list)


class MergeResult(BaseModel):
    findings: list[MergedFinding] = Field(default_factory=list)


def validate_fix(finding: Finding, source: str) -> Finding:
    """Drop a fix that cannot be applied, keeping the finding.

    Three mechanical checks, no model call: the range has to exist in the file,
    it has to contain the line the finding is anchored to, and the replacement has
    to actually differ from what is there. A no-op suggestion renders as a live
    "apply" button that changes nothing, which is worse than showing no button.
    """
    fix = finding.fix
    if fix is None:
        return finding

    lines = source.splitlines()
    ok = (
        1 <= fix.start_line <= fix.end_line <= len(lines)
        and fix.start_line <= finding.line <= fix.end_line
        and fix.replacement.strip()
        and fix.replacement.rstrip() != "\n".join(lines[fix.start_line - 1 : fix.end_line]).rstrip()
    )
    return finding if ok else finding.model_copy(update={"fix": None})
