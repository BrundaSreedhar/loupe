"""Checking that a finding quotes a line that exists.

A reviewer is asked to copy out the line its claim rests on. That quote is then
compared against the real file, which is free — no model call — and catches the
failure mode nothing else catches: a claim about code that is not there. The
reviewer is fluent either way, so a wrong citation reads exactly like a right one
until someone opens the file.

Two things it also buys:

- A comment you can check in five seconds. The quoted line goes into the report,
  so the claim and its evidence arrive together.
- Anchoring. Reviewers are often a line or two out, and a quote that matches
  exactly one nearby line says where the code actually is. The line is moved to
  match the quote rather than the finding being thrown away.

What it deliberately does not do is judge whether the quoted line supports the
argument. That is the verification gate's job and it costs a model call. This is
the mechanical half: does the cited code exist, and is it where the finding says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .fingerprint import normalise
from .schema import Finding

# `>123| code` and ` 123| code`, the shape the reviewer was shown. Models copy the
# prefix along with the line about half the time; stripping it is easier than
# arguing with them about it in the prompt.
_NUMBER_PREFIX = re.compile(r"^\s*>?\s*\d+\s*\|\s?")
_FENCE = re.compile(r"^\s*```")

# Below this, a quote carries no information: `}` or `else:` appears everywhere,
# so a match somewhere else in the file is not evidence the reviewer read it.
# Short quotes are still accepted, they just cannot be used to move a finding.
MIN_QUOTE_CHARS = 4


@dataclass(frozen=True)
class Citation:
    """The result of checking one quote against one file."""

    ok: bool
    line: int  # where the quoted code actually is; equals the claim when ok
    reason: str  # "" when ok, otherwise why it failed

    @property
    def moved(self) -> bool:
        return self.ok and self.reason == "moved"


def quote_lines(evidence: str) -> list[str]:
    """The quoted code, normalised, with the listing's decoration removed."""
    out: list[str] = []
    for raw in evidence.splitlines():
        if _FENCE.match(raw):
            continue
        text = normalise(_NUMBER_PREFIX.sub("", raw))
        if text:
            out.append(text)
    return out


def cited_text(evidence: str) -> str:
    """The quote as a person should see it: decoration removed, spacing kept.

    `quote_lines` normalises whitespace so two spellings of the same line compare
    equal, which is right for matching and wrong for display — the report shows
    the code as it is written in the file.
    """
    out = [
        _NUMBER_PREFIX.sub("", raw).rstrip()
        for raw in evidence.splitlines()
        if not _FENCE.match(raw) and raw.strip()
    ]
    return "\n".join(out)


def locate(quote: list[str], source_lines: list[str]) -> list[int]:
    """1-indexed starting lines where the quote appears as a consecutive block."""
    if not quote or len(quote) > len(source_lines):
        return []
    first = quote[0]
    return [
        i + 1
        for i in range(len(source_lines) - len(quote) + 1)
        if source_lines[i] == first and source_lines[i : i + len(quote)] == quote
    ]


def check(evidence: str, source: str, line: int, window: int) -> Citation:
    """Does `evidence` appear in `source`, at or near `line`?

    `window` is how far a finding may be moved to meet its own quote. Wide enough
    to absorb the usual off-by-a-few, narrow enough that a quote matching some
    unrelated line elsewhere in the file is not silently accepted as the target.
    """
    quote = quote_lines(evidence)
    if not quote:
        return Citation(False, line, "missing")

    source_lines = [normalise(text) for text in source.splitlines()]
    if not source_lines:
        # No source to check against. Not evidence the finding is wrong.
        return Citation(True, line, "")

    starts = locate(quote, source_lines)
    if not starts:
        return Citation(False, line, "mismatch")

    if any(start <= line <= start + len(quote) - 1 for start in starts):
        return Citation(True, line, "")

    if sum(len(q) for q in quote) < MIN_QUOTE_CHARS:
        # Matches, but somewhere else, and the quote is too common to relocate on.
        return Citation(False, line, "trivial")

    near = [start for start in starts if abs(start - line) <= window]
    if len(near) == 1:
        return Citation(True, near[0], "moved")
    if len(near) > 1:
        return Citation(False, line, "ambiguous")
    return Citation(False, line, "mismatch")


_EXPLAIN = {
    "missing": "quoted no source line",
    "mismatch": "quoted code that is not in the file",
    "trivial": "quoted a line too common to place, and not at the line claimed",
    "ambiguous": "quoted code that appears several times near the line claimed",
}


@dataclass
class GroundingOutcome:
    """What the check did to a batch of findings, in enough detail to report it."""

    kept: list[Finding]
    reasons: list[str]  # one human-readable line per dropped finding
    missing: int  # dropped only because no citation was given at all
    moved: int  # kept, but re-anchored to where the quoted code actually is
    skipped: bool = False  # the check did not run; nothing was dropped

    @property
    def dropped(self) -> int:
        return len(self.reasons)


def apply(
    findings: list[Finding], sources: dict[str, str], window: int
) -> GroundingOutcome:
    """Drop findings whose citation does not hold, and re-anchor the ones that moved.

    One exception, and it exists because this project has shipped the mistake
    before: if *no* finding carried a citation, that is a model that did not fill
    the field, not a batch of hallucinations. Dropping them all would turn a
    provider-side failure into a clean-looking review. Nothing is dropped in that
    case and the caller is told the check did not run.
    """
    if not findings:
        return GroundingOutcome([], [], 0, 0)

    checks = [(f, check(f.evidence, sources.get(f.file, ""), f.line, window)) for f in findings]
    if all(c.reason == "missing" for _, c in checks):
        return GroundingOutcome(list(findings), [], len(findings), 0, skipped=True)

    kept: list[Finding] = []
    reasons: list[str] = []
    missing = moved = 0
    for finding, result in checks:
        if not result.ok:
            missing += result.reason == "missing"
            reasons.append(
                f"{finding.file}:{finding.line} — {_EXPLAIN[result.reason]}"
            )
            continue
        if result.moved:
            moved += 1
            finding = finding.model_copy(update={"line": result.line})
        kept.append(finding)
    return GroundingOutcome(kept, reasons, missing, moved)
