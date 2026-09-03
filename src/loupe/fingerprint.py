"""Recognising a finding you have already seen.

`Finding.id` is a uuid generated per run — fine for matching a verdict to its
claim inside one review, useless across two. To know that today's finding is
yesterday's, a finding has to be identified by what it is *about*.

Deliberately not the line number. Lines move every time anything above them is
edited, so a line-keyed identity dissolves on contact with a rebase and every
finding looks new. What stays put is the code itself, so the fingerprint is taken
over a small window of normalised source around the flagged line — wide enough
that two identical lines elsewhere in the file do not collide, narrow enough that
an edit twenty lines away does not count as a different defect.
"""

from __future__ import annotations

import hashlib
import re

WINDOW = 2  # lines either side of the flagged line

_WS = re.compile(r"\s+")
# Spacing around punctuation and operators carries no meaning in any language this
# reviews, and a formatter reflowing a line must not turn one open finding into a
# resolved one plus a new one.
_AROUND_PUNCT = re.compile(r"\s*([^\w\s])\s*")


def normalise(line: str) -> str:
    """Whitespace-insensitive, so reformatting is not a new finding.

    Shared with `grounding`, which compares a reviewer's quoted line against the
    real file. The two need one notion of "the same line", or a citation would
    pass one check and fail the other over a space.
    """
    return _AROUND_PUNCT.sub(r"\1", _WS.sub(" ", line)).strip()


def code_window(source: str, line: int, window: int = WINDOW) -> str:
    lines = source.splitlines()
    if not lines:
        return ""
    lo = max(0, line - 1 - window)
    hi = min(len(lines), line + window)
    return "\n".join(normalise(text) for text in lines[lo:hi])


def compute(file: str, category: str, source: str, line: int) -> str:
    """A stable id for this defect, in this file, at this piece of code.

    Category is included because the same line can carry two genuinely different
    problems — a query that is both injectable and inside a loop is one finding
    for security and another for performance, and resolving one should not silence
    the other.
    """
    payload = " ".join((file, category, code_window(source, line)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
