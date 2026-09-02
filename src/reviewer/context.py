"""Context construction.

A reviewer that sees only +/- lines invents claims about the surrounding code, so
each file is rendered as a line-numbered window with changed lines marked. The
line numbers are not decoration: they are how a finding gets anchored well enough
to become an inline PR comment.

Token accounting is a two-step compromise. A cheap local estimate picks the
strategy per file (calling count_tokens once per file would add N network
round-trips to every review), then one real `count_tokens` call measures the
assembled context so the number reported to the user is not a guess.
"""

from __future__ import annotations

from .config import FILE_TOKEN_BUDGET, REVIEW_TOKEN_BUDGET, WINDOW_PADDING
from .schema import FileContext, FileDiff, ReviewRequest

# Deliberately pessimistic: over-estimating shrinks a window, which is safe.
# Under-estimating overflows the budget, which is not.
_CHARS_PER_TOKEN = 3.4


def _estimate(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN)


def render_numbered(content: str, changed: set[int], lo: int = 1, hi: int | None = None) -> str:
    lines = content.splitlines()
    hi = hi if hi is not None else len(lines)
    width = len(str(hi))
    out: list[str] = []
    for n in range(lo, min(hi, len(lines)) + 1):
        marker = ">" if n in changed else " "
        out.append(f"{marker}{n:>{width}}| {lines[n - 1]}")
    return "\n".join(out)


def build_file_context(fd: FileDiff) -> FileContext | None:
    if fd.is_binary or fd.change_type == "deleted" or not fd.content_after:
        return None

    changed = fd.changed_lines
    whole = render_numbered(fd.content_after, changed)

    if _estimate(whole) <= FILE_TOKEN_BUDGET:
        return FileContext(
            path=fd.path, content=whole, strategy="whole_file", tokens=_estimate(whole)
        )

    # Too big: keep padded windows around each hunk, joined by explicit gap markers
    # so the model can see that it is not looking at a contiguous file.
    total = len(fd.content_after.splitlines())
    spans: list[tuple[int, int]] = []
    for h in fd.hunks:
        lo = max(1, h.new_start - WINDOW_PADDING)
        hi = min(total, h.new_end + WINDOW_PADDING)
        if spans and lo <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], hi))
        else:
            spans.append((lo, hi))

    chunks: list[str] = []
    prev_hi = 0
    for lo, hi in spans:
        if lo > prev_hi + 1:
            chunks.append(f"      … lines {prev_hi + 1}-{lo - 1} omitted …")
        chunks.append(render_numbered(fd.content_after, changed, lo, hi))
        prev_hi = hi
    if prev_hi < total:
        chunks.append(f"      … lines {prev_hi + 1}-{total} omitted …")

    windowed = "\n".join(chunks)
    return FileContext(
        path=fd.path,
        content=windowed,
        strategy="windowed",
        tokens=_estimate(windowed),
        truncated=True,
    )


def build_contexts(request: ReviewRequest) -> tuple[dict[str, FileContext], list[str]]:
    """Returns (contexts, dropped_paths). Files are admitted largest-change-first
    until the review budget is spent; whatever doesn't fit is named in the report
    rather than quietly skipped."""
    candidates = sorted(
        request.reviewable, key=lambda f: len(f.changed_lines), reverse=True
    )
    contexts: dict[str, FileContext] = {}
    dropped: list[str] = []
    spent = 0

    for fd in candidates:
        ctx = build_file_context(fd)
        if ctx is None:
            continue
        if spent + ctx.tokens > REVIEW_TOKEN_BUDGET:
            dropped.append(fd.path)
            continue
        contexts[fd.path] = ctx
        spent += ctx.tokens

    return contexts, dropped


def measure(contexts: dict[str, FileContext]) -> int:
    """One real token count over the assembled context, via whichever provider is
    bound. Falls back to the local estimate rather than failing a review."""
    if not contexts:
        return 0
    blob = "\n\n".join(c.content for c in contexts.values())
    try:
        from .config import specialist_llm

        return specialist_llm().get_num_tokens(blob)
    except Exception:  # noqa: BLE001 — never fail a review over instrumentation.
        return sum(c.tokens for c in contexts.values())
