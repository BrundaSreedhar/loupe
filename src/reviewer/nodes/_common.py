"""Shared node helpers."""

from __future__ import annotations

from typing import Any

from ..config import PROVIDER
from ..schema import FileContext, Finding


def cached_block(text: str) -> str | list[dict[str, Any]]:
    """Mark the shared prefix as cacheable, in whatever way the provider understands.

    Anthropic needs an explicit `cache_control` breakpoint. Gemini caches matching
    prefixes implicitly and rejects unknown block keys, so it just gets the text —
    the prefix-ordering discipline in prompts/specialists.py is what makes the
    implicit cache hit, and that is provider-independent.
    """
    if PROVIDER != "anthropic":
        return text
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def valid_lines(ctx: FileContext) -> set[int]:
    """Line numbers actually present in what the reviewer was shown."""
    out: set[int] = set()
    for row in ctx.content.splitlines():
        head = row[:1]
        rest = row[1:].split("|", 1)
        if head in (">", " ") and rest and rest[0].strip().isdigit():
            out.add(int(rest[0].strip()))
    return out


def drop_ungrounded(
    findings: list[Finding], contexts: dict[str, FileContext]
) -> tuple[list[Finding], list[str]]:
    """Discard findings pointing at files or lines the reviewer was never shown.

    This is cheap, deterministic noise control that costs no tokens — a finding on
    a file outside the change is definitionally a hallucination, and one on a line
    that isn't in the context can't be anchored for an inline comment anyway."""
    kept: list[Finding] = []
    reasons: list[str] = []
    for f in findings:
        ctx = contexts.get(f.file)
        if ctx is None:
            reasons.append(f"{f.file}:{f.line} — file not in the reviewed change")
            continue
        if f.line not in valid_lines(ctx):
            reasons.append(f"{f.file}:{f.line} — line not present in the given context")
            continue
        kept.append(f)
    return kept, reasons
