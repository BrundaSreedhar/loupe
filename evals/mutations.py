"""Seeded defects.

Each mutator finds a candidate site in real source, breaks it in one specific way,
and records the line it broke. That recorded line is the ground truth — which is
the whole reason this is worth doing: unlike a hand-labelled corpus, recall here
is mechanical and cannot drift.

Mutators are line-based rather than AST-based on purpose, so the same corpus
builder works on Python, TypeScript and JavaScript without a parser per language.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Mutation:
    name: str
    category: str
    line: int  # 1-indexed, in the mutated file
    before: str
    after: str
    description: str


Mutator = Callable[[list[str], random.Random], Mutation | None]
_REGISTRY: dict[str, Mutator] = {}


def mutator(fn: Mutator) -> Mutator:
    _REGISTRY[fn.__name__] = fn
    return fn


def _pick(lines: list[str], pattern: re.Pattern[str], rng: random.Random) -> int | None:
    hits = [
        i
        for i, ln in enumerate(lines)
        if pattern.search(ln) and not ln.lstrip().startswith(("#", "//", "*"))
    ]
    return rng.choice(hits) if hits else None


def _apply(lines: list[str], idx: int, new: str, m: dict) -> Mutation:
    before = lines[idx]
    lines[idx] = new
    return Mutation(line=idx + 1, before=before.strip(), after=new.strip(), **m)


# ─── correctness ────────────────────────────────────────────────────────────


@mutator
def off_by_one(lines: list[str], rng: random.Random) -> Mutation | None:
    pat = re.compile(r"(?<![<>=!])<(?!=)")
    idx = _pick(lines, re.compile(r"(for|while|if).*(?<![<>=!])<(?!=)"), rng)
    if idx is None:
        return None
    return _apply(
        lines,
        idx,
        pat.sub("<=", lines[idx], count=1),
        {
            "name": "off_by_one",
            "category": "correctness",
            "description": "Loop or guard bound widened from < to <=, running one past the end.",
        },
    )


@mutator
def flip_equality(lines: list[str], rng: random.Random) -> Mutation | None:
    idx = _pick(lines, re.compile(r"(if|while|assert|return).*==(?!=)"), rng)
    if idx is None:
        return None
    return _apply(
        lines,
        idx,
        re.sub(r"==(?!=)", "!=", lines[idx], count=1),
        {
            "name": "flip_equality",
            "category": "correctness",
            "description": "Equality comparison inverted.",
        },
    )


@mutator
def drop_await(lines: list[str], rng: random.Random) -> Mutation | None:
    idx = _pick(lines, re.compile(r"\bawait\s+\w"), rng)
    if idx is None:
        return None
    return _apply(
        lines,
        idx,
        re.sub(r"\bawait\s+", "", lines[idx], count=1),
        {
            "name": "drop_await",
            "category": "correctness",
            "description": "await removed — the coroutine/promise is never resolved.",
        },
    )


@mutator
def remove_guard(lines: list[str], rng: random.Random) -> Mutation | None:
    """Delete a null/empty guard, leaving the body it protected."""
    pat = re.compile(r"^\s*if\s*\(?\s*(!|not\s)")
    hits = [
        i
        for i, ln in enumerate(lines[:-1])
        if pat.search(ln)
        and ("return" in lines[i + 1] or "raise" in lines[i + 1] or "throw" in lines[i + 1])
    ]
    if not hits:
        return None
    idx = rng.choice(hits)
    before = lines[idx]
    # Remove the guard and the single-statement body it wraps.
    del lines[idx : idx + 2]
    return Mutation(
        name="remove_guard",
        category="correctness",
        line=max(idx, 1),
        before=before.strip(),
        after="(guard deleted)",
        description="Null/empty guard and its early return removed.",
    )


@mutator
def swap_operands(lines: list[str], rng: random.Random) -> Mutation | None:
    pat = re.compile(r"(\w+)\s-\s(\w+)")
    idx = _pick(lines, pat, rng)
    if idx is None:
        return None
    return _apply(
        lines,
        idx,
        pat.sub(r"\2 - \1", lines[idx], count=1),
        {
            "name": "swap_operands",
            "category": "correctness",
            "description": "Operands of a subtraction swapped, negating the result.",
        },
    )


# ─── security ───────────────────────────────────────────────────────────────


@mutator
def sql_concat(lines: list[str], rng: random.Random) -> Mutation | None:
    idx = _pick(lines, re.compile(r"(SELECT|INSERT|UPDATE|DELETE)\s", re.IGNORECASE), rng)
    if idx is None:
        return None
    ln = lines[idx]
    if "%s" in ln:
        new = ln.replace("%s", '" + user_input + "', 1)
    elif "?" in ln:
        new = ln.replace("?", '" + user_input + "', 1)
    else:
        new = re.sub(r'(["\'])(\s*)$', r'" + user_input + "\1', ln, count=1)
        if new == ln:
            return None
    return _apply(
        lines,
        idx,
        new,
        {
            "name": "sql_concat",
            "category": "security",
            "description": "Parameterised query turned into string concatenation.",
        },
    )


@mutator
def broaden_except(lines: list[str], rng: random.Random) -> Mutation | None:
    idx = _pick(lines, re.compile(r"except\s+\w+Error"), rng)
    if idx is None:
        return None
    return _apply(
        lines,
        idx,
        re.sub(
            r"except\s+\w+(Error|Exception)(\s+as\s+\w+)?",
            r"except Exception\2",
            lines[idx],
            count=1,
        ),
        {
            "name": "broaden_except",
            "category": "correctness",
            "description": (
                "Narrow except widened to bare Exception, swallowing unrelated failures."
            ),
        },
    )


# ─── performance ────────────────────────────────────────────────────────────


@mutator
def hoist_to_loop(lines: list[str], rng: random.Random) -> Mutation | None:
    """Move a lookup inside a loop body — the N+1 shape."""
    idx = _pick(lines, re.compile(r"^\s*(for|while)\s"), rng)
    if idx is None or idx + 1 >= len(lines):
        return None
    body = lines[idx + 1]
    indent = re.match(r"\s*", body).group(0)
    injected = f"{indent}_row = db.query(f\"SELECT * FROM items WHERE id = {{item_id}}\")"
    lines.insert(idx + 1, injected)
    return Mutation(
        name="hoist_to_loop",
        category="performance",
        line=idx + 2,
        before="(none)",
        after=injected.strip(),
        description="Per-iteration database query introduced inside a loop.",
    )


# ─── benign controls ────────────────────────────────────────────────────────
# The clean half of the corpus. Any finding on these is a false positive.

BENIGN: dict[str, Mutator] = {}


def benign(fn: Mutator) -> Mutator:
    BENIGN[fn.__name__] = fn
    return fn


@benign
def add_comment(lines: list[str], rng: random.Random) -> Mutation | None:
    idx = _pick(lines, re.compile(r"^\s*(def|function|class|const|async)\s"), rng)
    if idx is None:
        return None
    indent = re.match(r"\s*", lines[idx]).group(0)
    marker = "#" if lines[idx].lstrip().startswith(("def", "class")) else "//"
    lines.insert(idx, f"{indent}{marker} Handles the common path.")
    return Mutation(
        name="add_comment",
        category="none",
        line=idx + 1,
        before="(none)",
        after="comment",
        description="Comment added. Semantics unchanged.",
    )


@benign
def widen_whitespace(lines: list[str], rng: random.Random) -> Mutation | None:
    idx = _pick(lines, re.compile(r"\w\s*=\s*\w"), rng)
    if idx is None:
        return None
    return _apply(
        lines,
        idx,
        re.sub(r"\s*=\s*", " = ", lines[idx], count=1),
        {
            "name": "widen_whitespace",
            "category": "none",
            "description": "Assignment spacing normalised. Semantics unchanged.",
        },
    )


@benign
def rename_local(lines: list[str], rng: random.Random) -> Mutation | None:
    """Rename a variable assigned and used on adjacent lines only."""
    pat = re.compile(r"^\s*(?:const |let |var )?([a-z_][a-z0-9_]{3,})\s*=\s*\S")
    for i, ln in enumerate(lines[:-1]):
        m = pat.match(ln)
        if not m:
            continue
        name = m.group(1)
        uses = [j for j, other in enumerate(lines) if re.search(rf"\b{re.escape(name)}\b", other)]
        if len(uses) == 2 and uses[1] == i + 1:
            new_name = f"{name}_value"
            lines[i] = re.sub(rf"\b{re.escape(name)}\b", new_name, lines[i])
            lines[i + 1] = re.sub(rf"\b{re.escape(name)}\b", new_name, lines[i + 1])
            return Mutation(
                name="rename_local",
                category="none",
                line=i + 1,
                before=name,
                after=new_name,
                description="Local variable renamed at both its definition and its only use.",
            )
    return None


def all_defect_mutators() -> dict[str, Mutator]:
    return dict(_REGISTRY)


def all_benign_mutators() -> dict[str, Mutator]:
    return dict(BENIGN)
