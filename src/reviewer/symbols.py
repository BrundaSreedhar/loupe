"""Resolve the definitions that changed code calls into.

The quality ceiling on a diff-only reviewer is that it cannot see what the code it
is reading calls. Told to review `if (validate(payload)) { commit(tx) }`, it either
invents a claim about `validate` or says nothing. Neither is useful.

This is a lightweight symbol index — a poor relation of a real call graph (SCIP,
LSIF, an LSP), and deliberately so: it needs no indexer, no daemon and no
per-language toolchain, and it recovers most of the value for a diff-sized
question. It resolves by name, so an overloaded or shadowed name may resolve to
the wrong definition; every returned definition therefore carries its own
file:line so the reviewer can see what it actually got.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .filters import SOURCE_SUFFIXES, is_reviewable_path

MAX_INDEX_FILES = 1200
MAX_FILE_BYTES = 400_000
MAX_BODY_LINES = 40

# Names that appear as calls everywhere and never repay a lookup.
_STOPWORDS = frozenset("""
if for while switch catch return typeof instanceof await async function class new
print len str int float dict list set tuple bool range enumerate zip map filter
super isinstance getattr setattr hasattr open type repr sorted sum min max abs any all
console require import export default this self None True False null undefined
describe it test expect beforeEach afterEach jest vi assert
""".split())

_DEF_PATTERNS = (
    # Python
    re.compile(r"^(?P<indent>\s*)(?:async\s+)?def\s+(?P<name>\w+)\s*\("),
    re.compile(r"^(?P<indent>\s*)class\s+(?P<name>\w+)\s*[(:]"),
    # TS/JS function and class declarations
    re.compile(
        r"^(?P<indent>\s*)(?:export\s+)?(?:default\s+)?(?:async\s+)?"
        r"function\s+(?P<name>\w+)\s*[(<]"
    ),
    re.compile(r"^(?P<indent>\s*)(?:export\s+)?(?:abstract\s+)?class\s+(?P<name>\w+)\b"),
    # TS/JS arrow and function-expression bindings
    re.compile(
        r"^(?P<indent>\s*)(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*"
        r"(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|\w+)\s*=>"
    ),
    # Go
    re.compile(r"^(?P<indent>)func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)\s*\("),
)

_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_MEMBER_CALL = re.compile(r"\.([A-Za-z_]\w*)\s*\(")


@dataclass(frozen=True)
class Definition:
    name: str
    path: str
    line: int
    source: str


def _body(lines: list[str], start: int, indent: str) -> str:
    """Extract a definition body: by dedent for indentation languages, by brace
    balance for the rest, capped either way."""
    head = lines[start]
    out = [head]

    if "{" in head:
        depth = head.count("{") - head.count("}")
        for ln in lines[start + 1 : start + 1 + MAX_BODY_LINES]:
            out.append(ln)
            depth += ln.count("{") - ln.count("}")
            if depth <= 0:
                break
        return "\n".join(out)

    for ln in lines[start + 1 : start + 1 + MAX_BODY_LINES]:
        if ln.strip() and not ln.startswith(indent + " ") and not ln.startswith(indent + "\t"):
            break
        out.append(ln)
    return "\n".join(out).rstrip()


def _scan(path: Path, rel: str) -> list[Definition]:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return []
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []

    lines = text.splitlines()
    found: list[Definition] = []
    for i, ln in enumerate(lines):
        for pattern in _DEF_PATTERNS:
            m = pattern.match(ln)
            if not m:
                continue
            found.append(
                Definition(
                    name=m.group("name"),
                    path=rel,
                    line=i + 1,
                    source=_body(lines, i, m.group("indent")),
                )
            )
            break
    return found


@lru_cache(maxsize=8)
def build_index(repo_root: str) -> dict[str, tuple[Definition, ...]]:
    """One pass over the repository's source files. Cached per root — a review
    resolves many names against the same tree."""
    root = Path(repo_root)
    index: dict[str, list[Definition]] = {}
    scanned = 0
    for path in sorted(root.rglob("*")):
        if scanned >= MAX_INDEX_FILES:
            break
        if path.suffix.lower() not in SOURCE_SUFFIXES or not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if not is_reviewable_path(rel):
            continue
        scanned += 1
        for d in _scan(path, rel):
            index.setdefault(d.name, []).append(d)
    return {k: tuple(v) for k, v in index.items()}


def called_names(snippet: str) -> set[str]:
    """Identifiers invoked as calls in the given text."""
    names = set(_CALL.findall(snippet)) | set(_MEMBER_CALL.findall(snippet))
    return {n for n in names if n not in _STOPWORDS and len(n) > 2}


def resolve(
    repo_root: str,
    changed_source: str,
    already_shown: set[str],
    limit: int = 12,
) -> list[Definition]:
    """Definitions called by the changed code that the reviewer would not otherwise
    see. `already_shown` holds paths already in the review context, so a function
    defined in a file under review is not sent twice.

    Ambiguous names are skipped rather than guessed: showing the wrong `validate`
    is worse than showing none, because the reviewer cannot tell it is wrong.
    """
    index = build_index(repo_root)
    out: list[Definition] = []
    for name in sorted(called_names(changed_source)):
        candidates = index.get(name)
        if not candidates or len(candidates) > 1:
            continue
        definition = candidates[0]
        if definition.path in already_shown:
            continue
        out.append(definition)
        if len(out) >= limit:
            break
    return out
