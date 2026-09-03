"""Reading the repository once, so a reviewer can follow what a change calls.

A reviewer sees the changed file and nothing else. When a changed line calls
`validate(payload)`, it has two options and both are bad: guess what `validate`
does, which is where a share of the false alarms come from, or say nothing and
miss a real defect. This module finds the definition and puts it in the prompt.

**Show nothing rather than the wrong thing.** If three functions are called
`validate` and the reviewer is handed the wrong one, it will be fluently,
confidently wrong, and neither the reviewer nor the person reading the review can
tell. So every step here fails closed: a name that resolves to more than one
definition is skipped, a name imported from outside the repository is skipped —
never resolved to a local function that happens to share it — and a language
without a parser here contributes nothing at all.

Which is why this is Python only, via the standard library's own parser. A regex
that approximates a parser across ten languages resolves the wrong `validate`
eventually, and produces exactly the failure this design exists to avoid. Files in
other languages are still reviewed; they just get no expansion.

The definitions are unchanged repository files, so they go through the same secret
scan as the diff before leaving the machine — otherwise a credential nobody
touched in this change would be sent by a feature that reads whole files.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .filters import is_reviewable_path
from .project import is_ignored, load_ignore
from .safety import redact_text, scan_text
from .schema import FileContext, ReviewRequest

log = logging.getLogger(__name__)

MAX_FILE_BYTES = 400_000


@dataclass(frozen=True)
class Definition:
    name: str
    kind: str  # "function" | "class" | "method"
    path: str  # repo-relative, posix
    start: int
    end: int
    module: str  # dotted, as an import would name it

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start}-{self.end} {self.kind} {self.name}"


@dataclass
class Index:
    by_name: dict[str, list[Definition]] = field(default_factory=dict)
    by_module: dict[str, list[Definition]] = field(default_factory=dict)
    files: int = 0
    partial: bool = False  # hit the file cap; some of the repo is not in here

    def add(self, defn: Definition) -> None:
        self.by_name.setdefault(defn.name, []).append(defn)
        self.by_module.setdefault(defn.module, []).append(defn)


@dataclass(frozen=True)
class Reference:
    """One definition, resolved and rendered, ready to go into the prompt."""

    definition: Definition
    source: str  # line-numbered, possibly truncated, already secret-scanned
    callers: int  # changed call sites that referenced it
    tokens: int


@dataclass
class Stats:
    """What the expansion did, in enough detail to say so out loud."""

    files_indexed: int = 0
    partial_index: bool = False
    names: int = 0  # distinct names called from changed lines
    resolved: int = 0
    ambiguous: int = 0  # several definitions share the name — deliberately skipped
    unresolved: int = 0  # not defined in this repository (a library, a builtin)
    over_budget: int = 0
    secrets: int = 0  # definitions carrying something that looked like a credential


def module_of(rel_path: str) -> str:
    """`a/b/c.py` -> `a.b.c`, `a/b/__init__.py` -> `a.b`."""
    parts = Path(rel_path).with_suffix("").as_posix().split("/")
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _definitions_in(tree: ast.Module, rel_path: str) -> Iterable[Definition]:
    module = module_of(rel_path)

    def make(node, kind: str, name: str) -> Definition:
        return Definition(
            name=name,
            kind=kind,
            path=rel_path,
            start=node.lineno,
            end=getattr(node, "end_lineno", node.lineno) or node.lineno,
            module=module,
        )

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield make(node, "function", node.name)
        elif isinstance(node, ast.ClassDef):
            yield make(node, "class", node.name)
            for child in node.body:
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                    # Methods are indexed under their bare name. That makes them
                    # ambiguous the moment two classes both define `save`, which
                    # is the intended outcome: an attribute call gives no receiver
                    # type, so guessing which `save` was meant is exactly the
                    # confident-and-wrong failure this module refuses to produce.
                    yield make(child, "method", child.name)


def build(repo_root: str | Path, max_files: int) -> Index:
    """Index every Python file in the repository, or the first `max_files` of them."""
    root = Path(repo_root).resolve()
    index = Index()
    ignore = load_ignore(root)

    for path in sorted(root.rglob("*.py")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not is_reviewable_path(rel) or is_ignored(rel, ignore):
            continue
        if index.files >= max_files:
            index.partial = True
            break
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
            # A file that will not parse is a file with no definitions to offer.
            # Not a failure of the review.
            continue
        index.files += 1
        for defn in _definitions_in(tree, rel):
            index.add(defn)

    return index


def called_names(source: str, lines: set[int]) -> dict[str, int]:
    """Names called from the given lines, and how often. `a.b(x)` counts as `b`."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # A multi-line call starts where its callee is written, which is the line
        # the reader is looking at; node.lineno can be the same or earlier.
        where = getattr(func, "lineno", node.lineno)
        if where not in lines and node.lineno not in lines:
            continue
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def imported_names(source: str, rel_path: str) -> dict[str, tuple[str, str | None]]:
    """Local name -> (module, original name). `None` means the module itself.

    Knowing where a name came from is most of the precision here: `from
    app.validators import validate` says which `validate` is meant, and an import
    from a third-party package says the repository's own `validate` is the wrong
    answer rather than the only one.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}

    package = module_of(rel_path).rsplit(".", 1)[0] if "." in module_of(rel_path) else ""
    out: dict[str, tuple[str, str | None]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out[alias.asname or alias.name.split(".")[0]] = (alias.name, None)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # Relative: climb out of this file's own package.
                parts = package.split(".") if package else []
                parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                module = ".".join([p for p in parts if p] + ([module] if module else []))
            for alias in node.names:
                out[alias.asname or alias.name] = (module, alias.name)
    return out


def on_screen(defn: Definition, visible: dict[str, set[int]]) -> bool:
    """Is this definition already in what the reviewers were shown?

    Sending it again buys nothing and spends budget that an unseen definition
    could have used. A windowed context may have omitted the region it sits in,
    and then it is worth including after all.
    """
    return defn.start in visible.get(defn.path, set())


def resolve(
    name: str,
    imports: dict[str, tuple[str, str | None]],
    index: Index,
    own_path: str,
    visible: dict[str, set[int]],
) -> Definition | None:
    """The one definition this name means, or None.

    None is the common and correct answer. It means one of: imported from outside
    this repository, several definitions share the name, or the definition is
    already on screen somewhere in this review.
    """
    if name in imports:
        module, original = imports[name]
        if original is None:
            return None  # `import x` — the name is a module, not a definition
        candidates = [d for d in index.by_module.get(module, []) if d.name == original]
        # Nothing under that module means the import leaves the repository. Falling
        # back to a same-named local function here is precisely how a reviewer ends
        # up reading the wrong `validate`.
        if len(candidates) != 1:
            return None
        return None if on_screen(candidates[0], visible) else candidates[0]

    candidates = index.by_name.get(name, [])
    # A definition in the calling file shadows one of the same name elsewhere.
    pool = [d for d in candidates if d.path == own_path] or candidates
    if len(pool) != 1:
        return None
    return None if on_screen(pool[0], visible) else pool[0]


def render(defn: Definition, repo_root: str | Path, max_lines: int) -> str | None:
    """The definition's source, line-numbered to match how the diff is shown."""
    path = Path(repo_root) / defn.path
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if defn.start > len(lines):
        return None

    end = min(defn.end, len(lines))
    body = lines[defn.start - 1 : end]
    truncated = len(body) > max_lines
    body = body[:max_lines]
    width = len(str(defn.start + len(body)))
    out = [f" {defn.start + i:>{width}}| {text}" for i, text in enumerate(body)]
    if truncated:
        out.append(f"      … {end - defn.start + 1 - max_lines} more line(s) …")
    return "\n".join(out)


def gather(
    request: ReviewRequest,
    contexts: dict[str, FileContext],
    *,
    max_files: int,
    max_lines: int,
    token_budget: int,
    estimate,
    visible_lines,
    on_secret: str = "redact",  # noqa: S107 — a policy name, not a credential
) -> tuple[list[Reference], Stats]:
    """Resolve what the changed Python lines call, and render what resolves.

    `estimate` and `visible_lines` are passed in rather than imported so this stays
    a pure function of its inputs — it is the piece most likely to need testing
    against a scratch repository on disk.
    """
    stats = Stats()
    python_files = [
        fd
        for fd in request.reviewable
        if fd.path.endswith((".py", ".pyi")) and fd.path in contexts and fd.content_after
    ]
    if not python_files:
        return [], stats

    index = build(request.repo_root, max_files)
    stats.files_indexed = index.files
    stats.partial_index = index.partial
    if not index.files:
        return [], stats

    # A definition called from three changed files is worth more room than one
    # called once, so count call sites across the whole change before choosing.
    # Every line of every file the reviewers can see, so a definition that is
    # already in front of them is not sent a second time.
    visible = {path: visible_lines(ctx) for path, ctx in contexts.items()}

    wanted: dict[Definition, int] = {}
    for fd in python_files:
        source = fd.content_after or ""
        imports = imported_names(source, fd.path)
        for name, count in called_names(source, fd.changed_lines).items():
            stats.names += 1
            defn = resolve(name, imports, index, fd.path, visible)
            if defn is None:
                if len(index.by_name.get(name, [])) > 1:
                    stats.ambiguous += 1
                else:
                    stats.unresolved += 1
                continue
            wanted[defn] = wanted.get(defn, 0) + count

    references: list[Reference] = []
    spent = 0
    for defn, callers in sorted(wanted.items(), key=lambda kv: (-kv[1], kv[0].label)):
        source = render(defn, request.repo_root, max_lines)
        if source is None:
            continue
        if scan_text(source, defn.path):
            # Same policy as the diff. This file was not touched by the change,
            # which makes it more surprising to find a key in, not less.
            stats.secrets += 1
            if on_secret == "block":  # noqa: S105 — a policy name, not a credential
                continue
            if on_secret == "redact":  # noqa: S105 — as above
                source = redact_text(source)
        tokens = estimate(source)
        if spent + tokens > token_budget:
            stats.over_budget += 1
            continue
        spent += tokens
        stats.resolved += 1
        references.append(Reference(defn, source, callers, tokens))

    return references, stats


def enabled(setting: str, source: str) -> bool:
    """`auto` means local only: a pull request's repository is not on this disk,
    so there is nothing to index and every name would resolve to None."""
    if setting == "on":
        return True
    if setting == "off":
        return False
    return source == "local"
