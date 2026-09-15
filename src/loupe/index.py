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
import re
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
    # The first sentence of its docstring, or "". Not for the model — for the
    # person reading a review of a codebase they have never seen, who needs to
    # know what `gather()` is before any finding about it means anything.
    doc: str = ""

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start}-{self.end} {self.kind} {self.name}"


@dataclass
class Index:
    by_name: dict[str, list[Definition]] = field(default_factory=dict)
    by_module: dict[str, list[Definition]] = field(default_factory=dict)
    # path -> the first sentence of the file's module docstring.
    summaries: dict[str, str] = field(default_factory=dict)
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


@dataclass(frozen=True)
class Edge:
    """One call from a changed line to somewhere it is defined.

    Recorded whether or not the definition went into a prompt: a call to a
    function defined in another file *of the same change* is the most useful
    thing to show a reader, and it is exactly the case the prompt leaves out
    because the reviewer can already see it.
    """

    caller: str  # the changed file the call is written in
    name: str
    definition: Definition
    # Whether the reviewers could read this definition's source — either because
    # it was injected, or because it is inside a file they were already given.
    # Not "was it injected": a callee living in another file of the same change
    # is deliberately not injected, and is the case this class exists to record.
    shown: bool


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
    changed: int = 0  # definitions this change edited


@dataclass
class Expansion:
    """Everything one pass of the index produced.

    A tuple was fine at three; the fourth is the point at which the caller starts
    unpacking positionally and getting it wrong.
    """

    references: list[Reference]
    stats: Stats
    edges: list[Edge]
    # path -> one line on what that file is for, for every file in `edges`.
    summaries: dict[str, str] = field(default_factory=dict)
    # The definitions this change edited. Not derived from the repository index —
    # only from the changed files' own trees — so it survives an index that found
    # nothing, and it is what the reverse lookups are keyed by.
    changed: list[Definition] = field(default_factory=list)


_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def summarise(doc: str | None, limit: int = 160) -> str:
    """The first sentence of a docstring, on one line.

    These files open with a paragraph explaining what they are for, which is
    exactly what someone new to the codebase needs and exactly what a diff does
    not give them. One sentence is the part that fits in a box.
    """
    if not doc:
        return ""
    first = _SENTENCE_END.split(doc.strip(), 1)[0]
    text = " ".join(first.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


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
            doc=summarise(ast.get_docstring(node)),
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
        index.summaries[rel] = summarise(ast.get_docstring(tree))
        for defn in _definitions_in(tree, rel):
            index.add(defn)

    return index


def parse(source: str) -> ast.Module | None:
    """A file's syntax tree, or None if it will not parse.

    Handed around rather than re-derived: the callers and the imports of one file
    are both read from it, and parsing twice doubles the cost of every review for
    nothing.
    """
    try:
        return ast.parse(source)
    except (SyntaxError, ValueError):
        return None


def called_names(source: str, lines: set[int], tree: ast.Module | None = None) -> dict[str, int]:
    """Names called from the given lines, and how often. `a.b(x)` counts as `b`."""
    tree = tree if tree is not None else parse(source)
    if tree is None:
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


def _definition_spans(
    node: ast.AST, rel_path: str, module: str, in_class: bool = False
) -> Iterable[tuple[int, int, Definition]]:
    """Every def/class in the tree as (span_start, span_end, Definition).

    `span_start` reaches back over the decorators, which `Definition.start` does
    not: a change to `@app.route(...)` is a change to the function under it, but
    `start` has to keep pointing at the `def` line because `render` slices source
    from it. Recursion carries `in_class` so a function in a class body is a
    method however many `if` blocks sit between them.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            is_class = isinstance(child, ast.ClassDef)
            kind = "class" if is_class else ("method" if in_class else "function")
            end = getattr(child, "end_lineno", child.lineno) or child.lineno
            yield (
                min([child.lineno, *(d.lineno for d in child.decorator_list)]),
                end,
                Definition(
                    name=child.name,
                    kind=kind,
                    path=rel_path,
                    start=child.lineno,
                    end=end,
                    module=module,
                    doc=summarise(ast.get_docstring(child)),
                ),
            )
            yield from _definition_spans(child, rel_path, module, in_class=is_class)
        else:
            yield from _definition_spans(child, rel_path, module, in_class=in_class)


def changed_definitions(
    source: str, lines: set[int], rel_path: str, tree: ast.Module | None = None
) -> list[Definition]:
    """The definitions this change edited — innermost first, in file order.

    A diff is a set of line numbers, which is the wrong unit for almost every
    question worth asking about a change. "Which functions did this touch" is the
    unit that a caller lookup, a test lookup and a history lookup all key on, and
    it is also what lets a windowed file show whole functions instead of fragments.

    Innermost wins: a change inside a nested helper is a change to the helper, not
    to the class three levels out. A line that sits in no definition at all — an
    import, a module-level constant — contributes nothing rather than the file,
    because "the whole module changed" is not a useful answer to any of them.
    """
    tree = tree if tree is not None else parse(source)
    if tree is None or not lines:
        return []

    spans = list(_definition_spans(tree, rel_path, module_of(rel_path)))
    if not spans:
        return []

    found: dict[Definition, None] = {}
    for line in sorted(lines):
        enclosing = [s for s in spans if s[0] <= line <= s[1]]
        if enclosing:
            # Latest start = deepest nesting, since a child always starts after
            # the parent it is written inside.
            found[max(enclosing, key=lambda s: s[0])[2]] = None
    return sorted(found, key=lambda d: (d.start, d.name))


def imported_names(
    source: str, rel_path: str, tree: ast.Module | None = None
) -> dict[str, tuple[str, str | None]]:
    """Local name -> (module, original name). `None` means the module itself.

    Knowing where a name came from is most of the precision here: `from
    app.validators import validate` says which `validate` is meant, and an import
    from a third-party package says the repository's own `validate` is the wrong
    answer rather than the only one.
    """
    tree = tree if tree is not None else parse(source)
    if tree is None:
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


def locate(
    name: str,
    imports: dict[str, tuple[str, str | None]],
    index: Index,
    own_path: str,
) -> Definition | None:
    """Where this name is defined in this repository, or None.

    None is the common and correct answer: imported from outside the repository,
    or several definitions share the name. Says nothing about whether the
    definition is worth putting in a prompt — that is `resolve`.
    """
    if name in imports:
        module, original = imports[name]
        if original is None:
            return None  # `import x` — the name is a module, not a definition
        candidates = [d for d in index.by_module.get(module, []) if d.name == original]
        # Nothing under that module means the import leaves the repository. Falling
        # back to a same-named local function here is precisely how a reviewer ends
        # up reading the wrong `validate`.
        return candidates[0] if len(candidates) == 1 else None

    candidates = index.by_name.get(name, [])
    # A definition in the calling file shadows one of the same name elsewhere.
    pool = [d for d in candidates if d.path == own_path] or candidates
    return pool[0] if len(pool) == 1 else None


def resolve(
    name: str,
    imports: dict[str, tuple[str, str | None]],
    index: Index,
    own_path: str,
    visible: dict[str, set[int]],
) -> Definition | None:
    """The definition worth sending to a reviewer, or None.

    Adds one rule to `locate`: a definition already on screen is not sent twice.
    The change map still wants that edge, which is why the two are separate.
    """
    found = locate(name, imports, index, own_path)
    if found is None or on_screen(found, visible):
        return None
    return found


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
) -> Expansion:
    """Resolve what the changed Python lines call, and render what resolves.

    Returns the definitions worth sending, what happened while finding them, and
    every edge discovered — the last of which is what the change map draws, and
    deliberately includes edges whose definition the prompt left out.

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
        return Expansion([], stats, [])

    # Computed before the repository index, and returned even when that index
    # comes back empty: this depends only on the changed files themselves.
    changed: list[Definition] = []
    trees: dict[str, ast.Module] = {}
    for fd in python_files:
        tree = parse(fd.content_after or "")
        if tree is None:
            continue
        trees[fd.path] = tree
        changed.extend(
            changed_definitions(fd.content_after or "", fd.changed_lines, fd.path, tree)
        )
    stats.changed = len(changed)

    index = build(request.repo_root, max_files)
    stats.files_indexed = index.files
    stats.partial_index = index.partial
    if not index.files:
        return Expansion([], stats, [], changed=changed)

    # A definition called from three changed files is worth more room than one
    # called once, so count call sites across the whole change before choosing.
    # Every line of every file the reviewers can see, so a definition that is
    # already in front of them is not sent a second time.
    visible = {path: visible_lines(ctx) for path, ctx in contexts.items()}

    wanted: dict[Definition, int] = {}
    found: list[tuple[str, str, Definition]] = []
    changed_summaries: dict[str, str] = {}
    for fd in python_files:
        source = fd.content_after or ""
        tree = trees.get(fd.path)
        if tree is None:
            continue
        changed_summaries[fd.path] = summarise(ast.get_docstring(tree))
        imports = imported_names(source, fd.path, tree)
        for name, count in called_names(source, fd.changed_lines, tree).items():
            stats.names += 1
            defn = locate(name, imports, index, fd.path)
            if defn is None:
                if len(index.by_name.get(name, [])) > 1:
                    stats.ambiguous += 1
                else:
                    stats.unresolved += 1
                continue
            found.append((fd.path, name, defn))
            if on_screen(defn, visible):
                # Already in front of the reviewer. Still an edge worth drawing.
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

    sent = {r.definition for r in references}
    edges = [
        Edge(
            caller=caller,
            name=name,
            definition=defn,
            shown=defn in sent or on_screen(defn, visible),
        )
        for caller, name, defn in sorted(found, key=lambda e: (e[0], e[1]))
    ]
    # Callers first so a changed file describes itself as it is now, then the
    # rest of the repository as it stands on disk.
    summaries = {
        path: index.summaries.get(path, "")
        for path in {e.definition.path for e in edges}
    }
    summaries.update(changed_summaries)
    return Expansion(references, stats, edges, summaries, changed)


def enabled(setting: str, source: str) -> bool:
    """`auto` means local only: a pull request's repository is not on this disk,
    so there is nothing to index and every name would resolve to None."""
    if setting == "on":
        return True
    if setting == "off":
        return False
    return source == "local"
