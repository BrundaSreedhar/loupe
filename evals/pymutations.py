"""Seeded defects that need a parser.

`mutations.py` is line-based on purpose, so one corpus builder covers Python,
TypeScript and JavaScript. That buys breadth and costs precision: a regex cannot
tell a call from a string containing one, cannot find where a call's arguments end
when they span lines, and cannot know whether the function being called is a
coroutine. Those are exactly the defects worth seeding, so this module gives them
up on other languages and uses `ast`.

Every mutator here obeys three rules, enforced centrally by the decorator rather
than trusted per-mutator:

- **The result must parse.** A case whose file is syntactically broken is not a
  test of the reviewer; it is a test of whether it notices a SyntaxError, and every
  reviewer notices.
- **The result must differ from the original**, or the case is a clean control
  mislabelled as a defect — which scores as a miss forever and looks like the
  reviewer's fault.
- **A deletion reports `after` as a parenthesised note**, following
  `mutations.remove_guard`. There is no "after" text when a line is removed, and a
  reader of the corpus needs to be able to tell the two shapes apart.
- **The reported line must be the line that changed.** That number is the ground
  truth the whole harness rests on; if it drifts, recall becomes unmeasurable
  without anyone noticing.

None of these seed a defect a linter would catch. `ruff` already runs before the
reviewers and they are told not to re-report what it found, so a mutator that
plants a B006 or a bare-except measures the pre-pass, not the panel.
"""

from __future__ import annotations

import ast
import random
import re
from collections.abc import Callable

from .mutations import Mutation

# (source, rng) -> (mutated source, Mutation) | None
SourceMutator = Callable[[str, random.Random], "tuple[str, Mutation] | None"]

_REGISTRY: dict[str, Callable] = {}


def _adapt(fn: SourceMutator) -> Callable:
    """Wrap a source-level mutator in the line-list protocol `corpus.attempt` uses,
    and enforce the three rules above on its way out."""

    def wrapped(lines: list[str], rng: random.Random) -> Mutation | None:
        source = "\n".join(lines)
        try:
            result = fn(source, rng)
        except (SyntaxError, ValueError, IndexError, AttributeError):
            # A mutator that trips over unusual source contributes no case. It
            # must never take the corpus build down with it.
            return None
        if result is None:
            return None
        mutated, mutation = result
        if mutated == source:
            return None
        try:
            # `ast.parse` raises rather than returning None, so this has to be a
            # try — written as a truthiness check the first time, which meant the
            # rule this decorator documents was not enforced at all.
            ast.parse(mutated)
        except (SyntaxError, ValueError):
            return None
        new_lines = mutated.splitlines()
        if not 1 <= mutation.line <= len(new_lines):
            return None
        lines[:] = new_lines
        return mutation

    wrapped.__name__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    # Read by `corpus.attempt`: these only apply to Python, and offering them a
    # TypeScript file wastes an attempt that another mutator could have used.
    wrapped.suffixes = (".py", ".pyi")
    return wrapped


def python_mutator(fn: SourceMutator) -> SourceMutator:
    _REGISTRY[fn.__name__] = _adapt(fn)
    return fn


def all_python_mutators() -> dict[str, Callable]:
    return dict(_REGISTRY)


# ─── source surgery ─────────────────────────────────────────────────────────


def _splice(source: str, node: ast.AST, new_text: str) -> tuple[str, int, str, str] | None:
    """Replace one single-line node's span. Returns (source, line, before, after).

    Byte offsets, not character offsets: `col_offset` indexes the UTF-8 encoding of
    the line, so slicing the `str` puts the edit in the wrong place the first time a
    file contains a non-ASCII character above the change.
    """
    start, end = node.lineno, getattr(node, "end_lineno", node.lineno)
    if start != end:
        return None
    lines = source.splitlines()
    idx = start - 1
    raw = lines[idx].encode("utf-8")
    head = raw[: node.col_offset].decode("utf-8")
    tail_text = raw[node.end_col_offset :].decode("utf-8")
    rebuilt = head + new_text + tail_text
    if rebuilt == lines[idx]:
        return None
    before = lines[idx].strip()
    lines[idx] = rebuilt
    tail = "\n" if source.endswith("\n") else ""
    return "\n".join(lines) + tail, start, before, rebuilt.strip()


def _seg(source: str, node: ast.AST) -> str | None:
    return ast.get_source_segment(source, node)


def _plain_params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str] | None:
    """Positional parameters, or None when the signature is not simple enough for a
    dropped or reordered argument to be unambiguously wrong."""
    a = fn.args
    if a.vararg or a.kwarg or a.defaults or a.kwonlyargs:
        return None
    names = [p.arg for p in a.posonlyargs + a.args]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names or None


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            # A name defined twice is ambiguous; drop both rather than guess which
            # one a call site meant.
            out[node.name] = None if node.name in out else node  # type: ignore[assignment]
    return {k: v for k, v in out.items() if v is not None}


def _pick(rng: random.Random, items: list):
    return rng.choice(items) if items else None


# ─── correctness ────────────────────────────────────────────────────────────


@python_mutator
def swap_call_arguments(source: str, rng: random.Random):
    """Pass the right arguments in the wrong order, to a function defined in this file.

    No crash and no linter warning — just a wrong answer. A regex cannot do this
    safely: it cannot find where the argument list ends, and it cannot check that
    the callee really takes two positionals, so it would plant non-defects.
    """
    tree = ast.parse(source)
    defs = _functions(tree)
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if len(node.args) < 2 or node.lineno != getattr(node, "end_lineno", node.lineno):
            continue
        target = defs.get(node.func.id)
        if target is None:
            continue
        params = _plain_params(target)
        if params is None or len(params) != len(node.args):
            continue
        segs = [_seg(source, a) for a in node.args]
        if any(s is None for s in segs) or segs[0] == segs[1]:
            continue
        candidates.append((node, [s for s in segs if s], params, target))

    chosen = _pick(rng, candidates)
    if chosen is None:
        return None
    node, segs, params, target = chosen
    func = _seg(source, node.func)
    if not func:
        return None
    swapped = [segs[1], segs[0], *segs[2:]]
    spliced = _splice(source, node, f"{func}({', '.join(swapped)})")
    if spliced is None:
        return None
    mutated, line, before, after = spliced
    return mutated, Mutation(
        name="swap_call_arguments",
        category="correctness",
        line=line,
        before=before,
        after=after,
        description=(
            f"First two arguments to {node.func.id}() swapped; it is defined at "
            f"line {target.lineno} as ({', '.join(params)})."
        ),
    )


@python_mutator
def unawaited_async_call(source: str, rng: random.Random):
    """Drop `await` from a call to an `async def` in this same file.

    The coroutine is created and never run: the work silently does not happen and
    the caller gets a coroutine object where it expected a value. `mutations.py`
    has a regex version, but it cannot confirm the callee is a coroutine, so it
    sometimes strips an `await` in a context where that is not a defect.
    """
    tree = ast.parse(source)
    coroutines = {n.name for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)}
    if not coroutines:
        return None

    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Await) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        name = func.id if isinstance(func, ast.Name) else None
        if name not in coroutines:
            continue
        inner = _seg(source, node.value)
        if inner is None or node.lineno != getattr(node, "end_lineno", node.lineno):
            continue
        candidates.append((node, inner, name))

    chosen = _pick(rng, candidates)
    if chosen is None:
        return None
    node, inner, name = chosen
    spliced = _splice(source, node, inner)
    if spliced is None:
        return None
    mutated, line, before, after = spliced
    return mutated, Mutation(
        name="unawaited_async_call",
        category="correctness",
        line=line,
        before=before,
        after=after,
        description=(
            f"`await` removed from a call to {name}(), which is an async def in "
            "this file. The coroutine is never awaited, so the work does not happen."
        ),
    )


@python_mutator
def invert_bool_operator(source: str, rng: random.Random):
    """Flip `and` to `or` (or back) inside a condition.

    Scoped by the parser to the *test* of an if/while, so it lands on control flow
    rather than on a value expression where either operator may be intended.
    """
    tree = ast.parse(source)
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If | ast.While):
            continue
        test = node.test
        if isinstance(test, ast.BoolOp) and len(test.values) >= 2:
            seg = _seg(source, test)
            if seg and test.lineno == getattr(test, "end_lineno", test.lineno):
                candidates.append((test, seg))

    chosen = _pick(rng, candidates)
    if chosen is None:
        return None
    test, seg = chosen
    was, now = ("and", "or") if isinstance(test.op, ast.And) else ("or", "and")
    flipped = re.sub(rf"\b{was}\b", now, seg, count=1)
    if flipped == seg:
        return None
    spliced = _splice(source, test, flipped)
    if spliced is None:
        return None
    mutated, line, before, after = spliced
    return mutated, Mutation(
        name="invert_bool_operator",
        category="correctness",
        line=line,
        before=before,
        after=after,
        description=f"`{was}` changed to `{now}` in a condition, inverting when the branch runs.",
    )


@python_mutator
def drop_call_argument(source: str, rng: random.Random):
    """Call a function in this file with one argument too few — a TypeError at run time.

    The unambiguous end of the scale, and deliberately an easy case: an eval needs
    defects a competent reviewer must catch, or a low score cannot distinguish a
    hard corpus from a broken reviewer.
    """
    tree = ast.parse(source)
    defs = _functions(tree)
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if len(node.args) < 2 or node.lineno != getattr(node, "end_lineno", node.lineno):
            continue
        target = defs.get(node.func.id)
        if target is None:
            continue
        params = _plain_params(target)
        if params is None or len(params) != len(node.args):
            continue
        segs = [_seg(source, a) for a in node.args]
        if any(s is None for s in segs):
            continue
        candidates.append((node, [s for s in segs if s], params, target))

    chosen = _pick(rng, candidates)
    if chosen is None:
        return None
    node, segs, params, target = chosen
    func = _seg(source, node.func)
    if not func:
        return None
    spliced = _splice(source, node, f"{func}({', '.join(segs[:-1])})")
    if spliced is None:
        return None
    mutated, line, before, after = spliced
    return mutated, Mutation(
        name="drop_call_argument",
        category="correctness",
        line=line,
        before=before,
        after=after,
        description=(
            f"Call to {node.func.id}() lost its last argument; it is defined at "
            f"line {target.lineno} taking {len(params)} ({', '.join(params)})."
        ),
    )


@python_mutator
def narrow_exception_catch(source: str, rng: random.Random):
    """Drop one exception type from a multi-type `except`, so it propagates.

    The opposite failure to a bare `except`, and unlike that one no linter reports
    it: catching fewer things looks tidier than catching more.
    """
    tree = ast.parse(source)
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        t = node.type
        if not isinstance(t, ast.Tuple) or len(t.elts) < 2:
            continue
        if t.lineno != getattr(t, "end_lineno", t.lineno):
            continue
        segs = [_seg(source, e) for e in t.elts]
        if any(s is None for s in segs):
            continue
        candidates.append((t, [s for s in segs if s]))

    chosen = _pick(rng, candidates)
    if chosen is None:
        return None
    t, segs = chosen
    dropped = segs[-1]
    kept = segs[:-1]
    new_text = f"({', '.join(kept)})" if len(kept) > 1 else kept[0]
    spliced = _splice(source, t, new_text)
    if spliced is None:
        return None
    mutated, line, before, after = spliced
    return mutated, Mutation(
        name="narrow_exception_catch",
        category="correctness",
        line=line,
        before=before,
        after=after,
        description=(
            f"`{dropped}` removed from the handled exception types, so it now "
            "propagates out of a path written to absorb it."
        ),
    )


@python_mutator
def swallow_exception(source: str, rng: random.Random):
    """Delete a `raise` from an `except` block, leaving the failure silent.

    The handler keeps its logging, so the code still looks as though it deals with
    the error. This is the shape behind most "it returned success and did nothing"
    bugs, and the one defect class this project has shipped three times itself.
    """
    tree = ast.parse(source)
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or len(node.body) < 2:
            continue
        for stmt in node.body:
            # Only a trailing re-raise: removing a raise from the middle can leave
            # unreachable code behind it, which is a different defect.
            if isinstance(stmt, ast.Raise) and stmt is node.body[-1]:
                if stmt.lineno == getattr(stmt, "end_lineno", stmt.lineno):
                    candidates.append(stmt)

    chosen = _pick(rng, candidates)
    if chosen is None:
        return None
    lines = source.splitlines()
    idx = chosen.lineno - 1
    before = lines[idx].strip()
    del lines[idx]
    tail = "\n" if source.endswith("\n") else ""
    mutated = "\n".join(lines) + tail
    return mutated, Mutation(
        name="swallow_exception",
        category="correctness",
        # The line the raise used to occupy; after deletion that is the line the
        # handler now ends on, which is where a reviewer would anchor the finding.
        line=min(chosen.lineno, len(lines)) or 1,
        before=before,
        after="(re-raise deleted)",
        description=(
            "A re-raise was removed from an except block. The exception is now "
            "swallowed and the caller is told the operation succeeded."
        ),
    )
