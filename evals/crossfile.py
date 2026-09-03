"""Defects that only exist between two files.

Every other mutator breaks a file in a way that file can be read for. These break
a *call*: the arguments at the call site stop matching the function's parameters,
and the function is defined somewhere else. Reading the changed file alone, there
is nothing visibly wrong — `charge(user, amount)` looks like every other line of
code. You have to know what `charge` takes.

Which makes this the measurement for repository expansion, and the only honest
one: if showing a reviewer the definitions its change calls is worth what it
costs, detection on these cases goes up and detection elsewhere does not move.

Ground truth is still mechanical — the line the call sits on. What is different
is that the corpus builder has to understand the repository well enough to find a
call whose callee it can locate, which is the same job the reviewer's index does,
done with the same code.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from pathlib import Path

from loupe.index import Definition, Index, imported_names

from .mutations import Mutation


@dataclass(frozen=True)
class Site:
    """One call, in one file, whose definition is in another."""

    path: str  # the caller, repo-relative
    source: str
    call: ast.Call
    callee: Definition
    params: list[str]

    @property
    def line(self) -> int:
        return self.call.lineno


def _required_params(defn: Definition, root: Path) -> list[str] | None:
    """Positional parameters with no default, or None if the shape is not simple.

    Anything with `*args`, `**kwargs` or defaults is skipped: dropping an argument
    from a call to one of those is not necessarily a defect, and a corpus entry
    that might not be a bug is worse than one fewer entry.
    """
    try:
        tree = ast.parse((root / defn.path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return None

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
            and node.name == defn.name
            and node.lineno == defn.start
        ):
            args = node.args
            if args.vararg or args.kwarg or args.defaults or args.kwonlyargs:
                return None
            names = [a.arg for a in args.posonlyargs + args.args]
            return names if names and names[0] not in ("self", "cls") else None
    return None


def find_sites(root: Path, caller: Path, index: Index) -> list[Site]:
    """Calls in `caller` to functions this repository defines elsewhere."""
    rel = caller.relative_to(root).as_posix()
    try:
        source = caller.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError, ValueError):
        return []

    imports = imported_names(source, rel)
    sites: list[Site] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        # One line only. A multi-line call cannot be edited by replacing a source
        # segment inside a single line, and the diff would be harder to read.
        if node.lineno != getattr(node, "end_lineno", node.lineno):
            continue
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            continue
        if node.func.id not in imports:
            continue
        module, original = imports[node.func.id]
        if original is None:
            continue
        candidates = [d for d in index.by_module.get(module, []) if d.name == original]
        if len(candidates) != 1 or candidates[0].path == rel:
            continue
        params = _required_params(candidates[0], root)
        if params is None or len(params) != len(node.args) or len(node.args) < 2:
            continue
        sites.append(Site(rel, source, node, candidates[0], params))
    return sites


def _rewrite(site: Site, new_args: list[str]) -> tuple[str, str, str] | None:
    """Rebuild the call with different arguments. Returns (source, before, after)."""
    segment = ast.get_source_segment(site.source, site.call)
    func = ast.get_source_segment(site.source, site.call.func)
    if not segment or not func:
        return None

    lines = site.source.splitlines()
    index = site.line - 1
    if segment not in lines[index]:
        return None

    before = lines[index]
    lines[index] = before.replace(segment, f"{func}({', '.join(new_args)})", 1)
    tail = "\n" if site.source.endswith("\n") else ""
    return "\n".join(lines) + tail, before.strip(), lines[index].strip()


def _arg_sources(site: Site) -> list[str] | None:
    args = [ast.get_source_segment(site.source, a) for a in site.call.args]
    return None if any(a is None for a in args) else [a for a in args if a is not None]


def drop_last_argument(site: Site, rng: random.Random) -> tuple[str, Mutation] | None:
    """Call the function with one argument too few — a TypeError at run time.

    Invisible in the changed file. `charge(user)` is only wrong if you know that
    `charge` takes two.
    """
    args = _arg_sources(site)
    if args is None:
        return None
    result = _rewrite(site, args[:-1])
    if result is None:
        return None
    source, before, after = result
    return source, Mutation(
        name="dropped_call_argument",
        category="correctness",
        line=site.line,
        before=before,
        after=after,
        description=(
            f"Call to {site.callee.name}() lost an argument; it is defined in "
            f"{site.callee.path} taking {len(site.params)} "
            f"({', '.join(site.params)})."
        ),
    )


def swap_first_arguments(site: Site, rng: random.Random) -> tuple[str, Mutation] | None:
    """Pass the right arguments in the wrong order. No crash, wrong answer."""
    args = _arg_sources(site)
    if args is None or args[0] == args[1]:
        return None
    result = _rewrite(site, [args[1], args[0], *args[2:]])
    if result is None:
        return None
    source, before, after = result
    return source, Mutation(
        name="swapped_call_arguments",
        category="correctness",
        line=site.line,
        before=before,
        after=after,
        description=(
            f"First two arguments to {site.callee.name}() swapped; it is defined "
            f"in {site.callee.path} as ({', '.join(site.params)})."
        ),
    )


MUTATORS = (drop_last_argument, swap_first_arguments)
