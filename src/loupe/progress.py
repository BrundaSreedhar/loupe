"""Saying what the review is doing while it does it.

A review is six to twelve model calls over a couple of minutes, and the terminal
used to show one spinner reading "Reviewing…" for all of it. When the result was
"no findings" there was no way to tell a clean review from a review where the
reviewers were shown nothing, the gate rejected everything, or a stage failed
quietly — three very different things that looked identical.

So each stage says what it did and with what. The unit is the stage, not the
file, because that is how the work is actually divided: reviewers read every
changed file in one call, while verification really is per file and says so.

Written as a callback handler rather than as print statements in the nodes: the
nodes are also run thousands of times by the eval harness, which wants none of
this, and a handler is opt-in per run. Node names arrive on the chain events and
the fan-out payloads carry the detail — which reviewer, which file — so nothing
has to be threaded through the graph to make this work.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
from rich.markup import escape

# Nodes worth narrating. Anything else in the graph is bookkeeping.
_ROUTERS = {"fan_out", "route_verify", "route_consensus"}


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one}" if n == 1 else f"{n} {many or one + 's'}"


def _safe(text: object) -> str:
    """A path from the diff, made safe to put in a markup string.

    File paths are attacker-controlled on a pull request, and these lines are
    printed with Rich markup switched on. An unmatched `[/bold]` in a filename
    raises MarkupError and takes the review down; a well-formed `[link=...]`
    renders as a clickable link the review never contained. Same rule as the
    source the reviewers read: it is data, not instructions — for the terminal
    as much as for the model.
    """
    return escape(str(text))


class Reporter(BaseCallbackHandler):
    """Prints a line per stage, in the order things actually finish."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._lock = threading.Lock()
        self._open: dict[UUID, tuple[str, dict]] = {}
        self._raw = 0

    # ─── plumbing ───────────────────────────────────────────────────────────

    def _say(self, text: str) -> None:
        with self._lock:
            self.console.print(f"  [dim]·[/dim] {text}")

    def on_chain_start(self, serialized, inputs, *, run_id=None, **kwargs) -> None:  # noqa: ANN001
        name = kwargs.get("name") or (serialized or {}).get("name") or ""
        if name in _ROUTERS or not name:
            return
        payload = inputs if isinstance(inputs, dict) else {}
        with self._lock:
            self._open[run_id] = (name, payload)
        line = self._starting(name, payload)
        if line:
            self._say(line)

    def on_chain_end(self, outputs, *, run_id=None, **kwargs) -> None:  # noqa: ANN001
        with self._lock:
            name, payload = self._open.pop(run_id, ("", {}))
        if not name or not isinstance(outputs, dict):
            return
        line = self._finished(name, payload, outputs)
        if line:
            self._say(line)

    # ─── what to say ────────────────────────────────────────────────────────

    def _starting(self, name: str, payload: dict) -> str | None:
        """Only the slow stages announce themselves. For the rest the result is
        the interesting part and arrives a moment later anyway."""
        if name == "specialist":
            role = payload.get("role", "generalist")
            files = len(payload.get("contexts") or {})
            return f"[cyan]{role}[/cyan] reviewer opens {_plural(files, 'file')}"
        if name == "verify":
            claims = len(payload.get("findings") or [])
            return (
                f"checking {_plural(claims, 'claim')} against the whole of "
                f"[bold]{_safe(payload.get('path', '?'))}[/bold]"
            )
        if name == "reconsider":
            finding = payload.get("finding")
            where = _safe(f"{finding.file}:{finding.line}") if finding is not None else "a finding"
            return f"asking a second time about {where} — the first two answers disagreed"
        if name == "warm_cache":
            return "writing the shared prefix to cache once, so the panel reads it"
        return None

    def _finished(self, name: str, payload: dict, outputs: dict) -> str | None:
        handler = getattr(self, f"_after_{name}", None)
        return handler(payload, outputs) if handler else None

    def _after_preflight(self, payload: dict, outputs: dict) -> str:
        issues = outputs.get("safety") or []
        secrets = sum(1 for i in issues if i.kind == "secret")
        planted = sum(1 for i in issues if i.kind == "injection")
        if not issues:
            return "nothing that looks like a credential, nothing addressed to the reviewer"
        parts = []
        if secrets:
            parts.append(f"[yellow]{_plural(secrets, 'possible credential')}[/yellow]")
        if planted:
            parts.append(f"[yellow]{_plural(planted, 'line')} written at the reviewer[/yellow]")
        return "found " + " and ".join(parts)

    def _after_prepare(self, payload: dict, outputs: dict) -> str:
        contexts = outputs.get("contexts") or {}
        dropped = outputs.get("dropped") or []
        names = ", ".join(_safe(p) for p in sorted(contexts)[:3])
        more = f" and {len(contexts) - 3} more" if len(contexts) > 3 else ""
        tail = f" · [yellow]{len(dropped)} left out, over budget[/yellow]" if dropped else ""
        return f"windowed {_plural(len(contexts), 'file')} — {names}{more}{tail}"

    def _after_expand(self, payload: dict, outputs: dict) -> str | None:
        references = outputs.get("references") or []
        edges = outputs.get("edges") or []
        if not edges:
            return None
        elsewhere = len({e.definition.path for e in edges})
        return (
            f"followed {_plural(len(edges), 'call')} out of the diff into "
            f"{_plural(elsewhere, 'file')}; {len(references)} definition(s) go in the prompt"
        )

    def _after_lint(self, payload: dict, outputs: dict) -> str | None:
        issues = outputs.get("lint_issues") or []
        if not issues:
            return None
        tools = "/".join(sorted({i.tool for i in issues}))
        return f"{tools} went first: {_plural(len(issues), 'issue')} the panel is told to skip"

    def _after_specialist(self, payload: dict, outputs: dict) -> str:
        role = payload.get("role", "generalist")
        found = len(outputs.get("findings") or [])
        with self._lock:
            self._raw += found
        verdict = "nothing" if not found else f"[bold]{_plural(found, 'finding')}[/bold]"
        return f"[cyan]{role}[/cyan] reviewer says {verdict}"

    def _after_dedupe(self, payload: dict, outputs: dict) -> str:
        merged = outputs.get("merged") or []
        return f"{self._raw} filed, {len(merged)} left after merging the duplicates"

    def _after_verify(self, payload: dict, outputs: dict) -> str:
        verdicts = outputs.get("verdicts") or []
        kept = sum(1 for v in verdicts if v.status == "CONFIRMED")
        gone = len(verdicts) - kept
        where = _safe(payload.get("path", "?"))
        if not kept:
            return f"[dim]{where}: nothing stood up ({_plural(gone, 'claim')} rejected)[/dim]"
        if not gone:
            return f"{where}: {_plural(kept, 'claim')} stood up"
        return f"{where}: {kept} stood up, {gone} rejected"

    def _after_finalize(self, payload: dict, outputs: dict) -> str:
        accepted = outputs.get("accepted") or []
        if not accepted:
            return "[dim]nothing survived — the report below says what was dropped[/dim]"
        return f"[bold]{_plural(len(accepted), 'finding')}[/bold] worth your time"


def reporter(console: Console | None) -> list[Any]:
    """Callbacks for one run. Empty when nobody is watching — the eval harness
    runs this graph thousands of times and wants none of it."""
    return [Reporter(console)] if console is not None else []
