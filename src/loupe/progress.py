"""Saying what the review is doing while it does it.

A review is six to twelve model calls over a couple of minutes, and the terminal
used to show one spinner reading "Reviewing…" for all of it. When the result was
"no findings" there was no way to tell a clean review from a review where the
reviewers were shown nothing, the gate rejected everything, or a stage failed
quietly — three very different things that looked identical.

Two halves, because they answer different questions:

- **What is running right now**, as a spinner per piece of work in flight. Four
  reviewers run at once and one model call takes the better part of a minute, so
  without this the screen is motionless long enough to look hung.
- **What each stage did**, printed above the spinners as it finishes and left
  there. That is the half you read afterwards.

Written as a callback handler rather than as print statements in the nodes: the
nodes are also run thousands of times by the eval harness, which wants none of
this, and a handler is opt-in per run. Node names arrive on the chain events and
the fan-out payloads carry the detail — which reviewer, which file — so nothing
has to be threaded through the graph to make this work.

Animation is off when the output is not a terminal. A redirected log full of
spinner frames is worse than no spinner, and the finished-stage lines are the
half worth keeping there anyway.
"""

from __future__ import annotations

import threading
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from rich.console import Console
from rich.markup import escape
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

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
    """A spinner on everything running, a line for everything that finishes."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._lock = threading.Lock()
        self._open: dict[UUID, tuple[str, dict]] = {}
        self._tasks: dict[UUID, int] = {}
        self._raw = 0
        self._closed = False
        # A pipe redraws nothing, so the frames would only pile up as noise.
        self._animate = console.is_terminal
        self._progress = (
            Progress(
                SpinnerColumn(style="cyan"),
                TextColumn("[dim]{task.description}[/dim]"),
                TimeElapsedColumn(),
                console=console,
                transient=True,  # the spinners are the live view, not the record
            )
            if self._animate
            else None
        )

    # ─── lifecycle ──────────────────────────────────────────────────────────

    def close(self) -> None:
        """Stop animating. Safe to call twice, and called even when the review
        raises — a spinner still turning over a traceback is its own bug."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            progress = self._progress
        if progress is not None:
            progress.stop()

    def __enter__(self) -> Reporter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _start_task(self, run_id: UUID, label: str) -> None:
        if self._progress is None:
            return
        with self._lock:
            if self._closed:
                return
            if not self._tasks:
                self._progress.start()
            self._tasks[run_id] = self._progress.add_task(label, total=None)

    def _end_task(self, run_id: UUID) -> None:
        if self._progress is None:
            return
        with self._lock:
            task = self._tasks.pop(run_id, None)
            if task is not None:
                self._progress.remove_task(task)

    def _say(self, text: str) -> None:
        """Print above the spinners, where it stays once they are gone."""
        with self._lock:
            if self._closed:
                return
            console = self._progress.console if self._progress else self.console
        console.print(f"  [dim]·[/dim] {text}")

    # ─── graph events ───────────────────────────────────────────────────────

    def on_chain_start(self, serialized, inputs, *, run_id=None, **kwargs) -> None:  # noqa: ANN001
        name = kwargs.get("name") or (serialized or {}).get("name") or ""
        if name in _ROUTERS or not name:
            return
        payload = inputs if isinstance(inputs, dict) else {}
        with self._lock:
            self._open[run_id] = (name, payload)
        label = self._label(name, payload)
        if not label:
            return
        if self._progress is None:
            # No terminal, so no spinner to carry this. Print it, or a piped log
            # loses which reviewer opened what and which file is being checked.
            self._say(label)
        else:
            self._start_task(run_id, label)

    def on_chain_end(self, outputs, *, run_id=None, **kwargs) -> None:  # noqa: ANN001
        self._end_task(run_id)
        with self._lock:
            name, payload = self._open.pop(run_id, ("", {}))
        if not name or not isinstance(outputs, dict):
            return
        handler = getattr(self, f"_after_{name}", None)
        line = handler(payload, outputs) if handler else None
        if line:
            self._say(line)

    def on_chain_error(self, error: BaseException, *, run_id=None, **kwargs) -> None:  # noqa: ANN001
        """A stage that raised must not leave its spinner turning."""
        self._end_task(run_id)
        with self._lock:
            name, _ = self._open.pop(run_id, ("", {}))
        if name:
            self._say(f"[red]{name} failed[/red] — {type(error).__name__}")

    # ─── what the spinner says while a stage runs ───────────────────────────

    def _label(self, name: str, payload: dict) -> str | None:
        if name == "specialist":
            role = payload.get("role", "generalist")
            files = len(payload.get("contexts") or {})
            return f"{role} reviewer reading {_plural(files, 'file')}"
        if name == "verify":
            claims = len(payload.get("findings") or [])
            return f"checking {_plural(claims, 'claim')} against {_safe(payload.get('path', '?'))}"
        if name == "reconsider":
            finding = payload.get("finding")
            where = _safe(f"{finding.file}:{finding.line}") if finding is not None else "a finding"
            return f"asking again about {where}"
        return {
            "preflight": "scanning for credentials and planted instructions",
            "prepare": "windowing the changed files",
            "expand": "following what the change calls",
            "lint": "letting the repo's own linters go first",
            "warm_cache": "warming the shared prefix",
            "dedupe": "merging findings that describe one defect",
            "finalize": "ranking what survived",
        }.get(name)

    # ─── what each stage leaves behind ──────────────────────────────────────

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


def reporter(console: Console | None) -> Reporter | None:
    """The watcher for one run, or None when nobody is looking — the eval harness
    runs this graph thousands of times, and `--output json` would be corrupted by
    anything else written to stdout."""
    return Reporter(console) if console is not None else None


def callbacks(watcher: Reporter | None) -> list[Any]:
    return [watcher] if watcher is not None else []
