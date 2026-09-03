"""Rendering and PR posting.

Deliberately outside the graph: posting is a side effect, and the eval harness
runs the graph thousands of times.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass

import httpx
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Text

from .grounding import cited_text
from .schema import ReviewRequest, ReviewResult

_SEVERITY_STYLE = {"high": "bold red", "medium": "yellow", "low": "cyan"}


def render_json(result: ReviewResult, console: Console | None = None) -> None:
    """Stable machine-readable review output for CI and coding agents."""
    console = console or Console()

    def encode(value):
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        raise TypeError(f"cannot encode {type(value).__name__}")

    console.print_json(json.dumps(result.model_dump(), default=encode, sort_keys=True))


def render_problems(result: ReviewResult, console: Console) -> None:
    """Things that went wrong but did not stop the review.

    Shown even when there are no findings, because "no findings" and "no findings
    because three stages failed" must not look the same."""
    if not result.problems:
        return
    console.print()
    console.print("  [bold]Problems during this review[/bold]")
    for p in result.problems:
        marker = "[red]![/red]" if p.severity == "error" else "[yellow]·[/yellow]"
        line = Text.from_markup(f"{marker} [dim]{p.stage}[/dim]  ")
        line.append(p.detail)
        # Padding keeps wrapped continuation lines under the text, not at column 0.
        console.print(Padding(line, (0, 0, 0, 4)))


def _short(n: int) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else str(n)


def render_tokens(result: ReviewResult, console: Console) -> None:
    """What the review cost. Printed whether or not it found anything — a clean
    review is not a free one, and the only way to know four reviewers are worth
    one is to see the bill next to the findings."""
    t = result.tokens
    if not t.calls:
        return
    line = (
        f"  [dim]{t.calls} call(s)   ·   {_short(t.input)} in · "
        f"{_short(t.output)} out"
    )
    if t.reasoning:
        line += f" ({_short(t.reasoning)} thinking)"
    if t.cache_read or t.cache_write:
        line += (
            f"   ·   cache {_short(t.cache_read)} read / "
            f"{_short(t.cache_write)} written"
        )
    if len(t.by_model) > 1:
        line += "   ·   " + ", ".join(
            f"{model} {_short(row['input'] + row['output'])}"
            for model, row in sorted(t.by_model.items())
        )
    console.print(line + "[/dim]")


def render_delta(result: ReviewResult, console: Console) -> None:
    """What changed since the last review of this branch.

    A reviewer that reprints the same list after every push gets muted, so
    anything already reported and still open is one line, not a panel."""
    delta = result.delta
    if delta is None or getattr(delta, "first_run", True):
        return
    bits = []
    if delta.resolved:
        bits.append(f"[green]{len(delta.resolved)} fixed since last review[/green]")
    if delta.persisting:
        bits.append(f"[dim]{len(delta.persisting)} still open from before[/dim]")
    if bits:
        console.print("  " + "   ·   ".join(bits))


def render(result: ReviewResult, request: ReviewRequest, console: Console | None = None) -> None:
    console = console or Console()

    if not result.accepted:
        console.print()
        errors = [p for p in result.problems if p.severity == "error"]
        if errors:
            console.print("  [yellow]No findings reported, but this review did not "
                          "complete cleanly.[/yellow]")
        else:
            console.print("  [green]No findings.[/green]  ", end="")
            console.print(
                f"[dim]{result.usage.get('raw_count', 0):.0f} raw → "
                f"{result.usage.get('merged_count', 0):.0f} merged → 0 confirmed[/dim]"
            )
        render_delta(result, console)
        render_tokens(result, console)
        render_problems(result, console)
        console.print()
        return

    console.print()
    by_file: dict[str, list] = {}
    for f in result.accepted:
        by_file.setdefault(f.file, []).append(f)

    for path, findings in by_file.items():
        console.print(f"  [bold]{path}[/bold]")
        for f in findings:
            style = _SEVERITY_STYLE[f.severity]
            head = Text()
            head.append(f"{f.severity.upper():<7}", style=style)
            head.append(f"{f.category}", style="dim")
            head.append(f"  line {f.line}", style="dim")
            body = Text()
            body.append(f.summary + "\n\n", style="bold")
            # The line the claim rests on, so the reader can check it here rather
            # than opening the file to find out what the finding is even about.
            quoted = cited_text(f.evidence)
            if quoted:
                for offset, text in enumerate(quoted.splitlines()):
                    body.append(f"{f.line + offset:>5}| ", style="dim")
                    body.append(text + "\n", style="cyan")
                body.append("\n")
            body.append("Fails when: ", style="dim")
            body.append(f.failure_scenario)
            if f.fix:
                if f.fix.note:
                    body.append("\n\nFix: ", style="dim")
                    body.append(f.fix.note)
                span = (
                    f"line {f.fix.start_line}"
                    if f.fix.start_line == f.fix.end_line
                    else f"lines {f.fix.start_line}-{f.fix.end_line}"
                )
                body.append(f"\n\nReplace {span} with:\n", style="dim")
                body.append(f.fix.replacement, style="green")
            console.print(Panel(body, title=head, title_align="left", border_style="dim"))
        console.print()

    render_delta(result, console)
    u = result.usage
    console.print(
        f"  [dim]{u.get('raw_count', 0):.0f} raw → {u.get('merged_count', 0):.0f} merged → "
        f"{u.get('accepted_count', 0):.0f} reported"
        + (f"   ·   gate rejected {u.get('rejection_rate', 0):.0%}" if result.verified else "")
        + (
            f"   ·   {u.get('dropped_files', 0):.0f} file(s) over budget"
            if u.get("dropped_files")
            else ""
        )
        + (
            f"   ·   read {u.get('references', 0):.0f} called definition(s)"
            if u.get("references")
            else ""
        )
        + "[/dim]"
    )
    render_tokens(result, console)
    render_problems(result, console)
    console.print()


def to_review_comments(result: ReviewResult) -> list[dict]:
    comments = []
    for f in result.accepted:
        body = (
            f"**{f.severity} · {f.category}** — {f.summary}\n\n"
            f"**Fails when:** {f.failure_scenario}"
        )
        comment: dict = {"path": f.file, "line": f.line, "side": "RIGHT"}

        if f.fix:
            if f.fix.note:
                body += f"\n\n{f.fix.note}"
            # A ```suggestion block renders in the PR as a one-click apply. GitHub
            # applies it to exactly the lines the comment spans, so the comment has
            # to be anchored to the fix's range, not to the finding's single line.
            body += f"\n\n```suggestion\n{f.fix.replacement.rstrip()}\n```"
            comment["line"] = f.fix.end_line
            if f.fix.end_line != f.fix.start_line:
                comment["start_line"] = f.fix.start_line
                comment["start_side"] = "RIGHT"

        comment["body"] = body
        comments.append(comment)
    return comments


def post_to_github(result: ReviewResult, request: ReviewRequest, token: str | None = None) -> str:
    """One batched review with inline comments — not N separate comments."""
    if request.repo is None or request.pr_number is None:
        raise ValueError("Not a GitHub review — no repo/pr_number on the request.")
    token = token or os.getenv("GITHUB_TOKEN") or ""
    if not token:
        raise RuntimeError("GITHUB_TOKEN is not set — posting needs pull_requests:write.")

    comments = to_review_comments(result)
    payload = {
        "commit_id": request.head_sha,
        "event": "COMMENT",
        "body": (
            f"Automated review — {len(comments)} finding(s) after verification."
            if comments
            else "Automated review — no findings."
        ),
        "comments": comments,
    }
    resp = httpx.post(
        f"https://api.github.com/repos/{request.repo}/pulls/{request.pr_number}/reviews",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json=payload,
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json().get("html_url", "")
