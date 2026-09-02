"""Rendering and PR posting.

Deliberately outside the graph: posting is a side effect, and the eval harness
runs the graph thousands of times.
"""

from __future__ import annotations

import os

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .schema import ReviewRequest, ReviewResult

_SEVERITY_STYLE = {"high": "bold red", "medium": "yellow", "low": "cyan"}


def render(result: ReviewResult, request: ReviewRequest, console: Console | None = None) -> None:
    console = console or Console()

    if not result.accepted:
        console.print()
        console.print("  [green]No findings.[/green]  ", end="")
        console.print(
            f"[dim]{result.usage.get('raw_count', 0):.0f} raw → "
            f"{result.usage.get('merged_count', 0):.0f} merged → 0 confirmed[/dim]"
        )
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
            body.append("Fails when: ", style="dim")
            body.append(f.failure_scenario)
            if f.suggested_fix:
                body.append("\n\nFix: ", style="dim")
                body.append(f.suggested_fix)
            console.print(Panel(body, title=head, title_align="left", border_style="dim"))
        console.print()

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
        + "[/dim]"
    )
    console.print()


def to_review_comments(result: ReviewResult) -> list[dict]:
    comments = []
    for f in result.accepted:
        body = (
            f"**{f.severity} · {f.category}** — {f.summary}\n\n"
            f"**Fails when:** {f.failure_scenario}"
        )
        if f.suggested_fix:
            body += f"\n\n**Suggested fix:** {f.suggested_fix}"
        comments.append({"path": f.file, "line": f.line, "side": "RIGHT", "body": body})
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
