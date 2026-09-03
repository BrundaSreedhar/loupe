"""Command line interface."""

from __future__ import annotations

import os

import typer
from rich.console import Console

from .adapters import github_pr, local_git
from .adapters.local_git import GitError
from .config import MODEL, PROVIDER, RPM, credentials_present, key_env_var
from .emit import post_to_github, render
from .logs import setup_logging
from .runner import run_review

app = typer.Typer(add_completion=False, help="Multi-agent code reviewer.")
console = Console()


def _preflight() -> None:
    """Fail before spending time on ingestion, and with a message that says what
    to do — a missing key otherwise surfaces as a stack trace from inside a node."""
    if not credentials_present():
        console.print(
            f"[red]No {PROVIDER} credential.[/red] Set {key_env_var()} in the "
            "reviewer's .env (copy .env.example) or export it in your shell.\n"
            "[dim]Switch providers with REVIEWER_PROVIDER=google|anthropic.[/dim]"
        )
        raise typer.Exit(1)

Mode = typer.Option("multi", help="'multi' runs the four-specialist panel; 'single' the baseline.")
NoVerify = typer.Option(False, "--no-verify", help="Skip the checking pass. For measurement.")
Verbose = typer.Option(
    0, "--verbose", "-v", count=True,
    help="-v shows what each stage did; -vv adds debug; -vvv adds HTTP traffic.",
)


@app.command()
def local(
    ref: str = typer.Argument("HEAD~1", help="Ref to diff against."),
    staged: bool = typer.Option(False, "--staged", help="Review the index instead."),
    repo_root: str = typer.Option(".", "--repo-root"),
    mode: str = Mode,
    no_verify: bool = NoVerify,
    verbose: int = Verbose,
) -> None:
    """Review a local diff."""
    setup_logging(verbose)
    _preflight()
    try:
        request = local_git.load(ref=ref, repo_root=repo_root, staged=staged)
    except GitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    if not request.reviewable:
        console.print("[yellow]Nothing to review.[/yellow]")
        raise typer.Exit(0)

    console.print(
        f"[dim]Reviewing {len(request.reviewable)} file(s) · mode={mode} · "
        f"{PROVIDER}/{MODEL}[/dim]"
    )
    if verbose:
        result = run_review(request, mode=mode, verify=not no_verify)
    else:
        with console.status("Reviewing…"):
            result = run_review(request, mode=mode, verify=not no_verify)
    render(result, request, console)


@app.command()
def pr(
    repo: str = typer.Argument(..., help="owner/repo"),
    number: int = typer.Argument(..., help="PR number"),
    mode: str = Mode,
    no_verify: bool = NoVerify,
    post: bool = typer.Option(
        False, "--post", help="Post findings to the PR. Off by default — this writes to GitHub."
    ),
    verbose: int = Verbose,
) -> None:
    """Review a GitHub pull request."""
    setup_logging(verbose)
    _preflight()
    request = github_pr.load(repo, number)
    if not request.reviewable:
        console.print("[yellow]Nothing to review.[/yellow]")
        raise typer.Exit(0)

    console.print(f"[dim]Reviewing {repo}#{number} · {len(request.reviewable)} file(s)[/dim]")
    if verbose:
        result = run_review(request, mode=mode, verify=not no_verify, run_name=f"{repo}#{number}")
    else:
        with console.status("Reviewing…"):
            result = run_review(
                request, mode=mode, verify=not no_verify, run_name=f"{repo}#{number}"
            )
    render(result, request, console)

    if not post:
        console.print("[dim]Dry run — nothing posted. Pass --post to publish.[/dim]")
        return
    if not result.accepted:
        console.print("[dim]No findings; nothing to post.[/dim]")
        return

    typer.confirm(
        f"Post {len(result.accepted)} inline comment(s) to {repo}#{number}?", abort=True
    )
    url = post_to_github(result, request)
    console.print(f"[green]Posted:[/green] {url}")


@app.command()
def doctor() -> None:
    """Show what is configured, before spending anything finding out."""
    ok = credentials_present()
    console.print(f"  provider      {PROVIDER}")
    console.print(f"  model         {MODEL}")
    console.print(
        f"  {key_env_var():<13} " + ("[green]set[/green]" if ok else "[red]missing[/red]")
    )
    console.print(f"  rate limit    {f'{RPM} req/min (client-side)' if RPM else 'none'}")
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes")
    has_ls = bool(os.getenv("LANGSMITH_API_KEY"))
    console.print(
        f"  tracing       {'on' if tracing and has_ls else 'off'}"
        + ("" if has_ls or not tracing else "  [yellow](LANGSMITH_TRACING set but no key)[/yellow]")
    )
    if not ok:
        raise typer.Exit(1)


@app.command()
def graph() -> None:
    """Print the compiled graph as mermaid."""
    from .graph import build_graph

    print(build_graph().get_graph().draw_mermaid())


if __name__ == "__main__":
    app()
