"""Command line interface."""

from __future__ import annotations

import os

import typer
from rich.console import Console
from rich.markup import escape

from .adapters import github_pr, local_git
from .adapters.local_git import GitError
from .config import MODE, MODEL, PROVIDER, RPM, TRACING, credentials_present, key_env_var
from .emit import post_to_github, render, render_json
from .logs import setup_logging
from .privacy import assurance, destinations
from .runner import run_review
from .schema import ReviewResult

app = typer.Typer(add_completion=False, help="Multi-agent code reviewer.")
console = Console()


def _logging_for(output: str, verbose: int) -> None:
    """Point the log handler at the same Console the spinners use.

    Rich coordinates a live region with its own Console and nothing else. Handed a
    second Console — which is what `setup_logging` builds when nobody passes it one
    — log lines go straight to the terminal at wherever the cursor happens to be,
    which is the middle of a spinner. In json mode they stay on stderr, because
    stdout has to hold one parseable document and nothing else.
    """
    setup_logging(verbose, console=console if output == "text" else None)


def _preflight() -> None:
    """Fail before spending time on ingestion, and with a message that says what
    to do — a missing key otherwise surfaces as a stack trace from inside a node."""
    if not credentials_present():
        console.print(
            f"[red]No {PROVIDER} credential.[/red] Set {key_env_var()} in the "
            "reviewer's .env (copy .env.example) or export it in your shell.\n"
            "[dim]Switch providers with LOUPE_PROVIDER=google|anthropic.[/dim]"
        )
        raise typer.Exit(1)

Mode = typer.Option("multi", help="'multi' runs the four-specialist panel; 'single' the baseline.")
NoVerify = typer.Option(False, "--no-verify", help="Skip the checking pass. For measurement.")
Fresh = typer.Option(
    False, "--fresh",
    help="Ignore what was reported before and show everything again.",
)
Verbose = typer.Option(
    0, "--verbose", "-v", count=True,
    help="-v shows what each stage did; -vv adds debug; -vvv adds HTTP traffic.",
)
Output = typer.Option("text", "--output", help="Output format: text or json.")


def _explain_nothing(request, target: str) -> None:
    """"Nothing to review" has two causes that need different fixes: an empty
    diff, or a diff whose files were all filtered out as non-source."""
    if not request.files:
        console.print(f"[yellow]No changes found in {target}.[/yellow]")
        console.print(
            "[dim]`loupe local <ref>` diffs a ref against your working tree, so "
            "HEAD means uncommitted changes only. Try `loupe local HEAD~1` for "
            "the last commit, or --staged for the index.[/dim]"
        )
        return

    console.print(
        f"[yellow]{len(request.files)} file(s) changed in {target}, but none are "
        "reviewable.[/yellow]"
    )
    for path in request.skipped:
        console.print(f"  [dim]skipped[/dim] {escape(path)}")
    console.print(
        "[dim]Docs, lockfiles, generated and vendored files are skipped — see "
        "filters.py. Binary and deleted files are skipped too.[/dim]"
    )


@app.command()
def local(
    ref: str = typer.Argument("HEAD~1", help="Ref to diff against."),
    staged: bool = typer.Option(False, "--staged", help="Review the index instead."),
    untracked: bool = typer.Option(
        True, "--untracked/--no-untracked", help="Include untracked source files."
    ),
    repo_root: str = typer.Option(".", "--repo-root"),
    mode: str = Mode,
    no_verify: bool = NoVerify,
    fresh: bool = Fresh,
    output: str = Output,
    verbose: int = Verbose,
) -> None:
    """Review a local diff."""
    if output not in {"text", "json"}:
        raise typer.BadParameter("--output must be text or json")
    _logging_for(output, verbose)
    _preflight()
    try:
        request = local_git.load(
            ref=ref, repo_root=repo_root, staged=staged, include_untracked=untracked
        )
    except GitError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    # Everything below that is written for a person goes to stdout, which in json
    # mode has to hold one parseable document and nothing else.
    talking = output == "text"

    if not request.reviewable:
        if talking:
            _explain_nothing(request, "staged changes" if staged else f"{ref}..working tree")
        else:
            # A machine asked a question and is owed an answer in the format it
            # asked for. "Nothing to review" is a result, not an error.
            render_json(ReviewResult(request_ref=ref, mode=mode, verified=not no_verify),
                        request, console)
        raise typer.Exit(0)

    if talking:
        if request.skipped:
            console.print(f"[dim]Skipping {len(request.skipped)} non-source file(s): "
                          f"{', '.join(escape(p) for p in request.skipped[:4])}"
                          f"{'…' if len(request.skipped) > 4 else ''}[/dim]")
        console.print(
            f"[dim]Reviewing {len(request.reviewable)} file(s) · mode={mode} · "
            f"{PROVIDER}/{MODEL}[/dim]"
        )
    result = run_review(
        request, mode=mode, verify=not no_verify, remember=not fresh,
        # A spinner reading "Reviewing…" for two minutes cannot tell a clean
        # review apart from one where every stage failed quietly. Off for JSON:
        # narration on stdout would be output nobody can parse.
        progress=console if output == "text" else None,
    )
    # Branched rather than picked with a ternary: the two renderers do not take
    # the same arguments, and a ternary that calls whichever it chose with one
    # signature type-checks fine and fails on every json run.
    if output == "json":
        render_json(result, request, console)
    else:
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
    output: str = Output,
    verbose: int = Verbose,
) -> None:
    """Review a GitHub pull request."""
    _logging_for(output, verbose)
    if MODE == "offline":
        console.print("[red]PR review is unavailable in offline mode.[/red] Use `loupe local`.")
        raise typer.Exit(1)
    if output not in {"text", "json"}:
        raise typer.BadParameter("--output must be text or json")
    _preflight()
    request = github_pr.load(repo, number)
    talking = output == "text"

    if not request.reviewable:
        if talking:
            _explain_nothing(request, f"{repo}#{number}")
        else:
            render_json(ReviewResult(request_ref=f"{repo}#{number}", mode=mode,
                                     verified=not no_verify), request, console)
        raise typer.Exit(0)

    if talking:
        console.print(f"[dim]Reviewing {repo}#{number} · {len(request.reviewable)} file(s)[/dim]")
    result = run_review(
        request, mode=mode, verify=not no_verify, run_name=f"{repo}#{number}",
        progress=console if talking else None,
    )
    if talking:
        render(result, request, console)
    else:
        render_json(result, request, console)

    if not post:
        if talking:
            console.print("[dim]Dry run — nothing posted. Pass --post to publish.[/dim]")
        return
    if not result.accepted:
        if talking:
            console.print("[dim]No findings; nothing to post.[/dim]")
        return

    typer.confirm(
        f"Post {len(result.accepted)} inline comment(s) to {repo}#{number}?", abort=True
    )
    url = post_to_github(result, request)
    console.print(f"[green]Posted:[/green] {url}")


@app.command()
def init(
    repo_root: str = typer.Option(".", "--repo-root"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files."),
) -> None:
    """Create a .loupe/ folder so this repo carries its own review settings."""
    from .project import scaffold

    directory, written, skipped = scaffold(repo_root, force=force)
    rel = os.path.relpath(directory, os.getcwd())
    for name in written:
        console.print(f"  [green]created[/green] {rel}/{name}")
    for name in skipped:
        console.print(f"  [dim]kept[/dim]    {rel}/{name} [dim](--force to replace)[/dim]")
    if written:
        console.print(
            f"\n[dim]Edit {rel}/rules.md with the mistakes this codebase actually "
            "makes — that is the file worth writing.[/dim]"
        )


@app.command()
def doctor(
    egress: bool = typer.Option(
        False, "--egress", help="Show every destination a review may contact."
    ),
) -> None:
    """Show what is configured, before spending anything finding out."""
    ok = credentials_present()
    console.print(f"  provider      {PROVIDER}")
    console.print(f"  model         {MODEL}")
    console.print(
        f"  {key_env_var():<13} " + ("[green]set[/green]" if ok else "[red]missing[/red]")
    )
    console.print(f"  rate limit    {f'{RPM} req/min (client-side)' if RPM else 'none'}")
    has_ls = bool(os.getenv("LANGSMITH_API_KEY"))
    console.print(
        f"  mode          {MODE}\n"
        f"  tracing       {'on' if TRACING and has_ls else 'off'}"
        + ("" if has_ls or not TRACING else "  [yellow](LOUPE_TRACING set but no key)[/yellow]")
    )
    console.print(f"  privacy       {assurance()}")
    if egress:
        console.print("\n  [bold]Possible egress[/bold]")
        for item in destinations():
            state = "[green]enabled[/green]" if item.enabled else "[dim]disabled[/dim]"
            console.print(
                f"  {state:<19} {item.destination}\n"
                f"                    [dim]{item.data}[/dim]"
            )
    if not ok:
        raise typer.Exit(1)


@app.command()
def pipeline() -> None:
    """Print loupe's own review pipeline as mermaid.

    Named `pipeline`, not `graph`: a code reviewer that indexes the repository it
    is reviewing will want `graph` to mean that repository's call graph, and
    having one name mean both is how people run the wrong command.
    """
    from .graph import build_graph

    print(build_graph().get_graph().draw_mermaid())


if __name__ == "__main__":
    app()
