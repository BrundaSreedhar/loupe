"""Eval runner.

Scores a reviewer configuration against a corpus of seeded defects and clean
controls. Arms vary two switches — `multi`/`single` reviewers and the verification
gate on/off — so the contribution of each can be read separately.

    python -m evals.run_eval --source ~/some-repo --arms all
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from loupe.config import (
    GROUNDING,
    INDEX,
    MODEL,
    PROVIDER,
    RPM,
    MissingCredential,
    require_credentials,
)
from loupe.quota import AuthenticationFailed, DailyQuotaExhausted
from loupe.runner import run_review
from loupe.schema import ReviewRequest

from .corpus import Case, build
from .scoring import Report, score_case, spread
from .seedset import describe, freeze, load, needs_source_repo

SEED_SET = Path("evals/seed")

app = typer.Typer(add_completion=False)
console = Console()

ARMS: dict[str, tuple[str, bool]] = {
    "multi+verify": ("multi", True),
    "multi": ("multi", False),
    "single+verify": ("single", True),
    "single": ("single", False),
}

# Rough model calls per review, per arm: warm + reviewers + merges + verifications.
# Deliberately on the high side — the point is to catch a run that will blow a
# daily quota before it starts, not to predict spend to the call.
# Consensus and the lint pre-pass are configured by environment, not by arm, so
# these are the baseline costs with both off.
CALLS_PER_REVIEW: dict[str, int] = {
    "multi+verify": 11,
    "multi": 6,
    "single+verify": 4,
    "single": 1,
}


def estimate_calls(selected: list[str], n_cases: int, repeats: int) -> int:
    return sum(CALLS_PER_REVIEW[a] for a in selected) * n_cases * repeats


def run_arm(cases: list[Case], mode: str, verify: bool, workers: int) -> Report:
    report = Report()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                run_review,
                case.request,
                mode=mode,
                verify=verify,
                run_name=f"eval:{mode}{'+v' if verify else ''}:{case.id}",
                # The mutated file exists only in this process. A linter would
                # read the unmutated one off disk and report on code the reviewer
                # was never shown, which is not a pre-pass, it is contamination.
                lint=False,
            ): case
            for case in cases
        }
        for fut in as_completed(futures):
            case = futures[fut]
            try:
                result = fut.result()
            except (DailyQuotaExhausted, AuthenticationFailed) as exc:
                # Two different walls, one correct response. A rejected key is
                # not discoverable before the first call the way a missing one
                # is, so it arrives here — and like an exhausted quota it is true
                # for every case still queued.
                label = (
                    "Out of quota" if isinstance(exc, DailyQuotaExhausted)
                    else "Credential rejected"
                )
                console.print(
                    f"[red]{label}:[/red] {exc}\n"
                    "Stopping. The remaining cases would fail identically, and "
                    "grinding through them turns a 2-minute failure into 20."
                )
                for pending in futures:
                    pending.cancel()
                break
            except Exception as exc:  # noqa: BLE001 — one bad case must not
                # abort an eval run that costs real money.
                console.print(f"[red]{case.id} failed:[/red] {type(exc).__name__}: {exc}")
                continue
            report.scores.append(score_case(case, result))
    return report


def check_did_work(results: dict[str, list[Report]]) -> bool:
    """Refuse to present numbers from a run that never reviewed anything."""
    ok = True
    for arm, reps in results.items():
        for i, r in enumerate(reps, start=1):
            if not r.scores:
                # The loudest failure of all, and the one the first version of
                # this guard skipped: every case raised, so there is nothing to
                # score, and every rate below is 0/0 rendered as a confident 0%.
                console.print(
                    f"[red]{arm} run {i}: every case failed — no result was "
                    "produced at all.[/red] Common cause: the daily quota is "
                    "exhausted. Re-run `loupe doctor` and check your limits."
                )
                ok = False
            elif r.fell_back:
                console.print(
                    f"[red]{arm} run {i}: {r.fell_back} of {len(r.scores)} cases "
                    "fell back to the spare model mid-run.[/red] These numbers mix "
                    "two models and measure neither. Unset LOUPE_FALLBACK_MODEL "
                    "and re-run when the quota resets."
                )
                ok = False
            elif r.reviewed == 0:
                console.print(
                    f"[red]{arm} run {i}: none of the {len(r.scores)} cases were "
                    "actually reviewed — every file was filtered out before any "
                    "model call.[/red] The numbers below are not a measurement."
                )
                ok = False
            elif r.scores and r.reviewed < len(r.scores):
                console.print(
                    f"[yellow]{arm} run {i}: {len(r.scores) - r.reviewed} of "
                    f"{len(r.scores)} cases were never shown to a reviewer.[/yellow]"
                )
    return ok


def render_comparison(results: dict[str, list[Report]]) -> None:
    table = Table(title="Arm comparison", title_style="bold", header_style="dim")
    table.add_column("arm")
    table.add_column("detection", justify="right")
    # The same catches, counted again but only when filed under the right
    # category. A wide gap means the headline column is being carried by
    # coincidence — something was flagged near the line for the wrong reason.
    table.add_column("right reason", justify="right")
    table.add_column("FP / clean diff", justify="right")
    table.add_column("silent on clean", justify="right")
    table.add_column("gate rejected", justify="right")
    table.add_column("merged away", justify="right")
    # The cost half of the trade. Detection alone cannot say whether an arm is
    # worth running — four reviewers that find one more defect for four times the
    # tokens is a different answer from four that find it for the same spend.
    table.add_column("tokens/review", justify="right")
    table.add_column("cache hit", justify="right")

    for arm, reps in results.items():
        det_m, det_s = spread([r.detection_rate for r in reps])
        strict_m, _ = spread([r.strict_detection_rate for r in reps])
        fp_m, fp_s = spread([r.fp_per_clean for r in reps])
        sil_m, _ = spread([r.clean_silence_rate for r in reps])
        rej_m, _ = spread([r.rejection_rate for r in reps])
        mrg_m, _ = spread([r.merge_rate for r in reps])
        tok_m, tok_s = spread([r.tokens_per_review for r in reps])
        cache_m, _ = spread([r.cache_hit_rate for r in reps])
        pm = len(reps) > 1
        # "0 tokens" would read as free rather than as unrecorded.
        metered = all(r.metered for r in reps)
        table.add_row(
            arm,
            f"{det_m:.0%}" + (f" ±{det_s:.0%}" if pm else ""),
            f"{strict_m:.0%}",
            f"{fp_m:.2f}" + (f" ±{fp_s:.2f}" if pm else ""),
            f"{sil_m:.0%}",
            f"{rej_m:.0%}" if "verify" in arm else "—",
            f"{mrg_m:.0%}",
            (f"{tok_m / 1000:.1f}k" + (f" ±{tok_s / 1000:.1f}k" if pm else ""))
            if metered else "not reported",
            f"{cache_m:.0%}" if metered else "—",
        )
    console.print()
    console.print(table)
    console.print()


@app.command()
def main(
    source: Path = typer.Option(
        Path("."), "--source",
        help="Repo to draw corpus source from. Ignored with --seed-set.",
    ),
    arms: str = typer.Option("multi+verify", "--arms", help="Comma-separated, or 'all'."),
    n_defect: int = typer.Option(20, "--n-defect"),
    n_clean: int = typer.Option(20, "--n-clean"),
    n_crossfile: int = typer.Option(
        0, "--n-crossfile",
        help="Defects visible only from another file — the measurement for "
             "repository expansion. Python callers only.",
    ),
    seed_set: bool = typer.Option(
        False, "--seed-set",
        help="Score the frozen corpus in evals/seed instead of generating one. The "
             "only way two runs weeks apart are comparable.",
    ),
    py_mutators: bool = typer.Option(
        True, "--py-mutators/--no-py-mutators",
        help="Include the parser-based Python defect mutators. Off for comparison "
             "against a run seeded only by the line-based ones.",
    ),
    repeats: int = typer.Option(1, "--repeats", help="Repeat runs to measure the noise floor."),
    workers: int = typer.Option(4, "--workers"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(Path("evals/results"), "--out"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build the corpus and stop."),
) -> None:
    selected = list(ARMS) if arms == "all" else [a.strip() for a in arms.split(",")]
    unknown = [a for a in selected if a not in ARMS]
    if unknown:
        raise typer.BadParameter(f"Unknown arm(s): {unknown}. Choose from {list(ARMS)}.")

    if seed_set:
        cases, manifest = load(SEED_SET)
        origin = manifest.get("source", "?")
        commit = manifest.get("source_commit") or "unknown commit"
        console.print(
            f"[dim]Frozen seed set: {len(cases)} case(s) from {SEED_SET}, cut from "
            f"{origin} @ {commit} on {manifest.get('frozen_at', '?')[:10]}[/dim]"
        )
        absent = needs_source_repo(cases) if not Path(origin).is_dir() else []
        if absent:
            # These carry the calling file but not the callee's, so without the
            # source checkout they score the reviewer with its context removed —
            # which is indistinguishable from a reviewer that missed them.
            console.print(
                f"[yellow]{len(absent)} cross-file case(s) need {origin} on disk "
                "and it is not there.[/yellow] They will be scored without "
                "repository context, which understates detection. Drop them or "
                "restore the checkout."
            )
    else:
        cases = build(
            source.expanduser().resolve(),
            n_defect=n_defect,
            n_clean=n_clean,
            seed=seed,
            n_crossfile=n_crossfile,
            python_mutators=py_mutators,
        )
    n_d = sum(1 for c in cases if c.kind == "defect")
    n_c = len(cases) - n_d
    n_x = sum(1 for c in cases if c.id.startswith("defect-x"))
    if not seed_set:
        console.print(
            f"[dim]Corpus: {n_d} seeded defects ({n_x} cross-file), {n_c} clean "
            f"controls, from {source}[/dim]"
        )
    # Both switches change what is being measured, and neither is visible in the
    # table below. A result file that does not say which run it was is a result
    # nobody can compare against anything.
    console.print(
        f"[dim]Citations {'on' if GROUNDING else 'off'} · repository expansion "
        f"{INDEX} · parser-based mutators {'on' if py_mutators else 'off'} · "
        f"lint pre-pass off (the corpus is in memory)[/dim]"
    )

    if not seed_set and (n_d - n_x < n_defect or n_c < n_clean):
        console.print(
            f"[yellow]Note:[/yellow] asked for {n_defect}/{n_clean}, got "
            f"{n_d - n_x}/{n_c} — mutators found fewer candidate sites than "
            "requested in this source tree."
        )
    if not seed_set and n_x < n_crossfile:
        console.print(
            f"[yellow]Note:[/yellow] asked for {n_crossfile} cross-file case(s), "
            f"got {n_x}. They need a Python call whose function is defined in "
            "another file of the same repository, with a plain signature and at "
            "least two arguments. A number built on very few of these is noise, "
            "not a measurement."
        )

    if dry_run:
        by_mutator: dict[str, int] = {}
        for c in cases:
            key = c.truth.name if c.truth else "clean"
            by_mutator[key] = by_mutator.get(key, 0) + 1
        console.print(json.dumps(by_mutator, indent=2))
        return

    # After the dry run, which builds a corpus and makes no calls, and before the
    # estimate below promises a number this run cannot produce. Checked once here
    # rather than left to the first review: without it a missing key is found
    # separately by every case, and the run ends having measured nothing while
    # looking like a reviewer that found nothing.
    try:
        require_credentials()
    except MissingCredential as exc:
        console.print(f"[red]{exc}[/red]")
        console.print(
            f"[dim]Nothing was run. {len(selected) * repeats * len(cases)} review(s) "
            "skipped.[/dim]"
        )
        raise typer.Exit(1) from exc

    total = len(selected) * repeats * len(cases)
    calls = estimate_calls(selected, len(cases), repeats)
    console.print(
        f"[dim]{len(selected)} arm(s) x {repeats} repeat(s) = {total} reviews "
        f"· ~{calls} model calls · {PROVIDER}/{MODEL}[/dim]"
    )
    if RPM:
        console.print(
            f"[dim]At {RPM} req/min that is roughly {calls / RPM / 60:.1f} hour(s) "
            "of wall clock.[/dim]"
        )
    if calls > 400:
        console.print(
            f"[yellow]~{calls} calls.[/yellow] Free tiers are typically capped in the "
            "low thousands per day and far lower per minute — check your own limits "
            "before committing to this. Shrink with --n-defect/--n-clean, fewer "
            "--arms, or --repeats 1."
        )
        typer.confirm("Continue?", abort=True)

    results: dict[str, list[Report]] = {}
    for arm in selected:
        mode, verify = ARMS[arm]
        reps: list[Report] = []
        for i in range(repeats):
            console.print(f"[dim]  {arm} — run {i + 1}/{repeats}[/dim]")
            reps.append(run_arm(cases, mode, verify, workers))
        results[arm] = reps

    if not check_did_work(results):
        raise typer.Exit(1)
    render_comparison(results)

    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"eval-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "source": str(source),
                "seed": seed,
                "n_defect": n_d,
                "n_crossfile": n_x,
                "n_clean": n_c,
                "grounding": GROUNDING,
                "index": INDEX,
                "python_mutators": py_mutators,
                "seed_set": str(SEED_SET) if seed_set else None,
                "model": MODEL,
                "provider": PROVIDER,
                "arms": {
                    arm: {
                        "runs": [r.as_dict() for r in reps],
                        "by_mutator": reps[0].by_mutator(),
                    }
                    for arm, reps in results.items()
                },
            },
            indent=2,
        )
    )
    console.print(f"[dim]Written to {path}[/dim]")


@app.command()
def freeze_set(
    source: Path = typer.Option(..., "--source", help="Repo to cut the cases from."),
    n_defect: int = typer.Option(20, "--n-defect"),
    n_clean: int = typer.Option(10, "--n-clean"),
    n_crossfile: int = typer.Option(0, "--n-crossfile"),
    seed: int = typer.Option(0, "--seed"),
    out: Path = typer.Option(SEED_SET, "--out"),
) -> None:
    """Generate a corpus once and write it out to be committed.

    Run this rarely. The whole point is that the cases stop moving, so re-freezing
    resets every comparison you have — a result against the old set and one against
    the new set are two different measurements wearing the same name.
    """
    root = source.expanduser().resolve()
    cases = build(root, n_defect=n_defect, n_clean=n_clean, seed=seed, n_crossfile=n_crossfile)
    written = freeze(cases, out, root)

    dist = describe(cases)
    n_d = sum(1 for c in cases if c.kind == "defect")
    console.print(f"[green]Froze {len(written)} case(s)[/green] to {out}")
    console.print(json.dumps(dist, indent=2))

    # A set where one mutator supplies half the defects measures that mutator.
    defects = {k: v for k, v in dist.items() if k != "clean"}
    if defects:
        worst, count = max(defects.items(), key=lambda kv: kv[1])
        # `>=`: at 20 defects the threshold is 6, and `>` let exactly 6 through —
        # which is the lopsided case the check exists to catch.
        if count >= max(3, n_d // 4):
            console.print(
                f"[yellow]{worst} supplies {count} of {n_d} defects.[/yellow] A "
                "lopsided set measures that one defect shape. Re-freeze with a "
                "different --seed, or a source repo with more variety."
            )
    if n_clean == 0:
        console.print(
            "[yellow]No clean controls.[/yellow] Detection without a "
            "false-positive column is not a measurement — a reviewer that flags "
            "every line scores 100%."
        )
    console.print(f"[dim]Commit {out} so the numbers stay comparable.[/dim]")


@app.command()
def push(
    source: Path = typer.Option(..., "--source"),
    dataset: str = typer.Option("code-review-seeded-bugs", "--dataset"),
    n_defect: int = typer.Option(20, "--n-defect"),
    n_clean: int = typer.Option(20, "--n-clean"),
    n_crossfile: int = typer.Option(
        0, "--n-crossfile",
        help="Defects visible only from another file — the measurement for "
             "repository expansion. Python callers only.",
    ),
    seed: int = typer.Option(0, "--seed"),
) -> None:
    """Push the corpus to LangSmith as a dataset, so experiments are comparable there."""
    from langsmith import Client

    if not os.getenv("LANGSMITH_API_KEY"):
        raise typer.BadParameter("LANGSMITH_API_KEY is not set.")

    cases = build(source.expanduser().resolve(), n_defect=n_defect, n_clean=n_clean, seed=seed)
    client = Client()
    if client.has_dataset(dataset_name=dataset):
        ds = client.read_dataset(dataset_name=dataset)
    else:
        ds = client.create_dataset(
            dataset_name=dataset,
            description="Seeded code defects plus semantics-preserving controls.",
        )

    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {
                "inputs": {"case_id": c.id, "request": c.request.model_dump(mode="json")},
                "outputs": {
                    "kind": c.kind,
                    "line": c.truth.line if c.truth else None,
                    "mutator": c.truth.name if c.truth else None,
                },
            }
            for c in cases
        ],
    )
    console.print(f"[green]Pushed {len(cases)} examples to '{dataset}'.[/green]")


def langsmith_target(inputs: dict) -> dict:
    """Target for `client.evaluate` against the pushed dataset."""
    request = ReviewRequest.model_validate(inputs["request"])
    result = run_review(request, mode="multi", verify=True)
    return {
        "findings": [
            {"file": f.file, "line": f.line, "summary": f.summary} for f in result.accepted
        ]
    }


def detected_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Recall on seeded defects. Mechanical — no judge, no drift."""
    if reference_outputs.get("kind") != "defect":
        return {"key": "detected", "score": None}
    target = reference_outputs["line"]
    hit = any(abs(f["line"] - target) <= 3 for f in outputs.get("findings", []))
    return {"key": "detected", "score": int(hit)}


def noise_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """False positives on semantics-preserving controls."""
    if reference_outputs.get("kind") != "clean":
        return {"key": "false_positives", "score": None}
    return {"key": "false_positives", "score": len(outputs.get("findings", []))}


if __name__ == "__main__":
    app()
